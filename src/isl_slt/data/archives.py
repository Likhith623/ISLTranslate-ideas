"""Seekable archive parts, pooled exact HTTP ranges and bounded selected-member windows."""

import bisect
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import requests
from urllib3.exceptions import HTTPError as BodyError

from isl_slt.utils.io import atomic_open, stream_identity


class _SafeSession(requests.Session):
    def rebuild_auth(self, prepared_request, response):
        super().rebuild_auth(prepared_request, response)
        old, new = urlsplit(response.request.url), urlsplit(prepared_request.url)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            prepared_request.headers.pop("Authorization", None)


class ArchiveHTTP:
    """Command-owned connection pool. Never expose tokens or signed URLs in errors."""
    def __init__(self, token=None):
        self.token, self.session = token, _SafeSession()
        self.requests = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.session.close()

    def get(self, url, *, byte_range=None):
        headers = {"Accept-Encoding": "identity"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        for attempt in range(3):
            response = None
            try:
                self.requests += 1
                response = self.session.get(url, headers=headers, stream=True, timeout=(10, 60))
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                    if response.status_code >= 400:
                        status = response.status_code
                        response.close()
                        raise OSError(f"Archive request failed with HTTP {status}; check access/revision")
                    return response
                retry_after = response.headers.get("Retry-After", "")
                delay = min(60, int(retry_after)) if retry_after.isdigit() else 0.5 * 2**attempt
                response.close()
            except requests.RequestException:
                if response is not None:
                    response.close()
                if attempt == 2:
                    raise OSError("Archive connection failed after bounded retries; retry acquisition") from None
                delay = 0.5 * 2**attempt
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")


@dataclass(frozen=True)
class ArchivePart:
    name: str
    size: int
    identity: dict
    open: Callable[[], io.IOBase]


def local_part(path) -> ArchivePart:
    path = Path(path)
    status = path.stat()
    return ArchivePart(path.name, status.st_size,
        {"path": str(path.resolve()), "size": status.st_size, "mtime_ns": status.st_mtime_ns, "inode": status.st_ino},
        lambda: path.open("rb"))


class _SeekableReader(io.RawIOBase):
    def __init__(self, size):
        super().__init__()
        self.size, self.position = size, 0

    def readable(self):
        self._checkClosed()
        return True

    def seekable(self):
        self._checkClosed()
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END):
            raise ValueError("Invalid archive seek mode")
        position = offset + (0 if whence == io.SEEK_SET else self.position if whence == io.SEEK_CUR else self.size)
        if position < 0:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def _read_size(self, size):
        self._checkClosed()
        remaining = max(0, self.size - self.position)
        return remaining if size is None or size < 0 else min(size, remaining)


class RangeFile(_SeekableReader):
    """Exact ranges only: reject HTTP 200 before reading a full body."""
    def __init__(self, url, size, token=None, *, client=None):
        super().__init__(size)
        self.url, self.client = url, client or ArchiveHTTP(token)
        self.owns_client, self.transferred = client is None, 0

    def read(self, size=-1):
        size = self._read_size(size)
        if not size:
            return b""
        start, end = self.position, self.position + size - 1
        try:
            with self.client.get(self.url, byte_range=(start, end)) as response:
                if (response.status_code != 206 or
                    response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}" or
                    response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise OSError("Remote server did not honor exact byte ranges; explicitly choose --transport staged")
                data = response.raw.read(size + 1)
        except (requests.RequestException, BodyError):
            raise OSError("Remote archive range body failed; retry acquisition") from None
        if len(data) != size:
            raise OSError("Remote archive range changed or was truncated")
        self.position += size
        self.transferred += size
        return data

    def close(self):
        if getattr(self, "owns_client", False):
            self.client.close()
        super().close()


class StagedParts:
    """LRU staging: temporary bytes count toward the two-part limit."""
    def __init__(self, parts, urls, directory, token=None, *, client=None):
        self.parts, self.urls, self.directory = parts, urls, Path(directory)
        self.cached, self.peak_bytes = OrderedDict(), 0
        self.client, self.owns_client = client or ArchiveHTTP(token), client is None
        self.lock = None

    def __enter__(self):
        import fcntl
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".staging.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            if self.owns_client:
                self.client.close()
            raise ValueError("Staging directory is already in use") from None
        for path in self.directory.iterdir():
            if re.fullmatch(r"(?:[0-9a-f]{64}\.part|\.[0-9a-f]{64}\.part\..+)", path.name) and path.is_file():
                path.unlink()
        return self

    def _progress(self, count):
        self.peak_bytes = max(self.peak_bytes, count + sum(p.stat().st_size for p in self.cached.values()))

    def ensure(self, number):
        part = self.parts[number]
        if number in self.cached:
            self.cached.move_to_end(number)
            return self.cached[number]
        while len(self.cached) >= 2:
            _, old = self.cached.popitem(last=False)
            old.unlink()
        name = hashlib.sha256(repr(part.identity).encode()).hexdigest()
        path = self.directory / f"{name}.part"
        if path.exists():
            path.unlink()
        try:
            with self.client.get(self.urls[number]) as response, atomic_open(path) as stream:
                if response.status_code != 200 or response.headers.get("Content-Encoding", "identity") != "identity":
                    raise OSError("Unexpected staged archive response")
                result = stream_identity(response.raw, destination=stream, max_bytes=part.size, on_progress=self._progress)
                if result["bytes"] != part.size or part.identity.get("sha256") not in (None, result["sha256"]):
                    raise OSError("Staged archive size/checksum mismatch")
        except (requests.RequestException, BodyError):
            raise OSError("Staged archive transfer failed; retry acquisition") from None
        self.cached[number] = path
        return path

    def open(self, number):
        owner = self
        class StagedFile(_SeekableReader):
            def __init__(self):
                super().__init__(owner.parts[number].size)
            def read(self, size=-1):
                size = self._read_size(size)
                if not size:
                    return b""
                with owner.ensure(number).open("rb") as stream:
                    stream.seek(self.position)
                    data = stream.read(size)
                self.position += len(data)
                return data
        return StagedFile()

    def __exit__(self, *args):
        try:
            for path in self.cached.values():
                path.unlink(missing_ok=True)
            self.cached.clear()
        finally:
            if self.lock:
                self.lock.close()
            if self.owns_client:
                self.client.close()


class MultipartReader(_SeekableReader):
    """Concatenate sized parts, with at most two 4 MiB selected-member windows."""
    window_bytes = 4 * 1024**2

    def __init__(self, parts):
        self.parts = [part if isinstance(part, ArchivePart) else local_part(part) for part in parts]
        suffixes = [re.search(r"iSign-poses_v1\.1_part_([a-z]{2})$", part.name) for part in self.parts]
        if any(suffixes) and [match.group(1) if match else None for match in suffixes] != ["aa", "ab", "ac", "ad"]:
            raise ValueError("Supply all four iSign pose ZIP parts in aa, ab, ac, ad order")
        if not self.parts or any(part.size <= 0 for part in self.parts):
            raise ValueError("Supply nonempty sized archive parts")
        self.streams, self.offsets = {}, [0]
        for part in self.parts:
            self.offsets.append(self.offsets[-1] + part.size)
        super().__init__(self.offsets[-1])
        self.spans, self.span_starts, self.cache = [], [], OrderedDict()

    @property
    def identity(self):
        return [part.identity for part in self.parts]

    def select_members(self, archive, members):
        """Enable read-ahead only inside consecutive selected ZIP entries."""
        wanted = set(members)
        entries = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
        spans = []
        for i, entry in enumerate(entries):
            if entry.filename not in wanted:
                continue
            start = entry.header_offset
            end = entries[i + 1].header_offset if i + 1 < len(entries) else archive.start_dir
            if not 0 <= start < end <= self.size:
                raise ValueError("Invalid ZIP member offsets")
            if spans and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        self.spans, self.span_starts = spans, [start for start, _ in spans]
        self.cache.clear()

    def _read_at(self, position, size):
        chunks = []
        while size > 0:
            number = bisect.bisect_right(self.offsets, position) - 1
            count = min(size, self.offsets[number + 1] - position)
            if number not in self.streams:
                self.streams[number] = self.parts[number].open()
            stream = self.streams[number]
            stream.seek(position - self.offsets[number])
            block = stream.read(count)
            if len(block) != count:
                raise OSError("ZIP part changed or was truncated during extraction")
            chunks.append(block)
            position += count
            size -= count
        return b"".join(chunks)

    def read(self, size=-1):
        size = self._read_size(size)
        chunks = []
        while size:
            i = bisect.bisect_right(self.span_starts, self.position) - 1
            if i >= 0 and self.position < self.spans[i][1]:
                start, end = self.spans[i]
                block_start = start + ((self.position - start) // self.window_bytes) * self.window_bytes
                block_end = min(end, block_start + self.window_bytes)
                key = (block_start, block_end)
                if key not in self.cache:
                    while len(self.cache) >= 2:
                        self.cache.popitem(last=False)
                    self.cache[key] = self._read_at(block_start, block_end - block_start)
                self.cache.move_to_end(key)
                count = min(size, block_end - self.position)
                block = self.cache[key][self.position - block_start:self.position - block_start + count]
            else:
                count = min(size, self.span_starts[i + 1] - self.position) if i + 1 < len(self.spans) else size
                block = self._read_at(self.position, count)
            chunks.append(block)
            self.position += count
            size -= count
        return b"".join(chunks)

    def close(self):
        for stream in getattr(self, "streams", {}).values():
            stream.close()
        getattr(self, "cache", {}).clear()
        super().close()
