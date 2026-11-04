import hashlib
import io
import zlib

import pytest

from isl_slt.utils.io import atomic_open, file_hash, stream_identity
from isl_slt.data.archives import MultipartReader, RangeFile


@pytest.mark.parametrize("payload", [b"", b"pose" * 400000])
def test_stream_identity_copies_and_counts_without_closing(payload):
    source, destination = io.BytesIO(payload), io.BytesIO()
    progress = []
    result = stream_identity(source, destination=destination, max_bytes=len(payload), on_progress=progress.append)
    assert result == {"bytes": len(payload), "crc32": zlib.crc32(payload) & 0xffffffff,
                      "sha256": hashlib.sha256(payload).hexdigest()}
    assert destination.getvalue() == payload
    assert not source.closed and not destination.closed
    assert (progress[-1] if progress else 0) == len(payload)


def test_stream_limit_and_checksum_failure_do_not_publish(tmp_path):
    path = tmp_path / "asset"
    with pytest.raises(OSError, match="declared size"):
        with atomic_open(path) as destination:
            stream_identity(io.BytesIO(b"oversize"), destination=destination, max_bytes=3)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
    path.write_bytes(b"valid")
    assert file_hash(path) == hashlib.sha256(b"valid").hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        with atomic_open(path) as destination:
            stream_identity(io.BytesIO(b"corrupt"), destination=destination)
            raise ValueError("checksum")
    assert path.read_bytes() == b"valid"


@pytest.mark.parametrize("kind", ["range", "multipart"])
def test_seek_modes_eof_and_closed_streams(tmp_path, kind):
    path = tmp_path / "part"
    path.write_bytes(b"abc")
    reader = RangeFile("http://unused", 3) if kind == "range" else MultipartReader([path])
    with reader:
        assert reader.readable() and reader.seekable()
        assert reader.seek(1) == 1
        assert reader.seek(1, io.SEEK_CUR) == 2
        assert reader.seek(-1, io.SEEK_END) == 2
        assert reader.seek(5) == 5
        assert reader.read() == b""
        with pytest.raises(ValueError):
            reader.seek(-1)
        with pytest.raises(ValueError):
            reader.seek(0, 9)
    with pytest.raises(ValueError):
        reader.tell()
