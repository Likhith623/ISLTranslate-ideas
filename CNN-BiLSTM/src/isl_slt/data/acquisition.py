"""Explicit acquisition and selective extraction from one ZIP split into ordered byte parts."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile
from contextlib import ExitStack

from isl_slt.config import Features, load_config
from isl_slt.data.manifests import Record, adapt_csv, load_manifest
from isl_slt.data.pose import decode_pose
from isl_slt.features.pose import selected_joints
from isl_slt.utils.io import atomic_open, directory_lock, file_hash, fingerprint, read_json, stream_identity, write_json


from isl_slt.data.archives import ArchiveHTTP, ArchivePart, MultipartReader, RangeFile, StagedParts


def pose_index(archive: zipfile.ZipFile) -> dict:
    index = {}
    for entry in archive.infolist():
        name = PurePosixPath(entry.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in entry.filename or "\x00" in entry.filename:
            raise ValueError("Unsafe archive member path")
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError("Archive symlinks are not accepted")
        if entry.is_dir() or name.suffix.lower() != ".pose":
            continue
        uid = name.stem
        if uid in index:
            raise ValueError(f"Ambiguous pose UID in archive: {uid}")
        index[uid] = {"member": entry.filename, "bytes": entry.file_size, "compressed_bytes": entry.compress_size,
                      "offset": entry.header_offset, "crc32": entry.CRC}
    return index


def inspect_pose_bytes(data: bytes, settings: Features | None = None) -> dict:
    from pose_format import Pose
    settings = settings or Features()
    pose = Pose.read(data)
    record = Record(uid="inspection", dataset_name="inspection", dataset_revision="provided-bytes",
                    text="", pose_path="provided.pose")
    sequence = decode_pose(pose, record, settings)
    required = selected_joints(settings)
    missing = [list(name) for name in required if name not in sequence.names]
    if missing:
        raise ValueError(f"Required pose landmarks are absent: {missing}")
    return {"frames": len(sequence.times), "fps": sequence.metadata["fps"],
            "width": sequence.width, "height": sequence.height,
            "components": sequence.metadata["components"], "points": [list(name) for name in sequence.names],
            "stored_components": [{"name": component.name, "format": component.format,
                                   "points": list(map(str, component.points))} for component in pose.header.components]}


def plan_selection(parts, records, *, target_records=None, budget_bytes=None, seed=2026):
    records = list(records)
    if not records or len({r.uid for r in records}) != len(records):
        raise ValueError("Planning requires nonempty unique annotation UIDs")
    excluded = []
    if target_records is not None:
        from isl_slt.data.splits import pretraining_candidates
        if budget_bytes is None or budget_bytes <= 0:
            raise ValueError("Pretraining selection requires an explicit positive --budget-bytes")
        target_records = list(target_records)
        if not target_records or any(not r.source_id or not r.source_verified for r in target_records):
            raise ValueError("Pretraining planning requires complete authoritative target source exclusions")
        records, excluded = pretraining_candidates(records, target_records)
    with MultipartReader(parts) as combined, zipfile.ZipFile(combined) as archive:
        index = pose_index(archive)
        missing = [r.uid for r in records if r.uid not in index]
        if missing:
            raise ValueError(f"Requested UIDs missing from archive: {missing}")
        identity = combined.identity
        archive_bytes = combined.offsets[-1]
        central_bytes = sum(46 + len(row.filename.encode()) + len(row.extra) + len(row.comment) for row in archive.infolist())
    if target_records is not None:
        groups = {}
        for record in records:
            groups.setdefault(record.source_id, []).append(record)
        selected, used = [], 0
        for source in sorted(groups, key=lambda source: fingerprint([seed, source])):
            rows = groups[source]
            size = sum(index[r.uid]["compressed_bytes"] for r in rows)
            if used + size <= budget_bytes:
                selected.extend(rows)
                used += size
        records = selected
        if not records:
            raise ValueError("No eligible complete source group fits the byte budget")
    records.sort(key=lambda r: index[r.uid]["offset"])
    selected = {r.uid: index[r.uid] for r in records}
    report = {"format": "isl-slt-acquisition-plan-v1", "archive_identity": identity,
              "archive_bytes": archive_bytes, "central_directory_bytes": central_bytes,
              "selected_count": len(records), "selected_sources": len({r.source_id for r in records}),
              "compressed_bytes": sum(row["compressed_bytes"] for row in selected.values()),
              "uncompressed_bytes": sum(row["bytes"] for row in selected.values()),
              "selection_hash": fingerprint(sorted(r.uid for r in records)), "selected_uids": [r.uid for r in records],
              "selected_members": selected, "excluded": excluded, "seed": seed, "budget_bytes": budget_bytes,
              "staging_peak_bound_bytes": sum(sorted((row["size"] for row in identity), reverse=True)[:2]),
              "transport_overhead": "ZIP end/index reads and selected local headers; no full-part range download"}
    return records, report


def annotation_request(config, path, dataset="target"):
    adapter = config.data.target if dataset == "target" else config.data.pretrain
    if adapter is None:
        raise ValueError("No pretraining adapter is configured")
    adapter = adapter.model_copy(update={"annotations": path, "metadata_csv": None, "corrections_jsonl": None,
        "source_mapping": adapter.source_mapping if adapter.source_mapping and Path(adapter.source_mapping).is_file() else None,
        "dataset_revision": adapter.dataset_revision if adapter.dataset_revision != "REQUIRED" else "unverified-extraction-request"})
    return adapt_csv(adapter)


def remote_parts(repo, revision, token=None, *, client=None):
    from huggingface_hub import HfApi, hf_hub_url
    info = HfApi(token=token).dataset_info(repo, revision=revision, files_metadata=True)
    files = {row.rfilename: row for row in info.siblings}
    parts, urls = [], []
    for suffix in ("aa", "ab", "ac", "ad"):
        name = f"iSign-poses_v1.1_part_{suffix}"
        if name not in files or files[name].size is None:
            raise ValueError("Expected sized pose parts absent; inspect explicit --list output")
        row = files[name]
        url = hf_hub_url(repo, name, repo_type="dataset", revision=info.sha)
        lfs = row.lfs
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        identity = {"repository": repo, "revision": info.sha, "name": name, "size": row.size,
                    "blob_id": row.blob_id, "sha256": sha}
        parts.append(ArchivePart(name, row.size, identity, lambda url=url, size=row.size: RangeFile(url, size, token, client=client)))
        urls.append(url)
    return parts, urls


def file_identity(path):
    with Path(path).open("rb") as stream:
        return stream_identity(stream)


def extract_selected(parts, records, output, *, settings=None, max_bytes=50 * 1024**3,
                     inspect_sample=True, resume=False, selection_sha256=None) -> dict:
    records = list(records)
    if not records or len({row.uid for row in records}) != len(records):
        raise ValueError("Extraction needs nonempty unique manifest UIDs")
    output = Path(output)
    with directory_lock(output, ".extraction.lock"), MultipartReader(parts) as combined, zipfile.ZipFile(combined) as archive:
        index = pose_index(archive)
        missing = [row.uid for row in records if row.uid not in index]
        if missing:
            raise ValueError(f"Manifest UIDs missing from archive: {missing}")
        selected = {row.uid: index[row.uid] for row in records}
        if sum(row["bytes"] for row in selected.values()) > max_bytes:
            raise ValueError("Selected extraction exceeds max_bytes")
        binding = {"archives": combined.identity, "selection": fingerprint(sorted(r.uid for r in records)),
                   "request_sha256": selection_sha256,
                   "datasets": sorted({(r.dataset_name, r.dataset_revision) for r in records}), "members": selected}
        binding = json.loads(json.dumps(binding, sort_keys=True))
        journal_path = output / "extraction.journal.json"
        if resume:
            if not journal_path.exists():
                raise ValueError("--resume requires the existing extraction journal")
            journal = read_json(journal_path)
            if journal.get("binding") != binding or journal.get("format") != "isl-slt-extraction-journal-v1":
                raise ValueError("Extraction revision/archive/selection differs from journal")
        else:
            if journal_path.exists() or any((output / f"{row.uid}.pose").exists() for row in records):
                raise ValueError("Extraction refuses to overwrite existing pose files/journal; use --resume")
            journal = {"format": "isl-slt-extraction-journal-v1", "binding": binding, "files": {}, "sample": None}
            write_json(journal_path, journal)
        written, reused = [], []
        for record in records:
            target = output / f"{record.uid}.pose"
            if target.is_symlink():
                raise ValueError(f"Extraction destination must not be a symlink: {record.uid}")
            saved = journal["files"].get(record.uid)
            if saved is not None and not target.is_file():
                raise ValueError(f"Journaled extraction file is missing: {record.uid}")
            if target.exists():
                actual = file_identity(target)
                expected = selected[record.uid]
                if actual["bytes"] != expected["bytes"] or actual["crc32"] != expected["crc32"] or (
                    saved is not None and any(actual[key] != saved[key] for key in actual)
                ):
                    raise ValueError(f"Existing extraction file is corrupt: {record.uid}")
                journal["files"][record.uid] = {"uid": record.uid, "member": expected["member"], "path": str(target), **actual}
                reused.append(record.uid)
        write_json(journal_path, journal)
        combined.select_members(archive, [row["member"] for row in selected.values()])
        first = selected[records[0].uid]
        if first["bytes"] > 512 * 1024**2:
            raise ValueError("Sample pose exceeds inspection memory bound")
        if inspect_sample and journal["sample"] is None:
            journal["sample"] = inspect_pose_bytes(archive.read(first["member"]), settings)
            write_json(journal_path, journal)
        for record in sorted(records, key=lambda r: selected[r.uid]["offset"]):
            target = output / f"{record.uid}.pose"
            if record.uid in journal["files"]:
                written.append(journal["files"][record.uid])
                continue
            expected = selected[record.uid]
            with archive.open(expected["member"]) as source, atomic_open(target) as stream:
                identity = stream_identity(source, destination=stream, max_bytes=expected["bytes"])
                if identity["bytes"] != expected["bytes"] or identity["crc32"] != expected["crc32"]:
                    raise ValueError(f"Extracted member size/CRC mismatch: {record.uid}")
            entry = {"uid": record.uid, "member": expected["member"], "path": str(target), **identity}
            journal["files"][record.uid] = entry
            write_json(journal_path, journal)
            written.append(entry)
        report = {"archive_identity": combined.identity, "sample": journal["sample"], "files": written,
                  "selection_hash": binding["selection"], "reused_uids": reused, "archive_index": index,
                  "layout": "flattened uid.pose; raw annotations unchanged", "complete": True}
        write_json(output / "extraction.json", report)
    return report


def fetch_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Exploration-Lab/iSign")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", default="data/raw/isign-acquisition")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Read archive index and print exact selected sizes; extract nothing")
    parser.add_argument("--fetch", action="store_true", help="Stream only selected remote members")
    parser.add_argument("--resume", action="store_true", help="Verify and continue an extraction journal")
    parser.add_argument("--transport", choices=["range", "staged"], default="range")
    parser.add_argument("--staging-dir", default="data/raw/isign-staging")
    parser.add_argument("--dataset", choices=["target", "pretrain"], default="target")
    parser.add_argument("--budget-bytes", type=int, help="Compressed-byte budget for whole pretraining source groups")
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--archives", nargs="+", help="Ordered local aa, ab, ac, ad byte parts")
    parser.add_argument("--manifest")
    parser.add_argument("--uids-csv", help="Target annotation CSV for extraction before immutable auditing")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--max-bytes", type=int, default=50 * 1024**3)
    args = parser.parse_args(argv)
    if args.resume and (args.plan or not (args.archives or args.fetch)):
        parser.error("--resume applies only to explicit local extraction or remote --fetch")
    if (args.plan or args.fetch or args.archives) and (args.list or args.csv or args.parts):
        parser.error("Selection planning/fetching cannot be combined with bulk/listing flags")
    if args.archives or args.plan or args.fetch:
        if bool(args.manifest) == bool(args.uids_csv):
            parser.error("Selection requires exactly one of --manifest or --uids-csv")
        if args.plan and args.fetch:
            parser.error("Choose --plan or --fetch, never both")
        if args.plan and args.transport == "staged":
            parser.error("--plan uses ranges only; staged acquisition must be explicitly enabled with --fetch")
        config = load_config(args.config)
        records = load_manifest(args.manifest) if args.manifest else annotation_request(config, args.uids_csv, args.dataset)
        target = annotation_request(config, config.data.target.annotations) if args.dataset == "pretrain" else None
        with ExitStack() as stack:
            if args.archives:
                if args.transport != "range":
                    parser.error("--transport staged applies only to explicit remote acquisition")
                parts = args.archives
            else:
                if not args.allow_network:
                    parser.error("Remote planning/fetching requires --allow-network")
                client = stack.enter_context(ArchiveHTTP(os.environ.get("HF_TOKEN")))
                parts, urls = remote_parts(args.repo, args.revision, os.environ.get("HF_TOKEN"), client=client)
                if args.transport == "staged":
                    print(json.dumps({"staged_part_peak_bound_bytes": sum(sorted((p.size for p in parts), reverse=True)[:2]),
                                      "notice": "Explicit staged fallback downloads archive parts; final part is indexed first."}), flush=True)
                    staging = stack.enter_context(StagedParts(parts, urls, args.staging_dir, client=client))
                    parts = [ArchivePart(p.name, p.size, p.identity, lambda i=i: staging.open(i)) for i, p in enumerate(parts)]
            records, plan = plan_selection(parts, records, target_records=target,
                                           budget_bytes=args.budget_bytes, seed=args.selection_seed)
            print(json.dumps(plan, indent=2), flush=True)
            if args.plan:
                return
            report = extract_selected(parts, records, args.out, settings=config.features, max_bytes=args.max_bytes,
                                      resume=args.resume, selection_sha256=file_hash(args.manifest or args.uids_csv))
            report.update(selection_sha256=file_hash(args.manifest or args.uids_csv),
                          selection_kind="canonical-manifest" if args.manifest else "unaudited-annotation-UID-request",
                          plan=plan)
            write_json(Path(args.out) / "extraction.json", report)
            # An explicit subset CSV lets subsequent audit adapt only acquired rows without editing raw inputs.
            import csv
            with atomic_open(Path(args.out) / "selected-annotations.csv", "w") as stream:
                writer = csv.writer(stream)
                writer.writerow(["uid", "text"])
                writer.writerows((r.uid, r.text) for r in records)
            print(json.dumps({"count": len(report["files"]), "report": str(Path(args.out) / "extraction.json")}))
        return
    if not args.allow_network or not (args.list or args.csv or args.parts):
        parser.error("Remote operations require --allow-network and an explicit --list, --csv or --parts")
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).dataset_info(args.repo, revision=args.revision, files_metadata=True)
    listing = [{"name": row.rfilename, "bytes": row.size} for row in info.siblings]
    expected = [f"iSign-poses_v1.1_part_{suffix}" for suffix in ("aa", "ab", "ac", "ad")]
    names = {row["name"] for row in listing}
    if args.parts and not set(expected).issubset(names):
        raise ValueError("Expected ordered iSign ZIP parts are absent; inspect the repository listing")
    downloaded = []
    for row in listing:
        if (args.csv and row["name"].endswith(".csv")) or (args.parts and row["name"] in expected):
            downloaded.append(hf_hub_download(args.repo, row["name"], repo_type="dataset",
                revision=info.sha, token=token, local_dir=args.out))
    report = {"repository": args.repo, "resolved_revision": info.sha, "files": listing,
              "downloaded": downloaded, "pose_part_order": expected}
    write_json(Path(args.out) / "acquisition.json", report)
    print(json.dumps(report, indent=2))


def inspect_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("pose")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_pose_bytes(Path(args.pose).read_bytes(), load_config(args.config).features), indent=2))
