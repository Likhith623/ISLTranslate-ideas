"""Fail-fast split and contamination contracts. Membership is immutable and independently hashed."""

from collections import defaultdict
from pathlib import Path

from isl_slt.config import Data
from isl_slt.data.manifests import Record, audit_records, load_manifest, manifest_hash, read_csv, save_manifest
from isl_slt.utils.io import fingerprint, read_json, write_json


def leakage_report(partitions: dict[str, list[Record]], require_verified: bool = True) -> dict:
    violations, unresolved = [], []
    uid_owner, source_owner, content_owner = {}, {}, {}
    for partition, rows in partitions.items():
        seen, seen_uids = set(), set()
        for row in rows:
            if row.key in seen:
                violations.append({"kind": "duplicate_record", "partition": partition, "uid": row.uid})
            seen.add(row.key)
            if row.uid in seen_uids:
                violations.append({"kind": "duplicate_uid", "partition": partition, "uid": row.uid})
            seen_uids.add(row.uid)
            if not row.source_id or not row.source_verified:
                unresolved.append({"partition": partition, "uid": row.uid, "provenance": row.source_provenance})
            for kind, values, owners in (
                ("uid", [row.uid], uid_owner),
                ("source", [row.source_id] if row.source_id else [], source_owner),
                ("fingerprint", row.duplicate_fingerprints, content_owner),
            ):
                for value in values:
                    owner = owners.get(value)
                    if owner is not None and owner != partition:
                        violations.append({"kind": kind, "identity": value, "partitions": [owner, partition]})
                    owners[value] = owner or partition
    return {"ok": not violations and (not require_verified or not unresolved), "violations": violations,
            "unresolved_mappings": unresolved, "verification_required": require_verified,
            "available_fingerprint_count": len(content_owner),
            "fingerprint_coverage": {name: sum(bool(r.duplicate_fingerprints) for r in rows)
                                     for name, rows in partitions.items()},
            "membership_hashes": {name: manifest_hash(rows) for name, rows in partitions.items()}}


def assert_leakage_free(partitions: dict[str, list[Record]], require_verified: bool = True) -> dict:
    report = leakage_report(partitions, require_verified)
    if not report["ok"]:
        raise ValueError(f"Leakage/provenance check failed: {report}")
    return report


def make_splits(records: list[Record], settings: Data) -> dict[str, list[Record]]:
    audit = audit_records(records)
    if audit["duplicate_uids"] or audit["invalid_targets"]:
        raise ValueError("Resolve duplicated identities or invalid targets before splitting")
    if any(r.source_id is None for r in records):
        raise ValueError("All records need a source identity; unresolved UIDs cannot be split safely")
    if settings.protocol != "exploratory-unverified" and any(not r.source_verified for r in records):
        raise ValueError("This protocol requires verified source-mapping provenance")
    groups = defaultdict(list)
    for record in records:
        groups[record.source_id].append(record)
    gold_sources = {r.source_id for r in records if r.interpreter_text is not None}
    result = {name: [] for name in ("train", "validation", "test", "gold")}
    assignment = {source: "gold" for source in gold_sources}
    if settings.protocol == "external-membership-gold-safe":
        _, external = read_csv(settings.original_splits_csv)
        membership = {row["uid"]: row["split"] for row in external}
        if len(membership) != len(external) or set(membership) != {r.uid for r in records}:
            raise ValueError("Original-paper membership must cover each target UID exactly once")
        for source, rows in groups.items():
            names = {membership[r.uid] for r in rows}
            if len(names) != 1 or not names.issubset(result):
                raise ValueError(f"External split violates source grouping: {source}")
            name = names.pop()
            if source in gold_sources and name != "gold":
                raise ValueError("External protocol overlaps reserved diagnostic parents; supply a separate safe protocol")
            assignment[source] = name
    else:
        remaining = [source for source in groups if source not in gold_sources]
        if len(remaining) < 3:
            raise ValueError("At least three nongold parent sources are needed for nonempty partitions")
        # Largest-first deficit filling with deterministic seeded tie breaks; not paper membership reproduction.
        remaining.sort(key=lambda source: (-len(groups[source]), fingerprint([settings.split_seed, source])))
        names = ("train", "validation", "test")
        total = sum(len(groups[source]) for source in remaining)
        targets = dict(zip(names, [ratio * total for ratio in settings.ratios]))
        counts = dict.fromkeys(names, 0)
        for index, source in enumerate(remaining):
            empty = [name for name in names if counts[name] == 0]
            candidates = empty if len(remaining) - index == len(empty) else names
            chosen = max(candidates, key=lambda name: ((targets[name] - counts[name]) / targets[name],
                                                      targets[name], -names.index(name)))
            assignment[source] = chosen
            counts[chosen] += len(groups[source])
    for source, rows in groups.items():
        name = assignment[source]
        result[name].extend(row.model_copy(update={"split": name}) for row in rows)
    for name in ("train", "validation", "test"):
        if not result[name]:
            raise ValueError(f"Empty partition {name}; supply a usable group allocation")
    assert_leakage_free(result, settings.protocol != "exploratory-unverified")
    return result


def normalized_text(text: str) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def pretraining_candidates(pool: list[Record], all_target: list[Record], *, require_poses: bool = False):
    """The same exclusion policy runs before acquisition and again after pose auditing."""
    if audit_records(pool)["conflicting_labels"]:
        raise ValueError("Resolve conflicting extra-iSign labels before deduplication")
    forbidden_sources = {r.source_id for r in all_target}
    forbidden_uids = {r.uid for r in all_target}
    forbidden_fingerprints = {value for r in all_target for value in r.duplicate_fingerprints}
    seen_text = {normalized_text(r.text) for r in all_target}
    kept, excluded, seen_uid, seen_content = [], [], set(), set()
    for row in sorted(pool, key=lambda r: (r.dataset_name, r.uid)):
        reason = None
        if not row.source_verified or not row.source_id:
            reason = "unresolved_identity"
        elif row.source_id in forbidden_sources:
            reason = "target_parent"
        elif row.uid in forbidden_uids or row.uid in seen_uid:
            reason = "duplicate_uid"
        elif set(row.duplicate_fingerprints) & (forbidden_fingerprints | seen_content):
            reason = "duplicate_fingerprint"
        elif normalized_text(row.text) in seen_text:
            reason = "duplicate_normalized_text"
        elif not row.text.strip():
            reason = "invalid_target"
        elif require_poses and not Path(row.pose_path).is_file():
            reason = "missing_pose_asset"
        if reason:
            excluded.append({"uid": row.uid, "reason": reason, "provenance": row.source_provenance})
        else:
            kept.append(row.model_copy(update={"split": "pretrain"}))
            seen_uid.add(row.uid)
            seen_content.update(row.duplicate_fingerprints)
            seen_text.add(normalized_text(row.text))
    return kept, excluded


def filter_pretraining(pool: list[Record], target: dict[str, list[Record]]) -> tuple[list[Record], dict]:
    kept, excluded = pretraining_candidates(pool, [row for rows in target.values() for row in rows], require_poses=True)
    report = assert_leakage_free({**target, "pretrain": kept})
    return kept, {**report, "pool_count": len(pool), "eligible_count": len(kept), "excluded": excluded,
                  "policy": "all-target-sources-and-normalized-text-v2",
                  "identity_scope": "Exclude every target parent and normalized text; missing perceptual hashes remain untested"}


def export_splits(partitions: dict[str, list[Record]], settings: Data, pretrain_report: dict | None = None) -> dict:
    root = Path(settings.split_dir)
    report = assert_leakage_free(partitions, settings.protocol != "exploratory-unverified")
    metadata = {"algorithm": "external-membership" if settings.protocol == "external-membership-gold-safe"
                else "largest-deficit-whole-source-v1", "seed": settings.split_seed,
                "protocol": settings.protocol, "ratios": list(settings.ratios),
                "hashes": report["membership_hashes"],
                "counts": {name: len(rows) for name, rows in partitions.items()},
                "source_ids": {name: sorted({r.source_id for r in rows}) for name, rows in partitions.items()}}
    if (root / "splits.json").exists() and read_json(root / "splits.json") != metadata:
        raise ValueError("Split directory already contains different immutable membership")
    for name, rows in partitions.items():
        save_manifest(root / f"{name}.jsonl", rows)
    write_json(root / "leakage.json", report, immutable=True)
    if pretrain_report is not None:
        write_json(root / "pretraining-filter.json", pretrain_report, immutable=True)
    write_json(root / "splits.json", metadata, immutable=True)
    return metadata


def checked_splits(settings: Data) -> tuple[dict[str, list[Record]], dict]:
    root = Path(settings.split_dir)
    metadata = read_json(root / "splits.json")
    required = {"train", "validation", "test", "gold"}
    if not required.issubset(metadata["hashes"]) or set(metadata["hashes"]) - required - {"pretrain"}:
        raise ValueError("Split metadata must identify train/validation/test/gold and optional pretrain only")
    if metadata["protocol"] != settings.protocol:
        raise ValueError("Configured protocol differs from immutable split metadata")
    partitions = {name: load_manifest(root / f"{name}.jsonl") for name in metadata["hashes"]}
    if any(not partitions[name] for name in ("train", "validation", "test")):
        raise ValueError("Train/validation/test manifests must be nonempty")
    report = assert_leakage_free(partitions, settings.protocol != "exploratory-unverified")
    if report["membership_hashes"] != metadata["hashes"] or read_json(root / "leakage.json") != report:
        raise ValueError("Split membership or leakage report was modified; refusing training")
    for name, rows in partitions.items():
        if any(r.split != name for r in rows):
            raise ValueError("Record partition does not match its manifest")
    if "pretrain" in partitions:
        report_path = root / "pretraining-filter.json"
        if (not report_path.exists() or read_json(report_path)["membership_hashes"] != metadata["hashes"]
            or read_json(report_path).get("policy") != "all-target-sources-and-normalized-text-v2"):
            raise ValueError("Missing or incompatible pretraining contamination report")
    return partitions, metadata
