"""Explicit CSV adapters and canonical manifests; annotations are never edited in place."""

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from isl_slt.config import Adapter, Strict
from isl_slt.utils.io import fingerprint, read_jsonl, write_jsonl


class Record(Strict):
    uid: str
    clip_type: Literal["sentence", "def", "unknown"] = "unknown"
    source_id: str | None = None
    source_verified: bool = False
    source_provenance: str | None = None
    dataset_name: str
    dataset_revision: str
    split: Literal["unassigned", "train", "validation", "test", "gold", "pretrain"] = "unassigned"
    text: str
    pose_path: str
    video_path: str | None = None
    interpreter_text: str | None = None
    interpreter_transcript: str | None = None
    sequence_index: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    source_duration_seconds: float | None = None
    timing_verified: bool = False
    fps: float | None = Field(default=None, gt=0)
    timestamps_path: str | None = None
    coordinate_space: Literal["image_pixels", "normalized_image"] = "image_pixels"
    # Affine mapping from pose-header pixel coordinates to decoded video pixels.
    image_transform: list[list[float]] | None = None
    person_index: int | None = Field(default=None, ge=0)
    annotation_revision: str = "original"
    correction_provenance: str | None = None
    duplicate_fingerprints: list[str] = Field(default_factory=list)
    asset_hashes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def invariants(self):
        if not self.uid or any(char in self.uid for char in ("/", "\\", "\x00")) or self.uid in (".", ".."):
            raise ValueError("UID must be a nonempty filename-safe identifier")
        if not self.dataset_name.strip() or not self.dataset_revision.strip() or not self.annotation_revision.strip():
            raise ValueError("Dataset and annotation identity/revisions must be nonempty")
        if self.source_verified and not (self.source_id and self.source_provenance):
            raise ValueError("Verified sources require both identity and provenance")
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("Supply both clip timing bounds or neither")
        if self.start_seconds is not None and not (0 <= self.start_seconds < self.end_seconds):
            raise ValueError("Clip timing bounds must be increasing and nonnegative")
        if self.timing_verified and self.start_seconds is None:
            raise ValueError("Verified timing requires clip bounds")
        for value in (self.start_seconds, self.end_seconds, self.source_duration_seconds, self.fps):
            if value is not None and not math.isfinite(value):
                raise ValueError("Timing/FPS values must be finite")
        if self.source_duration_seconds is not None and (
            self.source_duration_seconds <= 0 or self.end_seconds is not None and self.end_seconds > self.source_duration_seconds
        ):
            raise ValueError("Source duration must be positive and include the entire center clip")
        if self.image_transform is not None and (
            len(self.image_transform) != 2 or any(len(row) != 3 for row in self.image_transform)
        ):
            raise ValueError("image_transform must be a 2x3 affine matrix")
        return self

    @property
    def key(self) -> str:
        return fingerprint([self.dataset_name, self.dataset_revision, self.uid])


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError(f"Missing or duplicated CSV headers: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"Malformed CSV row with surplus columns: {path}")
    return headers, rows


def mapped_columns(headers: list[str], aliases: dict[str, list[str]]) -> dict[str, str]:
    result = {}
    for canonical, candidates in aliases.items():
        matches = [name for name in candidates if name in headers]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one explicit alias for {canonical}; found {matches}; headers={headers}")
        result[canonical] = matches[0]
    if len(set(result.values())) != len(result):
        raise ValueError("One CSV column cannot supply multiple canonical fields")
    return result


def source_mapping(path: str | None) -> dict[str, dict]:
    if path is None:
        return {}
    headers, rows = read_csv(path)
    required = {"uid", "source_id", "verified", "provenance"}
    if not required.issubset(headers):
        raise ValueError(f"Source mapping needs {sorted(required)}")
    result = {}
    for row in rows:
        if row["uid"] in result:
            raise ValueError(f"Duplicate source override: {row['uid']}")
        if row["verified"].lower() not in ("true", "false"):
            raise ValueError("Source verified must be true or false")
        result[row["uid"]] = {"source_id": row["source_id"].strip() or None,
                              "source_verified": row["verified"].lower() == "true",
                              "source_provenance": row["provenance"].strip() or None}
    return result


def source_candidate(uid: str, patterns: list[str] | str) -> tuple:
    for index, pattern in enumerate([patterns] if isinstance(patterns, str) else patterns):
        match = re.fullmatch(pattern, uid)
        if match is not None and "source" in match.groupdict():
            name = "def-tag" if "tag" in match.groupdict() else "sentence-suffix" if "sequence" in match.groupdict() else f"custom-{index}"
            return match, name, pattern
    return None, "unresolved", None


def resolve_source(uid: str, overrides: dict[str, dict], patterns: list[str] | str) -> dict:
    match, rule, pattern = source_candidate(uid, patterns)
    sequence = int(match["sequence"]) if match and "sequence" in match.groupdict() else None
    if uid in overrides:
        return {**overrides[uid], "sequence_index": sequence}
    if match is None or "source" not in match.groupdict():
        return {"source_id": None, "sequence_index": sequence, "source_verified": False,
                "source_provenance": "unresolved"}
    return {"source_id": match["source"], "sequence_index": sequence, "source_verified": False,
            "source_provenance": f"candidate:{rule}:{pattern}"}


def _metadata(path: str | None) -> dict[str, dict]:
    if path is None:
        return {}
    _, rows = read_csv(path)
    result = {}
    allowed = set(Record.model_fields) - {"text", "dataset_name", "dataset_revision", "split"}
    json_fields = {"image_transform", "duplicate_fingerprints", "asset_hashes"}
    bool_fields = {"timing_verified", "source_verified"}
    int_fields = {"sequence_index", "person_index"}
    float_fields = {"start_seconds", "end_seconds", "source_duration_seconds", "fps"}
    for row in rows:
        if "uid" not in row or row["uid"] in result:
            raise ValueError("Metadata requires unique uid rows")
        if set(row) - allowed:
            raise ValueError(f"Unknown metadata columns: {set(row) - allowed}")
        values = {}
        for key, value in row.items():
            if key == "uid" or value in (None, ""):
                continue
            if key in bool_fields:
                if value.lower() not in ("true", "false"):
                    raise ValueError(f"Metadata {key} must be true/false")
                value = value.lower() == "true"
            elif key in json_fields:
                value = json.loads(value)
            elif key in int_fields:
                value = int(value)
            elif key in float_fields:
                value = float(value)
            values[key] = value
        result[row["uid"]] = values
    return result


def adapt_csv(adapter: Adapter, audit: dict | None = None) -> list[Record]:
    if adapter.dataset_revision == "REQUIRED":
        raise ValueError("Set the exact dataset revision before creating a manifest")
    headers, rows = read_csv(adapter.annotations)
    columns = mapped_columns(headers, adapter.columns)
    if not {"uid", "text"}.issubset(columns):
        raise ValueError("Adapter must explicitly map uid and text")
    unique, seen, collapsed = [], {}, []
    for line_number, raw in enumerate(rows, start=2):
        uid = raw[columns["uid"]]
        if uid in seen:
            previous, kept_line = seen[uid]
            if previous != raw:
                raise ValueError(f"Conflicting duplicate UID {uid!r} at CSV rows {kept_line}, {line_number}")
            collapsed.append({"uid": uid, "kept_row": kept_line, "dropped_row": line_number,
                              "raw_row_sha256": fingerprint(raw)})
        else:
            seen[uid] = (raw, line_number)
            unique.append(raw)
    if audit is not None:
        audit["collapsed_exact_duplicates"] = collapsed
    overrides, metadata = source_mapping(adapter.source_mapping), _metadata(adapter.metadata_csv)
    records = []
    for raw in unique:
        values = {key: raw[column] for key, column in columns.items()}
        uid = values["uid"]
        values = {**resolve_source(uid, overrides, adapter.candidate_uid_pattern or adapter.candidate_uid_patterns),
                  "pose_path": str(Path(adapter.pose_root) / adapter.pose_pattern.format(uid=uid)),
                  "video_path": str(Path(adapter.video_root) / adapter.video_pattern.format(uid=uid))
                  if adapter.video_root else None,
                  "coordinate_space": adapter.coordinate_space, **metadata.get(uid, {}), **values}
        # Authoritative source mappings override any metadata or annotation source guesses.
        values.update(overrides.get(uid, {}))
        if "clip_type" not in values:
            _, rule, _ = source_candidate(uid, Adapter().candidate_uid_patterns)
            values["clip_type"] = {"sentence-suffix": "sentence", "def-tag": "def"}.get(rule, "unknown")
        records.append(Record.model_validate({"dataset_name": adapter.dataset_name,
            "dataset_revision": adapter.dataset_revision, "annotation_revision": adapter.annotation_revision, **values}))
    if adapter.corrections_jsonl:
        corrections = read_jsonl(adapter.corrections_jsonl)
        by_uid = {row["uid"]: row for row in corrections}
        if len(by_uid) != len(corrections) or set(by_uid) - {r.uid for r in records}:
            raise ValueError("Corrections must have unique known UIDs")
        corrected = []
        for record in records:
            correction = by_uid.get(record.uid)
            if correction:
                if set(correction) != {"uid", "text", "revision", "provenance"} or not correction["provenance"]:
                    raise ValueError("Corrections require uid, text, revision, provenance")
                record = Record.model_validate({**record.model_dump(), "text": correction["text"],
                    "annotation_revision": correction["revision"], "correction_provenance": correction["provenance"]})
            corrected.append(record)
        records = corrected
    return records


def attach_interpreter(records: list[Record], path: str, aliases: dict[str, list[str]]) -> list[Record]:
    headers, rows = read_csv(path)
    columns = mapped_columns(headers, aliases)
    annotations = {row[columns["uid"]]: row for row in rows}
    if len(annotations) != len(rows) or set(annotations) - {r.uid for r in records}:
        raise ValueError("Interpreter table contains duplicate or unknown target UIDs")
    return [record.model_copy(update={"interpreter_text": annotations[record.uid][columns["interpreter_text"]],
                                      "interpreter_transcript": annotations[record.uid][columns["text"]]})
            if record.uid in annotations else record for record in records]


def load_manifest(path: str | Path) -> list[Record]:
    rows = read_jsonl(path)
    if any(row.get("format") == "isl-slt-draft-v1" or row.get("frozen") is False for row in rows):
        raise ValueError("Draft manifests cannot be used for splitting, features, training or evaluation; audit real poses first")
    return [Record.model_validate(row) for row in rows]


def save_manifest(path: str | Path, records: list[Record], immutable: bool = True) -> None:
    write_jsonl(path, [r.model_dump(mode="json") for r in sorted(records, key=lambda r: (r.dataset_name, r.uid))],
                immutable=immutable)


def manifest_hash(records: list[Record]) -> str:
    return fingerprint([r.model_dump(mode="json") for r in sorted(records, key=lambda r: (r.dataset_name, r.uid))])


def audit_records(records: list[Record]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.dataset_name, record.uid)].append(record)
    repeated = Counter(r.text for r in records)
    return {"examples": len(records), "manifest_hash": manifest_hash(records),
            "duplicate_uids": [list(key) for key, rows in grouped.items() if len(rows) > 1],
            "conflicting_labels": [list(key) for key, rows in grouped.items() if len({r.text for r in rows}) > 1],
            "interpreter_transcript_conflicts": [r.uid for r in records if r.interpreter_transcript is not None
                                                   and r.interpreter_transcript.strip() != r.text.strip()],
            "repeated_target_texts": {text: count for text, count in repeated.items() if count > 1},
            "invalid_targets": [r.uid for r in records if not r.text.strip() or "\x00" in r.text],
            "missing_pose": [r.uid for r in records if not Path(r.pose_path).is_file()],
            "missing_video": [r.uid for r in records if r.video_path and not Path(r.video_path).is_file()],
            "source_count": len({r.source_id for r in records if r.source_id}),
            "unverified_sources": [r.uid for r in records if not r.source_verified],
            "verified_timing_examples": sum(r.timing_verified for r in records),
            "gold_examples": sum(r.interpreter_text is not None for r in records),
            "gold_source_count": len({r.source_id for r in records if r.interpreter_text is not None and r.source_id}),
            "gold_unresolved_source_examples": sum(r.interpreter_text is not None and not r.source_id for r in records)}
