"""Preserve real annotation targets and supplied splits; require prepared observations."""

import argparse
import csv
import io
import json
from pathlib import Path
import pickle


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"metadata contains an object/global: {module}.{name}; export plain JSON"
        )


def read_metadata(path, delimiter):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix in {".pkl", ".pickle"}:
        payload = MetadataUnpickler(io.BytesIO(path.read_bytes())).load()
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    rows = payload.get("info", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("metadata must be a list of records, optionally under info")
    return rows


def import_corpus(
    dataset,
    annotations,
    pose_root,
    output,
    split_map=None,
    split=None,
    rgb_root=None,
    pose_template="{id}.npz",
    rgb_template="{id}.npz",
    id_column=None,
    text_column=None,
    split_column="split",
    delimiter=None,
):
    if dataset == "cislr":
        raise ValueError(
            "CISLR is word-level recognition; it is not a continuous sentence-translation benchmark"
        )
    defaults = {
        "phoenix2014t": ("name", "translation", "|"),
        "isltranslate": ("uid", "text", ","),
        "csldaily": ("name", "text", "|"),
    }
    if dataset not in defaults:
        raise ValueError("unknown corpus")
    key, target, sep = defaults[dataset]
    key, target = id_column or key, text_column or target
    rows = read_metadata(annotations, delimiter or sep)
    assignments = {} if split_map is None else json.loads(Path(split_map).read_text(encoding="utf-8"))
    groups, seen = {"train": [], "val": [], "test": []}, set()
    root, output = Path(pose_root).resolve(), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("manifest output must be empty")
    for row in rows:
        uid = row.get(key)
        text = row.get(target)
        if not isinstance(uid, str) or not uid or uid in seen or Path(uid).name != uid:
            raise ValueError("annotation id must be unique, nonempty and path-free")
        seen.add(uid)
        if dataset == "csldaily" and isinstance(text, list) and all(isinstance(x, str) for x in text):
            text = "".join(text)  # Preserve a character-list written-language reference.
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{uid}: missing written-language target; do not substitute gloss labels")
        destination = assignments.get(uid, split or row.get(split_column))
        destination = {"dev": "val", "validation": "val"}.get(destination, destination)
        if destination not in groups:
            raise ValueError(f"{uid}: provide a split map/column or --split; no random split is invented")
        path = (root / pose_template.format(id=uid, split=destination)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"{uid}: missing prepared pose NPZ: {path}")
        record = {"id": uid, "text": text, "pose": str(path), "dataset": dataset, "task": "translation"}
        signer = row.get("signer", row.get("speaker"))
        if signer is not None:
            record["signer"] = str(signer)
        if row.get("source_video"):
            record["source_video"] = str(row["source_video"])
        elif dataset == "isltranslate" and uid.rsplit("-", 1)[-1].isdigit():
            record["source_video"] = uid.rsplit("-", 1)[0]
        if row.get("split_group"):
            record["split_group"] = str(row["split_group"])
        if rgb_root:
            rgb_base = Path(rgb_root).resolve()
            rgb = (rgb_base / rgb_template.format(id=uid, split=destination)).resolve()
            if not rgb.is_relative_to(rgb_base) or not rgb.is_file():
                raise FileNotFoundError(f"{uid}: missing prepared RGB archive: {rgb}")
            record["rgb"] = str(rgb)
        groups[destination].append(record)
    if not seen:
        raise ValueError("empty annotation source")
    output.mkdir(parents=True, exist_ok=True)
    for name, records in groups.items():
        if records:
            with (output / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {name: len(records) for name, records in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["phoenix2014t", "isltranslate", "csldaily", "cislr"], required=True
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--pose-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rgb-root")
    parser.add_argument("--split-map")
    parser.add_argument("--split", choices=["train", "val", "test", "dev"])
    parser.add_argument("--pose-template", default="{id}.npz")
    parser.add_argument("--rgb-template", default="{id}.npz")
    parser.add_argument("--id-column")
    parser.add_argument("--text-column")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--delimiter")
    args = parser.parse_args()
    print(json.dumps(import_corpus(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
