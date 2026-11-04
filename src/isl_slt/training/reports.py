"""Lightweight summaries of real event logs and checkpoint sizes."""

import argparse
import json
from pathlib import Path

from isl_slt.utils.io import write_json


def summarize_run(directory) -> dict:
    root = Path(directory)
    updates = examples = frames = epochs = 0
    update_seconds = train_seconds = validation_seconds = 0.0
    peak = None
    stages = {}
    with (root / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("kind") == "update":
                updates += int(event.get("updated", False))
                examples += event.get("examples", 0)
                frames += event.get("frames", 0)
                update_seconds += event.get("update_seconds", 0)
                measured = event.get("peak_gpu_memory_bytes")
                peak = max(peak or 0, measured) if measured is not None else peak
            elif event.get("kind") == "epoch":
                epochs += 1
                train_seconds += event["train_seconds"]
                validation_seconds += event["validation_seconds"]
                stage = stages.setdefault(event.get("stage", "unknown"), {
                    "completed_epochs": 0, "counted_epochs": 0, "examples": 0, "frames": 0,
                    "train_seconds": 0.0, "validation_seconds": 0.0})
                stage["completed_epochs"] += 1
                stage["train_seconds"] += event["train_seconds"]
                stage["validation_seconds"] += event["validation_seconds"]
                if "examples" in event and "frames" in event:
                    stage["counted_epochs"] += 1
                    stage["examples"] += event["examples"]
                    stage["frames"] += event["frames"]
    for stage in stages.values():
        complete = stage["counted_epochs"] == stage["completed_epochs"]
        stage["examples_per_second"] = stage["examples"] / stage["train_seconds"] if complete and stage["train_seconds"] else None
        stage["frames_per_second"] = stage["frames"] / stage["train_seconds"] if complete and stage["train_seconds"] else None
        stage["validation_seconds_per_epoch"] = stage["validation_seconds"] / stage["completed_epochs"]
    checkpoints = sorted((root / "checkpoints").rglob("*.pt"))
    unique_files = {(path.stat().st_dev, path.stat().st_ino): path.stat().st_size for path in checkpoints}
    return {"run_dir": str(root), "successful_updates": updates, "consumed_examples": examples,
            "consumed_frames": frames, "completed_epochs": epochs, "update_seconds": update_seconds,
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "update_examples_per_second": examples / update_seconds if update_seconds else None,
            "update_frames_per_second": frames / update_seconds if update_seconds else None,
            "validation_seconds_per_epoch": validation_seconds / epochs if epochs else None,
            "peak_gpu_memory_bytes": peak,
            "stages": stages,
            "unique_checkpoint_bytes": sum(unique_files.values()),
            "checkpoint_sizes": {str(path.relative_to(root)): path.stat().st_size
                                 for path in checkpoints},
            "measurement_scope": "Update throughput excludes input-window loading; epoch train time includes it. Setup/checkpoint I/O is excluded."}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = summarize_run(args.run)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
