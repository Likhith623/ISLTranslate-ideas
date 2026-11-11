"""Future offline CLI acceptance; random synthetic assets are never research results."""

from pathlib import Path

import pytest
import yaml

from isl_slt.cli import main
from isl_slt.data.splits import checked_splits
from isl_slt.utils.io import read_json, write_jsonl


@pytest.mark.slow
def test_offline_synthetic_cli_chain(synthetic_project, capsys):
    config, path = synthetic_project("t5")
    config.evaluation.warmup_iterations = 0
    config.evaluation.benchmark_iterations = 1
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    root = Path(config.run_dir)
    def command(stage, *arguments, config_file=path):
        main([stage, "--config", str(config_file), *map(str, arguments)])
        capsys.readouterr()
    for stage in ("plan", "env-check", "audit", "split", "check-leakage", "prepare-features", "verify-cache", "train"):
        command(stage)
    checkpoint = root / "checkpoints/best-bleu.pt"
    command("evaluate", "--checkpoint", checkpoint, "--partition", "validation", "--output", root / "validation-final")
    selection = root / "validation-final/decoding.json"
    with pytest.raises(SystemExit, match="final validation beam sweep"):
        command("evaluate", "--checkpoint", checkpoint, "--partition", "test", "--output", root / "no-decoding")
    for partition in ("test", "gold"):
        command("evaluate", "--checkpoint", checkpoint, "--partition", partition, "--decoding", selection,
                "--output", root / partition)
    command("diagnostics", "--checkpoint", checkpoint, "--partition", "test", "--decoding", selection,
            "--output", root / "diagnostics")
    partitions, _ = checked_splits(config.data)
    inputs = root / "visual-inputs.jsonl"
    write_jsonl(inputs, [{"uid": row.uid, "pose_path": row.pose_path} for row in partitions["test"][:2]])
    for stage, destination in (("predict", "raw-predictions.jsonl"), ("benchmark", "timing.json")):
        command(stage, "--checkpoint", checkpoint, "--inputs", inputs, "--decoding", selection, "--output", root / destination)
    averaged = root / "checkpoints/averaged.pt"
    main(["average", "--checkpoints", *map(str, sorted((root / "checkpoints").glob("epoch-*.pt"))), "--output", str(averaged)])
    command("evaluate", "--checkpoint", averaged, "--partition", "validation", "--output", root / "averaged-validation")
    command("evaluate", "--checkpoint", averaged, "--partition", "test", "--output", root / "averaged-test",
            "--decoding", root / "averaged-validation/decoding.json")
    raw = config.model_dump(mode="json")
    raw["run_dir"] = str(root / "debug")
    raw["model"]["dropout"] = 0
    raw["training"].update(debug_only=True, max_target_epochs=1, weight_decay=0,
                            stages=[{"name": "overfit", "epochs": 1, "freeze_t5": False}])
    raw["generation"].update(beams=[1], train_beams=[1])
    debug_config = root / "debug.yaml"
    debug_config.write_text(yaml.safe_dump(raw))
    uids = root / "inspected-uids.txt"
    uids.write_text("".join(row.uid + "\n" for row in partitions["train"][:32]))
    command("overfit", "--uids", uids, config_file=debug_config)
    main(["summarize", "--reports", str(root / "test/metrics.json"), "--output", str(root / "summary.json")])
    assert read_json(root / "summary.json")["individual"]
    assert read_json(root / "gold/metrics.json")["metrics"]["gold_subset_identity"]["uid_count"] == 1
