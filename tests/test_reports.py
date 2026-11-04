from isl_slt.training.reports import summarize_run
from isl_slt.utils.io import append_event


def test_summary_uses_only_measured_values(tmp_path):
    append_event(tmp_path / "events.jsonl", kind="update", updated=True, examples=4, frames=40,
                 update_seconds=2, peak_gpu_memory_bytes=None)
    append_event(tmp_path / "events.jsonl", kind="epoch", train_seconds=3, validation_seconds=1)
    report = summarize_run(tmp_path)
    assert report["update_examples_per_second"] == 2
    assert report["validation_seconds_per_epoch"] == 1
    assert report["peak_gpu_memory_bytes"] is None
    assert report["stages"]["unknown"]["examples_per_second"] is None


def test_completed_epoch_rate_uses_loading_and_ignores_partial_epoch(tmp_path):
    from isl_slt.cli import dispatch, parser
    from isl_slt.utils.io import write_json
    append_event(tmp_path / "events.jsonl", kind="epoch", stage="joint", train_seconds=10, validation_seconds=2,
                 examples=100, frames=1000)
    append_event(tmp_path / "events.jsonl", kind="epoch", stage="joint", train_seconds=30, validation_seconds=4,
                 examples=100, frames=1000)
    append_event(tmp_path / "events.jsonl", kind="update", updated=True, examples=50, frames=500, update_seconds=1)
    report = summarize_run(tmp_path)
    assert report["stages"]["joint"]["examples_per_second"] == 5
    assert report["stages"]["joint"]["examples"] == 200
    summary = tmp_path / "summary.json"
    write_json(summary, report)
    result = dispatch(parser().parse_args(["estimate", "--examples", "100", "--epochs", "2",
                                           "--run-summary", str(summary), "--stage", "joint"]))
    assert result["estimated_hours"] == 2 * (100 / 5 + 3) / 3600


def test_checkpoint_summary_counts_hardlinked_bytes_once(tmp_path):
    import os
    append_event(tmp_path / "events.jsonl", kind="setup")
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "best-bleu.pt").write_bytes(b"one snapshot")
    os.link(directory / "best-bleu.pt", directory / "best-chrf.pt")
    report = summarize_run(tmp_path)
    assert report["unique_checkpoint_bytes"] == len(b"one snapshot")
    assert len(report["checkpoint_sizes"]) == 2
