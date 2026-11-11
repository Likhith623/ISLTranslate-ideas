import torch

from isl_slt.utils.runtime import seed_everything


def test_strict_workspace_policy(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        seed_everything(7, True)
        import os
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        seed_everything(7, True, True)
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


def test_environment_probe_includes_local_t5_and_configured_precision():
    from isl_slt.config import Runtime
    from isl_slt.utils.runtime import environment_report
    report = environment_report(Runtime(device="cpu", precision="fp32"))
    assert report["probes"]["fp32"]["ok"]
    assert report["t5_probes"]["fp32"]["ok"]
    assert report["t5_probes"]["fp32"]["pretrained_weights"] is False
    assert report["configured_precision_ok"]
    assert not report["t5_probes"]["fp16"]["ok"]


def test_bf16_preflight_blocks_training_and_saves_failure(tmp_path, monkeypatch):
    import pytest
    from isl_slt.config import Config, Runtime
    from isl_slt.training import engine
    from isl_slt.utils.io import read_json
    config = Config(run_dir=str(tmp_path / "bf16"), runtime=Runtime(precision="bf16"))
    monkeypatch.setattr(engine, "environment_report", lambda policy: {"configured_precision_ok": False})
    def forbidden(*args):
        pytest.fail("Training started despite a failed preflight")
    monkeypatch.setattr(engine, "_run_training", forbidden)
    with pytest.raises(RuntimeError, match="BF16 environment probes failed"):
        engine.run_training(config)
    assert not read_json(tmp_path / "bf16/env-check.json")["configured_precision_ok"]


def test_bf16_preflight_reseeds_and_runs_on_resume(tmp_path, monkeypatch):
    from isl_slt.config import Config, Runtime
    from isl_slt.training import engine
    from isl_slt.utils.io import read_json
    config = Config(run_dir=str(tmp_path / "bf16"), runtime=Runtime(precision="bf16"))
    seed_everything(42)
    expected = torch.rand(3)
    calls = []
    def probe(policy):
        calls.append(policy.precision)
        seed_everything(0)
        torch.rand(10)
        return {"configured_precision_ok": True, "t5_probes": {"fp16": {"ok": False}}}
    def training(config, resume, *args):
        assert resume == "last.pt"
        assert torch.equal(torch.rand(3), expected)
        return {"ok": True}
    monkeypatch.setattr(engine, "environment_report", probe)
    monkeypatch.setattr(engine, "_run_training", training)
    assert engine.run_training(config, resume="last.pt")["ok"]
    assert calls == ["bf16"]
    assert read_json(tmp_path / "bf16/env-check.json")["configured_precision_ok"]


def test_bf16_autocast_keeps_visual_recurrence_fp32():
    from isl_slt.config import Config
    from isl_slt.models.encoders import VisualEncoder
    config = Config()
    config.model.cnn_channels = (8, 8)
    config.model.lstm_hidden = 4
    model = VisualEncoder(4, config)
    seen = []
    handle = model.recurrent.register_forward_pre_hook(lambda layer, args: seen.append(args[0].data.dtype))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, _ = model(torch.ones(2, 6, 4), torch.tensor([6, 4]))
            output.sum().backward()
    finally:
        handle.remove()
    assert seen == [torch.float32]
    assert torch.isfinite(output).all()
