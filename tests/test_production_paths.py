import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from isl_slt.config import Config, Features, Stage
from isl_slt.data.dataset import Batch
from isl_slt.features.pose import interpolate_short_gaps, motion, normalize_positions, prepare_pose, selected_joints
from isl_slt.training.engine import StageScheduler, optimizer_for, parameter_inventory, set_stage, update_window


def test_pose_format_roundtrip(pose_writer, record_factory, tmp_path):
    from isl_slt.data.pose import read_pose
    path = pose_writer(tmp_path / "real-format.pose")
    sequence = read_pose(record_factory(pose_path=str(path)), Features())
    assert sequence.names == selected_joints(Features())
    assert len(sequence.times) == 7
    assert not sequence.valid[..., 2].any()
    assert prepare_pose(sequence, Features()).values.shape == (7, 384)


def test_interpolation_anchor_and_velocity_boundaries():
    times = np.array([0., .04, .08])
    positions = np.array([[[0., 0.]], [[0., 0.]], [[2., 4.]]])
    valid = np.array([[True], [False], [True]])
    filled, mask = interpolate_short_gaps(positions, valid, times, np.zeros(3, int), .12)
    np.testing.assert_allclose(filled[1, 0], [1., 2.])
    assert mask[1, 0]
    assert not interpolate_short_gaps(positions, valid, times, np.array([0, 1, 1]), .12)[1].any()
    assert not interpolate_short_gaps(positions, valid, times, np.zeros(3, int), .01)[1].any()
    with pytest.raises(ValueError, match="shoulder anchors"):
        normalize_positions(np.zeros((3, 2, 2)), np.ones((3, 2), bool), np.zeros(3, int), 1e-6)
    velocity, velocity_mask = motion(filled, np.ones((3, 1), bool), times, np.array([0, 0, 1]))
    assert velocity_mask[1, 0] and not velocity_mask[2, 0]
    assert np.count_nonzero(velocity[2]) == 0


def test_missing_anchor_frame_keeps_positions_but_blocks_velocity():
    from types import SimpleNamespace
    coordinates = np.zeros((3, 48, 3), np.float32)
    coordinates[:, 0, 0], coordinates[:, 1, 0] = -1, 1
    valid = np.ones_like(coordinates, bool)
    valid[1, :2] = False
    sequence = SimpleNamespace(coordinates=coordinates, valid=valid, names=selected_joints(Features()),
        times=np.array([0., .04, .08]), segments=np.zeros(3, int), regions=np.zeros(3, int), metadata={})
    prepared = prepare_pose(sequence, Features())
    assert prepared.position_valid[:, 6].all()
    assert not prepared.velocity_valid[1:, 6].any()


class LinearTranslator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 7)
    def forward(self, labels, features, lengths):
        return {"logits": self.linear(features), "lexical_logits": None}


@pytest.mark.parametrize("sizes", [(1, 2), (1,)])
def test_production_update_window_matches_merged_tokens(sizes):
    config = Config()
    config.training.gradient_clip_norm = 1000000
    model = LinearTranslator()
    merged_model = copy.deepcopy(model)
    features = torch.randn(sum(sizes), 3, 3)
    labels = torch.tensor([[2, 1, -100], [3, 4, 1], [1, -100, -100]])[:sum(sizes)]
    def setup(module):
        optimizer = torch.optim.AdamW([{"params": list(module.parameters()), "base_lr": .01, "lr": .01}])
        scheduler = StageScheduler(optimizer, 0)
        scheduler.configure("test", 2)
        return optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False)
    batches, offset = [], 0
    for size in sizes:
        batches.append(Batch({"features": features[offset:offset + size], "lengths": torch.full((size,), 3)},
                             [], labels=labels[offset:offset + size]))
        offset += size
    merged = Batch({"features": features, "lengths": torch.full((len(features),), 3)}, [], labels=labels)
    update_window(model, *setup(model), batches, config, torch.device("cpu"), 0)
    update_window(merged_model, *setup(merged_model), [merged], config, torch.device("cpu"), 0)
    for expected, actual in zip(merged_model.parameters(), model.parameters()):
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)


def test_optimizer_registration_survives_t5_freeze_unfreeze():
    from transformers import T5Config, T5ForConditionalGeneration
    from isl_slt.models.encoders import VisualEncoder
    from isl_slt.models.t5 import T5Translator
    config = Config()
    config.model.cnn_channels = (4, 8)
    config.model.lstm_hidden = 4
    model = T5Translator(VisualEncoder(6, config), T5ForConditionalGeneration(T5Config(
        vocab_size=16, d_model=8, d_ff=16, d_kv=4, num_heads=2, num_layers=1, num_decoder_layers=1,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)))
    optimizer = optimizer_for(model, config)
    registered = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(registered) == len(set(registered)) == len(list(model.parameters()))
    set_stage(model, Stage(name="adapter", epochs=1, freeze_t5=True))
    assert not any(p.requires_grad for p in model.t5.parameters())
    assert all(id(p) in registered for p in model.t5.parameters())
    set_stage(model, Stage(name="joint", epochs=1))
    assert all(p.requires_grad for p in model.t5.parameters())
    assert registered == [id(p) for group in optimizer.param_groups for p in group["params"]]
    inventory = parameter_inventory(model, config)
    assert inventory["total"] == sum(p.numel() for p in model.parameters())
    assert inventory["total"] == inventory["visual"] + inventory["t5"] + inventory["other"]
    assert inventory["per_stage"]["adapter"]["t5_trainable"] == 0
    assert inventory["per_stage"]["joint"]["t5_trainable"] == inventory["t5"]
    assert all(p.requires_grad for p in model.t5.parameters())


def test_filter_pretraining_excludes_target_training_parent_and_text(record_factory):
    from isl_slt.data.splits import filter_pretraining
    target = {"train": [record_factory("target-1", text=" Same Caption ")]}
    pool = [record_factory("target-2", text="new"), record_factory("other-1", text="same caption"),
            record_factory("allowed-1", text="unique")]
    for row in pool:
        Path(row.pose_path).write_bytes(row.uid.encode())
    kept, report = filter_pretraining(pool, target)
    assert [r.uid for r in kept] == ["allowed-1"]
    assert {r["reason"] for r in report["excluded"]} == {"target_parent", "duplicate_normalized_text"}


def test_training_only_bpe_roundtrip_and_attention_generation(tmp_path, record_factory):
    from isl_slt.data.lexical import BPETokenizer
    from isl_slt.models.factory import build_model
    records = [record_factory(split="train")]
    tokenizer = BPETokenizer.fit(records, 300)
    tokenizer.save_pretrained(tmp_path)
    tokenizer = BPETokenizer.from_pretrained(tmp_path)
    assert tokenizer.decode(tokenizer.encode(records[0].text)) == records[0].text
    with pytest.raises(ValueError, match="training/pretraining"):
        BPETokenizer.fit([record_factory(split="test")], 300)
    config = Config()
    config.model.decoder = "lstm"
    config.model.cnn_channels = (4, 8)
    config.model.lstm_hidden = 4
    config.model.decoder_hidden = 8
    config.model.decoder_embedding = 8
    model = build_model(config, 6, tokenizer).eval()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(-100)
        model.output.bias[3] = 100
    output = model.generate(features=torch.randn(2, 7, 6), lengths=torch.tensor([7, 5]),
                            beams=2, max_new_tokens=4, length_penalty=1)
    assert output[:, 0].tolist() == [3, 3]


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    else:
        assert first == second


def test_full_loader_resume_and_average_predictor(synthetic_project, monkeypatch):
    from isl_slt import workflows
    from isl_slt.training import engine
    from isl_slt.training.checkpoint import average_checkpoints, load_checkpoint
    from isl_slt.inference import Predictor
    config, _ = synthetic_project()
    workflows.audit(config)
    workflows.split(config)
    workflows.prepare_features(config)
    engine.run_training(config)
    reference = load_checkpoint(Path(config.run_dir) / "checkpoints/last.pt")
    continued = config.model_copy(update={"run_dir": config.run_dir + "-resumed"}, deep=True)
    real_budget = engine.Budget
    class StopAfterOne:
        def __init__(self, seconds):
            self.calls = 0
        @property
        def expired(self):
            self.calls += 1
            return self.calls > 1
    monkeypatch.setattr(engine, "Budget", StopAfterOne)
    result = engine.run_training(continued)
    assert result["status"] == "wall_time_limit"
    monkeypatch.setattr(engine, "Budget", real_budget)
    checkpoint = Path(continued.run_dir) / "checkpoints/last.pt"
    engine.run_training(continued, str(checkpoint))
    resumed = load_checkpoint(checkpoint)
    for field in ("model", "optimizer", "scheduler", "scaler"):
        assert_nested_equal(reference[field], resumed[field])
    assert reference["state"]["updates"] == resumed["state"]["updates"]
    parents = sorted((Path(config.run_dir) / "checkpoints").glob("epoch-*.pt"))
    averaged = Path(config.run_dir) / "checkpoints/averaged.pt"
    average_checkpoints(list(map(str, parents)), str(averaged))
    output = Path(config.run_dir) / "averaged-validation"
    workflows.evaluate(config, str(averaged), "validation", str(output))
    predictor = Predictor.from_checkpoint(str(averaged), config.runtime, str(output / "decoding.json"), config.generation)
    assert predictor.beams in config.generation.beams
