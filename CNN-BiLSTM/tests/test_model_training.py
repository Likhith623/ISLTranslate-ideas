from types import SimpleNamespace

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from isl_slt.config import Config
from isl_slt.data.dataset import Batch
from isl_slt.evaluation.generation import generate_loader
from isl_slt.losses.objectives import translation_sum
from isl_slt.models.encoders import VisualEncoder
from isl_slt.models.t5 import T5Translator
from isl_slt.training.checkpoint import load_checkpoint, save_checkpoint, tokenizer_hash
from isl_slt.training.engine import StageScheduler, update_window
from isl_slt.utils.runtime import restore_rng, rng_state, seed_everything


def tiny_config(reduction=2):
    config = Config()
    config.runtime.device = "cpu"
    config.model.cnn_channels = (4, 8)
    config.model.lstm_hidden = 4
    config.model.dropout = 0
    config.model.downsample = reduction
    return config


@pytest.mark.parametrize("reduction", [2, 4])
def test_odd_lengths_masks_and_padding_invariance(reduction):
    encoder = VisualEncoder(6, tiny_config(reduction)).eval()
    values = torch.randn(1, 7, 6)
    short, mask = encoder(values, torch.tensor([7]))
    padded = torch.cat([values, torch.randn(1, 4, 6)], dim=1)
    longer, longer_mask = encoder(padded, torch.tensor([7]))
    expected = (7 + reduction - 1) // reduction
    assert int(mask.sum()) == int(longer_mask.sum()) == expected
    torch.testing.assert_close(short[:, :expected], longer[:, :expected], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(longer[:, expected:]) == 0


def test_frozen_t5_keeps_visual_gradient_path():
    t5 = T5ForConditionalGeneration(T5Config(vocab_size=16, d_model=8, d_ff=16, num_layers=1,
        num_decoder_layers=1, num_heads=2, dropout_rate=0, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    model = T5Translator(VisualEncoder(6, tiny_config()), t5)
    for parameter in t5.parameters():
        parameter.requires_grad_(False)
    labels = torch.tensor([[2, 3, 1], [4, 1, -100]])
    output = model(features=torch.randn(2, 7, 6), lengths=torch.tensor([7, 5]), labels=labels)
    translation_sum(output["logits"], labels).backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum()) for parameter in model.visual.parameters())
    assert all(parameter.grad is None for parameter in t5.parameters())


def test_token_normalized_accumulation_matches_merged_gradient():
    labels = torch.tensor([[1, 2, -100], [2, 1, 3]])
    logits = torch.randn(2, 3, 5, requires_grad=True)
    tokens = int((labels != -100).sum())
    (translation_sum(logits[:1], labels[:1]) / tokens).backward(retain_graph=True)
    (translation_sum(logits[1:], labels[1:]) / tokens).backward()
    accumulated = logits.grad.clone()
    merged = logits.detach().clone().requires_grad_()
    (translation_sum(merged, labels) / tokens).backward()
    torch.testing.assert_close(accumulated, merged.grad)


def test_generation_does_not_receive_labels(record_factory):
    class Generator:
        def eval(self):
            return self
        def generate(self, **kwargs):
            assert "labels" not in kwargs
            return torch.tensor([[2, 1]])
    batch = Batch({"features": torch.zeros(1, 3, 6), "lengths": torch.tensor([3])},
                  [record_factory()], labels=torch.tensor([[2, 1]]))
    tokenizer = SimpleNamespace(eos_token_id=1, decode=lambda tokens, **kwargs: "generated")
    rows = generate_loader(Generator(), tokenizer, [batch], tiny_config(), torch.device("cpu"), 1)
    assert rows[0]["prediction"] == "generated"


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.2)
        self.projection = torch.nn.Linear(3, 7)
    def forward(self, labels, features, lengths):
        return {"logits": self.projection(self.dropout(features)), "lexical_logits": None}


def test_two_window_resume_matches_uninterrupted(tmp_path):
    seed_everything(12)
    config = tiny_config()
    def setup():
        model = TinyTrainModel()
        optimizer = torch.optim.AdamW([{"params": list(model.parameters()), "base_lr": .01, "lr": .01}])
        schedule = StageScheduler(optimizer, .1)
        schedule.configure("tiny", 4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        return model, optimizer, schedule, scaler
    batches = [Batch({"features": torch.randn(1, 3, 3), "lengths": torch.tensor([3])},
                     [], labels=torch.tensor([[2, 3, 1]])) for _ in range(2)]
    model, optimizer, schedule, scaler = setup()
    update_window(model, optimizer, schedule, scaler, [batches[0]], config, torch.device("cpu"), 0)
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"format": "isl-slt-checkpoint-v1", "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(), "scaler": scaler.state_dict(),
        "rng": rng_state(), "tokenizer": {}, "compatibility": {"tokenizer": tokenizer_hash({})},
        "state": {"batch_offset": 1, "updates": 1}})
    update_window(model, optimizer, schedule, scaler, [batches[1]], config, torch.device("cpu"), 0)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    saved = load_checkpoint(path)
    resumed, resumed_optimizer, resumed_schedule, resumed_scaler = setup()
    resumed.load_state_dict(saved["model"])
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_schedule.load_state_dict(saved["scheduler"])
    resumed_scaler.load_state_dict(saved["scaler"])
    restore_rng(saved["rng"])
    update_window(resumed, resumed_optimizer, resumed_schedule, resumed_scaler,
                  [batches[1]], config, torch.device("cpu"), 0)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert resumed_schedule.state_dict() == schedule.state_dict()
