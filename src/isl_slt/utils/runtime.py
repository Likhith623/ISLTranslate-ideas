"""Backend-neutral precision and reproducibility. Operations run only on explicit calls."""

import random
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from isl_slt.config import Runtime


def seed_everything(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    # Configure NVIDIA cuBLAS before any CUDA seeding/initialization. HIP is probed separately.
    if deterministic and torch.version.hip is None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False


def device_for(runtime: Runtime) -> torch.device:
    name = "cuda" if runtime.device == "auto" and torch.cuda.is_available() else runtime.device
    if name == "auto":
        name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU PyTorch is unavailable; select the managed interpreter with a working GPU build")
    return torch.device(name)


def autocast(runtime: Runtime, device: torch.device):
    if runtime.precision == "fp32":
        return nullcontext()
    if runtime.precision == "fp16" and device.type == "cpu":
        raise ValueError("Use FP32 or BF16 on CPU")
    if runtime.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This backend does not report BF16 support; select FP32/FP16 explicitly")
    return torch.autocast(device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[runtime.precision])


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "gpu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["gpu"] is not None:
        if not torch.cuda.is_available() or len(state["gpu"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the saved GPU topology")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["gpu"]])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Budget:
    def __init__(self, seconds: float | None):
        self.seconds, self.started = seconds, time.monotonic()

    @property
    def expired(self) -> bool:
        return self.seconds is not None and time.monotonic() - self.started >= self.seconds


def tiny_t5_probe(policy: Runtime, device: torch.device) -> dict:
    """Random local architecture only: exercise text-model kernels without any pretrained assets."""
    from transformers import T5Config, T5ForConditionalGeneration
    model = T5ForConditionalGeneration(T5Config(vocab_size=16, d_model=16, d_ff=32, d_kv=8,
        num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)).to(device)
    inputs = torch.tensor([[2, 3, 1], [4, 5, 1]], device=device)
    with autocast(policy, device):
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=inputs)
        output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not bool(torch.isfinite(output.loss)) or not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("Tiny T5 produced nonfinite loss/gradients or no gradients")
    model.eval()
    with torch.no_grad(), autocast(policy, device):
        generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=4, num_beams=1)
        repeated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=4, num_beams=1)
    synchronize(device)
    if not torch.equal(generated, repeated):
        raise RuntimeError("Tiny T5 repeated greedy generation differs under the declared policy")
    return {"ok": True, "operation": "random tiny T5 forward/backward/repeated greedy generation",
            "logits_dtype": str(output.logits.dtype), "generated_shape": list(generated.shape),
            "finite_gradient_tensors": len(gradients), "pretrained_weights": False}


def environment_report(runtime: Runtime) -> dict:
    seed_everything(0, runtime.deterministic, runtime.determinism_mode == "warn")
    device = device_for(runtime)
    report = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
              "device": str(device), "precision": runtime.precision,
              "recurrent_fp32_policy": runtime.recurrent_fp32, "probes": {}, "t5_probes": {},
              "deterministic": runtime.deterministic, "determinism_mode": runtime.determinism_mode,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "hipblas_workspace_config": os.environ.get("HIPBLAS_WORKSPACE_CONFIG")}
    if device.type == "cuda":
        report.update(name=torch.cuda.get_device_name(device), memory_bytes=torch.cuda.get_device_properties(
            device).total_memory, bf16=torch.cuda.is_bf16_supported())
    for precision in ("fp32", "bf16", "fp16"):
        try:
            policy = runtime.model_copy(update={"precision": precision})
            layer = torch.nn.LSTM(4, 4, num_layers=2, bidirectional=True, batch_first=True).to(device)
            projection = torch.nn.Linear(4, 4).to(device)
            with autocast(policy, device):
                values = projection(torch.ones(2, 3, 4, device=device))
                recurrent_context = torch.autocast(device.type, enabled=False) if runtime.recurrent_fp32 else nullcontext()
                with recurrent_context:
                    packed = torch.nn.utils.rnn.pack_padded_sequence(values.float() if runtime.recurrent_fp32 else values,
                        torch.tensor([3, 2]), batch_first=True, enforce_sorted=False)
                    output, _ = layer(packed)
                    output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                output.float().sum().backward()
            synchronize(device)
            gradients = [p.grad for p in (*layer.parameters(), *projection.parameters()) if p.grad is not None]
            if (not bool(torch.isfinite(output).all()) or not gradients or
                any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)):
                raise RuntimeError("Packed LSTM produced nonfinite outputs/gradients or no gradients")
            report["probes"][precision] = {"ok": True, "projection_dtype": str(values.dtype),
                                           "recurrent_output_dtype": str(output.dtype),
                                           "operation": "two-layer packed bidirectional LSTM forward/backward"}
        except (RuntimeError, ValueError, NotImplementedError) as error:
            report["probes"][precision] = {"ok": False, "error": str(error)}
        try:
            seed_everything(0, runtime.deterministic, runtime.determinism_mode == "warn")
            report["t5_probes"][precision] = tiny_t5_probe(runtime.model_copy(update={"precision": precision}), device)
        except (RuntimeError, ValueError, NotImplementedError, ImportError) as error:
            report["t5_probes"][precision] = {"ok": False, "error": str(error)}
    report["configured_precision_ok"] = all(probes[runtime.precision]["ok"] for probes in (
        report["probes"], report["t5_probes"]))
    return report
