# PCAT-SLT v0.3

**Visual, multirate Progress-Coupled Area Transport**: PyTorch code, real RGB input, optional pretrained T5/mT5 text decoding, annotation importers and executable mathematical/integration tests.

This release implements the defensible parts of the supplied audit and corrects several of its prescriptions. A short Neumann inverse is not orthogonal; dynamic centering alone can erase body grammar; pretrained models do not guarantee a BLEU gain. Read [the audit response](docs/history/AUDIT_RESPONSE.md), [architecture/proofs](ARCHITECTURE.md) and [remaining limitations](docs/LIMITATIONS.md).

## Implemented upgrades

- Joint progress and **additional local articulator clocks**, retaining cross-stream order and timing.
- Dual fixed/local body coordinates with time-aware smoothing **and preserved global motion**.
- Reproducible low-dimensional **area sketches** with a shared full-state clock.
- Per-example **adaptive progress cells**, padded/masked independently of other batch members.
- **Solve-free palindromic Givens rotations**, preserving norm and reversal; exact Cayley remains available.
- Actual RGB crop encoding: spatial-grid CNN or optional pretrained **DINOv2**, observation masks and explicit temporal binding.
- Pretrained **T5/mT5/FLAN-T5 decoder** support, source/target masking, training/generation and portable tokenizer/weight checkpoints. Other text model families are rejected rather than guessing their token rules.
- Explicit planar/depth-ignore mode, plus opt-in RGB fallback when no usable pose reference exists.
- Real annotation importers for **PHOENIX-2014T, ISLTranslate and CSL-Daily**, with targets/splits preserved.
- Stricter RGB timing, content/path leakage checks, crop preprocessing, tests and archived failure evidence.

These are implemented capabilities, not measured real-world SLT superiority. The original PCAT invariant reference modes remain available for controlled research.

## Install and verify

Use Python 3.10+ and the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,multimodal]'
pytest -q
ruff check .
```

`multimodal` installs Transformers, Pillow and SentencePiece. The core pose/byte project also works with `pip install -e '.[dev]'`; optional pretrained tests skip when Transformers is absent. The reported full verification installed the optional dependencies. Tests create small local T5/DINO weight fixtures; they do not download large models or establish large-model translation quality.

## Complete offline pixel diagnostic

```bash
pcat-demo-visual --output data/visual_demo --samples 24
pcat-validate --config configs/multimodal_smoke.yaml --signer-independent
OMP_NUM_THREADS=1 pcat-train --config configs/multimodal_smoke.yaml
pcat-evaluate --checkpoint runs/v3_visual_final/best.pt --output evaluation/visual --max-new-tokens 8
```

The diagnostic makes pose static and labels simple observed pixel patterns with `a`/`b`. These are not signs, mouthings or sentences. It verifies that the full train/generate path can use pixels. The generator refuses a populated output folder; training refuses an existing `last.pt` without `--resume`. Original failed and revised diagnostic evidence is preserved in `reports/`.

For the original geometric task, use `pcat-demo-data` with `configs/smoke.yaml` or `configs/enhanced_smoke.yaml`. Historical results are under `reports/v1/` and `reports/v2/`; their numbers are not current real-corpus results.

## Documentation

| File | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current model, equations and guarantee boundaries |
| [TRAINING_AND_DATASETS.md](TRAINING_AND_DATASETS.md) | Single dataset catalog, memory/storage requirements and training-time/cost planning |
| [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md) | Accelerator selection and step-by-step Kaggle setup, training, saving and recovery |
| [notebooks/Kaggle_PCAT_SLT.ipynb](notebooks/Kaggle_PCAT_SLT.ipynb) | Executable diagnostic and prepared-data Kaggle workflows |
| [docs/DATA.md](docs/DATA.md) | Array formats, landmark layout, crop preparation and corpus import commands |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Baselines, ablations, initial settings and evaluation protocol |
| [docs/LITERATURE.md](docs/LITERATURE.md) | Research references and novelty assessment |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Remaining limitations and unverified claims |
| [reports/VERIFICATION.md](reports/VERIFICATION.md) | Executed tests, measurements and verification scope |

`docs/history/` retains earlier derivations and the supplied audit/response for provenance. Start with the current documents above; historical files describe earlier versions. Raw videos, pose extraction and complete corpus conversion are not bundled. Prepare observations using `docs/DATA.md` before using a real-data configuration.

## Configurations and modes

| File | Purpose |
|---|---|
| `configs/vpcat.yaml` | Full multimodal example: pixels, pretrained text, joint/local clocks, dual frames, sketches, adaptive cells and Givens transport |
| `configs/multimodal_smoke.yaml` | Download-free CNN/byte diagnostic using the new operator paths |
| `configs/base.yaml` | Preserved v0.2 enhanced pose/byte reference |
| `configs/invariant.yaml` | Fixed-reference geometric invariant research control |
| `configs/mps.yaml` | Explicit MPS entry point with FP32 settings and CPU integration fallback |

The full multimodal model is deliberately not covered by the strict geometric speed/view/subdivision invariance theorem. RGB encoders, explicit timing/frames and moving coordinates have different assumptions. Shape, masks and bounded transport properties are tested separately.

Dense reference configs use float64 integration to reduce cancellation; the resource-oriented multimodal example uses float32 sketches. `integration: clipped` retains the independent clipping implementation. Givens is a different orthogonal action, not an approximate Cayley matrix. CPU measurements do not establish accelerator latency.

## Checkpoints, inference and resume

```bash
pcat-train --config configs/vpcat.yaml --resume runs/vpcat_real/last.pt
pcat-predict --checkpoint runs/vpcat_real/best.pt --pose data/clip_pose.npz --rgb data/clip_rgb.npz
```

Format-3 checkpoints include active model weights, pretrained architecture snapshots and tokenizer files. Checkpoint inference/resume does not depend on the original model directories or fresh model downloads. Tests remove original assets before offline reload and exercise pretrained training/resume. Format-2 byte reference checkpoints remain supported; format-1 weights require the archived v0.1 source. All old source archives stay under `releases/` and historical weights/data were not overwritten.

The training system includes token-correct accumulation, finite-gradient checks, optional CUDA AMP, validation-NLL selection, atomic checkpoint writes, RNG state and training/validation input fingerprints (including RGB). Exact CPU epoch resume is tested for the reference modes. CUDA/MPS behavior is not certified by CPU tests. Actual MPS hardware is unavailable on this host, so its hardware test is explicitly skipped. CPU integration can add transfers on MPS; there is no promised Apple/GPU speedup.

## Evidence and research use

[reports/VERIFICATION.md](reports/VERIFICATION.md) records actual checks, diagnostics, failures and benchmarks. [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) specifies real-data and matched ablations.

```bash
python scripts/benchmark_transports.py
python scripts/run_ablations.py --config configs/vpcat.yaml --seeds 17 23 41
```

Ablation generation is nonexecuting by default; add `--execute` for sequential training. Some controls change modalities or model capacity and are diagnostic comparisons, not matched tests of the primitive. Use otherwise matched Cayley/Givens, sketches/dense areas and joint/local-clock comparisons to study the operator itself.

Everything stays inside this project folder. Build the clean source archive with:

```bash
python scripts/package.py --output releases/PCAT-SLT.zip
```

The ZIP excludes data, trained weights, environments, caches and recursive release archives. It contains a source hash manifest and has a companion archive checksum. `requirements-verified.txt` records the actual local verification environment. **No real-corpus SOTA, complete linguistic coverage, universal novelty or publication outcome is established.**
