# Implementation status and verification boundaries

## Included in this source release

| Area | Status | Verification |
|---|---|---|
| Specialist RGB appearance streams | Implemented | Native ViT synthetic tests; timm optional path not run in this environment |
| DINOv2/timm initializer | Implemented via optional timm | Requires `timm` and downloaded weights; not executed here |
| Sign-aware global/local self-distillation variant | Implemented | Synthetic forward/backward and EMA tests (synthetic suite) |
| 66-point graph pose expert & cross-articulator graph | Implemented | Synthetic shape, missing-node, missing-wrist centroid fallback, and gradient tests |
| Timestamp-aware pose alignment | Implemented | Exact/interpolated/gap tests |
| Aspect-preserving letterbox and coordinate transforms | Implemented | Synthetic pixel/point/box regression tests |
| 3-D RGB motion pathway | Implemented | Full-frame temporal length and padding invariance tests |
| Fingerspelling BiGRU-CTC & span-gated routing | Implemented | Codec, CTC, spotter routing and MIL/frame-span loss synthetic tests |
| Ten-stream reliability, continuous-time lag & uncertainty fusion | Implemented | Missing streams, uncertainty gates, weights and asynchronous offset tests |
| Source-aware RGB↔pose contrastive alignment | Implemented | Same-frame positive pairs, same-source negative masking and gradient tests |
| Fine/coarse & salience-adaptive temporal encoder | Implemented | Non-default stride, partial-group, single-frame and gradient finiteness tests |
| Latent plan/evidence attention & ordered dynamic planning | Implemented | Attention mask, ordering, diversity, dynamic routing and tensor tests |
| T5/FLAN-T5 bridge, plan semantic OT & counterfactual grounding | Implemented | Mock T5 adapter, duplicate-label fallback and train-step synthetic tests; real HF weights not loaded |
| Optional `fingerspelling_spans` dataset & collate support | Implemented | NPZ `[Tv]` masks, JSONL `[start_sec, end_sec]` intervals and synthetic dataset tests |
| Configs (`research.yaml`, `ablation_v1_baseline.yaml`, `smoke_v2.yaml`, `smoke.yaml`) | Implemented | Field/type/validator schema alignment checked statically |
| Training, visual SSL, validation and evaluation scripts | Implemented | Static compilation & stdlib checks; actual GPU/T5 runs require external ML runtime |
| Raw project `.pose` parser | Not included | Must use parent project parser/exporter |
| Actual dataset annotations/videos | Not included | User must provide authorized data |
| Trained weights | Not included | Must be trained by user |
| Human signer evaluation | Not automated | Required external research procedure |

## Explicit implementation limits

- The 18 face points are a defined adapter order; an extractor must map its own landmark IDs to it and that mapping must be manually verified.
- The pretraining code is an ISL-MEGA global-teacher/local-region-student distillation variant. It is not an exact SignDINO reimplementation.
- Both clip-level `fingerspelling_text` labels (supervised via weak clip-level MIL) and optional localized `fingerspelling_spans` (NPZ `[Tv]` boolean masks or JSONL `[start_sec, end_sec]` intervals) are supported; automated spotter evaluation on real continuous ISL corpora remains unverified.
- The latent planner returns ordered attention diagnostics, diversity/monotonicity regularizers, and dynamic halting scores, but has no direct human-rationale supervision loss.
- Semantic OT (frame and active-plan levels) and counterfactual visual sensitivity penalties are research hypotheses until validated on real ISL benchmarks.
- No reinforcement-learning attention steering or distributed trainer is included. The contrastive objective aligns same-frame anatomical RGB/pose pairs; it is not a human-labelled temporal grounding objective.
- The actual Hugging Face T5 checkpoint path, `timm` backbone and OpenCV decoder have not been executed on real sign benchmarks in this environment.

## Test suite and verification status

- The synthetic unit suites in `tests/test_architecture.py` (v1 core) and `tests/test_novel_v2.py` (v2 mechanisms & integrated smoke) are designed to run with `pytest -q` once PyTorch, NumPy, PyYAML, and pytest are installed (`pip install -e .[test]` or `.[all]`).
- On the current development host (without `torch`, `numpy`, `PyYAML`, or `pytest` installed), the latest v2 revisions were verified via Python bytecode compilation (`python3 -m compileall`), AST parsing, and YAML-to-`ISLMEGAConfig` schema checks; full `pytest -q` execution must be run in the target PyTorch environment before training.
- Real DINOv2/FLAN-T5 weights, GPU AMP (`bf16`/`fp16`) kernels, and benchmark datasets lie outside the synthetic regression suite.
