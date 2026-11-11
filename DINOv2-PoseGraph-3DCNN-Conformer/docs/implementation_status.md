# Implementation status and verification boundaries

## Included in this source release

| Area | Status | Verification |
|---|---|---|
| Specialist RGB appearance streams | Implemented | Native ViT synthetic tests; timm optional path not run in this environment |
| DINOv2/timm initializer | Implemented via optional timm | Requires `timm` and downloaded weights; not executed here |
| Sign-aware global/local self-distillation variant | Implemented | Synthetic forward/backward and EMA tests |
| 66-point graph pose expert | Implemented | Synthetic shape, missing-node and gradient tests |
| Timestamp-aware pose alignment | Implemented | Exact/interpolated/gap tests |
| Aspect-preserving letterbox and coordinate transforms | Implemented | Synthetic pixel/point/box regression tests |
| 3-D RGB motion pathway | Implemented | Full-frame temporal length and padding invariance tests |
| Fingerspelling BiGRU-CTC | Implemented | Codec, CTC edge cases and gradient tests |
| Ten-stream reliability fusion | Implemented | Missing streams, weights and asynchronous offset tests |
| Source-aware RGB↔pose contrastive alignment | Implemented | Same-frame positive pairs, same-source negative masking and gradient tests |
| Fine/coarse hierarchical temporal encoder | Implemented | Non-default stride, partial-group and padding tests |
| Latent plan/evidence attention | Implemented | Attention mask and tensor tests |
| T5/FLAN-T5 bridge | Implemented | Mock T5 adapter tests only; actual checkpoint not loaded |
| Training and checkpoint entry points | Implemented | Syntax/CLI checks; actual T5 training not executed |
| Evaluation and metrics entry points | Implemented | Syntax/CLI checks; optional metric packages absent here |
| Raw project `.pose` parser | Not included | Must use parent project parser/exporter |
| Actual dataset annotations/videos | Not included | User must provide authorized data |
| Trained weights | Not included | Must be trained by user |
| Human signer evaluation | Not automated | Required external research procedure |

## Explicit implementation limits

- The 18 face points are a defined adapter order; an extractor must map its own landmark IDs to it and that mapping must be manually verified.
- The pretraining code is an ISL-MEGA global-teacher/local-region-student distillation variant. It is not an exact SignDINO reimplementation.
- The optional fingerspelling label is clip-level; localized character spans are not represented in the manifest.
- The latent planner returns attention diagnostics but has no direct human-rationale supervision loss.
- Semantic OT is unconstrained, optional and unproven. It does not create gold temporal alignments.
- No reinforcement-learning attention steering or distributed trainer is included. The contrastive objective aligns same-frame anatomical RGB/pose pairs; it is not a human-labelled temporal grounding objective.
- The actual Hugging Face T5 checkpoint path, `timm` backbone and OpenCV decoder were not executed in this environment because the optional packages/weights were unavailable.

## Test environment for this release candidate

The software tests in this environment use Python 3.13.5, PyTorch 2.10.0 CPU, NumPy 2.3.5 and pytest 9.0.2. The full suite is designed to run with `pytest -q`. Real DINOv2/FLAN-T5 weights, GPU kernels and benchmark data are outside the synthetic regression suite.
