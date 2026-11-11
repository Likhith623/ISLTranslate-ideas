# ISL-MEGA release audit report

**Package version:** 0.2.0 (v3 prototype & audit remediation revision)  
**Audit date:** 2026-11-11  
**Historical v1 baseline audit:** 2026-11-08  
**Scope:** source modules, CLI scripts, configurations, tests, package manifest, example manifest, and documentation files under `DINOv2-PoseGraph-3DCNN-Conformer/`.

## Executive result

- **Independent Audit Verdict (2026-11-11):** Do not train `configs/research.yaml` unchanged (excessive VRAM footprint, 9 simultaneous loss objectives, 3B FLAN-T5-XL, unchunked 518px Conv3D). Retain the prototype and build a smaller, experimentally controlled v3 baseline focused on one primary falsifiable hypothesis:
  > *Confidence-aware, anatomically structured RGB–pose fusion for robust, signer-independent continuous Indian Sign Language translation under landmark loss, occlusion, and visual corruption.*
- **Previous Runtime Test Audit (PyTorch 2.10.0 CPU):** 55 passed, 1 failed (`tests/test_novel_v2.py::test_pose_v2_torso_normalization_shared_hands_and_cross_graph`).
  - *Root Cause:* `CrossArticulatorGraph.gate` was initialized to `0.0`, multiplying the message-passing residual by `tanh(0.0) = 0.0`. This zeroed forward cross-articulator perturbation effects and blocked first-step backward gradients to block parameters.
  - *Remediation:* Configurable `cross_articulator_gate_init: float = 0.1` added to `PoseConfig` and `CrossArticulatorGraph`. Added tests verifying (1) nonzero cross-articulator perturbation propagation, (2) first-step block parameter gradients, and (3) invalid node message isolation.
- **P1 Fingerspelling Negative Conflation Resolved:** Manifest loader converted `None` to `""`, causing `compute_fingerspelling_spotting_loss` to treat unannotated clips as explicit non-fingerspelling negatives. Added `fingerspelling_label_known` in `ManifestRecord`, `data.py`, and `losses.py` so unannotated clips are not penalized.
- **P2 Batch Loss Aggregation Resolved:** `run_epoch` in `scripts/train.py` aggregated batch losses uniformly; updated to example-weighted aggregation (`batch_examples`).
- **P0 Joint Source+Signer Splitting & Component Diagnostics:** `scripts/make_splits.py` and `data.py` now support `--group-key joint_source_signer`, using disjoint-set connected components over shared `source_id` and `signer_id` edges to guarantee zero cross-split leakage of either attribute. It writes `split_diagnostics.json` reporting total components, largest component %, median/p90 component sizes, and unique source/signer counts per split, warning if the largest component exceeds 50% to prevent sacrificing dataset balance for zero leakage.
- **Optional Motion Pathway Toggle:** `MotionConfig.enabled: bool` added to cleanly bypass the 3D Conv pathway in the primary compact baseline (`configs/research_v3_compact.yaml`), isolating the primary RGB-pose fusion hypothesis without confounding motion variables.
- **Version Metadata Synchronized:** Package metadata updated to `0.2.0` consistently across `pyproject.toml`, `CITATION.cff`, `src/isl_mega/__init__.py`, and this report.
- **v3 Baseline Config Created:** `configs/research_v3_compact.yaml` implements a lean, tractable baseline: shared ViT-B/14 (224px, frozen backbone), motion disabled (`motion.enabled: false`), confidence-aware 66-point pose graph, quality-conditioned RGB/pose fusion, compact 6-layer Conformer, and FLAN-T5-base with speculative losses disabled (`ot_weight: 0`, `counterfactual_weight: 0`, `fusion_entropy_weight: 0`).

This report covers software and experimental readiness. It does **not** establish real-data translation quality, real DINOv2 or FLAN-T5 checkpoint compatibility, GPU performance, research novelty, or Q1-journal acceptance.

## Target runtime vs local check environment

Optional and runtime packages declared in `pyproject.toml`: `torch`, `numpy`, `pyyaml`, `pytest`, `timm`, `transformers`, `sacrebleu`, `rouge-score`, `opencv-python`, `mediapipe`. On the current macOS check host, only Python stdlib is installed; pretrained DINOv2 (`timm`), Hugging Face FLAN-T5 (`transformers`), GPU AMP kernels (`bf16`/`fp16`), OpenCV decoding, and `pytest` runtime execution were therefore not executed locally after the v2 edits.

## File inventory and audit boundary

| File | Scope / Role | Verification / boundary |
|---|---|---|
| `.gitignore` | Local exclusions | Excludes caches, build artifacts, run outputs and private data |
| `pyproject.toml` | Package metadata & extras | Defines `isl-mega` entry point and core/optional dependencies; not rebuilt into a wheel locally |
| `README.md` | Package overview | Aligned with v3 compact baseline, NPZ schema, eval checkpoints and research limits |
| `CITATION.cff` | Citation metadata | Metadata only; does not imply publication |
| `LICENSE` | Code license | Applies to repository code, not third-party model weights or datasets |
| `configs/research_v3_compact.yaml` | Lean v3 research baseline config | Recommended starting point: shared ViT-B/14 (224px, frozen), T5-base, confidence-aware fusion, speculative losses disabled |
| `configs/research.yaml` | High-capacity v2 prototype config | Retained for reference; requires 80GB+ GPU cluster and chunking before real execution |
| `configs/ablation_v1_baseline.yaml` | Matched-capacity v1 ablation config | Disables all six v2 mechanisms at matched capacity; fields checked statically |
| `configs/smoke_v2.yaml` | Tiny v2 software smoke config | Compact native ViT + small dims with active cross-articulator gate; fields checked statically |
| `configs/smoke.yaml` | Tiny v1 software smoke config | Compact baseline config; fields checked statically |
| `docs/architecture.md` | Model architecture specification | Documents quality gate vs calibrated uncertainty, local temporal offset vs asynchronous clocks |
| `docs/data_contract.md` | Data & landmark schema | Canonical 66-point order, JSONL manifest, optional `fingerspelling_spans` in NPZ/JSONL |
| `docs/implementation_status.md` | Implementation & verification boundary | Separates implemented modules from unrun runtime/external weights |
| `docs/references.md` | Literature precedent notes | Precedents (ISLiM, SignBind-LLM, Think in Latent Thoughts) distinguished from exact reproductions |
| `docs/research_protocol.md` | Experimental & ablation protocol | Covers v3 primary hypothesis, robustness checks, and joint source+signer splits |
| `examples/manifest_record.json` | Sample JSONL record | Example schema only; not real data |
| `src/isl_mega/__init__.py` | Package exports | Verified static exports |
| `src/isl_mega/checkpoint.py` | Checkpoint I/O | Atomic temp-file save with cleanup on failure |
| `src/isl_mega/cli.py` | CLI entry point | Config, split and model-info commands |
| `src/isl_mega/config.py` | Config schema & validation | Finite-float bounds, `cross_articulator_gate_init: 0.1`, temporal dim divisibility |
| `src/isl_mega/data.py` | Dataset, timestamps, letterbox & collate | Timestamp alignment, `fingerspelling_label_known`, joint source+signer connected-component splitting |
| `src/isl_mega/decoder.py` | T5 bridge, semantic OT & counterfactual sensitivity | Detached target text embeds for OT, duplicate-label counterfactual fallback, beam/length checks |
| `src/isl_mega/fingerspelling.py` | Character CTC & span-gated spotting | BiGRU-CTC, FP32-safe calibrated `spot_prob` routing, MIL & localized span losses |
| `src/isl_mega/fusion.py` | 10-stream reliability, lag & uncertainty fusion | Continuous-time Gaussian lag bias, FP32 AMP-safe gating & lag softmax |
| `src/isl_mega/losses.py` | Auxiliary losses | InfoNCE, fingerspelling CTC + spotting with `label_known` filtering |
| `src/isl_mega/metrics.py` | Evaluation metric helpers | Corpus BLEU/chrF/TER + signature logging; optional metric packages external |
| `src/isl_mega/model.py` | Full `ISLMEGA` model integration | Encodes 10 streams, masks `diff_energy` on invalid adjacent frames, forward & decode |
| `src/isl_mega/motion.py` | 3-D RGB motion pathway | 3-D stem + temporal projection |
| `src/isl_mega/planning.py` | Latent plan/evidence attention | Ordered multi-scale queries, diversity/monotonicity losses, dynamic routing |
| `src/isl_mega/pose.py` | 66-point pose graph & cross-articulator interaction | Smooth epsilon norms, active residual gate (`gate_init: 0.1`), message isolation on invalid nodes |
| `src/isl_mega/pretraining.py` | Global-teacher / local-student SSL | AMP-safe FP32 temperature scaling; custom DINOv2 adaptation |
| `src/isl_mega/temporal.py` | Hierarchical & adaptive temporal encoder | Non-singular backward gradients on single-frame/static inputs, FP32 salience pooling |
| `src/isl_mega/vision.py` | Specialist RGB streams & optional timm ViT | AMP dtype-safe masked backbone scatter, cached patch embedding |
| `scripts/convert_video_pose_npz.py` | Video + pose NPZ converter | Float64 `video_timestamps`, shape/monotonicity validation, optional `fingerspelling_spans` |
| `scripts/create_synthetic_dataset.py` | Synthetic dataset generator | Generates toy videos, NPZs, and optional synthetic `fingerspelling_text`/`fingerspelling_spans` |
| `scripts/evaluate.py` | Evaluation CLI | Validates `--beams`/`--batch-size`, saves `metrics.json` & `diagnostics.json` |
| `scripts/make_splits.py` | Source/signer-disjoint splitter | Grouped splits, joint source+signer connected components (`joint_source_signer`) |
| `scripts/pretrain_visual.py` | Visual SSL pretraining CLI | Positive arg validation, CUDA-guarded `pin_memory`, atomic checkpoint save |
| `scripts/train.py` | Training CLI | Example-weighted loss aggregation, FS label semantics, logs `visual_lr` & counterfactual gaps |
| `scripts/validate_data.py` | Dataset/manifest validator | Optional `--config` check, validates `fingerspelling_text` & spans |
| `tests/test_architecture.py` | v1 unit suite (45 tests) | Passed in v1 snapshot |
| `tests/test_novel_v2.py` | v2/v3 mechanism & integration suite | First-step gradient checks, perturbation tests, invalid node isolation, mock T5 generation |

## Important scientific and engineering limits

1. No real ISL videos, annotations, pose tracks, signer IDs, or trained checkpoints are shipped.
2. The canonical pose schema expects 66 landmarks in the documented order. The adapter from the original ISLTranslate / MediaPipe 543-point data must be implemented and visually validated; the package deliberately does not guess an undocumented binary format.
3. The high-capacity research prototype config references `timm` DINOv2-L/14 (`vit_large_patch14_dinov2`, 518px) and `google/flan-t5-xl`. Neither external checkpoint was downloaded or executed here. The v3 compact baseline (`configs/research_v3_compact.yaml`) uses frozen ViT-B/14 (224px) and FLAN-T5-base.
4. The global-teacher/local-region-student pretraining is an ISL-MEGA variant inspired by sign-specific self-supervision; it is not an exact reproduction of SignDINO.
5. The face stream is visual face-region encoding, not validated phoneme/lipreading or ISL non-manual grammar understanding.
6. Fingerspelling supports explicit `fingerspelling_label_known` tracking so unannotated records do not trigger false negative suppression; character-localization precision on continuous ISL remains unverified on real data.
7. Both source-disjoint and joint source+signer connected-component splitting are implemented. Near-duplicate detection and corpus provenance must be audited separately.
8. The latent planner adds ordering, orthogonality, dynamic halting, and counterfactual sensitivity diagnostics, but has no human-labelled temporal grounding supervision. RGB↔pose InfoNCE aligns same-frame features, not human temporal rationales.
9. No distributed multi-GPU trainer is included. Real GPU CUDA/`bf16` throughput and memory scaling have not been profiled on real checkpoints.
10. No BLEU/chrF/TER scores, ablation results, signer ratings, confidence intervals or real-data accuracy claims are included.

## Publication statement

This archive is an implementation candidate and experiment framework. It does not guarantee novelty, state-of-the-art performance, a Q1 journal submission, or acceptance. A Q1-quality paper requires empirical ablations on real corpora (`configs/research_v3_compact.yaml` vs unweighted/single-modality baselines), exact baseline reproductions, source/signer leakage controls, multiple seeds, statistical uncertainty, qualified signer evaluation, data/model licensing checks, and transparent reporting.
