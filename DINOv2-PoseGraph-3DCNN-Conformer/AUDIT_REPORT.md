# ISL-MEGA release audit report

**Package version:** 0.1.0 (v2 research revision)  
**Latest revision date:** 2026-11-11  
**Historical v1 baseline audit:** 2026-11-08  
**Scope:** source modules, CLI scripts, configurations, tests, package manifest, example manifest, and documentation files under `DINOv2-PoseGraph-3DCNN-Conformer/`.

## Executive result

- **Historical v1 snapshot (2026-11-08):** Prior to v2 mechanisms and v2 hardening edits, the original v1 archive passed **45 tests** in `tests/test_architecture.py` on Python 3.13.5 / PyTorch 2.10.0 CPU / NumPy 2.3.5 / pytest 9.0.2 / OpenCV 4.13.0, along with editable install (`--no-build-isolation`), CLI help checks, and a one-epoch CPU SSL smoke run. That prior result does **not** cover subsequent v2 additions or post-2026-11-08 source edits. The stale pre-v2 artifact `dist/isl_mega-0.1.0-py3-none-any.whl` was removed.
- **Current v2 source & static audit (2026-11-11):** All modules, scripts, v2 mechanisms, matched ablation configs (`configs/ablation_v1_baseline.yaml`, `configs/smoke_v2.yaml`), AMP/autograd/geometric edge-case fixes, and `tests/test_novel_v2.py` have been inspected and checked statically (`compileall`, AST parsing, and YAML-to-`ExperimentConfig` schema alignment). Because PyTorch, NumPy, PyYAML, and pytest are not installed on the current local macOS host, runtime execution of `pytest -q` on the updated tree remains **pending** in a target Python/ML environment.

This report covers software/static readiness only. It does **not** establish real-data translation quality, real DINOv2 or FLAN-T5 checkpoint compatibility, GPU performance, research novelty, or Q1-journal acceptance.

## Target runtime vs local check environment

Optional and runtime packages declared in `pyproject.toml`: `torch`, `numpy`, `pyyaml`, `pytest`, `timm`, `transformers`, `sacrebleu`, `rouge-score`, `opencv-python`, `mediapipe`. On the current macOS check host, only Python stdlib is installed; pretrained DINOv2 (`timm`), Hugging Face FLAN-T5 (`transformers`), GPU AMP kernels (`bf16`/`fp16`), OpenCV decoding, and `pytest` runtime execution were therefore not executed locally after the v2 edits.

## File inventory and audit boundary

| File | Scope / Role | Verification / boundary |
|---|---|---|
| `.gitignore` | Local exclusions | Excludes caches, build artifacts, run outputs and private data |
| `pyproject.toml` | Package metadata & extras | Defines `isl-mega` entry point and core/optional dependencies; not rebuilt into a wheel locally |
| `README.md` | Package overview | Aligned with v1/v2 mechanisms, NPZ `fingerspelling_spans`, eval checkpoints and research limits |
| `CITATION.cff` | Citation metadata | Metadata only; does not imply publication |
| `LICENSE` | Code license | Applies to repository code, not third-party model weights or datasets |
| `configs/research.yaml` | Proposed v2 research config | `CrossArticulatorGraph`, lag+uncertainty fusion, span routing, adaptive consolidation, ordered/dynamic planning, plan OT & counterfactual loss; fields checked statically |
| `configs/ablation_v1_baseline.yaml` | Matched-capacity v1 ablation config | Disables all six v2 mechanisms at matched backbone/decoder capacity; fields checked statically |
| `configs/smoke_v2.yaml` | Tiny v2 software smoke config | Compact native ViT + small dims with v2 flags enabled; fields checked statically |
| `configs/smoke.yaml` | Tiny v1 software smoke config | Compact baseline config; fields checked statically |
| `docs/architecture.md` | Model architecture specification | Documents v1/v2 modules, counterfactual duplicate-label fallback, and `fingerspelling_spans` / MIL routing |
| `docs/data_contract.md` | Data & landmark schema | Canonical 66-point order, JSONL manifest, optional `fingerspelling_spans` in NPZ/JSONL |
| `docs/implementation_status.md` | Implementation & verification boundary | Separates implemented modules from unrun runtime/external weights |
| `docs/references.md` | Literature precedent notes | Precedents distinguished from exact reproductions |
| `docs/research_protocol.md` | Experimental & ablation protocol | Covers A1–A13 plus A14–A19 v2 ablations; plan only, no completed results |
| `examples/manifest_record.json` | Sample JSONL record | Example schema only; not real data |
| `src/isl_mega/__init__.py` | Package exports | Verified static exports |
| `src/isl_mega/checkpoint.py` | Checkpoint I/O | Atomic temp-file save with cleanup on failure |
| `src/isl_mega/cli.py` | CLI entry point | Config, split and model-info commands |
| `src/isl_mega/config.py` | Config schema & validation | Finite-float bounds, temporal dim divisibility and YAML coercion |
| `src/isl_mega/data.py` | Dataset, timestamps, letterbox & collate | Timestamp alignment, eager/runtime `fingerspelling_spans` validation, letterbox inverse transforms |
| `src/isl_mega/decoder.py` | T5 bridge, semantic OT & counterfactual sensitivity | Detached target text embeds for OT, duplicate-label counterfactual fallback, beam/length checks; actual HF T5 not loaded locally |
| `src/isl_mega/fingerspelling.py` | Character CTC & span-gated spotting | BiGRU-CTC, FP32-safe calibrated `spot_prob` routing, MIL & localized span losses |
| `src/isl_mega/fusion.py` | 10-stream reliability, lag & uncertainty fusion | Continuous-time Gaussian lag bias, FP32 AMP-safe gating & lag softmax |
| `src/isl_mega/losses.py` | Auxiliary losses | InfoNCE, fingerspelling CTC + spotting, temporal consistency |
| `src/isl_mega/metrics.py` | Evaluation metric helpers | Corpus BLEU/chrF/TER + signature logging; optional metric packages external |
| `src/isl_mega/model.py` | Full `ISLMEGA` model integration | Encodes 10 streams, masks `diff_energy` on invalid adjacent frames, forward & decode |
| `src/isl_mega/motion.py` | 3-D RGB motion pathway | 3-D stem + temporal projection |
| `src/isl_mega/planning.py` | Latent plan/evidence attention | Ordered multi-scale queries, diversity/monotonicity losses, `memory_mask.sum >= 2` guard, dynamic routing |
| `src/isl_mega/pose.py` | 66-point pose graph & cross-articulator interaction | Smooth epsilon norms (`sqrt(sq + 1e-8)`), torso normalization, visible hand-centroid fallback, non-persistent buffers |
| `src/isl_mega/pretraining.py` | Global-teacher / local-student SSL | AMP-safe FP32 temperature scaling; not an exact SignDINO reimplementation |
| `src/isl_mega/temporal.py` | Hierarchical & adaptive temporal encoder | Non-singular backward gradients on single-frame/static inputs, FP32 salience pooling |
| `src/isl_mega/vision.py` | Specialist RGB streams & optional timm ViT | AMP dtype-safe masked backbone scatter, cached patch embedding |
| `scripts/convert_video_pose_npz.py` | Video + pose NPZ converter | Float64 `video_timestamps`, shape/monotonicity validation, optional `fingerspelling_spans` |
| `scripts/create_synthetic_dataset.py` | Synthetic dataset generator | Generates toy videos, NPZs, and optional synthetic `fingerspelling_text`/`fingerspelling_spans` |
| `scripts/evaluate.py` | Evaluation CLI | Validates `--beams`/`--batch-size`, saves `metrics.json` & `diagnostics.json` |
| `scripts/make_splits.py` | Source/signer-disjoint splitter | Grouped splits + disjointness verifier |
| `scripts/pretrain_visual.py` | Visual SSL pretraining CLI | Positive arg validation, CUDA-guarded `pin_memory`, atomic checkpoint save |
| `scripts/train.py` | Training CLI | Unified `GradScaler`, filters frozen param groups, logs `visual_lr` & counterfactual NLL gaps |
| `scripts/validate_data.py` | Dataset/manifest validator | Optional `--config` check, validates `fingerspelling_text` & spans |
| `tests/test_architecture.py` | v1 unit suite (45 tests) | Passed in 2026-11-08 v1 snapshot; awaiting re-run in PyTorch environment after v2 edits |
| `tests/test_novel_v2.py` | v2 mechanism & integrated smoke suite | Added `MockT5.generate` and edge-case tests; awaiting `pytest -q` run in PyTorch environment |

## Important scientific and engineering limits

1. No real ISL videos, annotations, pose tracks, signer IDs, or trained checkpoints are shipped.
2. The canonical pose schema expects 66 landmarks in the documented order. The adapter from the original ISLTranslate `.pose` data must be implemented or validated against the actual parser; the package deliberately does not guess an undocumented binary format.
3. The high-capacity research and matched baseline configs reference `timm` DINOv2-L/14 (`vit_large_patch14_dinov2`, 518px) and `google/flan-t5-xl`. Neither external checkpoint was downloaded or executed here.
4. The global-teacher/local-region-student pretraining is an ISL-MEGA variant inspired by sign-specific self-supervision; it is not an exact reproduction of SignDINO.
5. The face stream is visual face-region encoding, not validated phoneme/lipreading or ISL non-manual grammar understanding.
6. Fingerspelling supports both clip-level `fingerspelling_text` labels (weak MIL spotting) and optional localized `fingerspelling_spans` intervals/masks; automated character-localization precision on continuous ISL remains unverified on real data.
7. Source-disjoint splitting is implemented. Signer-disjoint splitting is optional and depends on reliable signer metadata. Near-duplicate detection and corpus provenance must be audited separately.
8. The latent planner adds ordering, orthogonality, dynamic halting, and counterfactual sensitivity diagnostics, but has no human-labelled temporal grounding supervision. RGB↔pose InfoNCE aligns same-frame features, not human temporal rationales.
9. No distributed multi-GPU trainer is included. Real GPU CUDA/`bf16` throughput and memory scaling have not been profiled on real checkpoints.
10. No BLEU/chrF/TER scores, ablation results, signer ratings, confidence intervals or real-data accuracy claims are included.

## Publication statement

This archive is an implementation candidate and experiment framework. It does not guarantee novelty, state-of-the-art performance, a Q1 journal submission, or acceptance. A Q1-quality paper requires empirical ablations on real corpora (`configs/research.yaml` vs `configs/ablation_v1_baseline.yaml` and A1–A19), exact baseline reproductions, source/signer leakage controls, multiple seeds, statistical uncertainty, qualified signer evaluation, data/model licensing checks, and transparent reporting.
