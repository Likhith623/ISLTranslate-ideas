# Final Integration Audit & Morning Decision Report — ISL-MEGA

**Audit Role:** Independent Senior Research Engineer (Final Review of Prompts 1–9 + Direct Code/Repo Re-Verification)  
**Target Subproject:** [`DINOv2-PoseGraph-3DCNN-Conformer/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer) (`isl-mega` v`0.2.0`)  
**Repository Root:** [`ISLTranslate-ideas/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas)  
**Git Baseline (`main`):** `215766756ec379fa4001f2b6e01c74b5e824ec14` *(preserved untouched)*  
**Git Fix Branch (`HEAD`):** `research/verified-baseline-fixes` @ `04faef007b2da8c5595aa691ccb5fce80f0aeb4e`  
**Decision Verdict:** **`NOT READY`** *(locally blocked on missing PyTorch runtime, missing 543->66 data adapter, zero real continuous ISL assets, and unverified HF/timm execution)*

---

## Executive Summary

Across ten deep verification passes covering source code, autograd graph connectivity, dataset provenance, optimization/losses, local hardware physics, 2020–Nov 2026 sign-language literature, and surgical bug fixes on branch `research/verified-baseline-fixes`:

1. **Software Integrity Improved, Runtime Suite Still Needs PyTorch Verification:** All 7 confirmed baseline defects identified in earlier prompts (dead unconditional `FineDetailRetrieval` causing the `test_compact_v3_config_and_motion_disabled_forward_backward` failure, `fp16` `sqrt(... + 1e-8)` subnormal underflow, CUDA resume `cuda_rng` crash + `GradScaler` persistence, `error_if_nonfinite=True` `fp16` crash, unsafe full-clip CTC fallback + missing `label_known`, `evaluate.py` self-collision on `valid.jsonl`, `validate_data.py` `(0, 48)` body slice typo, and unweighted `pretrain_visual.py` epoch loss) are cleanly fixed on `research/verified-baseline-fixes` (`04faef0`) with `main` (`2157667`) untouched. Bytecode compilation, pure-Python config/metric imports, AST structural invariants, and all five YAML schema validations pass (`EXIT 0`), **but dynamic `pytest` execution (`66` collected items across `65` functions) cannot run on this macOS host because `torch` is not installed (`EXIT 2`/`EXIT 4`)**.
2. **Real Continuous ISL Data Is Absent & Workspace Assets Are Incompatible:** Zero real ISL videos, `.pose` files, CSV annotations, or real NPZ clips exist on disk. Sibling [`CNN-BiLSTM/data/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/data/README.md) contains only `.gitkeep` and Markdown placeholders; [`InceptionCNN-Transformer/preprocessing/output/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/InceptionCNN-Transformer/preprocessing/extract_keypoints.py) holds `3,543` **isolated ASL Citizen** `.npy` pose clips `(60, 85, 3)` incompatible with continuous ISL sentences and lacking wrists/nose/cheeks. Moreover, [`CNN-BiLSTM`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/src/isl_slt/features/pose.py#L17-L40) exports `[BODY, LEFT, RIGHT, FACE]` with eyelid indices, contradicting [`docs/data_contract.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md#L36-L82) (`[LEFT, RIGHT, BODY, FACE]`).
3. **Architecture & Compute Bottlenecks Require Discipline Before Cluster Training:** Even after Fix 1 removes `2,363,904` dead `detail` parameters from [`research_v3_compact.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml) (`142,122,080` internal + `~86.58M` frozen `ViT-B/14` + `~247.58M` `FLAN-T5-base` = **`~476.28M`** total), training `v3_compact` unmodified at `B=4, T=300` requires **`~37.86 GiB`** (`ViT-B/14` runs uncached on `4,800` crop passes/step = `97.4%` of FLOPs, while dense `[B*T, N, N, 512]` tensors in [`EdgeGraphBlock`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L128-L130) allocate `~20.55 GiB` activations). Full [`research.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research.yaml) (`~4,146.36M` params post-Fix-1, `518px`, `3x ViT-L/14`, `FLAN-T5-xl`) is infeasible on single academic GPUs.
4. **Scientific Positioning Must Shed Speculative Bloat:** Do **not** pursue `EvoHARE-Net` or copy `CARP-S2T`'s broken `TMC` Dirichlet equations or unimplemented `Bi-S6` Mamba. Retain the lean **`v3_compact` core archetype**, remove unused capacity (`planner_layers: 0` when planning losses are off, cache frozen ViT features, sparsify/shrink `PoseExpert` edge memory, replace `fusion_entropy_weight` / ungrounded ERF with `QMF`-consistent quality calibration + asymmetric single-modality counterfactual degradation), and anchor a Q1 paper cleanly against **`ISLiM` (`WSLP 2026`, `5.85` BLEU-4 on `iSign`)** and **`ISLTranslate` (`ACL 2023`, `6.09`)**.

---

## A. Repository Integrity & Baseline Preservation

Direct Git and filesystem checks confirm complete isolation of baseline history and strict containment of Prompt 9 modifications:

| Verification Item | Verified State at Audit Completion | Evidence / Command Check |
|---|---|---|
| **Original Baseline Branch (`main`)** | `215766756ec379fa4001f2b6e01c74b5e824ec14` (*unchanged*, zero force pushes or resets) | `git log -n 3 --oneline main research/verified-baseline-fixes` |
| **Fix Branch (`research/verified-baseline-fixes`)** | `04faef007b2da8c5595aa691ccb5fce80f0aeb4e` (`fix(isl-mega): apply verified baseline defect corrections and regression tests`) | Single clean commit on top of `main` |
| **Tracked Diff Scope (`main..HEAD`)** | **9 files (`+486 / -34` lines):** 8 surgical source/script edits + [`test_baseline_fixes.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py) | `git diff --stat main..research/verified-baseline-fixes` |
| **Unintended File Modifications** | **Zero.** All 5 YAML files under [`configs/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml), both original tests ([`test_architecture.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py), [`test_novel_v2.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py)), all tracked docs, and all sibling repos (`CNN-BiLSTM`, `InceptionCNN-Transformer`, `PoseGraph-Conformer`) match `main` byte-for-byte | `git -C .. diff --exit-code main -- CNN-BiLSTM InceptionCNN-Transformer PoseGraph-Conformer DINOv2-PoseGraph-3DCNN-Conformer/configs DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py` -> `EXIT 0` |
| **Checkpoint Integrity** | Zero `.pt`/`.pth`/`.ckpt`/`.safetensors` files exist inside `DINOv2-PoseGraph-3DCNN-Conformer/`. Only external workspace checkpoint [`best_model.pt`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/InceptionCNN-Transformer/model/checkpoints/best_model.pt) (`13.93 MB`, `Nov 11 16:16`) is untouched | `find . -maxdepth 5 \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" \) -ls` |
| **Secret & Credential Exposure** | **Clean.** Zero `.env` runtime files exist (only placeholder template [`CNN-BiLSTM/.env.example`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/.env.example) with empty `HF_TOKEN=`), and regex scan for HF/GitHub/OpenAI/AWS/private keys matched 0 real secrets | `grep -rnE "(hf_...\|ghp_...\|sk-...\|AKIA...\|BEGIN .*PRIVATE KEY)" .` -> zero matches |
| **Untracked Audit Deliverables** | Only [`REPOSITORY_CONTEXT_FOR_EXTERNAL_REVIEW.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/REPOSITORY_CONTEXT_FOR_EXTERNAL_REVIEW.md) and [`reports/overnight/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/09_VERIFIED_CODE_CHANGES.md) (`01`..`10` + `START_HERE_TOMORROW.md`) | `git -C DINOv2-PoseGraph-3DCNN-Conformer status` |

---

## B. Verified Code Fixes & Exact Validation Results

### Summary of Applied Baseline Fixes (Commit `04faef0`)

All 7 fixes and 8 regression tests documented in [`09_VERIFIED_CODE_CHANGES.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/09_VERIFIED_CODE_CHANGES.md) were re-verified directly against committed files:

1. **Temporal Dead-Parameter Fix ([`temporal.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L218-L233)):** `self.detail` is now `None` when `adaptive_consolidation=True`, eliminating `10` un-used parameter tensors (`p.grad is None`) that caused `test_compact_v3_config_and_motion_disabled_forward_backward` to fail (`-16,768` in `smoke_v2`, `-2,363,904` in `v3_compact`, `-4,200,448` in `research`).
2. **FP16 Subnormal `sqrt` Guards ([`temporal.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L256-L260), [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L69-L255)):** All 5 Euclidean/Frobenius `sqrt(... + 1e-8)` ops now evaluate inside `.float()` before casting back to input dtype, preventing `0.5 / 0.0 -> Inf -> NaN` backward under `amp_dtype: fp16`, and `raw_dt` matches `step_dt.dtype`.
3. **CUDA Resume RNG + FP16 `GradScaler` Crash Fixes ([`checkpoint.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/checkpoint.py#L12-L68), [`train.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L144-L150)):** Normalized `cuda_rng` tensors to CPU on save/load with multi-to-single GPU fallback, persisted `scaler.state_dict()`, and set `error_if_nonfinite=False` when `scaler.is_enabled()` so FP16 overflow steps back off cleanly instead of crashing in `clip_grad_norm_`.
4. **Unsafe Full-Clip CTC Fallback + `label_known` Wiring ([`losses.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L22-L47), [`train.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L98-L102)):** Added optional `label_known` filter and stopped `span_known=True` clips with short/empty spans from silently falling back to full-sentence CTC alignment.
5. **Evaluation Self-Collision Fix ([`evaluate.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py#L17-L71)):** Added `build_eval_split_records` so `--manifest valid.jsonl` or `train.jsonl` does not duplicate itself into `"test"` while keeping cross-split disjointness checks active.
6. **Pose Body Validation Slice Fix ([`validate_data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/validate_data.py#L63-L64)):** Fixed `"body"` region coverage slice from `(0, 48)` to `(42, 48)`.
7. **SSL Crop-Weighted Loss Logging ([`pretrain_visual.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/pretrain_visual.py#L37-L62)):** Weighted epoch loss averages by active valid crop count (`crop_count`) rather than unweighted batch count.
8. **8 Targeted Regression Tests ([`test_baseline_fixes.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L1-L406)):** Brings the suite to **65 test functions / 66 pytest items** (`44/45` in `test_architecture.py` + `13/13` in `test_novel_v2.py` + `8/8` in `test_baseline_fixes.py`).

### Independent Validation Ledger Executed on `04faef0`

Every check below was re-run freshly on the final branch state:

| Validation Check | Exact Command Executed | Observed Result & Boundary |
|---|---|---|
| **Python Bytecode Compilation** | `cd DINOv2-PoseGraph-3DCNN-Conformer && /opt/homebrew/bin/python3 -m compileall -q src scripts tests` | **`PASS` (`EXIT 0`)** across all 27 Python files (`src/`, `scripts/`, `tests/`). |
| **AST Parse Across Repo** | `"/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/python"` `ast.parse` on all 27 `.py` files + invariant checks | **`PASS` (`EXIT 0`)**: 27/27 parsed; `.float()` on all `sqrt(... + 1e-8)` in `src/`; `detail` mutual exclusion; `(42, 48)` slice in `validate_data.py`; `65` test defs (`66` pytest items). |
| **Config Schema Validation** | `load_config()` loaded cleanly via `importlib` (without `torch`) across all 5 files in `configs/*.yaml` | **`PASS` (`EXIT 0`)**: `ablation_v1_baseline`, `research`, `research_v3_compact`, `smoke`, `smoke_v2` all validate. |
| **Package & Optional Metric Import Boundary** | `PYTHONPATH=src /opt/homebrew/bin/python3 -c 'import isl_mega'` vs direct load of `config.py` / `metrics.py` | **Expected Local Fail (`EXIT 1`):** `import isl_mega` eagerly imports `model.py` (`import torch`) -> `ModuleNotFoundError: No module named 'torch'`. Standalone `config.py` and `metrics.py` load cleanly (`metrics.py` raises intended `ImportError` on missing `sacrebleu`). |
| **Targeted Failing Test (`pytest`)** | `PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q tests/test_novel_v2.py::test_compact_v3_config_and_motion_disabled_forward_backward` | **`BLOCKED / COLLECTION ERROR` (`EXIT 4`)**: `tests/test_novel_v2.py:9: ModuleNotFoundError: No module named 'torch'`. |
| **New Regression Tests (`pytest`)** | `PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q tests/test_baseline_fixes.py` | **`BLOCKED / COLLECTION ERROR` (`EXIT 2`)**: `tests/test_baseline_fixes.py:10: ModuleNotFoundError: No module named 'torch'`. |
| **Full Unit & Forward/Backward Suite (`pytest`)** | `PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q` | **`BLOCKED / 3 COLLECTION ERRORS` (`EXIT 2`)**: `test_architecture.py`, `test_baseline_fixes.py`, `test_novel_v2.py` all fail at `import torch`. |

> [!WARNING]
> **Strict Honesty Boundary:** Neither the existing 58 items nor the 8 new regression tests—nor real forward/backward passes, GPU `fp16`/`bf16`, `timm` DINOv2, or Hugging Face `FLAN-T5` tokenizers/weights—have been executed at runtime on `04faef0` on this host due to absent `torch` and external dependencies. That dynamic PyTorch test pass is **mandatory step #1** on the target Python/GPU environment tomorrow.

---

## C. Architecture Validation & Unresolved Defects

### 1. Actual Implemented Architecture vs. Design Documents

Tracing [`src/isl_mega/model.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/model.py#L19-L227) end-to-end confirms:

- **Input & Alignment ([`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L31-L429)):** Reads pre-packed `.npz` (`frames [Tv, H, W, 3]`, `video_timestamps [Tv]`, `pose_xy [Tp, 66, 2]`, `pose_timestamps [Tp]`), retains **all `Tv` frames** (zero temporal downsampling), linearly interpolates pose within `<= max_pose_gap_seconds` (`0.20s`), and applies uniform letterboxing `[B, T, 3, S_img, S_img]`. If `boxes` are absent, `derive_rois` generates bounding boxes from pose, collapsing face/hand RGB crops to invalid when `< 2` keypoints survive.
- **Appearance Stream ([`vision.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L20-L243)):** Bilinearly crops 4 ROIs `[B, T, 4, 3, S_crop, S_crop]` (`face, left_hand, right_hand, body`) via `F.grid_sample`. Encodes valid ROIs via `TimmBackbone` (CLS vector `[N, D_vis]` only—spatial patch tokens discarded; **no Region-LoRA adapters exist**). Notice `v3_compact` keeps `freeze_backbone: true` with shared ViT-B/14 yet re-evaluates DINOv2 on all `4 * B * T` images every training step (`97.4%` of FLOPs!).
- **Pose Stream ([`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L13-L434)):** Expects `[LEFT(0:21), RIGHT(21:42), BODY(42:48), FACE(48:66)]`, computes `11`-channel kinematics (`x, y, rel_x, rel_y, tanh(vx/5), tanh(ax/25), conf, valid, confidence_known`), runs intra-region `EdgeGraphBlock` attention, optional `CrossArticulatorGraph` over `36` phonological landmarks (`gate_init=0.1`), and pools into 4 `[B, T, D_pose]` streams. **Activation Trap:** `EdgeGraphBlock` instantiates `[B*T, N, N, D_pose]` dense edge tensors across all node pairs (`~20.55 GiB` saved activations at `B=4, T=300`!).
- **Motion & Fingerspelling ([`motion.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/motion.py), [`fingerspelling.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fingerspelling.py)):** `MotionExpert` (`Conv3d` + `Conv1d`) is cleanly disabled (`self.motion = None`) in `v3_compact`; `FingerspellingExpert` concatenates `[left_rgb, right_rgb, left_pose, right_pose]` into a 2-layer BiGRU + 40-class CTC + span-routing spotter `[B, T, D_fs]`.
- **Fusion, Temporal & Planner ([`fusion.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py), [`temporal.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py), [`planning.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/planning.py)):**
  - `ReliabilityAwareFusion` projects all 10 `STREAM_NAMES` (`D_fus`), adds `type_embedding`, runs Pre-LN self-attention (injecting `self.null` if all streams missing), gates weights using `quality_gate + log(sigmoid(reliability_net(q)))`, and applies `ContinuousTimeLagBias` neighbor cross-attention -> `fused [B, T, D_fus]`. *(Note: **Zero Dirichlet concentration parameters, Subjective Logic uncertainty, or Neural CDE differential solvers exist**).*
  - `HierarchicalTemporalEncoder` runs `MultiScaleTemporalStem` (`[B, T, D_t]`), adaptive event consolidation (`M = ceil(T/2)`), `GlobalTemporalBlock` Conformer layers (**zero Mamba/Bi-S6 code exists**), `AdaptiveDetailRetrieval`, transpose unpooling, and final concatenation `[B, T, D_t]`.
  - `LatentPlanGrounder` runs `PlanGroundingBlock` cross+self attention -> `plan_tokens [B, K, D_t]`.
- **Decoder ([`decoder.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py)):** Concatenates `bridge(memory)` + `plan_bridge(plan)` `[B, T+K, D_t5]` into full `FLAN-T5` encoder + decoder -> translation logits `[B, L, |V|]` (`OT` & `counterfactual` disabled in `v3_compact`).

### 2. Post-Fix-1 Exact Parameter Counts & Gradient Audit (`04faef0`)

Recomputed directly from module constructors in `src/isl_mega/*.py` and all 5 YAML configs **after** Fix 1 removed dead `FineDetailRetrieval` from adaptive runs:

| Config Profile | Post-Fix-1 Exact `ISL-MEGA` Internals | External `timm` ViT *(Est.)* | External `FLAN-T5` *(Est.)* | Combined Total *(Est.)* | Combined Trainable *(Est.)* | Combined Frozen *(Est.)* | Key Gradient / Dead-Param State |
|---|---:|---:|---:|---:|---:|---:|---|
| [`smoke.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/smoke.yaml) | **556,290** | `0` *(3x native)* | ~76.96M (`small`) | ~77.52M | ~77.52M | `0` | All internal weights active |
| [`smoke_v2.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/smoke_v2.yaml) | **561,937** *(-16,768 fix)* | `0` *(3x native)* | ~76.96M (`small`) | ~77.52M | ~77.52M | `0` | Temporal dead `detail` eliminated |
| [`research_v3_compact.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml) | **142,122,080** *(-2,363,904 fix)* | ~86.58M (`1x ViT-B/14`) | ~247.58M (`base`) | **~476.28M** | **~389.70M** | **~86.58M** | `395,520` zero-grad weights remain in `fusion.projections["motion"]`; `28.37M` planner params active without plan losses |
| [`ablation_v1_baseline.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/ablation_v1_baseline.yaml) | **388,993,150** | ~913.11M (`3x ViT-L/14`) | ~2,849.76M (`xl`) | ~4,151.86M | ~4,151.86M | `0` | Prohibitively large (`>1 TB` un-checkpointed activations) |
| [`research.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research.yaml) | **383,498,969** *(-4,200,448 fix)* | ~913.11M (`3x ViT-L/14`) | ~2,849.76M (`xl`) | **~4,146.36M** | **~4,146.36M** | `0` | Unrunnable single-GPU + conflicting losses |

*(Note: In `research_v3_compact.yaml`, `with_decoder=False` internal parameter count is `140,937,824` + `1,184,256` T5 bridges = `142,122,080` total internals: `PoseExpert` `34,224,132`, `HierarchicalTemporalEncoder` `54,433,934`, `LatentPlanGrounder` `28,368,384`, `ReliabilityAwareFusion` `22,571,940`, `FingerspellingExpert` `1,337,898`, `VisualBackbone.norm` `1,536`.)*

### 3. Remaining Unresolved Defects Intentionally Deferred From Prompt 9

Per [`09_VERIFIED_CODE_CHANGES.md` §5](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/09_VERIFIED_CODE_CHANGES.md#L275-L290), these confirmed architectural/training defects were **not** modified yet because they affect configs, existing unit contracts, or model designs and require deliberate research sign-off:

- **P0 Training/Runtime Gaps:**
  1. **Zero real `FLAN-T5` + `timm` DINOv2 verification:** All decoder tests use [`DummyT5`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py) / [`MockT5`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py#L66-L106).
  2. **Uncached Frozen ViT (`97.4%` FLOPs) & Dense `EdgeGraphBlock` (`20.55 GiB` activations):** `v3_compact` defaults (`B=4, T=300`) consume **`~37.86 GiB` VRAM** until frozen ViT embeddings are cached offline (or chunked in `no_grad`) and `EdgeGraphBlock` replaces `[B*T, N, N, D]` tensors with a lightweight head-bias projection.
  3. **Newly initialized `decoder.bridge` & `decoder.plan_bridge` assigned tiny `decoder_lr`:** [`train.py:251-254`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L251-L254) matches `"decoder."` instead of `"decoder.model."`, starving `bridge` and `plan_bridge` at 4x–10x lower LR (`5e-5` vs `2e-4` in `v3_compact`).
  4. **Checkpoint selection & scheduler anomalies:** [`train.py:331`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L331) picks `best_valid.pt` using composite 9-loss `valid_loss` rather than `valid_translation` / `chrF++`; `--max-steps` overwrites `last.pt` / `best_valid.pt`; `build_epoch_scheduler` holds `0.5x` LR for all of Epoch 1 (and skips warmup when `warmup_epochs == 1`); `DataLoader` lacks frame caps / length bucketing.
- **P1 Scientific & Mathematical Design Tensions:**
  5. **`fusion.projections["motion"]` & `PlanGroundingBlock` bloat:** When `motion.enabled=False`, `fusion.py` keeps `395,520` zero-grad params shrinking under `AdamW` weight decay; `PlanGroundingBlock` inadvertently runs **two** 4x FFNs per layer (`nn.TransformerEncoderLayer` FFN + trailing `self.ffn`), doubling planner capacity (`28.37M`) even when `plan_*` weights are `0`.
  6. **Objective & Anatomical Weaknesses:** In [`losses.py:105-125`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L105-L125), `compute_fusion_entropy_loss` fights reliability downweighting (fortunately disabled in `v3_compact`); `compute_fingerspelling_spotting_loss` (`L97`) equates character count / frame count; [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L378) snaps the body anchor by `0.5 * torso_scale` when one shoulder disappears, uses per-frame `estimate_torso_scale` (jumping on single-frame misses), and mirrors right-hand pose `x` inside `PoseExpert` while leaving right-hand RGB crops unflipped inside `AppearanceExpert`.

---

## D. Dataset Provenance, Real-Data Readiness & Hardware Feasibility

### 1. Physical Data Reality & Provenance Corrections

Direct inspection verified in [`05_DATASET_AND_REAL_DATA_READINESS.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/05_DATASET_AND_REAL_DATA_READINESS.md) establishes:

- **Physical Workspace Files:** **Zero continuous ISL assets on disk.** Only synthetic random noise ([`create_synthetic_dataset.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/create_synthetic_dataset.py)), an example JSON line ([`examples/manifest_record.json`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/examples/manifest_record.json)), empty `CNN-BiLSTM/data/` folders, and **3,543 ASL Citizen isolated-gloss `(60, 85, 3)` `.npy` files** under `InceptionCNN-Transformer/preprocessing/output/` (missing wrists, timestamps, RGB frames, and English sentences).
- **Corrected Official Corpus Facts & Licensing:**
  - **`ISLTranslate` (`ACL Findings 2023`):** **31,222** sentence pairs (`11,655` vocab, `291` human-validated pairs), `CC-BY-NC-4.0`.
  - **`iSign v1.1` (`ACL Findings 2024`):** **118,228** pairs (`~252h`), `CC-BY-NC-SA-4.0` (gated HF access). Subsumes `~31K` ISLTranslate + `~7K` CISLR.
  - **`GloSL-ISL` (`pedroodb/glosl-isl`):** **126,846** pose-only `.pose` samples (`269.03h`), **`CC-BY-NC-4.0`** (*previous project docs falsely wrote `CC-BY-4.0`!*), empty `signer` metadata, zero RGB videos (incompatible with RGB-required `ISLVideoDataset`).
  - **`ISH-NEWS` (*Sci. Rep.* July 2026):** **4,222** sentence clips; raw broadcast video redistribution restricted.
  - **`ISL-Fingerspelling` / `ISL-FS-Dense` (`WSLP 2025`):** `1,308` manual segments from 499 ISH News videos + `11,700+` mined from `iSign`, `CC-BY-NC-4.0`.
- **Missing Adapter & Silent Corruption Hazard:** `ISL-MEGA` contains **no `.pose` binary reader, no MediaPipe Holistic extractor, and no 543->66 converter**. Directly reusing [`CNN-BiLSTM`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/src/isl_slt/features/pose.py#L17-L40)'s output silently scrambles `[BODY 0:6, LEFT 6:27, RIGHT 27:48, FACE_eyelid 48:66]` against `ISL-MEGA`'s `[LEFT 0:21, RIGHT 21:42, BODY 42:48, FACE_contract 48:66]`, while `Phase 1` doc slice `468:489` extracts face mesh points instead of left hand `501:522`!
- **Split Leakage Exposure:** All shipped YAMLs set **`require_signer_disjoint: false`**; DSU splitting falls back to `source_id` alone when `signer_id` is `None`; no perceptual video hashing (pHash), English text deduplication, cross-corpus YouTube parent-ID canonicalization, or `pretrain_visual.py` valid/test exclusion check exists yet.

### 2. Hardware Feasibility Summary ([`07_HARDWARE_AND_PERFORMANCE_REPORT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/07_HARDWARE_AND_PERFORMANCE_REPORT.md))

| Environment / Config | Measured / Calculated Capacity | Verdict |
|---|---|---|
| **Local Host (Apple M2 Air)** | **`8.00 GiB` unified RAM (`7.44 GiB` usable), `11 GiB` free disk (`95%` full), `0` CUDA GPUs, `torch` absent, scripts lack `mps` device routing** | **Impossible for real training or `v3_compact`** (`6.16 GiB` static weights+AdamW alone; two `v3` checkpoints = `9.41 GiB`). Viable **only** for CI `pytest` (`smoke.yaml` / `smoke_v2.yaml` < `1.5 GiB`) after freeing `>=25 GiB` disk & creating a `.venv`. |
| **Unmodified `research.yaml` (`v2`)** | **`~4.15B` params**, `61.85 GiB` static FP32+AdamW, `~1–2.4 TB` un-checkpointed `518px` ViT-L activations | **Infeasible** on single `80 GiB` A100/H100 GPUs. |
| **Unmodified `research_v3_compact.yaml`** | **`~476.3M` params** (`6.16 GiB` static), `~228.3 TFLOPs/step` & **`~37.86 GiB` peak VRAM** at `B=4, T=300` | Needs `>= 40–48 GiB` GPU (A6000 / A100-40/80GB) unless cached ViT & sparse pose edge bias are added. |
| **Refactored Lean Real-Data Run (`v3.1`)** | Offline cached `ViT-B/14` features (`-97.4%` FLOPs, `~1.3 MiB/clip`), compact `PoseExpert` (`d=256`, head-bias edge MLP -> `<0.25 GiB` pose acts), `planner_layers=0`, `flan-t5-small`/`base` (`119.5M`/`290.3M` train) | **`~2.84 GiB` (`flan-t5-small`) / `~5.38 GiB` (`flan-t5-base`) peak VRAM at `B=4, T=300` -> trains comfortably on a single `16–24 GiB` GPU (`L4`/`RTX 4090`/`A10G`)!** |

---

## E. Training Readiness Verdict

### Verdict: **`NOT READY`**

Chosen from `{NOT READY, READY FOR SYNTHETIC TESTING, READY FOR TINY REAL-DATA EXPERIMENT, READY FOR BASELINE TRAINING, READY FOR FULL EXPERIMENTS}`.

**Why `NOT READY` Today:**
1. **Runtime verification missing (`torch` absent locally):** Commit `04faef0` fixes have passed static AST/compile/YAML checks, **not** live `pytest` execution.
2. **Real data & ingestion absent:** Zero continuous ISL datasets exist locally, zero 543->66 pose/video NPZ exporters exist, and real `timm` DINOv2 + `FLAN-T5` checkpoints have zero integrated test runs.

**Exact Transition Gates Tomorrow:**
- **`-> READY FOR SYNTHETIC TESTING`** *(~15 min on target PyTorch GPU host):* Install `.[all]` in a clean venv and get `66/66` `pytest -v` green + run `create_synthetic_dataset.py` + `train.py` (`smoke_v2.yaml`, real `MockT5`/HF tokenizer) end-to-end.
- **`-> READY FOR TINY REAL-DATA EXPERIMENT`** *(Day 1–2):* Implement & visually overlay-verify canonical 543->66 `.pose`+MP4 -> NPZ converter (fixing `[LEFT, RIGHT, BODY, FACE]` & documented face IDs), validate real `google/flan-t5-base` + `vit_base_patch14_dinov2`, cache frozen ViT features (`--cache-viT` or pre-extracted shard), and verify 200–500 real `iSign` clips overfit sanity + leakage-free DSU+text/YT splits.
- **`-> READY FOR BASELINE TRAINING` -> `READY FOR FULL EXPERIMENTS`:** Refactor `EdgeGraphBlock` edge tensor memory (`20.55 GiB -> <0.25 GiB`), fix `decoder.bridge` LR group + `valid_translation` / `chrF++` best-checkpoint selector + step scheduler, run single-seed baseline parity vs `ISLiM`, then scale `>= 3` seeds + C1–C5 corruption + `Beyond BLEU` evaluations.

---

## F. Scientific Novelty & Strategy Assessment (Direct Answers 1–7)

### 1. Should the compact core architecture be retained?
**Yes — keep the `v3_compact` core design, strip the speculative bloat.** Specifically retain: shared frozen/light-adapted DINOv2 `ViT-B/14` region crops (`224px`) + canonical 66-point intra-/cross-articulator pose graph + observable quality/provenance-gated temporal-lag fusion (`ReliabilityAwareFusion`) + stride-2 `MultiScaleTemporalStem` & `Conformer` + pretrained `FLAN-T5-small/base` (`motion.enabled: false` initially). **Drop/bypass** `LatentPlanGrounder` (`planner_layers: 0` in the clean baseline so `28.37M` extra params do not confound attribution), keep `ot_weight = 0`, `fusion_entropy_weight = 0`, and implement ViT caching + low-memory `EdgeGraphBlock`.

### 2. What is the biggest technical or scientific weakness?
1. **Experimental Vacuum & Missing Adapter:** Zero runs exist on any real ISL dataset or real `FLAN-T5`/`DINOv2` weights, and no verified 543->66 adapter bridges [`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py) to `iSign`/`ISLTranslate`.
2. **Conceptual Contradiction in Quality Gating:** Adding `reliability_net(q)` without explicit degradation/calibration supervision while upstream [`compute_fusion_entropy_loss`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L105-L125) pulls all stream weights toward uniform `1/S` directly undermines selective noise rejection, leaving no proven separation from `Uni-Sign` (`ICLR 2025`) raw confidence gating.

### 3. What is the strongest publishable hypothesis?
> **Provenance-Calibrated Quality-Aware Multimodal Reliability for Degraded Gloss-Free ISL Translation:**  
> In unconstrained gloss-free ISL (`iSign` and `ISLTranslate`), modulating 66-point anatomical pose graphs and DINOv2 region crops via **observable, provenance-aware quality calibration** (`QMF`-consistent weighting separating detector confidence from binary/missing proxies, aligned across variable timestamps) trained with **asymmetric single-modality counterfactual degradation** preserves translation quality (`chrF++`, semantic QA accuracy, `BLEU-4` retention ratio) under realistic landmark jitter, unilateral hand occlusion, and RGB motion blur significantly better than both pose-only translators (`ISLiM`) and unweighted or scalar-confidence fusion—without relying on gloss labels or decoder language priors.

### 4. Should `CARP-S2T` be adopted as currently designed?
**No.** As documented across [`08_SCIENTIFIC_NOVELTY_REVIEW.md` §4.1](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/08_SCIENTIFIC_NOVELTY_REVIEW.md#L548-L566) and [`04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md):
- Its **Dirichlet Evidential Fusion** copies `TMC`/`EDL` class-simplex equations (`alpha = e + 1`, MSE + variance + KL) onto unlabelled complementary streams (`vis, pose, fs`), where ground-truth reliability `y_m` does not exist and modalities are non-exclusive. Replace it with **Quality-Aware Multimodal Fusion (`QMF`, ICML 2023)** style predictive confidence calibration.
- Its **`Bi-S6` (Mamba)** has zero code, runs **1.8x more MACs** than stride-2 Conformer at typical `T <= 300` (`Conformer` `O(M^2)` is only `1.4%–2.7%` of temporal FLOPs and `0.03%` of total model FLOPs), and loses native macOS/MPS portability.
- Its blueprint [`reports/05` L78-L88](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/05_IMPLEMENTATION_BLUEPRINT.md#L78-L88) reintroduces the wrong `0..5` body slice order (`left_hand=6:27`) and naive `-x` mirroring without dominance role handling.

### 5. Should `EvoHARE-Net` be pursued?
**No — permanently reject.** Dynamic structural plasticity and Continual Backpropagation (*Nature* 2024; `NISPA` 2022) target **non-stationary sequential multi-task streams**. Applying error-triggered module growth on stationary offline YouTube captions (`iSign`/`ISLTranslate`) simply memorizes noisy transcripts, destabilizes `FLAN-T5` cross-attention manifolds, and breaks compiled graphs—while **Yazdani et al. (`NAACL 2025`)** already demonstrated standard fine-tuning beats continual-learning architectures even in cross-lingual SLT.

### 6. What evidence is still missing before writing a paper?
Seven mandatory requirements before drafting any Q1 manuscript:
1. Verified runtime pass (`pytest -v` 66/66 green) + live `FLAN-T5` & `DINOv2` integration on GPU.
2. Verified 543->66 `.pose`+video NPZ exporter with plotted skeleton overlay confirmation (`LEFT`, `RIGHT`, `BODY`, `FACE`).
3. Cross-corpus leakage-clean splits on `iSign` + `ISLTranslate` (YouTube canonical parent ID + normalized text dedup + DSU component diagnostics + `291`-clip gold validation subset preserved).
4. Objective/architecture cleanup (`fusion_entropy_weight=0`, `QMF`-aligned stream calibration / asymmetric modality counterfactual dropout, single-GPU memory fixes, `decoder.bridge` LR routing).
5. Matched-split head-to-head baselines (`>= 3` seeds, paired bootstrap significance): reproduced `ISLiM` pose-only (`5.85` ref on `iSign`), `ISLTranslate Pose-SLT` (`6.09` ref), `RGB-only DINOv2`, `Unweighted Multimodal`, `Scalar-Confidence Multimodal`, and proposed `Provenance Quality+Counterfactual Fusion`.
6. Clean + 5 corruption stress suites (`C1` landmark jitter, `C2` chunk dropout, `C3` unilateral dominant-hand occlusion, `C4` RGB blur, `C5` combined noise) plus calibration metrics (risk-coverage AURC / negative stream-weight correlation with corruption).
7. `Beyond BLEU` evaluation (`chrF++`, ROUGE-L, LLM/semantic QA, zero-visual/shuffled-visual causal ablation proving visual reliance, and qualified Deaf ISL signer human evaluation).

### 7. What exact experiment should be executed first?
**Two-phase Day-1 -> Week-1 progression (never skip Step A for Step B):**
- **Immediate Sanity Run (`Day 1`, `GPU + Tiny Subset`):** Run `pytest -v` (`66/66` pass), then extract and inspect **`500` source-disjoint clips from `iSign`** (`200–400` train / `50` valid / `50` test, including `291`-pair gold checks if applicable) into cached `224px` crops + verified 66-point poses (`validate_data.py`), and verify that **unweighted RGB+Pose `flan-t5-small` overfits a `32`-clip batch cleanly** (checking real loss drop, finite `bf16` grads, and non-trivial `model.generate()` English outputs—no repetitive collapsed tokens).
- **First Scientific Experiment (`Week 1`, `iSign` dev split / `ISLTranslate`, `16–24 GB` GPU, cached frozen `ViT-B/14` + `flan-t5-small`):** Execute the **4-arm clean vs. C1–C5 corruption baseline ladder** under fixed splits & optimizer:
  1. `Pose-only` (matched `PoseExpert` + `Conformer-6` + `FLAN-T5`)
  2. `RGB-only` (shared frozen `ViT-B/14` crops + `Conformer-6` + `FLAN-T5`)
  3. `Unweighted Fusion` (`ReliabilityAwareFusion` without quality descriptor/reliability gate)
  4. `Provenance Quality + Reliability Fusion` (`5-descriptor` quality + `reliability_net` + asymmetric single-stream dropout)
  Only after this `~3 GiB` GPU ladder confirms clean separation and stable `chrF++`/`BLEU-4` gains should capacity scale to `google/flan-t5-base` (`~5.4 GiB`) and full multi-seed runs.

---

## G. Sequenced Next Actions (Strict Order)

1. **[CUDA/PyTorch Workstation — Day 1, 15 min]:** Check out `research/verified-baseline-fixes` (`04faef0`), install `pip install -e ".[all]"` in an isolated virtualenv, run `PYTHONPATH=src pytest -v` (`66` items), and execute the synthetic `smoke_v2.yaml` train + eval loop.
2. **[Data Engineering — Day 1–2]:** Build a tested `543->66` `.pose`+video converter script grounded strictly in [`docs/data_contract.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md), draw overlaid skeletons on 10 random `iSign` clips to verify hand/body/face orientation visually, and run [`validate_data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/validate_data.py).
3. **[Optimization & Memory PR — Day 2–3]:** Add frozen ViT feature caching (or chunked `torch.no_grad()` visual encoding), replace dense `[B*T, N, N, D]` tensors in [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L128-L130) with low-rank/head-bias edge attention, route `decoder.bridge`/`plan_bridge` to `visual_lr` in [`train.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L251-L254), switch `best_valid.pt` selection to `valid_translation` / `chrF++`, and expose `--no-save` for `--max-steps`.
4. **[Split Hygiene — Day 3]:** Add canonical YouTube parent-ID + normalized translation dedup checks + non-null `signer_id` reporting before freezing official `train`/`valid`/`test` `.jsonl` manifests (and ensure `pretrain_visual.py` excludes `valid`/`test`).
5. **[Real-Data Verification & Baseline Ladder — Day 4–7]:** Run real `FLAN-T5` overfit sanity test, then execute the 4-arm controlled ablation ladder (`Pose-only`, `RGB-only`, `Unweighted`, `Quality+Reliability`) on clean + degraded splits.

