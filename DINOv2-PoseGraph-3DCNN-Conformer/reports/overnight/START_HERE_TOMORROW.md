# START HERE TOMORROW — ISL-MEGA Morning Action Sheet

## 1. Current Project Status

- **Branches:** `main` preserved cleanly at `215766756ec379fa4001f2b6e01c74b5e824ec14`; **7 verified baseline fixes + 8 regression tests** are committed on `research/verified-baseline-fixes` at `04faef007b2da8c5595aa691ccb5fce80f0aeb4e` (plus [`reports/overnight/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/10_FINAL_MORNING_REPORT.md) `01`–`10`). Zero configs or checkpoints were overwritten.
- **Training Verdict:** **`NOT READY`** — static checks (`compileall`, AST checks, all 5 YAML config validations) pass (`EXIT 0`), **but runtime `pytest` could not run locally because `torch` is absent on this macOS host (`EXIT 2`/`4`)**, zero real continuous ISL videos/poses exist on disk, and `google/flan-t5-*` / `timm` DINOv2 weights remain untested end-to-end.

---

## 2. Three Most Important Findings

1. **Temporal Test Bug Fixed, Runtime PyTorch Pass Still Mandatory:** Adding `self.detail = FineDetailRetrieval(...)` unconditionally inside [`temporal.py#L218-L233`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L218-L233) left 10 dead tensors (`2.36M` in `v3_compact`) with `p.grad is None`. Fix 1 + `fp16` `sqrt` guards + CUDA `cuda_rng`/`GradScaler` + CTC span/eval/slice fixes are now in `04faef0`, **awaiting first live `pytest -v` (`66` items) in a PyTorch environment**.
2. **Zero Real ISL Data & Silent 66-Point Contract Mismatch:** Zero `ISLTranslate` / `iSign` / `GloSL-ISL` files exist locally (sibling [`CNN-BiLSTM`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/src/isl_slt/features/pose.py#L17-L40) exports `[BODY, LEFT, RIGHT, FACE]` with eyelids, colliding silently with `ISL-MEGA`'s `[LEFT(0:21), RIGHT(21:42), BODY(42:48), FACE(48:66)]`; [`InceptionCNN-Transformer`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/InceptionCNN-Transformer/preprocessing/extract_keypoints.py) only holds isolated ASL `(60, 85, 3)` skeletons; `glosl-isl` is `CC-BY-NC-4.0` pose-only, not `CC-BY-4.0`).
3. **`v3_compact` Needs 2 Efficiency Fixes Before Training (`37.86 GiB` -> `~5.4 GiB`):** On [`research_v3_compact.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml) (`B=4, T=300`), uncached frozen `ViT-B/14` eats **`97.4%` of FLOPs** (`222.3 TFLOPs/step`) and dense `[B*T, N, N, 512]` tensors in [`pose.py#L128-L130`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L128-L130) allocate **`20.55 GiB` activations** (`~37.86 GiB` total peak). Caching frozen ViT features + slimming `EdgeGraphBlock` + setting `planner_layers=0` drops peak VRAM to **`~5.38 GiB` (`flan-t5-base`) / `~2.84 GiB` (`flan-t5-small`)** for a single `16–24 GB` GPU.

---

## 3. Exact First Command to Run Tomorrow

On your **GPU / PyTorch workstation** (once `pip install -e ".[all]"` is active inside `DINOv2-PoseGraph-3DCNN-Conformer`):

```bash
cd DINOv2-PoseGraph-3DCNN-Conformer && git checkout research/verified-baseline-fixes && PYTHONPATH=src pytest -v tests/test_architecture.py tests/test_novel_v2.py tests/test_baseline_fixes.py
```

*(Expected first verification check: `66 passed` — specifically confirming [`test_compact_v3_config_and_motion_disabled_forward_backward`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py#L467-L486) and all 8 tests in [`test_baseline_fixes.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py) pass dynamically.)*

---

## 4. Exact File to Inspect First

Inspect [`scripts/train.py#L244-L344`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L244-L344) alongside [`docs/data_contract.md#L34-L85`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md#L34-L85):
- [`train.py#L251-L344`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L251-L344) still assigns untrained `decoder.bridge`/`plan_bridge` to tiny `decoder_lr`, saves `best_valid.pt` by composite `valid_loss`, holds `0.5x` LR for all of Epoch 1, and misses frame-length capping/bucketing.
- [`docs/data_contract.md#L34-L85`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md#L34-L85) defines the exact `66`-point `(LEFT 0:21, RIGHT 21:42, BODY 42:48, FACE 48:66)` contract needed for your ingestion adapter.

---

## 5. Next Implementation Task

1. **Confirm PyTorch `pytest` (`66/66`) + Real `FLAN-T5`/`DINOv2` Overfit Test:** Verify [`build_model(cfg, with_decoder=True)`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/model.py#L215-L227) loads genuine `google/flan-t5-small`/`base` + `vit_base_patch14_dinov2`, backward-passes cleanly in `bf16`, and generates non-degenerate text on synthetic data.
2. **Build & Visually Validate the `543->66` Real-Data Exporter:** Implement a verified MediaPipe `.pose` + MP4 -> NPZ converter enforcing `docs/data_contract.md` (`[LEFT, RIGHT, BODY, FACE]` + exact `18`-point face mapping, canonical YouTube ID + text-dedup + DSU splitting), inspect 10 rendered skeleton overlays visually, and pass [`validate_data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/validate_data.py) on a `500`-clip `iSign` pilot subset.
3. **Prepare Lean Training Optimization:** Cache frozen `ViT-B/14` ROI embeddings (or `no_grad` frame chunking), refactor [`EdgeGraphBlock`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L89-L141) to avoid `[B*T, N, N, D]` allocation, fix `decoder.bridge` LR grouping + `best_valid` (`valid_translation` / `chrF++`), and then run the 4-arm baseline ablation (`Pose-only`, `RGB-only`, `Unweighted`, `Quality+Reliability`) on clean + corrupted validation splits.

---

## 6. What Must NOT Be Done Yet

- **DO NOT launch full-scale training or [`configs/research.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research.yaml):** Requires `> 61.85 GiB` static + multi-TB activations and mixes 9 conflicting objectives.
- **DO NOT train locally on this 8 GB MacBook Air (`11 GiB` free disk):** Physically OOMs/fills SSD before epoch 1 completes.
- **DO NOT blindly pipe [`CNN-BiLSTM`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/src/isl_slt/features/pose.py#L38-L40) `.npy` poses or [`reports/phase_1_verification/07` (`468:489`)](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/phase_1_verification/07_ACTIONABLE_IMPLEMENTATION_PLAN.md#L34-L41) slices into `ISL-MEGA`:** Both silently permute joints / corrupt hands and face.
- **DO NOT implement speculative `CARP-S2T` (`TMC` Dirichlet ERF, unbenchmarked `Bi-S6` Mamba) or `EvoHARE-Net`:** Dirichlet class-simplex equations lack valid reliability targets `y_m`, Mamba offers zero gain at `T_c <= 90`, and dynamic structural plasticity harms stationary offline SLT.
- **DO NOT claim Q1 readiness, commercial licensing (`iSign` = `CC-BY-NC-SA-4.0`, `GloSL-ISL` = `CC-BY-NC-4.0`), or BLEU gains** before multi-seed real-data baselines (`ISLiM` parity), C1–C5 degradation curves, and `Beyond BLEU` causal evaluations exist.

