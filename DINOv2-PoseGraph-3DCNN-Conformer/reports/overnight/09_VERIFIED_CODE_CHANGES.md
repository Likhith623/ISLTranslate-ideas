# PROMPT 9 — VERIFIED CODE CHANGES AND CONTROLLED BASELINE FIXES

## 1. Repository State, Branch Provenance, and Scope Guardrails

- **Repository Root:** `/Users/likhith/Codes - Github./ISLTranslate-ideas`
- **Subproject Scope:** `/Users/likhith/Codes - Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer` (`isl-mega` v`0.2.0`)
- **Preserved Original Branch & Commit (`main`):** `215766756ec379fa4001f2b6e01c74b5e824ec14`
- **Active Fix Branch:** `research/verified-baseline-fixes`
- **Verified Fixes Commit Hash:** `04faef007b2da8c5595aa691ccb5fce80f0aeb4e` (`fix(isl-mega): apply verified baseline defect corrections and regression tests`)
- **Pre-Existing Untracked Assets Preserved Untouched:** `REPOSITORY_CONTEXT_FOR_EXTERNAL_REVIEW.md`, `reports/overnight/01` through `08`, and all five YAML files under `configs/` (`git diff --exit-code -- configs/ reports/` exited `0`).
- **Explicit Scope Prohibitions Observed:**
  - Zero speculative architectural additions (`CARP-S2T`, `EvoHARE-Net`, Dirichlet/evidential fusion, Bi-Mamba, Region-LoRA, structural growth).
  - Zero removal of optional or v1/v2/v3 functional modules (`MotionExpert`, `FineDetailRetrieval` when `adaptive_consolidation=False`, 10-stream `ReliabilityAwareFusion` interface, `LatentPlanGrounder`).
  - Zero test weakening (`tests/test_architecture.py` and `tests/test_novel_v2.py`, including `test_compact_v3_config_and_motion_disabled_forward_backward`, are untouched).
  - Ambiguous architectural/hyperparameter/protocol trade-offs were excluded from source/config edits and documented for review in [Section 5](#5-ambiguous--deferred-changes-requiring-review).

---

## 2. Complete File Inventory of Changes (`main..research/verified-baseline-fixes`)

- **Modified files (`8`):**
  1. [temporal.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L218-L261) (`+4 / -5`)
  2. [pose.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L69-L255) (`+4 / -4`)
  3. [checkpoint.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/checkpoint.py#L12-L68) (`+17 / -4`)
  4. [losses.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L22-L47) (`+5 / -4`)
  5. [train.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L98-L343) (`+6 / -5`)
  6. [evaluate.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py#L17-L71) (`+33 / -9`)
  7. [validate_data.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/validate_data.py#L63-L64) (`+1 / -1`)
  8. [pretrain_visual.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/pretrain_visual.py#L37-L62) (`+8 / -4`)
- **Added files (`1` tracked code/test file + this report):**
  - [test_baseline_fixes.py](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L1-L406) (`+406` lines, 8 targeted regression tests)
  - [09_VERIFIED_CODE_CHANGES.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/09_VERIFIED_CODE_CHANGES.md) (untracked overnight report matching `01`–`08`)
- **Deleted files (`0`):** None.
- **Total Commit Diff Stat (`04faef007b2da8c5595aa691ccb5fce80f0aeb4e`):** `9 files changed, 486 insertions(+), 34 deletions(-)`.

---

## 3. Detailed Record of Verified Fixes

> [!NOTE]
> **Local Execution Boundary:** Consistent with [02_TEST_FAILURE_ROOT_CAUSE.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/02_TEST_FAILURE_ROOT_CAUSE.md) and [07_HARDWARE_AND_PERFORMANCE_REPORT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/07_HARDWARE_AND_PERFORMANCE_REPORT.md), no PyTorch package is installed in any Python interpreter on the macOS host (`ModuleNotFoundError: No module named 'torch'`), and host constraints prohibit unapproved package installation. Accordingly, bytecode compilation, Python AST parsing, YAML configuration validation, and AST structural invariants were executed and verified (`PASS`), whereas all dynamic `pytest` / `torch` runtime checks remain **unexecuted locally (`ERROR` during collection on missing `torch`)** and must be executed on the target GPU/PyTorch environment.

---

### Fix 1 — Eliminate Dead Unconditional `self.detail` in `HierarchicalTemporalEncoder` (`adaptive_consolidation=True`)

1. **Exact Defect:** `HierarchicalTemporalEncoder.__init__` unconditionally instantiated `self.detail = FineDetailRetrieval(...)` even when `config.adaptive_consolidation=True`. In `forward()`, `adaptive_consolidation=True` routes strictly through `self.adaptive_detail` (`src/isl_mega/temporal.py:316`) and never invokes `self.detail` (`src/isl_mega/temporal.py:363`). Consequently, 10 trainable parameter tensors (`FineDetailRetrieval.{q,k,v,out,norm}`: `16,768` params in `smoke_v2`, `2,363,904` params in `research_v3_compact`, `4,200,448` params in `research`) stayed dead with `p.grad is None`.
2. **Evidence Reference:**
   - [02_TEST_FAILURE_ROOT_CAUSE.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/02_TEST_FAILURE_ROOT_CAUSE.md)
   - [03_ARCHITECTURE_AND_TENSOR_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/03_ARCHITECTURE_AND_TENSOR_AUDIT.md)
   - Failing assertion: [test_novel_v2.py#L467-L485](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py#L467-L485)
3. **Explanation of Correction:** Moved `self.detail = FineDetailRetrieval(...)` into the `else:` branch of `if config.adaptive_consolidation:` and set `self.detail = None` inside the `if config.adaptive_consolidation:` branch—matching the existing mutual-exclusion pattern already used for `salience_net`, `log_slot_sigma`, `slot_spread_proj`, and `adaptive_detail`. Preserves `enc.detail` intact whenever `adaptive_consolidation=False` (e.g., `configs/smoke.yaml`, `configs/ablation_v1_baseline.yaml`, and [test_architecture.py#L645-L669](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py#L645-L669)).
4. **Minimal Code Change:** [temporal.py#L218-L233](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L218-L233)

```python
        if config.adaptive_consolidation:
            self.detail = None
            ...
            self.adaptive_detail = AdaptiveDetailRetrieval(config.dim, config.heads)
        else:
            self.detail = FineDetailRetrieval(config.dim, config.heads, config.detail_window, config.coarse_stride)
            ...
            self.adaptive_detail = None
```

5. **Regression Test Added/Updated:**
   - Original strict test preserved untouched: [test_compact_v3_config_and_motion_disabled_forward_backward](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py#L467-L485).
   - Added explicit mutual-exclusion and full-module backward gradient check for both `smoke.yaml` (`adaptive=False`) and `smoke_v2.yaml` (`adaptive=True`): [test_temporal_encoder_has_no_unused_trainable_parameters](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L86-L112).
6. **Relevant Tests / Checks Executed:**
   - `/opt/homebrew/bin/python3 -m compileall -q src scripts tests`
   - AST mutual-exclusion invariant verification (`AST check 2`)
   - `PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q tests/test_novel_v2.py::test_compact_v3_config_and_motion_disabled_forward_backward`
   - `PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q tests/test_baseline_fixes.py`
7. **Recorded Output & Result:**
   - Bytecode / AST / config validation: `PASS` (`AST check 2 PASS: HierarchicalTemporalEncoder mutually excludes self.detail and adaptive modules`).
   - PyTorch runtime pytest: **BLOCKED (`exit 4` / `exit 2`)** due to missing host `torch` (`ModuleNotFoundError: No module named 'torch'`).
8. **Remaining Limitation:** Existing pretrained checkpoints saved with `adaptive_consolidation=True` from commit `2157667` will contain orphaned `temporal.detail.*` keys; loading them into the corrected module requires `strict=False` or stripping unused `temporal.detail.*` keys on load. *(Notice also that in `test_compact_v3_config_and_motion_disabled_forward_backward`, `loss = enc["memory"].sum()` reaches every remaining active `temporal` parameter—specifically `stem`, `_adaptive_pool` including `salience_net`, `log_slot_sigma`, `slot_spread_proj`, `coarse_projection`, `time_projection`, `time_bias`, `global_blocks`, `adaptive_detail`, and `final`—once `self.detail` is `None`; runtime PyTorch verification will confirm empirically).*

---

### Fix 2 — Guard `float16` Subnormal Underflow in `torch.sqrt(... + 1e-8)` (`temporal.py` & `pose.py`)

1. **Exact Defect:** Five Euclidean/Frobenius norm expressions used `torch.sqrt(... + 1e-8)` directly in input dtype (`temporal.py:257`, `pose.py:69, 77, 124, 255`). Under IEEE-754 `float16` (`amp_dtype: fp16`, explicitly suggested at `scripts/train.py:209`), the smallest positive subnormal is ~`5.96e-8`, so `+ 1e-8` rounds to `0.0`. Whenever `delta == 0` (`delta[:, 0]` at `t = 0`, padded frames, or repeated static frames in `temporal._adaptive_pool`) or two landmarks coincide in `pose.py`, forward evaluates `sqrt(0.0)` and backward produces `0.5 / 0.0 = Inf -> NaN` gradients. (`temporal.py:284` already used `.float()` for `slot_std`). Additionally, `temporal.py:256` assigned float32 `raw_dt` directly into `step_dt` without explicit `.to(step_dt.dtype)`.
2. **Evidence Reference:**
   - [03_ARCHITECTURE_AND_TENSOR_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/03_ARCHITECTURE_AND_TENSOR_AUDIT.md)
   - [04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md)
   - [06_TRAINING_AND_DECODER_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/06_TRAINING_AND_DECODER_AUDIT.md)
3. **Explanation of Correction:** Evaluated `sqrt(.float().square().sum(...) + 1e-8)` strictly in `float32` and cast back `.to(<input>.dtype)` at all 5 locations (`temporal.py:258-260`, `pose.py:69, 77, 124, 255`), and aligned `raw_dt.to(step_dt.dtype)` at `temporal.py:256`. Because `.float()` and `.to(float32)` are identity operations on `float32` inputs, all `float32` numerical values and tolerances are unchanged while `float16` backward division-by-zero is eliminated.
4. **Minimal Code Change:**
   - [temporal.py#L256-L260](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L256-L260)
   - [pose.py#L69-L77](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L69-L77), [pose.py#L124](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L124), [pose.py#L255](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L255)
5. **Regression Test Added:**
   - [test_fp16_sqrt_zero_delta_and_pose_coords_no_nan](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L115-L136) (tests `float16` zero-difference `estimate_torso_scale` backward unconditionally on CPU and `HierarchicalTemporalEncoder` identical-frame `float16` autocast backward when CUDA is available).
6. **Relevant Tests / Checks Executed:**
   - AST scan of every `.sqrt(` call containing `1e-8` across `src/` (`AST check 1`)
   - `pytest -q tests/test_baseline_fixes.py`
7. **Recorded Output & Result:**
   - AST invariant: `PASS` (`AST check 1 PASS: all torch.sqrt(... + 1e-8) in src/ cast to .float()`).
   - Runtime PyTorch check: **BLOCKED (`exit 2`)** locally due to missing `torch`.
8. **Remaining Limitation:** `float16` `delta_norm` cast back to `fp16` at exact zero difference (`1e-4 / sqrt(1024) = 3.125e-6`) falls in `fp16` subnormal range (nonzero representable above `5.96e-8` up to `d <= 2.6e6`); runtime gradient computation in `fp32` remains exact (`0.0`). On CPU-only PyTorch environments, the CUDA `autocast('cuda', fp16)` branch in the regression test is guarded by `torch.cuda.is_available()`.

---

### Fix 3 — Fix CUDA RNG Resume Crash, FP16 `GradScaler` Non-Finite Clip Crash, and `GradScaler` State Persistence

1. **Exact Defects:**
   - *(3A — `src/isl_mega/checkpoint.py:54-55`):* `scripts/train.py:274` calls `load_checkpoint(..., map_location=device)`. On CUDA resume, `torch.load` moves `state["cuda_rng"]` `ByteTensor`s onto CUDA, and `torch.cuda.set_rng_state_all(state["cuda_rng"])` crashes because PyTorch requires CPU `ByteTensor`s (unlike line 53, which called `.cpu()` on `torch_rng`). Also, `GradScaler` state was never persisted across checkpoints.
   - *(3B — `scripts/train.py:146-148`):* Under FP16 AMP (`amp_dtype: fp16`), calling `torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm, error_if_nonfinite=True)` right after `scaler.unscale_(optimizer)` raises `RuntimeError` on the first normal FP16 dynamic loss-scale overflow before `scaler.step(optimizer)` / `scaler.update()` can skip the step and halve `scale`.
2. **Evidence Reference:**
   - [06_TRAINING_AND_DECODER_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/06_TRAINING_AND_DECODER_AUDIT.md)
3. **Explanation of Correction:**
   - In `checkpoint.py`: normalized `cuda_rng` tensors to CPU both on save (`[s.cpu() for s in ...]`) and on load (`[s.cpu() for s in state["cuda_rng"]]`), added device-count mismatch fallback via `torch.cuda.set_rng_state(...)`, and added optional backward-compatible `scaler=None` argument to `save_checkpoint` and `load_checkpoint`.
   - In `scripts/train.py`: changed `error_if_nonfinite=True` to `error_if_nonfinite=False` inside the `if scaler is not None and scaler.is_enabled():` branch (retaining `error_if_nonfinite=True` for unscaled `fp32`/`bf16` training), and passed `scaler=scaler` into `load_checkpoint` and all three `save_checkpoint` calls. Because `scaler.unscale_(optimizer)` records `found_inf` before `clip_grad_norm_`, `scaler.step(optimizer)` skips parameter updates on overflow and `scaler.update()` backs off the scale cleanly.
4. **Minimal Code Change:**
   - [checkpoint.py#L12-L68](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/checkpoint.py#L12-L68)
   - [train.py#L144-L150](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L144-L150), [train.py#L274](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L274), [train.py#L335-L343](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L335-L343)
5. **Regression Tests Added:**
   - [test_checkpoint_save_load_restores_scaler_and_converts_cuda_rng_to_cpu](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L170-L220)
   - [test_train_epoch_passes_label_known_and_allows_amp_scaler_overflow](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L223-L319)
6. **Relevant Tests / Checks Executed:**
   - `compileall`, `ast.parse`, and `pytest -q tests/test_baseline_fixes.py`
7. **Recorded Output & Result:**
   - `compileall` & AST parse: `PASS`.
   - Runtime PyTorch check: **BLOCKED (`exit 2`)** locally due to missing `torch`.
8. **Remaining Limitation:** If `GradScaler` overflow happens inside `train.py`, `run_epoch` still increments `count` and averages the finite-or-overflow step's `loss` scalar into epoch metrics, and `optimizer.zero_grad(set_to_none=True)` relies on the next step cleanly clearing skipped gradients. Single-GPU only (DDP remains unimplemented; see Section 5).

---

### Fix 4 — Remove Unsafe Full-Clip CTC Fallback on Short Spans and Wire `label_known` Guard

1. **Exact Defect:**
   - In `compute_ctc_batch` (`src/isl_mega/losses.py:40-46`), when `span_known[i]` was `True` and `spanned.sum() < minimum_steps` (including explicit `spans: []` or very short spans), the `if int(spanned.sum()) >= minimum_steps:` guard simply fell through with `valid = full_clip_valid`. If the full clip length satisfied `len(row) >= minimum_steps`, CTC silently aligned a short fingerspelling token over unrelated whole-sentence frames instead of skipping the unusable span.
   - `compute_ctc_batch` also lacked an optional `label_known` parameter (unlike `compute_fingerspelling_spotting_loss` at `losses.py:70`), and `scripts/train.py:98-101` did not pass `label_known=label_known` to `compute_ctc_batch`.
2. **Evidence Reference:**
   - [06_TRAINING_AND_DECODER_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/06_TRAINING_AND_DECODER_AUDIT.md)
3. **Explanation of Correction:** Added optional `label_known: torch.Tensor | None = None` (default `None` to preserve compatibility with all existing callers and synthetic batch fixtures), skipping items with `label_known[i] == False`. Inside `if span_targets is not None and span_known is not None and bool(span_known[i]):`, unconditionally restricted `valid = valid & span_targets[i, :length].bool()` so items with `< minimum_steps` valid spanned frames are skipped at `if len(row) < minimum_steps: continue`. Passed `label_known=label_known` at `scripts/train.py:101`.
4. **Minimal Code Change:**
   - [losses.py#L22-L47](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L22-L47)
   - [train.py#L98-L102](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L98-L102)
5. **Regression Tests Added:**
   - [test_compute_ctc_batch_respects_span_constraint_and_label_known](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L139-L167)
   - [test_train_epoch_passes_label_known_and_allows_amp_scaler_overflow](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L223-L319)
6. **Relevant Tests / Checks Executed:**
   - `compileall`, `ast.parse`, and `pytest -q tests/test_baseline_fixes.py`
7. **Recorded Output & Result:**
   - `compileall` & AST parse: `PASS`.
   - Runtime PyTorch check: **BLOCKED (`exit 2`)** locally due to missing `torch`.
8. **Remaining Limitation:** Pre-loop `[encode_characters(x) for x in texts]` still validates character vocabulary strictly even if `label_known[i]` is `False` (which is appropriate since unknown clips carry `""`, and prevents malformed strings from hiding). Upstream clip-level CTC alignment remains coarse-grained whenever `span_known=False`.

---

### Fix 5 — Fix Self-Collision Crash in `scripts/evaluate.py --manifest valid.jsonl` / `train.jsonl`

1. **Exact Defect:** When invoking `scripts/evaluate.py --manifest data/manifests/valid.jsonl` (or `train.jsonl`), lines 39–43 loaded `cfg.data.train_manifest` and `cfg.data.valid_manifest` into `split_records["train"]` / `split_records["valid"]` and unconditionally placed the evaluated manifest `ds.records` into `split_records["test"]`. `validate_split_disjointness` then immediately crashed with `ValueError: Duplicate uid ... appears in splits 'valid' and 'test'`.
2. **Evidence Reference:**
   - [05_DATASET_AND_REAL_DATA_READINESS.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/05_DATASET_AND_REAL_DATA_READINESS.md)
   - [06_TRAINING_AND_DECODER_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/06_TRAINING_AND_DECODER_AUDIT.md)
3. **Explanation of Correction:** Extracted `build_eval_split_records(manifest, cfg, ds_records)` using resolved manifest paths: matches `ds_records` cleanly to its own split slot (`"train"`, `"valid"`, `"test"`, or `"eval"` for external held-out manifests) while still loading the remaining distinct configured splits (including optional existing `test_manifest` when evaluating `valid` or `train`) so genuine cross-split leakage continues to fail loudly.
4. **Minimal Code Change:** [evaluate.py#L17-L71](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py#L17-L71)
5. **Regression Test Added:** [test_evaluate_build_split_records_handles_valid_and_train_without_self_collision](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L322-L341)
6. **Relevant Tests / Checks Executed:**
   - `compileall`, `ast.parse`, and `pytest -q tests/test_baseline_fixes.py`
7. **Recorded Output & Result:**
   - `compileall` & AST parse: `PASS`.
   - Runtime PyTorch check: **BLOCKED (`exit 2`)** locally due to missing `torch`.
8. **Remaining Limitation:** Evaluating an arbitrary external subset file that intentionally slices records from `test.jsonl` works (checked against `train` and `valid`), whereas evaluating a symlink with non-matching canonical path resolves to `"eval"` cleanly without crashing unless `train`/`valid` share UIDs/sources.

---

### Fix 6 — Fix `"body"` Keypoint Slice `(0, 48)` -> `(42, 48)` in `scripts/validate_data.py`

1. **Exact Defect:** `scripts/validate_data.py:63` computed `"mean_region_pose_coverage"` using `zip(REGION_ORDER, ((48, 66), (0, 21), (21, 42), (0, 48)))`. Because `REGION_ORDER[3]` is `"body"`, slicing `0:48` averaged all `42` hand landmarks (`0:42`) plus the `6` body landmarks (`42:48`)—silently masking completely missing upper-body landmarks (`42:48`) whenever hands were present.
2. **Evidence Reference:**
   - Canonical 66-point schema: [pose.py#L14-L17](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L14-L17), [data_contract.md#L34-L80](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md#L34-L80)
   - [05_DATASET_AND_REAL_DATA_READINESS.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/05_DATASET_AND_REAL_DATA_READINESS.md)
3. **Explanation of Correction:** Corrected the fourth slice in `validate_data.py:63` from `(0, 48)` to `(42, 48)` so `"mean_region_pose_coverage"` reports true per-region pose validity (`face=48:66`, `left_hand=0:21`, `right_hand=21:42`, `body=42:48`). (`data.py:148` `derive_rois` continues to use `0:48` intentionally to bound the upper-body visual RGB crop around shoulders + hands).
4. **Minimal Code Change:** [validate_data.py#L63-L64](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/validate_data.py#L63-L64)
5. **Regression Test Added:** [test_validate_data_uses_canonical_body_slice_42_to_48](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L344-L370)
6. **Relevant Tests / Checks Executed:**
   - AST / source slice check (`AST check 3`), `compileall`, `pytest -q tests/test_baseline_fixes.py`
7. **Recorded Output & Result:**
   - AST check & `compileall`: `PASS` (`AST check 3 PASS: validate_data.py maps body to (42, 48)`).
   - Runtime PyTorch check: **BLOCKED (`exit 2`)** locally due to missing `torch`.
8. **Remaining Limitation:** `validate_data.py` still validates schema/transform integrity rather than semantic anatomical accuracy of externally extracted poses.

---

### Fix 7 — Weight Logged Epoch Loss by Active Valid Crops in `scripts/pretrain_visual.py`

1. **Exact Defect:** `scripts/pretrain_visual.py:57-58` accumulated unweighted batch means (`running += float(loss.detach()); count += 1`), distorting reported epoch SSL loss across batches with highly variable valid frame and ROI counts.
2. **Evidence Reference:**
   - [pretraining.py#L68-L81](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pretraining.py#L68-L81) (`loss` is mean over `keep.sum()` active crops)
   - [06_TRAINING_AND_DECODER_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/06_TRAINING_AND_DECODER_AUDIT.md)
3. **Explanation of Correction:** Weighted each batch loss by `crop_count = int(valid_local_mask.sum().item())` and normalized the epoch summary by `max(crop_total, 1)`, reporting `batches={count} crops={crop_total}`. Preserves `SignAwareDistillation.forward` API and parameter gradients identically.
4. **Minimal Code Change:** [pretrain_visual.py#L37-L62](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/pretrain_visual.py#L37-L62)
5. **Regression Test Added:** [test_pretrain_visual_reports_crop_weighted_epoch_loss](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_baseline_fixes.py#L373-L406)
6. **Relevant Tests / Checks Executed:**
   - `compileall`, `ast.parse`, and `pytest -q tests/test_baseline_fixes.py`
7. **Recorded Output & Result:**
   - `compileall` & AST parse: `PASS`.
   - Runtime PyTorch check: **BLOCKED (`exit 2`)** locally due to missing `torch`.
8. **Remaining Limitation:** See Section 5 regarding `body` ROI (`index=3`) inclusion in student local crops and missing `valid`/`test` split disjointness flags in `pretrain_visual.py`.

---

## 4. Complete Recorded Verification Log

### 4.1 Checks Executed Successfully (`PASS`)

```text
=== 1. COMPILEALL ===
Command: cd DINOv2-PoseGraph-3DCNN-Conformer && /opt/homebrew/bin/python3 -m compileall -q src scripts tests
Result:  compileall: EXIT 0

=== 2. AST & CONFIG VALIDATION ===
Command: "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/python" -c '... ast.parse + cfg_mod.load_config ...'
Output:
AST parse: OK (27 files)
load_config(configs/ablation_v1_baseline.yaml): OK (adaptive=False, motion=True, amp=bf16)
load_config(configs/research.yaml): OK (adaptive=True, motion=True, amp=bf16)
load_config(configs/research_v3_compact.yaml): OK (adaptive=True, motion=False, amp=bf16)
load_config(configs/smoke.yaml): OK (adaptive=False, motion=True, amp=none)
load_config(configs/smoke_v2.yaml): OK (adaptive=True, motion=True, amp=none)

=== 3. AST STRUCTURAL INVARIANTS & UNMODIFIED CONFIGS CHECK ===
Output:
AST check 1 PASS: all torch.sqrt(... + 1e-8) in src/ cast to .float()
AST check 2 PASS: HierarchicalTemporalEncoder mutually excludes self.detail and adaptive modules
AST check 3 PASS: validate_data.py maps body to (42, 48)
AST check 4 PASS: 8 regression tests collected statically: ['test_temporal_encoder_has_no_unused_trainable_parameters', 'test_fp16_sqrt_zero_delta_and_pose_coords_no_nan', 'test_compute_ctc_batch_respects_span_constraint_and_label_known', 'test_checkpoint_save_load_restores_scaler_and_converts_cuda_rng_to_cpu', 'test_train_epoch_passes_label_known_and_allows_amp_scaler_overflow', 'test_evaluate_build_split_records_handles_valid_and_train_without_self_collision', 'test_validate_data_uses_canonical_body_slice_42_to_48', 'test_pretrain_visual_reports_crop_weighted_epoch_loss']
=== UNMODIFIED CONFIGS CHECK ===
configs/ untouched: EXIT 0
```

### 4.2 PyTorch Test Suite Execution (`BLOCKED` Locally — Missing `torch`)

```text
=== 3. PYTEST TARGETED FAILING TEST ===
Command: PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q tests/test_novel_v2.py::test_compact_v3_config_and_motion_disabled_forward_backward
Output:
ERROR collecting tests/test_novel_v2.py
tests/test_novel_v2.py:9: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
pytest_targeted_exit=4

=== 4. PYTEST BASELINE FIXES ===
Command: PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q tests/test_baseline_fixes.py
Output:
ERROR collecting tests/test_baseline_fixes.py
tests/test_baseline_fixes.py:10: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
pytest_fixes_exit=2

=== 5. PYTEST FULL SUITE ===
Command: PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q
Output:
ERROR collecting tests/test_architecture.py
ERROR collecting tests/test_baseline_fixes.py
ERROR tests/test_novel_v2.py
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!!
pytest_full_exit=2
```

**Required GPU / PyTorch Verification Command (to run in the project's PyTorch environment):**

```bash
cd DINOv2-PoseGraph-3DCNN-Conformer
PYTHONPATH=src pytest -v tests/test_architecture.py tests/test_novel_v2.py tests/test_baseline_fixes.py
```

---

## 5. Ambiguous / Deferred Changes Requiring Review

The following issues identified across Prompts 1–8 involve architectural trade-offs, hyperparameter choices, existing test contracts, or missing external datasets/checkpoints, and were **intentionally left unmodified** pending explicit research/maintainer sign-off:

| # | Area / File | Finding & Why Deferred | Recommended Review Decision |
|---|---|---|---|
| 1 | `configs/*.yaml` | Prompt 9 explicitly prohibits overwriting existing configs (`configs/research.yaml` / `research_v3_compact.yaml`). Parameter mismatches (e.g., `fusion.projections["motion"]` `395k` params when `motion.enabled=False`, `use_semantic_ot` vs `ot_weight` settings) must be tuned via explicit experiment configs, not silent edits. | Keep `configs/*.yaml` pinned; add new experiment YAMLs only when running controlled ablations. |
| 2 | `scripts/train.py:172-186` (`build_epoch_scheduler`) | First-epoch `LambdaLR` factor starts at `1.0` for `warmup_ratio=0.25, epochs=4` (and `max(1.0 - ...)` decays linearly on epochs rather than steps). Existing strict unit test [test_epoch_scheduler_is_stepped_per_epoch](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py#L457-L473) explicitly asserts `values[0] == pytest.approx(1.0)`. | Approve whether to migrate to a per-step scheduler (updating the epoch-scheduler unit test intentionally) or retain epoch stepping. |
| 3 | `scripts/train.py:250-253` (`decoder.` vs `decoder.model.`) | `decoder.bridge` and `decoder.plan_bridge` are randomly initialized linear adapters currently grouped under `decoder_lr` (`1e-5`/`2e-5`) instead of `visual_lr` (`2e-4`), slowing adapter warmup especially when `freeze_decoder=True`. | Design/optimizer decision: approve moving `decoder.bridge` & `decoder.plan_bridge` into the `visual_lr` group. |
| 4 | `scripts/train.py:88-90, 330-345` (`--max-steps`, validation CF, `best_valid`) | (a) `--max-steps` overwrites `last.pt` / `best_valid.pt` / `metrics.csv` in `output_dir`; skipping checkpoint save could break smoke tests asserting checkpoint serialization unless gated by `--no-save`. (b) Validation runs `counterfactual_grounding` because `metrics.csv` explicitly logs `valid_cf_nll_gap`. (c) `best_valid.pt` selects on `valid_loss` (which mixes auxiliary losses) rather than `valid_translation` / validation BLEU-chrF. | Add explicit `--no-save` / `--debug-dir` flag for smoke runs and decide whether checkpoint selection tracks `valid_translation`. |
| 5 | `src/isl_mega/pose.py:370-380` (body anchor) & `61-86` (`estimate_torso_scale`) & `396-404` (right-hand mirror) | (a) Single-shoulder visibility (`shoulder_valid.any()`) matches both inline comment (`L372`) and `docs/architecture.md:46`, though single-shoulder fallback snaps body origin asymmetrically vs midpoint. (b) `estimate_torso_scale` switches discretely across `shoulder_span`, `body_diag*0.65`, `face_diag*1.25`. (c) Velocity computes finite difference of raw image `xy` rather than body-relative coordinates; shared hand graph mirrors `xy[..., 0]` while `VisualBackbone` does not horizontally flip right-hand RGB crops. (d) Intra-hand 3-layer graph radius (`3 < diameter 8`) limits distal fingertip-to-fingertip hops. | Treat as v3.1/v4 anatomical ablations rather than silent baseline edits so published baseline comparability stays intact. |
| 6 | `src/isl_mega/fusion.py:66-70, 145-196` & `temporal.py:57` | (a) `fusion.py` retains all 10 `STREAM_NAMES` (`motion` projection even if disabled) to preserve unit-test contracts and v1/v2/v3 modularity. (b) `fusion.py` intentionally permits neighbor-lag imputation on dropped frames and relies on `MultiScaleTemporalStem` (`temporal.py:38`) for exact padding zeroing. (c) `reliability_net` shares 1 small MLP across heterogeneous stream quality descriptors (e.g., CTC blank prob in channel 3). (d) `RelativeTimeBias` allocates ~53–55% of buckets below `28 ms`. | Evaluate in controlled ablations; do not remove functional components. |
| 7 | `src/isl_mega/planning.py:17-21` (`PlanGroundingBlock`) | Contains both `nn.TransformerEncoderLayer` (which includes a 4x FFN) and an explicit trailing `self.ffn` (`ffn_norm` + 4x FFN), doubling per-block FFN capacity. Removing either alters checkpoint state-dict keys and parameter counts. | Keep checkpoints compatible; evaluate single-FFN simplification in an ablation branch. |
| 8 | `src/isl_mega/losses.py:97, 11, 160-188` & `fingerspelling.py:24` | (a) `compute_cross_modal_alignment_loss` pools all 4 anatomical regions into one `(4BT)x(4BT)` matrix rather than per-region contrastive matrices and returns zero whenever batch has `< 2` unique `source_ids`. (b) `compute_fingerspelling_spotting_loss` compares fraction-of-frames to `len(text)/len(logits)`. (c) `prepare_ctc_targets` and `decode_characters` are unused helper utilities kept for API continuity. | Confirm contrastive design mathematically and retain utility functions cleanly. |
| 9 | `scripts/pretrain_visual.py:43-50` & `scripts/evaluate.py:71-86` | (a) `pretrain_visual.py` passes `crops[:, :, :4]` (including `body` ROI index `3`, which becomes full-frame `[0,0,1,1]` when keypoints vanish) as student local views despite docstring `"local face/hand crops"`, and lacks split-disjointness flags. (b) `evaluate.py` runs in FP32 without AMP/TF32 option and computes only corpus text metrics (does not run `greedy_ctc_decode` CER). | Add explicit `--exclude-body-local` / split-manifest CLI options in a dedicated pretraining/eval cleanup PR. |
| 10 | Missing Real Data Extractors & Weight Verification | No MediaPipe/`.pose` -> 66-point NPZ converter exists in repo (blocking end-to-end real dataset ingestion on `ISLTranslate`/`iSign`/`GloSL-ISL`), and real `google/flan-t5-*` / `timm` DINOv2 integration remains unverified offline. | Implement separate dataset-specific extraction adapters with documented landmark/license checks before cluster training. |

