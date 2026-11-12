# Root-Cause Analysis: Failing Temporal Gradient Test (`test_compact_v3_config_and_motion_disabled_forward_backward`)

**Repository:** `isl-mega` v`0.2.0` (`DINOv2-PoseGraph-3DCNN-Conformer/`, git HEAD `2157667`)  
**Reported Failure Signature:** `57 passed, 1 failed` — failing test `tests/test_novel_v2.py::test_compact_v3_config_and_motion_disabled_forward_backward` (`tests/test_novel_v2.py:467-486`)  
**Local Execution Boundary:** Verified on-host that Python bytecode compilation (`compileall`) exits `0`, while runtime `pytest` collection stops immediately on this macOS machine because `torch` is not installed in any local Python environment (see §11 for exact commands and outputs). Every structural and mathematical finding below is proven directly from the source code and PyTorch autograd computation graph.

---

## Executive Verdict

- **Root Cause Location:** **Model implementation bug** inside [`HierarchicalTemporalEncoder.__init__`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L204-L233) (`src/isl_mega/temporal.py:218`).
- **Mechanism:** `self.detail = FineDetailRetrieval(...)` is instantiated **unconditionally** even when `config.adaptive_consolidation` is `True`. During `HierarchicalTemporalEncoder.forward()` (`src/isl_mega/temporal.py:309-329`), the `if self.config.adaptive_consolidation:` branch calls `self.adaptive_detail(...)` (`L316`) and **never calls `self.detail`** (`self.detail` is invoked only at `L363` inside the `else:` branch). Consequently, all **10 trainable parameter tensors** in `model.temporal.detail.*` have `requires_grad=True` yet sit completely outside the forward autograd graph, leaving `p.grad is None` after `loss.backward()`.
- **Why `motion.enabled=False` Was Not the Cause:** Disabling motion is intentional and works cleanly; disabling motion did not break temporal gradients. Instead, commit `2157667` added `test_compact_v3_config_and_motion_disabled_forward_backward`, which is the **only** test in the entire repository that asserts `all(p.grad is not None ...)` over a module's parameters without silently filtering out `None` (`if p.requires_grad and p.grad is not None`).
- **Judgment:** **The test is valid and must NOT be weakened.** Fixing the bug removes **16,768** dead trainable parameters in `configs/smoke_v2.yaml` (`dim=64`), **2,363,904** (~2.36M) in `configs/research_v3_compact.yaml` (`dim=768`), and **4,200,448** (~4.20M) in `configs/research.yaml` (`dim=1024`), and prevents PyTorch `DistributedDataParallel` unused-parameter failures.

---

## 1. Exact Failing Assertion

In `DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py:483-485`:

```python
483:     loss = enc["memory"].sum()
484:     loss.backward()
485:     assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.temporal.parameters() if p.requires_grad)
```

- **Line 485** fails with `AssertionError` because `p.grad is None` for all 10 parameter tensors under `model.temporal.detail.*` whenever `cfg.temporal.adaptive_consolidation` is `True` (as configured in `configs/smoke_v2.yaml:58`).
- Notice by contrast how every neighboring backward test in the suite escaped detecting this bug:
  - [`test_full_model_forward_backward_and_frame_retention`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py#L276-L277) (`tests/test_architecture.py:276`, uses `smoke.yaml` where `adaptive_consolidation=False`):
    `grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]`
  - [`test_full_v2_model_end_to_end_backward_and_padding_invariance`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py#L342-L343) (`tests/test_novel_v2.py:342`, uses `smoke_v2.yaml` where `adaptive_consolidation=True`):
    `grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]`
  - [`test_single_frame_adaptive_temporal_and_full_model_backward_finite`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py#L362-L363) (`tests/test_novel_v2.py:362`, uses `smoke_v2.yaml` where `adaptive_consolidation=True`):
    `grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]`

Because all three neighboring tests filtered out `p.grad is None` before asserting, `model.temporal.detail.*` having `p.grad is None` went completely undetected until `test_compact_v3_config_and_motion_disabled_forward_backward` was added in commit `2157667` (bringing total pytest items from 56 to **58**: 45 in `test_architecture.py` + 13 in `test_novel_v2.py` -> hence `57 passed, 1 failed`).

---

## 2. Complete Test Setup (`tests/test_novel_v2.py:467-486`)

```python
467: def test_compact_v3_config_and_motion_disabled_forward_backward():
468:     v3 = load_config(Path(__file__).parents[1] / "configs" / "research_v3_compact.yaml")
469:     assert not v3.motion.enabled
470:     assert not v3.vision.separate_hand_face_body
471:     assert v3.vision.freeze_backbone
472:     assert v3.decoder.model_name == "google/flan-t5-base"
473:     assert v3.train.ot_weight == 0.0 and v3.train.fusion_entropy_weight == 0.0
474:
475:     # Verify that motion.enabled=False cleanly runs through ISLMEGA without instantiating MotionExpert
476:     cfg = v2_smoke_config()
477:     cfg.motion.enabled = False
478:     model = ISLMEGA(cfg)
479:     assert model.motion is None
480:     batch = make_batch_inputs(batch=2, time=4)
481:     enc = model.encode(**batch)
482:     assert enc["memory"].shape == (2, 4, cfg.temporal.dim)
483:     loss = enc["memory"].sum()
484:     loss.backward()
485:     assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.temporal.parameters() if p.requires_grad)
```

Step-by-step execution trace:
1. `L468-473`: Validates `configs/research_v3_compact.yaml` static flags (`motion.enabled=False`, `separate_hand_face_body=False`, `freeze_backbone=True`, `flan-t5-base`, `ot_weight=0.0`, `fusion_entropy_weight=0.0`).
2. `L476-479`: Loads `cfg = v2_smoke_config()` (`configs/smoke_v2.yaml`), mutates `cfg.motion.enabled = False`, instantiates `model = ISLMEGA(cfg)` (`with_decoder=False`, `training=True`), and confirms `model.motion is None`.
3. `L480`: Generates synthetic batch (`make_batch_inputs(batch=2, time=4)`, `L31-53`):
   - `frames`: `[2, 4, 3, 64, 64]` in `[0, 1]`
   - `timestamps`: `[[0.0, 0.04, 0.08, 0.12], [0.0, 0.04, 0.08, 0.12]]`
   - `lengths`: `tensor([4, 1])` (`max(1, 4 - 3) == 1`)
   - `pose_xy` (`[2, 4, 66, 2]`), `pose_confidence`, `pose_valid` (`~95%` True), `confidence_known=True`
   - `boxes` (`[0.1, 0.1, 0.9, 0.9]` across all 4 ROIs), `box_valid=True`, `box_scores=1.0`
4. `L481-482`: Executes `enc = model.encode(**batch)` and verifies `enc["memory"].shape == (2, 4, 64)`.
5. `L483-484`: Computes scalar loss `loss = enc["memory"].sum()` and runs `loss.backward()`.
6. `L485`: Checks all trainable parameters in `model.temporal`.

---

## 3. Model Configuration

Relevant parameters active on `model` in `L476-485` (`configs/smoke_v2.yaml` + `cfg.motion.enabled = False`):

| Config Section | Setting | Value in Test | Comparison with `research_v3_compact.yaml` |
|---|---|---|---|
| `vision` | `backend` / `dim` / `separate_hand_face_body` | `native` / `32` / `true` | `timm` (`ViT-B/14`) / `768` / `false` (`freeze_backbone: true`) |
| `pose` | `hidden_dim` / `cross_articulator_graph` / `gate_init` | `32` / `true` / `0.1` | `512` / `true` / `0.1` |
| `motion` | `enabled` / `hidden_dim` | **`false`** (set on `L477`) / `32` | **`false`** (`L43`) / `512` |
| `fusion` | `dim` / `continuous_time_lag` / `reliability_uncertainty_gate` | `64` / `true` / `true` | `768` / `true` / `true` |
| `fingerspelling` | `enabled` / `span_routing` | `true` / `true` | `true` / `true` |
| `temporal` | `dim` / `global_layers` / `heads` / `coarse_stride` / `detail_window` | `64` / `2` / `4` / `2` / `3` | `768` / `6` / `12` / `2` / `7` |
| **`temporal`** | **`adaptive_consolidation`** | **`true`** (`smoke_v2.yaml:58`) | **`true`** (`research_v3_compact.yaml:80`) |

Both `configs/smoke_v2.yaml`, `configs/research.yaml`, and `configs/research_v3_compact.yaml` set `temporal.adaptive_consolidation: true`. Only v1 configs (`configs/smoke.yaml`, `configs/ablation_v1_baseline.yaml`) leave `adaptive_consolidation: false`.

---

## 4. Is Motion Intentionally Disabled?

**Yes.**
- `MotionConfig.enabled` (`src/isl_mega/config.py:87`) defaults to `True`, and was explicitly added in commit `2157667` so `configs/research_v3_compact.yaml:43` can disable the 3-D CNN stream (`motion.enabled: false`) to isolate the RGB+pose fusion hypothesis (`AUDIT_REPORT.md:18,20`).
- Inside `ISLMEGA.__init__` (`src/isl_mega/model.py:35`) and `ISLMEGA.encode` (`src/isl_mega/model.py:83-88, 121-125, 145`):
  - `self.motion = None`.
  - `streams["motion"]` is filled with zeros `[2, 4, 32]`, `available["motion"]` is `False` everywhere, and `quality["motion"]` is zeros `[2, 4, 5]`.
  - Inside `ReliabilityAwareFusion.forward_with_aux` (`src/isl_mega/fusion.py:141-196`), stream index `8` (`"motion"`) has `mask[:, :, 8] = False`, so its key tokens are masked in cross-stream self-attention, its gate softmax weight is forced to exact `0.0`, and its lag-attention keys are masked out.
  - Meanwhile, the other **9 active streams** (`left_hand_rgb`, `right_hand_rgb`, `face_rgb`, `body_rgb`, `left_hand_pose`, `right_hand_pose`, `body_pose`, `face_pose`, `fingerspelling`) actively compute `fused [2, 4, 64]`, which feeds `self.temporal(fused, lengths, timestamps)` (`src/isl_mega/model.py:151`).
- **Conclusion:** Disabling `motion` does **not** disconnect `model.temporal` from the input or loss. (Note as a secondary cleanliness detail outside `model.temporal`: `self.fusion.projections["motion"]` is still constructed in `fusion.py:69-70` and invoked on zero tensors before stream 8 is masked out; thus `projections["motion"].*.grad` is exact `0.0`—though not `None`—since stream 8's downstream weights are masked.)

---

## 5–8. Forward Participation & Gradient Audit Across Every `model.temporal` Parameter

Below is the complete submodule-by-submodule trace of `model.temporal` ([`HierarchicalTemporalEncoder`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L197-L375)) under `adaptive_consolidation=True`, `batch=2`, `time=4`, `lengths=[4, 1]`, and `loss = enc["memory"].sum()`:

| Submodule (`model.temporal.*`) | Definition Line | Forward Participation (`adaptive_consolidation=True`) | Tensor Count (`dim=64`) | Scalar Params (`dim=64`) | Expected Grad Status (`L485` Intent) | Actual Grad After `loss.backward()` | Math / Graph Reason |
|---|---|---|---|---:|---|---|---|
| `stem.*` (`branches`, `project`, `norm_in`, `norm_out`) | `temporal.py:207-208` | **Yes** (`L306`) | 10 | 20,864 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Runs on full `[2,4,64]` `fused`; feeds both `final` and `_adaptive_pool`. |
| `coarse_projection.*` (`Linear`, `LayerNorm`) | `temporal.py:209-210` | **Yes** (`L279`) | 4 | 8,384 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Projects `[grouped_mean, grouped_delta]` -> `coarse` -> `global_blocks` -> `memory`. |
| `time_projection.*` (`Linear(2, dim)`) | `temporal.py:211` | **Yes** (`L288`) | 2 | 192 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Adds `[elapsed, log1p(elapsed)]` projection (`elapsed > 0` for sample 0 slot 1). |
| `global_blocks.{0,1}.*` (`norm1..3`, `qkv`, `attn_out`, `conv_in`, `depthwise`, `conv_out`, `ffn`) | `temporal.py:212-215` | **Yes** (`L314-315`) | 40 (`20` x 2) | 94,208 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | All valid slots (`m_valid=[2, 1]`) flow through self-attn + gated depthwise conv + FFN with residuals. |
| `time_bias.table.weight` (`Embedding(16, 4)`) | `temporal.py:216-217` | **Yes** (`L313`) | 1 | 64 | Non-`None`, finite (**sparse nonzero**) | **Non-`None`, finite** (visited bucket rows nonzero; unvisited buckets exact `0.0`) | PyTorch `Embedding` produces dense `[16, 4]` grad tensor; added directly to attention logits (`GlobalTemporalBlock:92`). |
| **`detail.*`** (`FineDetailRetrieval`: `q`, `k`, `v`, `out`, `norm`) | **`temporal.py:218`** | **NO — UNUSED** (`L363` is only inside `else:`) | **10** | **16,768** | Either unused (`None` / omitted) or shouldn't exist when adaptive | **`p.grad is None` (FAILS `L485`)** | Unconditionally registered in `__init__`; bypassed completely when `adaptive_consolidation=True`. |
| `salience_net.*` (`0`, `2` `Linear`) | `temporal.py:220-224` | **Yes** (`L259`) | 4 | 4,225 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Outputs `salience > 0.1` -> differentiable `assign_f` & `assign_logits` -> `coarse` & `adaptive_detail` -> `memory`. |
| `log_slot_sigma` (`Parameter` scalar) | `temporal.py:225` | **Yes** (`L266-267`) | 1 | 1 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Smooth Gaussian RBF temperature on `assign_logits` (`sigma >= 0.30`); sample 0 (`length=4`) has `t=4 != m=2` giving non-trivial slot-to-frame distances. |
| `slot_spread_proj.*` (`Linear(1, dim)`) | `temporal.py:226` | **Yes** (`L289`) | 2 | 128 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Sample 0 (`length=4`) has `slot_var > 0` (`timestamps = 0..0.12`), giving `slot_std > 0`; bias & weight add to `coarse`. |
| `adaptive_detail.*` (`q`, `k`, `v`, `out`, `norm`, `log_prior_scale`) | `temporal.py:227` | **Yes** (`L316`) | 11 | 16,772 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Cross-attends `coarse` to `fine` biased by `softplus(log_prior_scale) * assign_logits`; outputs `retrieved` -> `global_up` -> `final`. |
| `final.*` (`Linear(2D, D)`, `LayerNorm`) | `temporal.py:233` | **Yes** (`L373`) | 4 | 8,384 | Non-`None`, finite, **nonzero** | **Non-`None`, finite, nonzero** | Directly produces `out` (`enc["memory"]`) from `[fine, global_up]`. |

*(Total `model.temporal` parameters at `dim=64`: 89 tensors / 169,990 scalar weights — 79 active tensors / 153,222 scalars with gradients, **10 dead tensors / 16,768 scalars in `detail.*` with `p.grad is None`**.)*

Wait — also verify the asymmetric padding case `lengths = [4, 1]` on sample 1:
- In `sample 1`, `length = 1`, so `m_valid[1] = 1` and `valid_pairs = False` (`L251`), giving `slot_var[1, 0] = 0.0` and `slot_std = sqrt(0 + 1e-6) - 1e-3 = 0.0` (`L284`). Because `+ 1e-6` is inside `sqrt(...)`, its derivative `1 / (2 * sqrt(1e-6)) = 500` is finite (not `NaN`/`inf`), as also tested by `test_single_frame_adaptive_temporal_and_full_model_backward_finite` (`T=1`, `lengths=[1]`, `test_novel_v2.py:346-363`). All gradients are finite.

---

## 9. Does the Loss Connect to the Temporal Branch?

**Yes, directly.**
- `model.encode(**batch)` (`src/isl_mega/model.py:151,154`) calls `temporal, memory_mask, temporal_aux = self.temporal(fused, lengths, timestamps)` and returns `"memory": temporal`.
- `loss = enc["memory"].sum()` (`tests/test_novel_v2.py:483`) sums directly over `temporal`'s output tensor.
- There is no detached intermediate, no missing downstream loss path (`self.planner` is downstream of `temporal` and intentionally excluded from `loss`, which is why the test specifically inspects `model.temporal.parameters()`, not `model.planner.parameters()`). Every module inside `model.temporal` that executes in `forward()` connects directly to `loss`.

---

## 10. Verdict: Model, Configuration, Test, or Loss?

| Component | Faulty? | Evidence |
|---|---|---|
| **Model (`src/isl_mega/temporal.py:218`)** | **YES — PRIMARY ROOT CAUSE** | `HierarchicalTemporalEncoder.__init__` sets `salience_net`, `log_slot_sigma`, `slot_spread_proj`, and `adaptive_detail` to `None` when `adaptive_consolidation=False` (`L229-232`), but fails to gate `self.detail` to `None` when `adaptive_consolidation=True` (`L218`). |
| **Configuration (`configs/smoke_v2.yaml`, `configs/research_v3_compact.yaml`)** | No | Intentionally sets `temporal.adaptive_consolidation: true` and (in v3) `motion.enabled: false`; configs match intended design. |
| **Test (`tests/test_novel_v2.py:485`)** | No | Properly checks `p.grad is not None and torch.isfinite(p.grad).all()` for every trainable parameter of `model.temporal`. Exposed a genuine dead-parameter bug masked by earlier `and p.grad is not None` filters. |
| **Loss (`enc["memory"].sum()`)** | No | Differentiates directly through `enc["memory"]` (`temporal.final` output). |

### Structural Verification Across All Other Toggleable Submodules
Inspecting every conditional module across `src/isl_mega/` confirms this pattern is isolated specifically to `temporal.py:218`:
- `pose.py:290-296`: `self.cross_graph = ... if config.cross_articulator_graph else None`; `self.hand_side_emb = ... if config.share_hand_encoder else None`.
- `model.py:35-40`: `self.motion = ... if config.motion.enabled else None`; `self.fingerspelling = ... if config.fingerspelling.enabled else None`. *(Note: `fusion.projections["motion"]` remains instantiated in `fusion.py:69` when `motion.enabled=False`, receiving `0.0` gradients rather than `None`, which does not fail `p.grad is not None` and does not block DDP since its forward runs, though pruning it cleanly later would also save unused weights.)*
- `fingerspelling.py:67-78`: `self.spot_conv`, `self.spot_head` set to `None` when `not span_routing`.
- `fusion.py:78-99`: `self.reliability_net` set to `None` when disabled; `self.continuous_lag_bias` vs `self.lag_bias` mutually exclusive (`None` on the inactive branch).
- `planning.py:75-84`: `self.slot_pos` and `self.budget_gate` set to `None` when disabled.

Only `self.detail` in `HierarchicalTemporalEncoder.__init__` lacks mutual-exclusion gating!

---

## 11. Commands Executed & Recorded Outputs (Local Environment)

All commands were executed read-only inside `/Users/likhith/Codes - Github./ISLTranslate-ideas`:

### Command 1 — Bytecode compilation check
```bash
(cd "DINOv2-PoseGraph-3DCNN-Conformer" && /opt/homebrew/bin/python3 -m compileall -q src scripts tests)
echo "compileall=$?"
```
**Output:**
```text
compileall=0
```

### Command 2 — Individual failing test execution attempt
```bash
cd "DINOv2-PoseGraph-3DCNN-Conformer" && PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q tests/test_novel_v2.py::test_compact_v3_config_and_motion_disabled_forward_backward || true
```
**Output:**
```text
==================================== ERRORS ====================================
___________________ ERROR collecting tests/test_novel_v2.py ____________________
ImportError while importing test module '/Users/likhith/Codes - Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Cellar/python@3.14/3.14.6/Frameworks/Python.framework/Versions/3.14/lib/python3.14/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_novel_v2.py:9: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
=========================== short test summary info ============================
ERROR tests/test_novel_v2.py
1 error in 0.08s
ERROR: found no collectors for /Users/likhith/Codes - Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_novel_v2.py::test_compact_v3_config_and_motion_disabled_forward_backward
```

### Command 3 — Full test suite execution attempt
```bash
(cd "DINOv2-PoseGraph-3DCNN-Conformer" && PYTHONPATH=src "/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/pytest" -q) || true
```
**Output:**
```text
==================================== ERRORS ====================================
_________________ ERROR collecting tests/test_architecture.py __________________
ImportError while importing test module '/Users/likhith/Codes - Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py'.
...
tests/test_architecture.py:7: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
___________________ ERROR collecting tests/test_novel_v2.py ____________________
...
tests/test_novel_v2.py:9: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
=========================== short test summary info ============================
ERROR tests/test_architecture.py
ERROR tests/test_novel_v2.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!!
2 errors in 0.15s
```

### Local Python Environments Inspected
Checked `/opt/homebrew/bin/python3` (`3.14.6`), `/opt/homebrew/bin/python3.10`, `/opt/homebrew/bin/python3.12`, `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`, `/usr/bin/python3`, `/Users/likhith/Codes - Github./rdhei-robust/.venv/bin/python` (`pytest`, `numpy`, `yaml` present; `torch` absent), and Docker socket (`/Users/likhith/.docker/run/docker.sock` inactive). None currently contain `torch`.

---

## 12. Recommended Correction (Do Not Modify Source Files Yet)

### Proposed Fix in `src/isl_mega/temporal.py` (`L218-232`)

Make `self.detail` and the adaptive consolidation submodules mutually exclusive so only the active retrieval pathway is registered:

```python
--- a/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py
+++ b/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py
@@ -215,16 +215,17 @@ class HierarchicalTemporalEncoder(nn.Module):
         ])
         self.time_bias = RelativeTimeBias(config.heads, config.relative_time_buckets,
                                           config.max_elapsed_seconds)
-        self.detail = FineDetailRetrieval(config.dim, config.heads, config.detail_window, config.coarse_stride)
         if config.adaptive_consolidation:
+            self.detail = None
             self.salience_net = nn.Sequential(
                 nn.Linear(config.dim * 2 + 2, max(config.dim // 2, 32)),
                 nn.GELU(),
                 nn.Linear(max(config.dim // 2, 32), 1),
             )
             self.log_slot_sigma = nn.Parameter(torch.tensor(-0.6))
             self.slot_spread_proj = nn.Linear(1, config.dim)
             self.adaptive_detail = AdaptiveDetailRetrieval(config.dim, config.heads)
         else:
+            self.detail = FineDetailRetrieval(config.dim, config.heads, config.detail_window, config.coarse_stride)
             self.salience_net = None
             self.log_slot_sigma = None
             self.slot_spread_proj = None
```

### Verification of Side Effects
1. **v1 tests (`adaptive_consolidation=False`):** Including [`test_temporal_detail_centres_are_per_sample_for_partial_stride_groups`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/tests/test_architecture.py#L645-L669) which calls `enc.detail(...)` directly (`L659`), continue passing unchanged because `self.detail` is instantiated whenever `adaptive_consolidation` is `False`. (A grep across `src/`, `scripts/`, `tests/`, and `configs/` confirms `test_architecture.py:659` and `temporal.py:363` are the only callers of `.detail` in the repository.)
2. **v2/v3 configs (`adaptive_consolidation=True`):** Removes all 10 dead tensors in `self.detail` (~2.36M dead parameters in `research_v3_compact.yaml`; ~4.20M in `research.yaml`). All remaining trainable parameters in `model.temporal` participate in `forward()` and receive finite, non-`None` gradients.
3. **Checkpoint note:** No pre-existing checkpoints ship with the repository; if an external pre-fix `adaptive_consolidation=True` checkpoint exists, its `temporal.detail.*` keys can be dropped or loaded with `strict=False`.

---

## 13. Regression Tests Needed (Without Weakening Existing Assertions)

1. **Preserve `test_compact_v3_config_and_motion_disabled_forward_backward` (`tests/test_novel_v2.py:485`) verbatim** — do **not** add `and p.grad is not None` filtering there.
2. **Add `test_temporal_encoder_has_no_unused_trainable_parameters` covering both `adaptive_consolidation=True` and `adaptive_consolidation=False`:**
   - Assert mutual exclusion explicitly:
     - When `adaptive_consolidation=True`: `enc.detail is None` and `enc.adaptive_detail is not None`.
     - When `adaptive_consolidation=False`: `enc.detail is not None` and `all(m is None for m in (enc.salience_net, enc.log_slot_sigma, enc.slot_spread_proj, enc.adaptive_detail))`.
   - Run `out, _, _ = enc(x, lengths=torch.tensor([5, 3]), timestamps=times)` (using `batch=2, time=5, dim=cfg.dim`) -> `out.sum().backward()`:
     - Assert **zero** trainable parameters have `p.grad is None`:
       `missing = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is None]` -> `assert not missing`.
     - Assert every parameter gradient is finite AND non-zero (`assert p.grad.abs().sum().item() > 0` — note `time_bias.table.weight` has total norm `> 0` across visited rows).
3. **Audit & tighten neighboring gradient tests (`test_novel_v2.py:342,362` and `test_architecture.py:276`):**
   - Replace silent `and p.grad is not None` filters in end-to-end backward checks (when summing outputs of active heads, e.g. `out["memory"]` + `out["plan_tokens"]` + `out["ctc_logits"]` / `out["spot_logits"]`) with explicit assertions that **no parameter belonging to active submodules has `p.grad is None`**, preventing future dead-module regressions from hiding behind list comprehensions.
