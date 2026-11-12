# Hardware Feasibility & Performance Profiling Report

**Target Repository:** `DINOv2-PoseGraph-3DCNN-Conformer` (`isl-mega` v`0.2.0`)  
**Audited Configurations:** [`smoke_v2.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/smoke_v2.yaml), [`research_v3_compact.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml), [`research.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research.yaml), [`ablation_v1_baseline.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/ablation_v1_baseline.yaml)  
**Strict Epistemic Rule:** Throughout this report, all claims are partitioned into **(M) Measured on Host**, **(C) Calculated Estimate** (derived deterministically from exact layer constructors and tensor shapes in `src/isl_mega/*.py` and `scripts/*.py`), and **(H) Hypothetical Projection**. No cloud pricing, synthetic PyTorch runtime timings, or unmeasured GPU TFLOPS/training hours are invented.

---

## Executive Feasibility Verdict

**Can `ISL-MEGA` realistically be trained or full-scale evaluated on this local environment?**  
**No.** Training any real-data configuration—including the compact [`research_v3_compact.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml) baseline—is **physically blocked on four independent axes** on this host:

1. **(M) Zero Deep-Learning Stack Installed:** `torch`, `timm`, `transformers`, `sacrebleu`, `PyYAML`, `numpy`, `cv2`, and `pytest` are absent across all 5 host Python runtimes (`import isl_mega` fails in `18.65 ms` with `ModuleNotFoundError: No module named 'torch'`). Package installation was strictly withheld per audit rules.
2. **(M) Critical Storage Deficit (`11 GiB` free / 95% full):** Available disk on `/System/Volumes/Data` (`11 GiB`) cannot simultaneously accommodate `vit_base_patch14_dinov2` (`~330 MiB`), `flan-t5-base` (`~944 MiB`), one pair of `last.pt` + `best_valid.pt` checkpoints (`9.41 GiB` calculated), and raw video `.npz` clips (`~14.0 GiB` to `206 GiB` per 1,000 clips).
3. **(C) Severe Memory Overflow on `8.00 GiB` Unified RAM (`7.44 GiB` usable):** Even with `freeze_backbone: true`, [`scripts/train.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L243-L261) keeps all weights, gradients, and AdamW states in `FP32`, requiring **`6.16 GiB`** purely for static states before a single video frame is loaded (`~10.10 GiB` at `B=1, T=150`; `~37.86 GiB` at config default `B=4, T=300`). Meanwhile `research.yaml` requires **`61.85 GiB` static alone** (`> 1 TB` un-checkpointed activations at `B=2, T=150`).
4. **(M + C) No CUDA GPU & Missing MPS Support:** The host has an Apple M2 (`8`-core Metal GPU) with zero NVIDIA CUDA devices, while [`train.py:207`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L207), [`evaluate.py:56`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py#L56), and [`pretrain_visual.py:27`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/pretrain_visual.py#L27) hardcode `"cuda" if torch.cuda.is_available() else "cpu"` without an `mps` path.

> [!IMPORTANT]
> **Two Hidden Implementation Bottlenecks Discovered by Analytical Tensor Profiling:**
> 1. **`AppearanceExpert` (`ViT-B/14`) consumes `97.4%` of `v3_compact` FLOPs (`222.3 TFLOPs` of `228.3 TFLOPs/step` at `B=4, T=300`)** despite `freeze_backbone: true`, because frozen DINOv2 is re-run on `4 * B * T = 4,800` images every training step without feature caching.
> 2. **`PoseExpert` (`EdgeGraphBlock`) consumes `64.8%` of `v3_compact` activation memory (`~20.55 GiB` in BF16 at `B=4, T=300`)**, **3.3x larger than all static weights + optimizer states (`6.16 GiB`)**, because [`pose.py:128-130`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L128-L130) materializes `[B*T, N, N, D=512]` pairwise edge tensors across `5,022` pairs/frame! Fixing those two bottlenecks cuts per-step compute by **~150x** and single-GPU VRAM from `~37.9 GiB` down to **`~5.4 GiB`** (`flan-t5-base`) / **`~2.8 GiB`** (`flan-t5-small`).

---

## 1. Host Machine & Software Environment Inventory **(M — Measured)**

| Resource / Component | Verified Host Measurement (`2026-11-12`) | Verification Command / Evidence |
|---|---|---|
| **Machine Model & OS** | `MacBook Air` (`Mac14,2`), `macOS 26.5.2` (`Build 25F84`), Darwin `25.5.0` `arm64` | `sw_vers`, `uname -a`, `system_profiler SPHardwareDataType` |
| **CPU** | `Apple M2`, `8` physical/logical cores (`4 Performance` + `4 Efficiency`); P-core L1i `192 KiB` / L1d `128 KiB` / L2 `16 MiB`; E-core L1i `128 KiB` / L1d `64 KiB` / L2 `4 MiB` | `sysctl -n machdep.cpu.brand_string`, `sysctl -a` |
| **System RAM (Unified)** | **`8.00 GiB`** physical (`8,589,934,592 B`), **`7.44 GiB`** usable (`7,989,641,216 B`) | `sysctl hw.memsize hw.memsize_usable` |
| **Live Memory Pressure** | Page size `16 KiB`; `4,302` free (`~67 MiB`), `79,031` active (`~1.21 GiB`), `76,463` inactive (`~1.17 GiB`), `104,019` wired (`~1.59 GiB`) | `vm_stat` |
| **GPU & Accelerator** | `Apple M2` built-in **`8-core GPU`** (`Metal 4`), shares `8 GiB` unified RAM; **`0` NVIDIA CUDA GPUs** | `system_profiler SPDisplaysDataType` |
| **MPS / CUDA Software** | **Unavailable.** `torch` not installed; additionally `scripts/{train,evaluate,pretrain_visual}.py` omit `torch.backends.mps` and route macOS execution to `cpu` | [`train.py:207`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L207), [`evaluate.py:56`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py#L56) |
| **Available Storage** | `/dev/disk3s1` (`/System/Volumes/Data`): `228 GiB` total, `187 GiB` used, **`11 GiB` available (`95%` capacity)**. Workspace `ISLTranslate-ideas`: `427 MiB` (`DINOv2-PoseGraph-3DCNN-Conformer`: `1.7 MiB`) | `df -h`, `du -sh` |
| **Python Interpreters** | `/opt/homebrew/bin/python3` (`3.14.6`); `python@3.12` (`3.12.14`); `python@3.10` (`3.10.20`); `/Library/Frameworks/...` & `/usr/local/bin/python3` (`3.14.6`); `/usr/bin/python3` (`3.9.6`) | `which -a python3`, `python3 -V` |
| **PyTorch Version** | **Not installed (`None`)** across all 5 host Pythons and all sibling virtualenvs (`pyproject.toml` requires `torch>=2.2`) | `pip list` + `find ~ -name torch`; `import isl_mega` fails with `ModuleNotFoundError: No module named 'torch'` |
| **Core Repo Dependencies** | **Missing in active project** (`no .venv` in repo): `numpy`, `PyYAML`, `timm`, `transformers`, `sentencepiece`, `safetensors`, `opencv-python-headless`, `sacrebleu`, `rouge-score`, `pytest`, `ffmpeg` are all absent from system Pythons (only `pip`/`wheel`/`pillow`/`lxml` present). Sibling `rdhei-robust/.venv` has `numpy 2.5.3`, `PyYAML 6.0.3`, `opencv-python 5.0.0.93`, `pytest 9.1.1` (still **zero `torch`/`timm`/`transformers`**) | `pip list` across all 5 Pythons & 2 sibling `.venv`s; [`pyproject.toml:11-24`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/pyproject.toml#L11-L24) |
| **Previous Report Discrepancy** | Note that [`05_BASELINE_TEST_AND_RUNTIME_REPORT.md:84`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/phase_1_verification/05_BASELINE_TEST_AND_RUNTIME_REPORT.md#L84) listed `14.00 GB` free disk; live `df -h` today measures **`11 GiB`** | Direct `df -h` measurement |

### Local Measurement Boundary (`1–10` Metrics Check)

- **Measured Directly on Host:** Source discovery (`26` Python files in `1.56 ms`), bytecode compilation (`python3 -m compileall -q src scripts tests` -> `exit 0`), `isl_mega` import probe (`ModuleNotFoundError: No module named 'torch'` in `18.65 ms`), physical CPU/RAM/GPU/disk capacities.
- **Unmeasurable on Host Without Prohibited Installs/Downloads:** PyTorch model construction latency, CPU/MPS forward/backward wall-clock latency, and live `torch.cuda.max_memory_allocated()` / RSS peaks cannot be measured natively while `torch`, `timm`, and `transformers` are absent. Everything in Sections 2–5 below uses **exact deterministic calculations** from constructor dimensions and tensor shapes in `src/isl_mega/*.py`, explicitly separated from hypothetical hardware projections.

---

## 2. Parameter Count, Optimizer Memory & Checkpoint Footprint **(C — Calculated Estimates)**

Notice how [`scripts/train.py:243-262`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py#L243-L262) instantiates the model in `float32` (`build_model(cfg, with_decoder=True).to(device)`) and applies mixed precision via `torch.autocast` rather than `.to(torch.bfloat16)` weight casting. Consequently:
- Every model parameter stays **`4 B` (`FP32`)** in memory (even frozen `ViT-B/14`).
- Every trainable parameter requires **`4 B` (`FP32` gradient)** + **`8 B` (`AdamW` first & second moments `m, v` in `FP32`)** = **`16 B / trainable param`** (`4 B` per frozen param).
- Every checkpoint written by [`checkpoint.py:21-35`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/checkpoint.py#L21-L35) serializes `model.state_dict()` (including frozen backbones!) + `optimizer.state_dict()` = **`4 * P_total + 8 * P_train` bytes**.

| Configuration Profile | Exact Internal Params | External ViT Params (Est.) | External FLAN-T5 Params (Est.) | Total Params | Trainable Params | Static Train Mem (`FP32 W + Grad + AdamW`) | Eval Weights (`FP32` / `BF16`) | Single Checkpoint Size (`last` + `best`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [`smoke_v2.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/smoke_v2.yaml) | **578,705** *(16.8k dead)* | `0` (`3x` native ViT) | `~76,961,152` (`flan-t5-small`) | **~77.54M** | **~77.54M** | **`1.16 GiB`** (`0.29+0.29+0.58`) | `0.29 GiB` / `0.14 GiB` | `0.87 GiB` (`1.73 GiB`) |
| [`research_v3_compact.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml) | **144,485,984** *(2.36M dead `detail`; 396k `motion_proj`)* | `~86,579,712` (`1x ViT-B/14`, **frozen**) | `~247,577,856` (`flan-t5-base`) | **~478.64M** | **~392.06M** | **`6.16 GiB`** (`1.78+1.46+2.92`) | `1.78 GiB` / `0.89 GiB` | **`4.70 GiB`** (**`9.41 GiB`**) |
| [`research.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research.yaml) | **387,699,417** *(4.20M dead `detail`)* | `~913,102,848` (`3x ViT-L/14`, train) | `~2,849,757,184` (`flan-t5-xl`) | **~4,150.56M** | **~4,150.56M** | **`61.85 GiB`** (`15.46+15.46+30.92`) | `15.46 GiB` / `7.73 GiB` | `46.39 GiB` (`92.77 GiB`) |
| [`ablation_v1_baseline.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/ablation_v1_baseline.yaml) | **388,993,150** | `~913,102,848` (`3x ViT-L/14`, train) | `~2,849,757,184` (`flan-t5-xl`) | **~4,151.85M** | **~4,151.85M** | **`61.87 GiB`** | `15.47 GiB` / `7.73 GiB` | `46.40 GiB` (`92.80 GiB`) |

*(Note: Previous document [`05_BASELINE_TEST_AND_RUNTIME_REPORT.md:83`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/phase_1_verification/05_BASELINE_TEST_AND_RUNTIME_REPORT.md#L83) misstated `research_v3_compact.yaml` static+training RAM as `5.63 GB` by undercounting internal parameters (`77M` vs exact `144.49M`) and assuming pure BF16 storage without `train.py`'s FP32 parameter/AdamW tensors.)*

### Per-Module Internal Parameter & Weight Breakdown (`research_v3_compact.yaml`)

- **PoseEncoder (`PoseExpert`):** `34,224,132` params (`130.6 MiB` FP32; `522.2 MiB` with grads+AdamW).
- **Temporal Backbone (`HierarchicalTemporalEncoder`):** `56,797,838` params (`2,363,904` dead in unused `self.detail`, `216.7 MiB` FP32; `866.7 MiB` total train state).
- **Latent Planner (`LatentPlanGrounder`):** `28,368,384` params (`108.2 MiB` FP32; `432.9 MiB` train state; all planning losses disabled in `v3_compact`).
- **Multimodal Fusion (`ReliabilityAwareFusion`):** `22,571,940` params (`395,520` zero-grad in `projections["motion"]`, `86.1 MiB` FP32; `344.4 MiB` train state).
- **Fingerspelling (`FingerspellingExpert`):** `1,337,898` params (`5.1 MiB` FP32).
- **FLAN-T5 Bridges (`bridge` + `plan_bridge`):** `1,184,256` params (`4.5 MiB` FP32).
- **Visual Wrapper (`VisualBackbone.norm`):** `1,536` params (`+ 86.58M` frozen `ViT-B/14` = `330.3 MiB` static weights).

---

## 3. Compute (FLOPs), Batch/Sequence Scaling & Memory Profiling **(C — Calculated Estimates)**

### A. Forward + Backward Memory & FLOP Sweep Across Batch Size `B` and Sequence Length `T`

Computed analytically in `BF16` autocast (`2 B` activations, `L_dec = 32` target tokens) directly from `ISLMEGA.forward`:
- Notice in [`vision.py:48`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L48), `crop_regions` materializes `frames[:, :, None].expand(-1, -1, 4, ...).reshape(4*B*T, 3, H, W)` (`4x` raw frame copies) before `F.grid_sample`.
- In `research_v3_compact.yaml`, `freeze_backbone: true` means PyTorch autograd does **not** retain all `12` ViT blocks for backward, **but** [`AppearanceExpert.forward`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L194-L198) still feeds all `N = B * T` crops simultaneously in one ViT call, creating a transient single-block forward peak (`[BT, 12, 257, 257]` + `[BT, 257, 3072]`).
- In [`pose.py:128-130`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L128-L130), every `EdgeGraphBlock` layer saves intermediate `[B*T, N, N, 512]` pairwise tensors (`edge` Linear1 output, `Tanh` output, `eb`/`pair`) across `5,022` total node-pairs per frame (`3*(21^2+21^2+6^2+18^2) + 36^2`), **dominating training activation RAM**!

#### `research_v3_compact.yaml` Sweep (`Static Train Base = 6.16 GiB`, `BF16` Activations)

| Batch `B` | Frames `T` (~@25fps) | Total Fwd FLOPs (`GFLOPs`) | Visual ViT FLOPs | Pose Graph FLOPs | Fusion + Temporal + Plan FLOPs | T5 Enc+Dec FLOPs | Input+Crops + ViT Forward Peak | **Pose Saved Activations** | Fusion+Tmp + T5 + InfoNCE Acts | **Est. Peak Train Memory** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | `64` (~2.6s) | **12,185** (`12.2 TF`) | `11,859` (**97.3%**) | `265` (2.2%) | `37.5` | `24.1` | `522 MiB` | **`1.10 GiB`** | `84 MiB` | **`~7.85 GiB`** |
| **1** | `150` (~6.0s) | **28,544** (`28.5 TF`) | `27,794` (**97.4%**) | `621` (2.2%) | `86.9` | `42.1` | `1.19 GiB` | **`2.57 GiB`** | `173 MiB` | **`~10.10 GiB`** |
| **1** | `300` (~12s) | **57,078** (`57.1 TF`) | `55,587` (**97.4%**) | `1,242` (2.2%) | `173.1` | `74.6` | `2.39 GiB` | **`5.14 GiB`** | `349 MiB` | **`~14.04 GiB`** |
| **1** | `600` (~24s) | **114,152** (`114.2 TF`) | `111,174` (**97.4%**) | `2,485` (2.2%) | `346.7` | `144.8` | `4.78 GiB` | **`10.27 GiB`** | `768 MiB` | **`~21.98 GiB`** |
| **2** | `300` (~12s) | **114,156** (`114.2 TF`) | `111,174` (**97.4%**) | `2,485` (2.2%) | `346.3` | `149.3` | `4.78 GiB` | **`10.27 GiB`** | `732 MiB` | **`~21.95 GiB`** |
| **4** *(default)* | `64` (~2.6s) | **48,742** (`48.7 TF`) | `47,434` (**97.3%**) | `1,060` (2.2%) | `149.9` | `96.6` | `2.04 GiB` | **`4.38 GiB`** | `340 MiB` | **`~12.93 GiB`** |
| **4** *(default)* | `150` (~6.0s) | **114,176** (`114.2 TF`) | `111,174` (**97.4%**) | `2,485` (2.2%) | `347.4` | `168.3` | `4.78 GiB` | **`10.27 GiB`** | `743 MiB` | **`~21.96 GiB`** |
| **4** *(default)* | `300` (~12s) | **228,312** (`228.3 TF`) | `222,348` (**97.4%**) | `4,970` (2.2%) | `692.8` | `298.6` | `9.55 GiB` | **`20.55 GiB`** | `1.60 GiB` | **`~37.86 GiB`** |
| **4** *(default)* | `600` (~24s) | **456,607** (`456.6 TF`) | `444,696` (**97.4%**) | `9,939` (2.2%) | `1,386.7` | `579.0` | `19.11 GiB` | **`41.09 GiB`** | `3.87 GiB` | **`~70.22 GiB`** |
| **8** | `300` (~12s) | **456,623** (`456.6 TF`) | `444,696` (**97.4%**) | `9,939` (2.2%) | `1,385.5` | `597.1` | `19.11 GiB` | **`41.09 GiB`** | `3.72 GiB` | **`~70.08 GiB`** |

#### `research.yaml` (`v2`, `518px`, `B=2`, `freeze_backbone: false`) Reference Scaling

- **Static Train Base (`4.15B` params):** **`61.85 GiB`**.
- **At `B=2, T=64`:** `~538.2 TFLOPs/step`; saved ViT-L/14 backward activations (`24` layers x `4*2*64 = 512` passes of `1,370` patches) = **`~1,008 GiB`** (`+ 61.85 GiB` static).
- **At `B=2, T=150`:** `~1,260.3 TFLOPs/step`; saved ViT-L/14 backward activations = **`~2,364 GiB`** (`+ 30 GiB` Motion3D + `5.1 GiB` Pose).
- **Conclusion:** Without gradient checkpointing and frame subsampling, `research.yaml` is mathematically **impossible on any single `80 GiB` H100/A100 GPU by >30x** (`crop_size=518`, `separate_hand_face_body=true`, and zero subsampling drive ViT FLOPs up **~28x per ROI pass** vs `224px` + `~2.4 TB` of saved ViT graphs).

### B. Visual vs. Pose Encoder Overhead Breakdown

1. **Visual Encoder (`AppearanceExpert` + `crop_regions`):**
   - Even with `freeze_backbone: true` in `v3_compact`, evaluating `vit_base_patch14_dinov2` across `4` ROIs per frame at `25–30 fps` (`257` tokens/crop) takes **`97.4%` of all model FLOPs** (`~46.4 GFLOPs/frame/batch-item` x `4 = 185.3 GFLOPs/frame`) and copies `[4*B*T, 3, H, W]` inside [`vision.py:48`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L48).
   - Because `DINOv2-B/14` is **completely frozen** during `v3_compact` training, pre-extracting `[T, 4, 768]` CLS embeddings offline (or chunking ViT in `torch.no_grad()` every `64` frames when reading raw video) eliminates **`97.4%` of training compute** and removes the `6.3 GiB` ViT peak completely!
2. **Pose Encoder (`PoseExpert`):**
   - Accounts for only **`2.2%` of FLOPs** (`~4.14 GFLOPs/frame` across 10 `EdgeGraphBlock`s at `D=512`), yet generates **`64.8%` of all saved backward activation RAM** (`~68.5 MiB` per `(B, T)` frame-step -> `20.55 GiB` at `B=4, T=300`) because `eb = self.edge(edge)` allocates `[B*T, N, N, D]` (`21x21` even though 86% of hand edges are masked off, and `36x36` in cross graph!).
   - Also check `Fusion` lag attention (`fusion.py:187-188`): computing `lag_k(keys)` and `lag_v(keys)` over all `K = 9 * 10 = 90` neighbor keys per frame allocates `[B, T, 90, D]` tensors (`738 MiB` at `B=4, T=300`).

### C. Inference Memory (`scripts/evaluate.py`, `num_beams=4`, `max_new_tokens=128`)

Because [`scripts/evaluate.py:65-76`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py#L65-L76) omits `torch.autocast`, inference currently runs in **full `FP32`** (`2x` memory vs `BF16`):

| Profile & Precision | `B` | `T` | Weights | Peak `encode()` (`crops` + 1-stream ViT block) | `T5` Beam=4 KV-Cache (`cross + self`) | **Est. Peak Eval Memory** |
|---|---:|---:|---:|---:|---:|---:|
| `v3_compact` (**current `FP32`**) | `1` | `150` | `1.78 GiB` | `689 MiB` + `1,583 MiB` | `83 MiB` | **`~4.00 GiB`** |
| `v3_compact` (**current `FP32`**) | `1` | `300` | `1.78 GiB` | `1,378 MiB` + `3,166 MiB` | `125 MiB` | **`~6.22 GiB`** |
| `v3_compact` (**current `FP32`**) | `4` | `300` | `1.78 GiB` | `5,512 MiB` + `12,663 MiB` | `500 MiB` | **`~19.53 GiB`** |
| `v3_compact` (fixed `BF16` autocast) | `1` | `300` | `0.89 GiB` | `689 MiB` + `1,583 MiB` | `62 MiB` | **`~3.11 GiB`** |
| `v3_compact` (fixed `BF16` autocast) | `4` | `300` | `0.89 GiB` | `2,756 MiB` + `6,332 MiB` | `250 MiB` | **`~9.77 GiB`** |
| `research_v2` (**current `FP32`**) | `1` | `300` | `15.46 GiB` | `7,370 MiB` + `50,422 MiB` | `690 MiB` | **`~71.90 GiB`** |

Again, beam-search KV cache (`62–500 MiB`) is tiny—[`evaluate.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py#L70) peaks inside `AppearanceExpert` because it evaluates all `B * T` ROI images in one ViT batch.

---

## 4. Conformer vs. Temporal Convolution vs. Mamba Comparison

### Repository Reality Check **(M — Measured in Source)**
- **Temporal Convolution (Present):** [`MultiScaleTemporalStem`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L17-L45) implements 4 parallel dilated `Conv1d` branches (`k=(3,5,9,15), d=(1,1,2,4)`) + `1x1` projection (`3.19 GFLOPs` at `B=1, T=300, D=768`), and [`MotionExpert`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/motion.py#L11-L74) implements `Conv3d` + depthwise `Conv1d` (`disabled` in `v3_compact`).
- **Conformer Backbone (Present):** [`GlobalTemporalBlock`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L64-L108) (self-attention with `RelativeTimeBias` + GLU depthwise `Conv1d(k=15)` + Pre-LN FFN) runs on **consolidated slots `M = ceil(T/2)`**, followed by [`AdaptiveDetailRetrieval`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L155-L194) (`M -> T` cross-attention).
- **Mamba / Selective State Space (ABSENT):** **Zero lines of Mamba/S6 code** exist anywhere in `DINOv2-PoseGraph-3DCNN-Conformer` or sibling workspaces (`CNN-BiLSTM`, `InceptionCNN-Transformer`, `PoseGraph-Conformer`), despite claims in untracked design docs (`reports/04..07`).

### Analytical Complexity & Hardware Comparison (`D=768, L=6, B=1`, Fair Same-Depth Benchmark Formula)

Below compares the implemented `MultiScaleTemporalStem` (`1` multiscale layer), an equivalent `6`-layer depthwise-separable `TCN`, the implemented `6`-layer `GlobalTemporalBlock` Conformer (`stride=2`, `M=ceil(T/2)`) + `AdaptiveDetailRetrieval`, and a theoretical `6`-layer Bidirectional Mamba (`Bi-S6`, `expand=2 -> E=1536`, `d_state=16`) operating at full length `T`:

| Sequence Length `T` (`M = ceil(T/2)`) | Implemented `Stem` (`Conv1d`) **(C)** | `6L` Equivalent `TCN` (`Conv1d+FFN`) **(C)** | Implemented `6L` Conformer + `DetailRet` **(C)** | Conformer Quadratic `M^2` Attn Share & Buffer **(C)** | Theoretical `6L` `Bi-Mamba` (`Bi-S6`) FLOPs **(C)** | Hardware / Portability Assessment **(H)** |
|---:|---:|---:|---:|---:|---:|---|
| `64` (`M=32`) | `0.68 GF` | `2.88 GF` | **`3.12 GF`** | `0.6%` (`0.2 MiB`) | `5.94 GF` (**1.9x higher**) | Mamba offers zero win; sequential scan kernel overhead hurts at `T <= 64`. |
| `150` (`M=75`) | `1.59 GF` | `6.74 GF` | **`7.40 GF`** | `1.4%` (`1.0 MiB`) | `13.91 GF` (**1.9x higher**) | Typical ~6s ISL sentence: Conformer is < `0.03%` of end-to-end `v3` FLOPs! |
| `300` (`M=150`) | `3.19 GF` | `13.49 GF` | **`15.08 GF`** | `2.7%` (`4.1 MiB`) | `27.82 GF` (**1.8x higher**) | Conformer `O(M^2)` still negligible vs `linear` FFN & depthwise conv. |
| `600` (`M=300`) | `6.37 GF` | `26.98 GF` | **`31.27 GF`** | `5.3%` (`16.5 MiB`) | `55.65 GF` (**1.8x higher**) | At `24s` clip, `T5` cross/self-attn (`S_enc=616`) grows larger than Conformer `M^2`. |
| `1200` (`M=600`) | `12.74 GF` | `53.96 GF` | **`66.97 GF`** | `9.9%` (`65.9 MiB`) | `111.30 GF` (**1.7x higher**) | Pure-PyTorch Bi-S6 scan without `mamba-ssm` CUDA kernels runs `O(T)` serial launches; fails native Apple MPS. |

**Why Mamba Should NOT Be Assumed Faster Here:**
1. **(C)** Because `HierarchicalTemporalEncoder` halves resolution to `M = ceil(T/2)` before global attention, quadratic attention represents **only `1.4%–2.7%` (`1–4 MiB`)** of Conformer work at `T = 150..300`, whereas bidirectional S6 doubles the projection/scan count at full resolution `T` (`1.8x` more MACs at typical sign-sentence lengths).
2. **(C)** Across the full `v3_compact` model (`B=4, T=300`), `HierarchicalTemporalEncoder` is **`0.03%` of total FLOPs (`74.5` of `228,312 GFLOPs`)** uncached (or `< 5%` even with cached ViT + refactored Pose). Replacing Conformer with Mamba yields near-zero end-to-end speedup.
3. **(H)** Fused `mamba-ssm` selective-scan kernels require NVIDIA CUDA GPUs (`Linux/x86_64`), whereas native PyTorch `MultiScaleTemporalStem` + `GlobalTemporalBlock` + SDPA execute portably on CPU, Apple MPS (once wired), and CUDA.

---

## 5. Dataset & Checkpoint Disk Feasibility **(M + C)**

[`scripts/convert_video_pose_npz.py:76-81`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/convert_video_pose_npz.py#L76-L81) stores raw decoded `uint8` RGB frames `[T, H, W, 3]` inside zlib-compressed `.npz` archives (`np.savez_compressed`) rather than keeping H.264/HEVC MP4 decoding on the fly or caching frozen ViT features:

| Storage Format (`T = 200` frames/clip, `~8s @ 25fps`) | Calculated Size / Clip | `1,000` Clips | `10,000` Clips (Sub-corpus) | Fits on Host (`11 GiB` free)? **(M + C)** |
|---|---:|---:|---:|---|
| `720p uint8 NPZ` (`1280x720x3`, `~2.5x` zlib) | `~210.9 MiB` | `~206.0 GiB` | `~2.01 TiB` | **No** (`~19x` over disk even for 1k clips) |
| `480p uint8 NPZ` (`640x480x3`, `~2.5x` zlib) | `~70.3 MiB` | `~68.7 GiB` | `~686.6 GiB` | **No** |
| Pre-resized `224x224 uint8 NPZ` (`~2.0x` zlib) | `~14.4 MiB` | `~14.0 GiB` | `~140.2 GiB` | **No** (`1k` clips alone > `11 GiB` free) |
| **Pose + Pre-Cached Frozen `ViT-B` (`[T, 4, 768]` FP16)** | **`~1.3 MiB`** | **`~1.3 GiB`** | **`~12.9 GiB`** | Marginal for 1k micro-test ONLY if freed space + deps allow |
| `research_v3_compact` Checkpoints (`last.pt` + `best_valid.pt`) | `4.70 GiB` each | `9.41 GiB` (`2` files) | — | **Barely 2 files fill 85% of free disk!** |

---

## 6. Separation Summary: Measured vs. Calculated vs. Hypothetical

| Category | Verified Statements Only |
|---|---|
| **(M) Measured on Host** | Apple M2 (`4P+4E`, 8-core GPU), `8.00 GiB` RAM (`7.44 GiB` usable, `~67 MiB` free pages), `11 GiB` free SSD (`95%` used), Python `3.9.6`/`3.10.20`/`3.12.14`/`3.14.6`, zero `torch`/`timm`/`transformers`/`pytest`/`sacrebleu`/`ffmpeg`, `compileall` exits `0`, `import isl_mega` fails in `18.65 ms` (`No module named 'torch'`). |
| **(C) Calculated Estimates** | Exact internal params (`578,705` / `144.49M` / `387.70M`), combined totals (`77.5M` / `478.6M` / `4.15B`), FP32 train static bases (`1.16 GiB` / `6.16 GiB` / `61.85 GiB`), `4.70 GiB` per `v3` checkpoint, FLOP & activation sweeps (`228.3 TFLOPs` & `~37.86 GiB` for `v3_compact` `B=4, T=300`), `ViT` `97.4%` FLOP dominance, `PoseExpert` `64.8%` activation dominance (`20.55 GiB`), `evaluate.py` `FP32` peaks (`6.22–19.53 GiB`), Conformer `M^2` attn FLOP share (`0.6%–9.9%`), `.npz` vs feature-cache disk sizes. |
| **(H) Hypothetical Projections** | Pure-PyTorch selective-scan latency behavior on short `T <= 300` clips without `mamba-ssm` CUDA kernels; expected portability gains once `torch.backends.mps` is wired; single-GPU batch throughput scalability after caching ViT features and refactoring `EdgeGraphBlock`. |

---

## 7. Most Practical Configuration for the First Real-Data Experiment

Do **not** launch `configs/research.yaml` (`4.15B` params, `61.85 GiB` static + multi-TB ViT graphs) and do **not** run [`research_v3_compact.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml) un-modified (`37.86 GiB` VRAM at `B=4, T=300` due to uncached ViT + dense `[B*T, N^2, 512]` `EdgeGraphBlock` tensors).

Instead, separate **local Mac debugging** from the **real-data single-GPU research run**:

### A. Primary Real-Data Experiment Config (Single `16–24 GiB` CUDA GPU, `>= 250 GiB` Disk)

| Setting / Module | Recommended Change vs `research_v3_compact.yaml` | Exact Engineering Impact **(C)** |
|---|---|---|
| **Visual Pipeline (`freeze_backbone: true`)** | Keep shared `vit_base_patch14_dinov2.lvd142m` (`224x224`); **pre-extract `[T, 4, 768]` `FP16`/`BF16` ROI embeddings offline** (or chunk `AppearanceExpert` in `torch.no_grad()` every `64` frames); subsample long clips to `<= 256` frames (`~12.5–15 fps` or window cap) during batching | Cuts training compute by **~97% (`228.3 -> ~1.5 TFLOPs/step`)**, eliminates `6.3 GiB` ViT spike + `~90%` of disk I/O (`~1.3 MiB/clip` vs `~70–210 MiB/clip` video `.npz`), and removes duplicated frozen ViT weights from checkpoints. |
| **Pose Encoder ([`pose.py:97,128`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L97-L130))** | Set `pose.hidden_dim: 256`, `layers: 2`, `cross_articulator_layers: 1`; refactor `EdgeGraphBlock.edge` MLP (`edge_dim -> 64 -> heads`) as an additive `[B*T, H, N, N]` attention-logit bias instead of `[B*T, N, N, D]` | Cuts `PoseExpert` from `34.22M -> ~5.56M` params and shrinks saved pose backward activations at `B=4, T=300` from **`20.55 GiB -> < 0.25 GiB`** (**>80x** VRAM reduction!). |
| **Fusion & Temporal** | `fusion.dim: 512`, `layers: 2`, `heads: 8`; `temporal.dim: 512`, `global_layers: 6`, `ffn_dim: 1536`. Fix `temporal.detail` bug ([`temporal.py:218`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L218)) and omit disabled `motion` projection in [`fusion.py:69-70`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L69-L70) (`motion.enabled: false`) | Reduces Fusion+Temporal from `79.37M -> 35.54M` params (`0` dead params), retains `MultiScaleTemporalStem` + `ContinuousTimeLagBias` + `reliability_uncertainty_gate` core hypothesis cleanly. |
| **Latent Planner ([`planning.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/planning.py))** | **Bypass in clean fusion baseline** (`planner_layers: 0`, removing `28.37M` un-regularized params when `plan_*` weights are `0.0`); test Planner in a dedicated ablation **only after** fixing `PlanGroundingBlock`'s double-FFN | Saves `28.37M` trainable params (`~433 MiB` static optimizer RAM) and isolates the core RGB-pose reliability hypothesis without confounding capacity bloat. |
| **Decoder & Optimisation** | Use **`google/flan-t5-small` (`76.96M`)** for fast `Model 1–5` controlled ablations (`RGB-only`, `Pose-only`, `Unweighted`, `Scalar-Conf`, `5-Desc Quality+Rel`), then scale winner to **`google/flan-t5-base` (`247.58M`)**. Route `decoder.bridge` to `visual_lr: 2e-4`, `decoder.model` to `5e-5`, `amp_dtype: bf16`, `batch_size: 4` with `grad_accum_steps: 4` (`eff B=16`), select `best_valid.pt` by **`valid_translation`** / `chrF++` | Calculated static+activation training footprint at `B=4, T=300` (with cached ViT): **`2.84 GiB` (`flan-t5-small`, `119.5M` trainable)** or **`5.38 GiB` (`flan-t5-base`, `290.3M` trainable)** — comfortably runnable with headroom on a single **`16 GiB` (`L4`/`T4`) or `24 GiB` (`A10G`/`RTX 4090`) GPU**, even without multi-GPU DDP! |

### B. Permissible Role of Local Apple M2 Air (`8 GiB` RAM, `11 GiB` Free Disk)

- **Only for code CI / unit smoke tests** after freeing **`>= 25–30 GiB`** disk space and installing an isolated `.venv` (`torch`, `PyYAML`, `transformers`, `pytest`): run `pytest -q` ([`smoke.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/smoke.yaml) / [`smoke_v2.yaml`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/smoke_v2.yaml) synthetic tensors, `~1.16 GiB` static + `< 0.3 GiB` activation -> **`< 1.5 GiB`** total) and manifest/split validation scripts ([`validate_data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/validate_data.py), [`make_splits.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/make_splits.py)).
- **Never** attempt full `ISLTranslate`/`iSign` training, raw video `.npz` storage, or `ViT-B/14` + `flan-t5-base` fine-tuning locally on this 8 GB MacBook Air.
