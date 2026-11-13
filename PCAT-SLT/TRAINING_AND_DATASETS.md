# PCAT-SLT: training location, datasets, Kaggle hardware and compute budget

Research checked **14 November 2026**, using provider pages, original papers and the current PCAT-SLT v0.3 source. Linked sources establish external facts; proposed experiments and hardware choices are engineering recommendations. Release contents, access terms and platform allocations can change.

## 1. Recommended path

**Start on Kaggle T4 ×2 with a pose-only pilot. Use PHOENIX-2014T for the first full controlled comparison. For an Indian Sign Language product, make iSign the main ISL training resource and use source-disjoint ISL evaluation. For sustained multimodal experiments, prefer an institution-approved server with one A100 40/80 GB or equivalent CUDA GPU and persistent storage.**

You do not need every dataset in this guide. A sensible initial study is PHOENIX-2014T + iSign, adding CSL-Daily or How2Sign if making broader generalization claims. Other sign languages do not replace ISL supervision. Train separate corpus models first; joint multilingual training requires explicit language conditioning and sampling that this trainer does not supply.

No real-corpus training or CUDA benchmark was performed for this guide. The local host has no available CUDA GPU. Software tests and toy-task accuracy do not establish real SLT quality, convergence time or publication readiness. Runtime tables below are **conditional calculations**, not measured T4/A100 speeds.

| Goal | Dataset | Location | Evidence obtained |
|---|---|---|---|
| Installation, gradients, generation, resume | Included synthetic pose/pixel diagnostics | CPU, then CUDA smoke test | Software operation only |
| Test the temporal primitive | PHOENIX-2014T official splits | Kaggle, initially pose-only | Controlled narrow-domain SLT comparison |
| ISL → English | iSign; WSLP SLT held-out evaluation when accessible | Approved institutional CUDA machine preferred for full data | Language-specific learning and source generalization |
| Broader architectural evidence | Add CSL-Daily and/or How2Sign | One larger-memory CUDA GPU | Separately trained cross-corpus comparisons |
| Large-scale pretraining | BOBSL / OpenASL / YouTube-ASL or SL-25, as appropriate | Approved server/cloud with substantial storage | A separate, larger research stage |

## 2. Main datasets and priorities

Counts refer to the cited release, before conversion/exclusion. Record the actual release, usable counts and assignment files: a dataset name alone does not define an experiment.

| Dataset | Signing → text | Published size / splits | Role |
|---|---|---|---|
| **PHOENIX-2014T** | DGS → German | **7,096 train / 519 dev / 642 test** | First full continuous-translation benchmark |
| **iSign** | ISL → English | Original paper: **118,228** sentence/phrase pairs, about **252 h**; describes an 80/10/10 translation split | Main ISL resource; version/source grouping must be explicit |
| **ISLTranslate** | ISL → English | About **31k** sentence/phrase pairs in the 2023 paper | Legacy comparison or smaller development resource; audit iSign overlap |
| **CSL-Daily** | Chinese Sign Language → Chinese | **18,401 / 1,077 / 1,176**, about 23 h total | Strong second conventional benchmark, subject to access |
| **How2Sign** | ASL → English | Original green-screen split: **31,128 / 1,741 / 2,322** clips, 79.12 h; separate approximately 3 h Panoptic subset | Broader translation and limited multiview analysis |
| **WSLP 2025/2026 SLT** | ISL → English | Public/iSign training resources and task-specific held-out evaluation | External ISL protocol; obtain the exact translation release |
| **ISH-NEWS (2026)** | ISL → English | Paper reports **4,222** sentence clips | Optional small additional-domain experiment after overlap audit |

Count sources: [PHOENIX/CSL-Daily original experiment tables](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf), [iSign paper](https://arxiv.org/html/2407.05404v1), [ISLTranslate paper](https://aclanthology.org/2023.findings-acl.665/), [How2Sign paper, Table 2](https://openaccess.thecvf.com/content/CVPR2021/papers/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.pdf), [WSLP tasks](https://exploration-lab.github.io/WSLP/task/), [ISH-NEWS paper](https://www.nature.com/articles/s41598-026-60893-0).

### PHOENIX-2014T: first full experiment

Use the [RWTH provider page](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/). It supplies parallel video, gloss and written translations. Select the **T** translation corpus, not a recognition-only PHOENIX release. The [provider file index](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/) lists `phoenix-2014-T.v3.tar.gz` at **39G**; extraction and preprocessing need additional space.

Preserve official splits and written `translation` targets. Glosses are not the translation target. Its weather domain is useful for controlled comparisons but cannot establish conversational ISL performance. Observe provider terms and attribution before any hosted dataset copy.

**Code:** annotation importer implemented; landmarks and aligned visual crops still need preparation. See [REAL_DATA.md](docs/REAL_DATA.md).

### iSign: main ISL resource, with important access conditions

Use the [authors' project](https://exploration-lab.github.io/iSign/) and [current dataset card](https://huggingface.co/datasets/Exploration-Lab/iSign/blob/main/README.md). The card describes v1.1, `.pose` observations in `pose-format`, and `iSign_v1.1.csv`. It recommends splitting by `video_id`: all `[video_id]-[sequence_number]` segments stay together. The original paper's ratio is not an exact current assignment file. Do not infer official v1.1 train/dev/test counts from the older headline total.

The card lists **CC-BY-NC-SA-4.0** and additionally asks users to agree to research-only use and no further sharing/uploading. **A private Kaggle upload is not automatically permitted.** Establish that the provider permits the intended hosted processing before transferring restricted files or derivatives. An approved institutional machine avoids assuming that permission.

The paper describes caption/automatic-transcription noise in parts of the data. Audit target boundaries and tracker failures, and source IDs across pretraining and evaluation.

**Code:** no dedicated iSign `.pose` converter exists. Convert units, layout, timestamps and confidence to [the NPZ contract](docs/DATA.md). A normalized `uid,text` CSV with explicit split map can use the ISLTranslate-schema importer; this is schema reuse, not a claim that the corpora are identical.

### ISLTranslate: establish overlap before combining

The [2023 paper](https://aclanthology.org/2023.findings-acl.665/) reports about 31k pairs. The [authors' repository](https://github.com/Exploration-Lab/ISLTranslate) provides the original `uid,text` CSV and describes MediaPipe archives. Its current download link points to iSign. Verify which release you obtained and compare clip IDs, source videos and time intervals before combining them. Never pretrain on a test recording through a second corpus.

**Code:** annotation import exists and requires an explicit split map. Raw MediaPipe ordering is not the default 66-joint layout. A verified source-ID prefix is a useful leakage group, but does not establish signer identity.

### CSL-Daily: secondary benchmark and Chinese scoring

The [official access page](https://ustc-slr.github.io/datasets/2021_csl_daily/) releases data to universities/research institutes for research and requires a full-time staff member's signature; a student signature alone is insufficient. Check download size and hosted-processing terms against your granted release. Do not rely on unverified mirrors.

**Code:** the documented annotation schemas have an importer; observations need conversion. Choose an appropriate Chinese-capable T5-family decoder, for example an mT5 variant, or evaluate the byte baseline explicitly. Check target tokenization first. Changing the decoder/dataset is a new experiment, not exact resume from English FLAN-T5 training.

**Metric caveat:** current `metrics.py` uses SacreBLEU `13a` and whitespace-token ROUGE-L. These are not automatically comparable to Chinese character-tokenized benchmark results. Export predictions and use the benchmark's documented Chinese normalization/scorer for the primary comparison. Retain built-in signatures as secondary diagnostics.

### How2Sign: broader ASL with larger storage needs

The [provider page](https://how2sign.github.io/) lists frontal sentence clips at **31G train + 1.7G val + 2.2G test**, versus **290G + 16G + 23G** for frontal full recordings. Its 2D body/face/hand clip keypoints are **21G + 1.2G + 1.6G**. These are provider labels, not measured expanded sizes. The resource is research-only, under **CC-BY-NC-4.0**.

Original clips can have imperfect sentence alignment. Manually realigned timestamp annotations can be used to segment full videos; do not pair those targets with differently segmented clips without checking. Declare the alignment version. The Panoptic subset does not make every clip calibrated 3D. The original paper also shows signer overlap in standard splits: do not call the whole benchmark signer-independent.

**Code:** generic manifests work after annotation/observation conversion; no dedicated How2Sign adapter is bundled.

### Recent ISL evaluation resources

[WSLP 2026 findings](https://aclanthology.org/2026.wslp-1.2/) distinguish sentence translation, isolated recognition and word presence, and discuss weak grounding despite fluent T5 outputs. Obtain the actual **SLT** data and scorer from organizers. The recognition task's 4,398/109/526 examples are not sentence-translation counts. Preserve the held-out-source policy described in a [2026 SLT participant report](https://aclanthology.org/2026.wslp-1.13/).

[ISH-NEWS](https://www.nature.com/articles/s41598-026-60893-0), published July 2026, uses 162 source recordings. Its availability statement offers annotations, splits, timestamps and preprocessing, while raw video redistribution is excluded for copyright reasons. It is optional until retrieval, terms and overlap with other ISH-derived corpora are checked. Its existence does not negate earlier ISLTranslate/iSign resources.

## 3. Optional resources: targeted questions, not mandatory downloads

| Resource | What it supplies | Appropriate use / limitation |
|---|---|---|
| [OpenASL release](https://github.com/chevalierNoir/OpenASL) / [paper](https://arxiv.org/abs/2205.12870) | About 288 h ASL, over 200 signers; repository license CC-BY-NC-ND-4.0 | Later ASL expansion; acquisition, conversion and overlap audit required |
| [BOBSL v1.4](https://www.robots.ox.ac.uk/~vgg/data/bobsl/) | 1,447 h excluding challenge; about 1.2M subtitles; train/val/test 993K/20K/165K | Large BSL pretraining, with weak subtitle alignment. Provider lists 331 GB MP4 / 1.5 TB frame LMDB. BBC approval and eligibility restrictions apply, including no independent-researcher/commercial use |
| [YouTube-ASL](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md) | 984 h, 610,193 captions | Later ASL weakly supervised pretraining; metadata access does not grant universal video redistribution rights |
| [YouTube-SL-25 paper](https://arxiv.org/abs/2407.11144) / [release](https://github.com/google-research/google-research/tree/master/youtube_sl_25) | More than 3,000 h across more than 25 sign languages | Later multilingual pretraining; language identification, caption quality and new training/data engineering needed |
| [FLEURS-ASL](https://aclanthology.org/2025.naacl-long.314/) | ASL by five Certified Deaf Interpreters, supporting multilingual evaluation | Reserve as external ASL evaluation rather than a large training source |
| [SignNet-1M release status](https://github.com/openhe-hub/SignNet-1M) | 2026 synthetic view/background/identity augmentation; table currently lists 350,530 How2Sign variants, 134 GiB | Optional view stress test. Full approximately 1M corpus is not all marked released. Keep every derivative in its original source split; generated views are not independent recordings or physical 3D ground truth |
| [CISLR](https://github.com/Exploration-Lab/CISLR) | ISL isolated word recognition | Optional separate lexical study; translation importer deliberately rejects it as a sentence corpus |
| [ISL-FS / ISL-FS-Dense](https://kirandevraj.github.io/ISL-Fingerspelling/) | 1,308 original spelling segments; site also announces dense annotations and 11,700+ discovered segments | ISL fingerspelling diagnostic; audit ISH-source overlap; not sentence SLT |
| [FSboard](https://openaccess.thecvf.com/content/CVPR2025/html/Georg_FSboard_Over_3_Million_Characters_of_ASL_Fingerspelling_Collected_via_CVPR_2025_paper.html) | More than 3M ASL fingerspelling characters, more than 250 h, 147 signers | Optional ASL spelling research; distinct from ISL and general sentence translation |

WLASL, MS-ASL, AUTSL and INCLUDE are not required for the first translation study. Classification, CTC recognition, contrastive pretraining and multi-task sampling are not implemented merely by listing their datasets. Do not report isolated-word accuracy as sentence-translation quality.

## 4. Kaggle hardware: current recommendation

**Select GPU T4 ×2; use one card initially. Do not select TPU for this release.** The [Kaggle team announcement](https://www.kaggle.com/product-announcements/735239) retired P100 on **15 September 2026**, replacing it with T4 ×2, **16 GB per card**. Advice to select a Kaggle P100 is now stale.

| Accelerator | Current project support | Recommendation |
|---|---|---|
| T4 ×2 | Single CUDA device; actual free memory shown by `nvidia-smi` | Best free start; batch 1 initially |
| TPU | No TPU/XLA training/device path | Do not use with this source release |
| A100 40/80 GB / H100 if actually available to your account | One CUDA device | More capacity for full multimodal runs; still profile |
| CPU | Data checks and diagnostics | Avoid full-corpus multimodal training |

Two T4s are not one 32 GB device. The trainer implements neither DDP nor model sharding; normal training leaves card 2 idle. `torchrun` alone does not add distribution and may cause conflicting writes. Two independently GPU-bound experiments require separate output directories and sufficient host RAM; validate one first.

The [official Kaggle CLI docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) map `NvidiaTeslaT4` to T4 ×2 and `TpuV5E8` to TPU v5e-8. Other listed IDs can be competition/admin restricted; a listed A100/H100 ID does not guarantee free access. The [CLI reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md) documents replacement of retired TPU-v3 IDs. Old TPU-v3 tutorials do not establish today's allocation.

PyTorch TPU training requires an actual backend port, including runtime/device setup, optimizer synchronization, distributed input handling and compilation/precision checks. Adaptive shapes, variable clip lengths and Python scalar decisions need recompilation/synchronization profiling. See [PyTorch/XLA conversion docs](https://docs.pytorch.org/xla/master/learn/xla-examples.html). `device: tpu` is currently rejected.

### Quota, sessions and persistence

[Kaggle notebook specifications](https://www.kaggle.com/docs/notebooks) list **12 h CPU/GPU sessions**, **9 h TPU sessions**, **20 GB saved working output**, and about **29 GB host RAM / four CPU cores** for T4. Some hardware descriptions on that page still mention retired accelerators; use the newer announcement/CLI for hardware identity. Check the actual session timer, memory and output allowance.

The [official GPU guide](https://www.kaggle.com/docs/efficient-gpu-usage) describes **30 h/week or sometimes higher**. [Staff's floating-quota explanation](https://www.kaggle.com/discussions/product-feedback/173129) establishes variability. Budget on the quota displayed in your account, with 30 h as a planning baseline. Do not promise a fixed TPU quota from old tutorials. Extra-hours promotions are not assumed here.

Use authorized attached inputs under `/kaggle/input`, and keep project/output files under `/kaggle/working/PCAT-SLT`. Save a completed notebook version with outputs and retrieve checkpoints. Verify that the saved version actually contains `last.pt`, `best.pt`, config and logs; a file existing in a live session is not evidence it survived termination. Scratch outside saved output is temporary. Keep restricted data and credentials out of public notebooks/outputs.

## 5. Memory: inspect the actual model, not only the GPU name

`configs/vpcat.yaml` is an experimental starting point, **not a certified T4 fit**: frozen DINOv2-small, trainable FLAN-T5-base decoder, width 320, four crops/frame, batch 2 and up to 400 frames. HF decoder depth comes from its pretrained architecture, not `model.decoder_layers` (which controls the native decoder).

In visual + dual-body + detail mode, decoder memory contains coarse/local tokens, four visual tokens/frame, one motion token/frame and one detail token/frame. The example's coarse/local upper bound is approximately 162 tokens, giving **about `162 + 6T` source tokens**: approximately 2,562 at 400 frames, not merely 64. Adaptive cells can lower the coarse part. Padding and target lengths still matter.

Float32 RGB alone requires:

```text
bytes = batch × frames × regions × 3 × height × width × 4
B=1, T=400, R=4, H=W=112: 240,844,800 bytes ≈ 230 MiB
B=2, same dimensions:                         ≈ 459 MiB
```

This excludes CPU copies, decoding buffers, activations, weights and optimizer states. Doubling both crop dimensions multiplies pixel volume by four. About **16 bytes/trainable float32 parameter** is an AdamW weights/gradients/two-moment baseline before activations/overhead. Current native AMP retains float32 parameters/states. Frozen vision saves gradient/state memory but retains weights and forward compute; frozen text still needs gradients through its activations to the sign encoder.

Checkpoint reload can temporarily hold checkpoint and model tensors on the GPU. **Profile resume peak as well as fresh training.** Pretrained initialization can also peak above steady state. Optimizer steps and generation have different peaks from the first forward pass.

| Supplied profile | Purpose | Starting settings, unbenchmarked on CUDA |
|---|---|---|
| `configs/kaggle_pose.yaml` | Cheap first mechanism study | Native byte decoder, no RGB, width 320, batch 1, accumulation 8, float32 integration, CUDA AMP |
| `configs/kaggle_visual.yaml` | First visual memory/learning pilot | Frozen DINOv2-small, **FLAN-T5-small**, four offline-prepared 112px crops, batch 1, accumulation 8, vision chunks 8, CUDA AMP |
| `configs/vpcat.yaml` | Larger multimodal hypothesis | FLAN-T5-base; profile on larger GPU or adapt before a new T4 run |

Both Kaggle profiles expect `data/real/{train,val,test}.jsonl`, allow 400 prepared frames and reject overlength targets. They retain the v0.3 mechanism and global detail memory. They are starting configurations, not tuned results. Compare the primitive with a fixed decoder/backbone: byte versus pretrained text is a separate capacity comparison.

Pretrained resource references: [FLAN-T5-small](https://huggingface.co/google/flan-t5-small), [DINOv2-small](https://huggingface.co/facebook/dinov2-small), [mT5-small, including Chinese in its pretraining coverage](https://huggingface.co/google/mt5-small). A multilingual tokenizer is necessary for coverage but does not guarantee accurate translation. mT5's large vocabulary can make even its small variant materially different in memory from FLAN-T5-small; profile it separately.

On OOM, first use batch 1, smaller vision chunks and a smaller T5-family decoder if needed. Reduce padding/observation lengths only through an audited, consistently applied data policy. **Accumulation cannot shrink a single clip's activations.** There are no implemented LoRA, gradient-checkpointing, 8-bit-optimizer, cached-DINO-feature input, bucketing or FlashAttention YAML switches.

`max_frames` rejects longer clips; it does not resample them. Changing it to 128 is not a working long-video solution. If offline temporal sampling is necessary, apply identical frame indices to pose/confidence/RGB/masks/timestamps, preserve physical timestamps and cover the whole sentence. Report frame-rate/duration distributions and exclusions. Front truncation can remove meaning; sparse sampling can destroy fingerspelling. Fix this policy before matched experiments.

## 6. Data preparation before consuming GPU quota

1. Acquire an authorized, pinned release. Record file hashes, terms, versions and assignment files. Raw corpora are not included in the source ZIP.
2. Keep source recording, interval and signer metadata. All camera views/augmentations of a recording stay together. Unknown signer metadata does not prove signer independence.
3. Map landmarks explicitly. Default 66-joint ordering, shoulder/neck anchors and configured hand/face/body regions are not raw MediaPipe/OpenPose ordering. Use consistent image-plane x/y units; do not mix hand-relative and image-relative coordinates. Use `depth_policy: ignore` for unreliable monocular z.
4. Prepare pose NPZ: `[T,J,3]`, confidence `[T,J]` in [0,1], strictly increasing seconds `[T]`. Visual NPZ: `[T,4,3,112,112]` RGB in [0,1], bool mask `[T,4]`, exactly matching frame counts/timestamps/origin. Crop order: left hand, right hand, mouth, eyes. Missing observations remain marked absent.
5. `scripts/prepare_rgb.py` takes frames and explicitly aligned image-plane boxes; it does not detect hands/faces or extract pose. A raw-video pose/box extractor and iSign `.pose` converter are not bundled. Crop normalization happens inside the encoder.
6. Compute usable counts, min/median/p95/p99/max frame/target lengths, missing-joint/crop fractions, unanchored fraction, signer/source overlap and duration histograms. Inspect real clip/target alignment and tracker failures in every split without tuning on test labels.
7. Follow [REAL_DATA.md](docs/REAL_DATA.md) to import and validate. PHOENIX split files are imported into separate fresh output directories. Manifest-relative paths resolve from their parent. Validator checks exact decoded/path leakage; near duplicates and upstream corpus overlap require provenance/content analysis too.
8. Preserve official splits for comparisons and separately name any stronger source-/signer-held-out experiment. `--signer-independent` requires known nonoverlapping signer identities. Do not label a modified split as the official benchmark.

### Storage: float crops can dominate everything

Original compressed-video size is not prepared-array size. Before compression, pose/confidence costs approximately `T × 66 × 4 × 4` bytes plus timestamps: about 0.21 MB at 200 frames. Four float32 112px RGB crops cost **120.4 MB/clip at 200 frames**.

| Hypothetical 200-frame clips | Raw pose/confidence | Raw float RGB crops |
|---|---:|---:|
| 1,000 | 0.21 GB | 120 GB |
| 7,096 | 1.50 GB | 854 GB |
| 31,128 | 6.57 GB | 3.75 TB |
| 118,228 | 25.0 GB | 14.2 TB |

These are arithmetic scenarios, **not measured corpus sizes**. They omit metadata/padding and assume no compression savings. `prepare_rgb.py` writes compressed NPZ, but measure 100–500 representative processed clips before extrapolating; do not assume a compression ratio. The loader reopens/decompresses arrays each epoch. Attaching MP4s does not make float-crop training storage cheap.

Start pose-only, then a visual subset, then measure full-corpus storage/I/O feasibility. Uint8 crop storage, streaming decode or cached frozen features could help but require loader/model changes and validation; they are not existing flags.

Budget downloads **plus** extraction, processed arrays, HF cache and checkpoints. `best.pt` and `last.pt` contain full state, and atomic replacement creates a temporary third file. A rough checkpoint floor is `4 × total_model_parameters + 8 × parameters_with_Adam_state` bytes, excluding metadata. Measure the real file after updates. Keep at least three checkpoint-sized units free plus caches/logs. Do not put hundreds of GB of inputs into a 20 GB saved-output allowance.

## 7. Concrete Kaggle workflow

Use CPU sessions for preparation where practical. Create a private notebook, attach only authorized data and enable internet for package/model access when needed. No upload, remote training or dataset acquisition was performed while preparing this guide.

Extract the source ZIP's single **PCAT-SLT** directory under `/kaggle/working`. Notebook setup cells:

```python
%cd /kaggle/working/PCAT-SLT
```

```bash
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
!python -m pip install -e '.[dev,multimodal]'
!python -m pytest -q
!python -m pip freeze > reports/kaggle_environment.txt
```

Retain Kaggle's CUDA-enabled torch if it meets project requirements. `requirements-verified.txt` records the local CPU environment; blindly imposing it can replace a working CUDA stack. Record Python, torch, transformers, driver and GPU. Profiles use HF `main`; pin immutable pretrained commit revisions before final research and retain tokenizer/config snapshots in checkpoints.

Create `data/real` manifests using documented adapters after preparing observations. Stable absolute `/kaggle/input/...` paths may refer to authorized attached data. Keep generated project/config/report/weight files under PCAT-SLT.

```bash
!pcat-validate --config configs/kaggle_pose.yaml > reports/real_data_validation.json
```

For visual mode, validate with `configs/kaggle_visual.yaml`; it requires actual RGB and matching tokenizer assets. Validate all splits before training. A development subset needs separate manifests/config/output, leaving benchmark assignments intact. First overfit 32–128 training examples as a failure diagnostic, then use a 500–2,000-example training pilot. Neither establishes full-benchmark quality.

Use the intended total epoch count in YAML from the start:

```bash
!pcat-train --config configs/kaggle_pose.yaml --stop-after-epoch 1
```

Next session, restore identical inputs/config/output and continue:

```bash
!pcat-train --config configs/kaggle_pose.yaml --resume runs/kaggle_pose/last.pt --stop-after-epoch 2
```

Increase the stop epoch for later chunks. Visual mode uses its own config and `runs/kaggle_visual/last.pt`. Stop chunks with time remaining to save outputs. Final frozen test evaluation:

```bash
!pcat-evaluate --checkpoint runs/kaggle_pose/best.pt --manifest data/real/test.jsonl --device cuda --output evaluation/kaggle_pose_test
```

Evaluate dev separately for selection. Built-in `best.pt` selection uses **validation NLL**, not BLEU. Autoregressive generation evaluation is separate and can cost much more than teacher-forced validation. Report length/EOS behavior and input dependence, not just fluency.

### Resume constraints on a session-limited platform

Checkpoints occur **at complete epoch boundaries**, not every step. An interrupted epoch is redone. Exact resume checks config except `output_dir`, manifest bytes and input fingerprints. Changing epochs, batch size, AMP, model revision, dataset or paths is rejected. Set total epochs before training so the cosine schedule stays consistent. Changing datasets/decoders is not `--resume`; a general weight-only fine-tuning-stage CLI is not bundled.

Every startup/resume fingerprints all referenced pose/RGB files, adding considerable I/O. Absolute resolved paths participate; keep input layout stable. Preserve config, logs and both checkpoints, not just decoder weights. Profile reload memory separately.

Operational recommendation: keep an epoch plus validation/writes below about **8–9 h**, leaving margin within a nominal 12 h session for setup/output capture. This margin is not an official limit. If one epoch exceeds the session, use longer-lived compute or implement/test true mid-epoch resume. Changing corpus shards between sessions is not exact resume.

## 8. Training time and budget

There is no measured fixed T4/A100 runtime yet. Prepared frames, crop size, missing observations, targets, dtype, decoder, NPZ decompression and storage throughput all affect it. CPU operator microbenchmarks cannot predict full GPU training.

For training examples `N`, microbatch `B`, accumulation `A`, epochs `E`:

```text
microbatches/epoch = ceil(N / B)
optimizer updates/epoch = ceil(ceil(N / B) / A)
training hours ≈ E × ceil(N / B) × measured_mean_seconds_per_microbatch / 3600
total = training + validation + generation evaluation + startup/hashing + checkpoint I/O
```

Measure representative batches and long cases with CUDA synchronization; include optimizer updates amortized over accumulation. Accumulation does not divide the number of forward/backward passes.

### Conditional scenarios, NOT measured T4 speeds

Assume **batch 2, 40 epochs**, and hypothetical means of **0.25 / 1 / 2 seconds per training microbatch**, including amortized updates but excluding validation/setup. These are sensitivity points, not hardware bounds; actual performance can be outside them. Recalculate for the supplied batch-1 profiles.

| Dataset | N used | 0.25 s/batch | 1 s/batch | 2 s/batch |
|---|---:|---:|---:|---:|
| PHOENIX-2014T | 7,096 | 9.9 h | 39.4 h | 78.8 h |
| CSL-Daily | 18,401 | 25.6 h | 102.2 h | 204.5 h |
| How2Sign original | 31,128 | 43.2 h | 172.9 h | 345.9 h |
| Illustrative iSign train partition | 94,582 (approximately 80% of original total) | 131.4 h | 525.5 h | 1,050.9 h |

The iSign row is **not** an official current split. Replace N with your manifest count. Forty epochs is an example planning horizon, not a validated optimum; justify stopping/model selection with dev curves.

An explicit budgeting assumption of **25% additional nontraining overhead** on the middle column gives about **49 / 128 / 216 / 657 h**. This overhead is not measured; slow I/O or beam evaluation can exceed it. At an available 30 h/week, this is roughly **2 / 5 / 8 / 22 quota weeks for one run**, before queueing/failures. Do not apply a guessed A100 speed multiplier.

A first diagnostic with 1,000 examples × 10 epochs × batch 1 × a measured 1 s/batch would take **2.8 h training-only**. Its convergence is not evidence about the full corpus.

### Measure an actual epoch using the existing trainer

Run this notebook cell with real prepared inputs and an unused output directory. It writes timing/memory/checkpoint evidence inside the project. The CUDA branch has not been exercised on the local CPU host.

```python
import json
import time
from pathlib import Path
import torch
from pcat_slt.config import load_config
from pcat_slt.train import train

cfg = load_config('configs/kaggle_pose.yaml')  # or kaggle_visual.yaml
assert torch.cuda.is_available(), 'Select a Kaggle GPU first'
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
checkpoint = train(cfg, stop_after_epoch=1)
torch.cuda.synchronize()
report = {
    'device': torch.cuda.get_device_name(0),
    'torch': str(torch.__version__),
    'total_planned_epochs': cfg.train.epochs,
    'completed_epochs': 1,
    'first_call_seconds': time.perf_counter() - started,
    'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
    'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
    'checkpoint_bytes': Path(checkpoint).stat().st_size,
    'batch_size': cfg.data.batch_size,
    'accumulation': cfg.train.accumulation,
    'config': cfg.to_dict(),
}
Path('reports').mkdir(exist_ok=True)
Path('reports/kaggle_first_epoch.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
```

This measures downloads/loading, hashing, an epoch, validation and writes, not pure steady-state training. In a fresh next session, similarly time `train(cfg, resume='runs/kaggle_pose/last.pt', stop_after_epoch=2)` (use the visual path when appropriate). That includes reload/hashing overhead. Do not rerun the non-resume cell over existing output.

Use whole-call measurements conservatively for session planning. Measure generation and preparation separately. Keep `nvidia-smi` logs: PyTorch allocation excludes other CUDA/process memory, while reserved memory includes allocator cache. One short successful batch does not prove longest-clip or resume fit. Check validation/test lengths too. Evaluation reloads in normal precision; it does not automatically use training AMP.

### Whole-study compute

Five matched variants × three seeds = **15 training runs**, before pilots. Under the illustrative PHOENIX 1 s/batch case, this is about **591 training h**, or **739 h** with the assumed 25% overhead: about **25 quota weeks** at 30 h/week. Preprocessing, failed runs, human review and writing add calendar time. Kaggle is useful for iteration but inconvenient for a broad study.

For paid capacity, obtain a quote for the exact GPU/host RAM/storage and permitted data use. Calculate `instance hours × current instance rate + storage + transfer`. An instance with two paid GPUs can charge for both even if this trainer uses one. No price quote or spending authorization is implied. Compare a representative 1–2 h pilot before renting for long runs.

### Alternatives to Kaggle and example paid costs

An approved university/research server is the first choice when its data-access agreement covers the corpus. For permitted hosted processing, compare a single-GPU [Runpod Pod](https://www.runpod.io/pricing) with [Lambda instances](https://lambda.com/instances), checking capacity, persistent storage, interruptions and the actual invoice rate. Colab can also run CUDA code, but [Google's FAQ](https://research.google.com/colaboratory/faq.html) states that resources/limits vary; do not buy a notebook subscription assuming a guaranteed particular GPU or uninterrupted long job.

The Runpod public Pods page, checked 14 November 2026 and labeled updated 12 November, displayed **A100 PCIe 80 GB at $2.09/h**, **H100 PCIe 80 GB at $2.89/h**, and **L40S 48 GB at $1.09/h**. These are page snapshots, not reserved availability or a checkout quote; pool, region and selected offering can change the rate. Verify the deployment screen and storage charges before committing. [Source: Runpod pricing](https://www.runpod.io/pricing).

If a run actually measures **100 billed hours on that A100 offering**, compute alone at the displayed rate would be **$209**; 500 h would be **$1,045**, before storage/other charges. These examples do not convert the T4 timing scenarios into A100 runtimes. L40S's larger capacity is useful, but its lower listed hourly price does not prove a lower cost per completed experiment. Measure throughput on the exact allocated device.

## 9. Research plan and initial hyperparameters

1. **Data/software gate:** validate data, inspect alignment, pass tests on the selected CUDA environment and overfit a tiny training subset. Show input dependence so a pretrained decoder cannot win through fluent hallucination.
2. **PHOENIX mechanism pilot:** compare PCAT with otherwise matched temporal/first-order controls, fixing data, decoder/backbone, schedule, tuning budget and seeds. Use generated dev metrics as well as NLL.
3. **Main ablations:** joint versus additional local clocks; dense versus sketched area; first-order/no-transport; exact Cayley versus Givens as distinct maps. Match capacity as closely as possible and disclose differences. Byte versus larger pretrained text is not proof of the primitive.
4. **Visual extension:** add actual crops under a fixed observation policy. Test RGB/pose masking and temporal shuffling; report peak memory, throughput, generation latency and parameters. The included pixel toy is not an SLT benchmark.
5. **ISL study:** iSign with documented source-held-out assignments and deduplicated external evaluation. Preserve order/timestamps; separate cross-language transfer from native ISL learning.
6. **Broader robustness:** independently train CSL-Daily or How2Sign if claiming breadth. Test frame-rate changes, missing landmarks/crops and known held-out signers. Distinguish conditional mathematical invariants from physical perspective robustness.
7. **Final evidence:** at least three seeds for main conclusions; paired bootstrap intervals on fixed predictions; explicit scoring normalization/signatures; correct Chinese scoring; exclusions/versions; fluent/Deaf-signer adequacy review where available, especially negation, entities, non-manuals and fingerspelling. Test outputs must not drive hyperparameter tuning.

Starting configs use AdamW, clipping 1, accumulation for nominal effective batch 8, initial seed 17 (expand to 17/23/41). Native-byte pilot LR is `3e-4`; pretrained-text pilot LR is `5e-5`. These are proposed starting values, not tuned optima. Make a small dev-only search with equal budgets for controls. Keep target smoothing/selection consistent. Nominal batch equality does not imply equal token counts across languages. Track EOS, length, rare terms and input grounding.

## 10. Practical work still needed before scaling

| Issue | Immediate action | Not currently supplied |
|---|---|---|
| Access / hosted restrictions | Confirm actual provider/release terms | Universal upload permission |
| Videos / `.pose` | Explicit layout, crop, timestamp conversion | All-corpus extraction/conversion pipeline |
| RGB storage/I/O | Measure compression and decode throughput first | Streaming/cached-feature training input |
| Session termination | Complete-epoch chunks and verified saved outputs | Mid-epoch preemption resume |
| Two T4 devices | Use one initially | DDP/FSDP/sharding |
| TPU | Choose CUDA | TPU integration/tests |
| Changed dataset after pretraining | Separate weight-transfer stage design | General weight-only fine-tune CLI |
| Long clips | Audit lengths and sampling policy | Automatic bucketing/long-sequence strategy |
| Chinese scoring | External benchmark scorer on exported predictions | Language-specific CLI scorer selection |
| Quality/efficiency claims | Measure real-data results and CUDA performance | Proven SOTA/publication guarantees |

For larger runs, **one A100 40 GB is a reasonable proposed capacity target; 80 GB gives more room**, with 64–128 GB host RAM and storage sized from actual prepared inputs, potentially hundreds of GB or TB. These are provisioning targets, not measured minimums. [NVIDIA's specification](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf) confirms the 40/80 GB variants. Check that an allocation is a full GPU rather than a small MIG slice. This single-device trainer gains nothing automatically from paying for eight GPUs.

Immediate sequence: **authorized data → pose pilot on T4 → measured complete epoch and resume → PHOENIX matched comparison → iSign on approved persistent compute → budgeted visual expansion**.
