# PCAT-SLT: training location, datasets, Kaggle hardware and compute budget

Research checked **14 November 2026**, using provider pages, original papers and the current PCAT-SLT v0.3 source. Linked sources establish external facts; proposed experiments and hardware choices are engineering recommendations. Release contents, access terms and platform allocations can change.

Dataset descriptions and budgets live in this file. Follow [the Kaggle walkthrough](docs/KAGGLE_RUNBOOK.md) for accelerator selection and executable setup/save/resume instructions, [data preparation](docs/DATA.md) for arrays/crops/importers, and [the experiment plan](docs/EXPERIMENTS.md) for comparisons and initial hyperparameters. The [notebook](notebooks/Kaggle_PCAT_SLT.ipynb) implements the walkthrough. Third-party datasets are linked with access instructions, not redistributed here.

## 1. Recommended path

**Start on Kaggle T4 ×2 with a pose-only pilot. Use PHOENIX-2014T for the first full controlled comparison. For an Indian Sign Language product, make iSign the main ISL training resource and use source-disjoint ISL evaluation. For sustained multimodal experiments, prefer an institution-approved server with one A100 40/80 GB or equivalent CUDA GPU and persistent storage.**

You do not need every dataset in this guide. A sensible initial study is PHOENIX-2014T + iSign, adding CSL-Daily or How2Sign if making broader generalization claims. Other sign languages do not replace ISL supervision. Train separate corpus models first; joint multilingual training requires explicit language conditioning and sampling that this trainer does not supply.

No real-corpus training or CUDA benchmark was performed for this guide. The local host has no available CUDA GPU. Software tests and toy-task accuracy do not establish real SLT quality, convergence time or publication readiness. Runtime tables below are **conditional calculations**, not measured T4/A100 speeds.

## 2. Datasets

This is the single dataset catalog for the project. It covers relevant translation, pose, recognition, spelling and auxiliary resources; it is not an exhaustive list of every corpus worldwide. Sizes/counts are source-reported, not locally verified downloads.

| Priority | What to obtain | Reason |
|---|---|---|
| First ISL experiment | **iSign poses + translation CSV**, subject to access terms | Existing pose extraction avoids starting from raw video |
| First controlled benchmark | **PHOENIX-2014T** translation release | Smaller conventional benchmark with official splits |
| Second continuous benchmark | **CSL-Daily** or **How2Sign** | Chinese daily communication or broader ASL translation |
| ISL external evaluation | WSLP **translation** task; optionally ISH-NEWS after overlap audit | Held-out source/domain evidence |
| Existing lightweight isolated pose data | OpenHands labeled collection, especially INCLUDE | Useful only for a separately implemented lexical objective |
| Later large-scale pretraining | Relevant-language OpenASL, BOBSL, CSL-News, YouTube corpora | More data, significantly more engineering and compute |
| Specific failure analysis | ISL-FS/FSboard/ChicagoFSWild; optional geometry data | Spelling, tracking or viewpoint questions rather than ordinary sentence SLT |

**No dataset below is automatically a raw-file drop-in for PCAT-SLT.** Current training consumes prepared NPZ observations and JSONL written-text targets. The included synthetic diagnostics are the only fully generated, immediately executable datasets supplied by the project.

### A. Continuous translation and linguistic corpora

#### A1. iSign — ISL → English, first choice for the target language

- Source: [authors' project](https://exploration-lab.github.io/iSign/), [current release/card](https://huggingface.co/datasets/Exploration-Lab/iSign/blob/main/README.md), [paper](https://arxiv.org/html/2407.05404v1).
- Original publication: 118,228 video-sentence/phrase pairs, approximately 252 hours, with an 80/10/10 translation split described in the paper. That ratio does not identify the exact assignments or counts of a current release.
- Existing inputs: v1.1 split video archives, four pose archive parts, `.pose` format and `iSign_v1.1.csv`. Keep only the modalities needed for your first pilot.
- Preparation: read `.pose` headers/components, select joints explicitly, map coordinates/confidence and preserve frame timing; export NPZ and stable source-grouped manifests. The repository does not yet include that converter. Generic `uid,text` annotation import can reuse the `isltranslate` schema after normalization.
- Splits/terms: group by source `video_id`; the card has research/noncommercial and further-upload restrictions. Obtain the intended hosted-processing permission before placing it on Kaggle; private hosting is not automatically permitted.
- Compute: small pose pilot first; full visual data needs a storage survey and actual GPU measurements.

#### A2. ISLTranslate — legacy ISL → English translation

- Source: [original repository](https://github.com/Exploration-Lab/ISLTranslate), [2023 paper](https://aclanthology.org/2023.findings-acl.665/).
- About 31k published sentence/phrase pairs. Repository describes MediaPipe holistic features and a `uid,text` CSV; its current dataset link points to iSign.
- Preparation: identify the actual archive/version, convert its landmark convention, and supply an explicit split map to `pcat-import-corpus --dataset isltranslate`.
- Audit: source IDs/intervals may overlap other ISL corpora. Do not sum headline counts or use it as independent test data without checking. Verify the terms of the archive actually supplied.
- Compute: appropriate for a smaller ISL experiment after conversion; no measured training ETA is available.

#### A3. PHOENIX-2014T — DGS → German

- Source: [RWTH release](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), [archive index](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/).
- Standard translation split: 7,096 train / 519 dev / 642 test; see [original experiment table](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf). Provider lists the T archive at 39 GB.
- Existing inputs: signing frames, gloss and written-language annotations. Directory name `features/fullFrame-210x260px` does not mean PCAT-compatible pose features.
- Preparation: extract/obtain mapped landmarks; optionally create aligned crops; import official pipe-delimited `name, translation` records using the implemented adapter, with separate train/dev/test invocations.
- Terms/splits: retain official assignments and provider attribution/terms; verify hosted-use rights. Narrow weather-domain results do not establish ISL conversational performance.
- Compute: recommended first complete pose benchmark on Kaggle, after a measured pilot.

#### A4. CSL-Daily — Chinese signing → Chinese

- Source: [official access page](https://ustc-slr.github.io/datasets/2021_csl_daily/), [original paper/statistics](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf).
- 18,401 train / 1,077 dev / 1,176 test; gloss and written Chinese annotations.
- Access: research institutions; release agreement requires a full-time staff signature. Check archive contents and hosted processing against your approved agreement.
- Preparation: mapped observations + the implemented `csldaily` annotation adapter. Use a suitable Chinese-capable text model or declared byte baseline. Score exported text with the benchmark's Chinese normalization; built-in whitespace ROUGE/13a BLEU is not an automatic equivalent.
- Compute: second-stage benchmark; pose pilot on T4, larger-memory GPU for expensive visual variants if profiling requires it.

#### A5. How2Sign — ASL → English with existing keypoints

- Source: [provider downloads](https://how2sign.github.io/), [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.pdf).
- Original green-screen split: 31,128 / 1,741 / 2,322 clips, 79.12 h; separate approximately 3 h Panoptic material.
- Existing modalities: videos, body/face/hand 2D keypoints and English annotations. Provider lists frontal clips at 31G train / 1.7G val / 2.2G test and keypoints at 21G / 1.2G / 1.6G. Full frontal recordings are 290G / 16G / 23G. These are provider download labels, not measured expanded sizes.
- Preparation: map OpenPose-style keypoints and validity to the project layout, convert targets to generic JSONL. Choose original versus manually realigned segmentation consistently; different segmentation can change clip counts.
- Terms: provider lists CC-BY-NC-4.0 and research-only use. Standard splits contain signer overlap; do not label the whole benchmark signer-independent.
- Compute: a useful pose-first ASL option. RGB expansion and repeated NPZ decode need measurement before full visual training.

#### A6. WSLP 2025/2026 translation shared tasks — ISL evaluation protocol

- Sources: [2025 task page](https://exploration-lab.github.io/WSLP/task/), [2026 findings](https://aclanthology.org/2026.wslp-1.2/), [2026 translation report](https://aclanthology.org/2026.wslp-1.13/).
- Choose the **SLT** track. Recognition and word-presence are different datasets/objectives; their split sizes are not translation counts.
- Obtain the actual release, permitted training resources and scorer from the organizers. Availability of test labels, submissions and task downloads can differ after a challenge ends.
- Preparation: convert supplied pose/video observations and exact task targets; retain disjoint source-video evaluation. No dedicated WSLP adapter exists.
- Role: held-out ISL evaluation rather than another corpus to concatenate indiscriminately. The 2026 report discusses fluency versus grounding failures, so include input-dependence controls.

#### A7. ISH-NEWS (2026) — smaller ISL news-domain study

- Source: [original paper and data-availability section](https://www.nature.com/articles/s41598-026-60893-0).
- Reports 4,222 sentence clips from 162 source recordings. Annotations, splits, timestamps and preprocessing are linked; raw videos are not redistributed for copyright reasons.
- Preparation: obtain permissible source recordings, reconstruct exact intervals, map landmarks and targets. No dedicated adapter exists.
- Role: optional extra-domain experiment after auditing overlap with ISH-sourced iSign, ISLTranslate, OpenHands and fingerspelling material. Publication open access is not blanket permission for the original videos.

#### A8. Public DGS Corpus — German signing beyond weather

- Source: [Hamburg project](https://www.sign-lang.uni-hamburg.de/dgs-korpus/public-dgs-corpus.html).
- Public portals offer more than 50 h of DGS material with translations/annotations, including downloadable files. The full approximately 560 h parent corpus is not the same public release.
- Preparation: convert linguistic annotation time spans to stable translation examples, handle conversational participants/views explicitly, extract landmarks and create your own documented source/signer split unless a chosen published protocol supplies one.
- Access/terms: public portal access does not replace the release-specific reuse conditions. No ready PCAT adapter or official PCAT benchmark is bundled.
- Role/compute: later domain-generalization research; preparation is more involved than PHOENIX.

#### A9–A13. Larger translation/pretraining sources

| ID / resource | Existing material / scale | Integration, access and sensible role |
|---|---|---|
| A9 [OpenASL](https://github.com/chevalierNoir/OpenASL), [paper](https://arxiv.org/abs/2205.12870) | About 288 h ASL, over 200 signers, translations/metadata | ASL expansion; repository CC-BY-NC-ND-4.0. Acquire observations, audit surviving sources, convert annotations/landmarks and deduplicate external pretraining. No PCAT adapter |
| A10 [BOBSL v1.4](https://www.robots.ox.ac.uk/~vgg/data/bobsl/) | About 1.2M BSL subtitle sentences; 1,447 h excluding challenge; 331 GB listed MP4 / 1.5 TB frame LMDB | BBC agreement/eligibility restrictions. Interpret subtitles as potentially weak alignment and use task-specific aligned evaluation. Approved persistent compute, not the first Kaggle run |
| A11 [YouTube-ASL](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md) | 984 h, 610,193 captions in the release description | Later ASL pretraining; source retrieval/rights, temporal alignment, missing URLs and evaluation contamination need checks. No downloader/converter/objective-specific adapter supplied |
| A12 [YouTube-SL-25](https://github.com/google-research/google-research/tree/master/youtube_sl_25), [paper](https://arxiv.org/abs/2407.11144) | More than 3,000 h across more than 25 sign languages | Metadata access does not make all raw videos freely redistributable. Language-conditioned sampling/model design is needed; do not flatten different languages into an unlabeled mixture |
| A13 [CSL-News pose release](https://huggingface.co/datasets/ZechengLi19/CSL-News_pose) | Current card shows 722,711 rows and 256 GB repository files; pose pickle dictionaries contain `keypoints`, `scores`, `w_h`; written Chinese labels supplied | CC-BY-NC-4.0. Existing poses save detection work, but layout/timing/safe format conversion remains. Convert trusted-source object-bearing files in a separate preparation environment; runtime NPZ loader rejects pickles. Large-scale Chinese pretraining, not an ISL substitute |

Headline counts from papers and newer cards can differ. For CSL-News use the card's exact version/hash rather than mixing its counts with a third-party repack's split policy. No single download-size figure predicts prepared RGB volume or training memory.

#### A14. AfriSign — multilingual African signing

[Original research abstract/paper record](https://mlanthology.org/iclrw/2023/gueuwou2023iclrw-afrisign/), [OpenReview](https://openreview.net/forum?id=EHldk3J2xk). Bible-verse translations in six African sign languages provide a possible cross-language/domain research resource. This audit verified the research description, not unrestricted bulk archive download or hosted processing rights. Obtain the authors' current data instructions and terms first. New annotation/pose conversion and explicit language design are required; it is not a priority for the first ISL study.

### B. External evaluation and synthetic robustness

| Resource | Available resource / role | Important boundary |
|---|---|---|
| B1 [FLEURS-ASL](https://aclanthology.org/2025.naacl-long.314/) | ASL translated by five Certified Deaf Interpreters; multilingual sentence/discourse evaluation | Reserve held-out evaluation; convert observations according to exact task protocol. It is not a large supervised training corpus |
| B2 [SignNet-1M](https://github.com/openhe-hub/SignNet-1M) | 2026 generated viewpoint/background/identity variants; release table lists 350,530 How2Sign variants, 134 GiB | Full approximately 1M release is not all marked downloadable. Keep derivatives with original source splits; synthetic identities are not independent human signers. Generated views do not prove physical 3D invariance |
| B3 Included synthetic **pose order** | `pcat-demo-data`: reproducible nonlinguistic hand/face event-order codes | Fully generated offline, native NPZ/JSONL; ideal software check, never a real-language result |
| B4 Included synthetic **pixel patterns** | `pcat-demo-visual`: actual observed pixel inputs with artificial labels | Tests visual learning/input dependence only; not mouthings, gaze or spelling accuracy |

### C. Isolated recognition: optional, requires a separate experiment

Current sentence trainer has no isolated classifier/CTC/retrieval objective. These datasets cannot establish continuous SLT performance. A deliberately designed clip-to-label text diagnostic is possible after observation conversion, but its metric and claim remain lexical recognition; it is not an implemented standard recognition benchmark.

| Resource | Already available | Size / terms / preparation notes |
|---|---|---|
| C1 [CISLR](https://github.com/Exploration-Lab/CISLR) | ISL word-level clips and labels | Target-language lexical resource; current corpus importer rejects it for continuous translation. Check provider's noncommercial/research conditions; source/video overlap audit still needed |
| C2 [INCLUDE](https://huggingface.co/datasets/ai4bharat/INCLUDE), [raw-video record](https://zenodo.org/records/4010759) | ISL isolated videos, 263 sign labels; card has split metadata and download instructions | Card lists CC-BY-4.0. Existing OpenHands pose derivative is listed below. Dataset schema/class labels do not become sentence translations |
| C3 [WLASL](https://github.com/dxli94/WLASL/blob/master/README.md) | ASL lexical annotations/download tools; processed/missing clips can be requested | Source URLs can expire. Agree to the provider's applicable use terms; do not interpret available annotation IDs as all videos downloaded. OpenHands pose derivative exists |
| C4 [MS-ASL](https://www.microsoft.com/en-us/download/details.aspx?id=100121), [paper](https://arxiv.org/abs/1812.01053) | ASL recognition metadata with video acquisition information | The small Microsoft ZIP is metadata, not all raw videos. Respect its license and original content rights; verify actual available samples. OpenHands pose derivative exists |
| C5 [AUTSL](https://cvml.ankara.edu.tr/datasets/) | Turkish isolated RGB/depth/body skeleton | Original provider: 226 signs, 43 signers, 38,336 samples; challenge versions can differ. Skeleton has 25 body points, not full PCAT hand/face observations. Use the exact agreement/version and map missing modalities honestly |
| C6 [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/), [datasheet](https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/), [license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/) | Consented isolated ASL recordings | Datasheet: 83,399 videos, 2,731 signs, 52 participants. Microsoft noncommercial research license; not general sentence translation. Need pose extraction/conversion |
| C7 [LSA64](https://facundoq.github.io/datasets/lsa64/) | Argentinian isolated signs; raw/cut videos and preprocessed hand/head trajectories/crops | Provider lists approximately 1.9 GB raw, 1.5 GB cut, 800 MB preprocessed. Hand/head-only trajectories do not supply default torso anchors or face/hand joint structure. New model/layout policy required |
| C8 [ASLLVD](https://www.bu.edu/asllrp/av/dai-asllvd.html) | ASL lexicon videos with annotation/access instructions | Citation-form lexicon, not a sentence corpus. Inspect camera/file formats and BU reuse terms. OpenHands skeleton derivative exists |

#### C9. OpenHands labeled pose collection: useful existing preprocessing

[Authors' Zenodo record](https://zenodo.org/records/6674324) lists approximately 6.0 GB total, including **INCLUDE 662.9 MB**, **WLASL 1.0 GB**, **MS-ASL 1.3 GB**, **AUTSL 1.5 GB**, **ASLLVD skeleton 264.3 MB**, **GSL 725.8 MB**, and **LSA64 cut/full 215.0/226.6 MB**. These are archive labels, not PCAT-converted sizes. The small `RWTH-PHOENIX-Weather-Signer03-cutout` archive is not the full PHOENIX-2014T translation benchmark.

Useful because detection has already been done; still inspect layout, coordinates, scores, timing and upstream terms. A derivative record does not erase original conditions. GSL in this collection must be interpreted using that derivative's recognition metadata, not assumed to provide an arbitrary Greek translation split. No OpenHands loader is bundled here.

#### C10. OpenHands unlabeled Indian-SL pose collection

[Authors' record](https://zenodo.org/records/5558897), [pretraining documentation](https://github.com/AI4Bharat/OpenHands/blob/main/docs/instructions/self_supervised.rst): unlabeled pose material from ISH News, MBM Vadodara, NewzHook, NIOS and Sign Library; record lists **114.5 GB**. Documentation describes HDF5 keypoints/visibility for its pretraining pipeline.

There are no sentence targets supplied merely by having unlabeled poses. Current PCAT supervised trainer cannot consume it without a new self-supervised objective/data path. Audit source overlap before pretraining. Inspect derivative and upstream permissions before cloud copies. This is optional future work, not a faster substitute for iSign translation labels.

### D. Fingerspelling and entity accuracy

| Resource | Existing material | How to use responsibly |
|---|---|---|
| D1 [ISL-FS](https://kirandevraj.github.io/ISL-Fingerspelling/) | 1,308 ISL spelling segments from 499 ISH News videos, aligned text | Specialized spelling experiment; verify source-disjoint splits, displayed-text shortcuts and access terms |
| D2 [ISL-FS-Dense](https://kirandevraj.github.io/ISL-Fingerspelling/) | Same project's dense letter/frame labels and 11,700+ automatically discovered segments announced for 2026 | Distinguish manually verified base annotations from discovered segments; release availability/terms must be checked. Not independent of ISL-FS/ISH sources |
| D3 [FSboard](https://openaccess.thecvf.com/content/CVPR2025/html/Georg_FSboard_Over_3_Million_Characters_of_ASL_Fingerspelling_Collected_via_CVPR_2025_paper.html) | More than 3M ASL characters, more than 250 h, 147 signers | Smartphone fingerspelling; requires dedicated task protocol, observation conversion and character-error evaluation. Not ISL lexical supervision |
| D4 [ChicagoFSWild](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) | 7,304 ASL spelling sequences, 160 signers; 14 GB listed archive | Frames, processed spelling labels, partitions/signer metadata; hand-box annotations for a subset. Convert observations and preserve documented splits |
| D5 [ChicagoFSWild+](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) | 55,232 ASL spelling sequences, 260 signers; 82 GB listed archive | Larger related release; do not assume it is independent of the smaller corpus. Review release terms and original-video permissions |

Spelling can use an explicitly evaluated sequence-to-character decoder after conversion, but this repository does not supply the official specialized adapters, CTC head or CER reporting pipeline. Generic sentence BLEU is not adequate evidence of fingerspelling success.

### E. Geometry/gesture resources: auxiliary research only

| Resource | Available supervision | Role / boundary |
|---|---|---|
| E1 [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/), [official repository/license](https://github.com/facebookresearch/InterHand2.6M/blob/main/README.md) | Interacting-hand pose images/3D annotations; CC-BY-NC-4.0 | Separate hand-tracker/geometry pretraining. No sign-language sentence targets or full torso/face observations; cannot be passed directly to the SLT trainer |
| E2 [HaGRID / HaGRIDv2](https://github.com/hukenovs/hagrid/blob/master/README.md) | Gesture images/boxes and related recognition resources | Optional detector/gesture work, not sign-language translation. Provider specifies a variant of CC-BY-SA-4.0; read the actual license. No corresponding auxiliary objective is bundled |

## 3. Model memory and starting configurations

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

For larger runs, **one A100 40 GB is a reasonable proposed capacity target; 80 GB gives more room**, with 64–128 GB host RAM and storage sized from actual prepared inputs, potentially hundreds of GB or TB. These are provisioning targets, not measured minimums. [NVIDIA's specification](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf) confirms the 40/80 GB variants. Check that an allocation is a full GPU rather than a small MIG slice. This single-device trainer gains nothing automatically from paying for eight GPUs.

## 4. Storage budget

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

## 5. Training time and cost

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

### Measuring your run

Use the timing cell in [the Kaggle notebook](notebooks/Kaggle_PCAT_SLT.ipynb), following [the walkthrough](docs/KAGGLE_RUNBOOK.md). It records a complete train/validation/checkpoint call and CUDA peak allocation/reservation. Measure a resumed call and generation separately; initial download/hashing/load time is not steady-state training throughput.

### Whole-study compute

Five matched variants × three seeds = **15 training runs**, before pilots. Under the illustrative PHOENIX 1 s/batch case, this is about **591 training h**, or **739 h** with the assumed 25% overhead: about **25 quota weeks** at 30 h/week. Preprocessing, failed runs, human review and writing add calendar time. Kaggle is useful for iteration but inconvenient for a broad study.

For paid capacity, obtain a quote for the exact GPU/host RAM/storage and permitted data use. Calculate `instance hours × current instance rate + storage + transfer`. An instance with two paid GPUs can charge for both even if this trainer uses one. No price quote or spending authorization is implied. Compare a representative 1–2 h pilot before renting for long runs.

### Alternatives to Kaggle and example paid costs

An approved university/research server is the first choice when its data-access agreement covers the corpus. For permitted hosted processing, compare a single-GPU [Runpod Pod](https://www.runpod.io/pricing) with [Lambda instances](https://lambda.com/instances), checking capacity, persistent storage, interruptions and the actual invoice rate. Colab can also run CUDA code, but [Google's FAQ](https://research.google.com/colaboratory/faq.html) states that resources/limits vary; do not buy a notebook subscription assuming a guaranteed particular GPU or uninterrupted long job.

The Runpod public Pods page, checked 14 November 2026 and labeled updated 12 November, displayed **A100 PCIe 80 GB at $2.09/h**, **H100 PCIe 80 GB at $2.89/h**, and **L40S 48 GB at $1.09/h**. These are page snapshots, not reserved availability or a checkout quote; pool, region and selected offering can change the rate. Verify the deployment screen and storage charges before committing. [Source: Runpod pricing](https://www.runpod.io/pricing).

If a run actually measures **100 billed hours on that A100 offering**, compute alone at the displayed rate would be **$209**; 500 h would be **$1,045**, before storage/other charges. These examples do not convert the T4 timing scenarios into A100 runtimes. L40S's larger capacity is useful, but its lower listed hourly price does not prove a lower cost per completed experiment. Measure throughput on the exact allocated device.
