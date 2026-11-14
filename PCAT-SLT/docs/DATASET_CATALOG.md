# Dataset catalog for PCAT-SLT

Checked **14 November 2026**. This catalog covers the practical translation, pose-pretraining, isolated-sign, fingerspelling and geometry resources considered for this project. It is a curated catalog, not a claim to enumerate every sign-language dataset worldwide. It contains source links and integration requirements, not redistributed third-party datasets. An accessible page does not establish that all archive files can be downloaded anonymously or that cloud upload is allowed.

Read the [training/budget guide](../TRAINING_AND_DATASETS.md) and [Kaggle walkthrough](KAGGLE_RUNBOOK.md) alongside this catalog. Counts and sizes below are provider/paper figures, not the number of files verified locally. Downloaded release IDs, hashes, surviving URLs and actual usable counts must be recorded for your experiment.

## Choose by the question you are testing

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

## A. Continuous translation and linguistic corpora

### A1. iSign — ISL → English, first choice for the target language

- Source: [authors' project](https://exploration-lab.github.io/iSign/), [current release/card](https://huggingface.co/datasets/Exploration-Lab/iSign/blob/main/README.md), [paper](https://arxiv.org/html/2407.05404v1).
- Original publication: 118,228 video-sentence/phrase pairs, approximately 252 hours. Do not apply that number blindly to another release.
- Existing inputs: v1.1 split video archives, four pose archive parts, `.pose` format and `iSign_v1.1.csv`. Keep only the modalities needed for your first pilot.
- Preparation: read `.pose` headers/components, select joints explicitly, map coordinates/confidence and preserve frame timing; export NPZ and stable source-grouped manifests. The repository does not yet include that converter. Generic `uid,text` annotation import can reuse the `isltranslate` schema after normalization.
- Splits/terms: group by source `video_id`; the card has research/noncommercial and further-upload restrictions. Obtain the intended hosted-processing permission before placing it on Kaggle. See the master guide for details.
- Compute: small pose pilot first; full visual data needs a storage survey and actual GPU measurements.

### A2. ISLTranslate — legacy ISL → English translation

- Source: [original repository](https://github.com/Exploration-Lab/ISLTranslate), [2023 paper](https://aclanthology.org/2023.findings-acl.665/).
- About 31k published sentence/phrase pairs. Repository describes MediaPipe holistic features and a `uid,text` CSV; its current dataset link points to iSign.
- Preparation: identify the actual archive/version, convert its landmark convention, and supply an explicit split map to `pcat-import-corpus --dataset isltranslate`.
- Audit: source IDs/intervals may overlap other ISL corpora. Do not sum headline counts or use it as independent test data without checking. Verify the terms of the archive actually supplied.
- Compute: appropriate for a smaller ISL experiment after conversion; no measured training ETA is available.

### A3. PHOENIX-2014T — DGS → German

- Source: [RWTH release](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/), [archive index](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/).
- Standard translation split: 7,096 train / 519 dev / 642 test; see [original experiment table](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf). Provider lists the T archive at 39 GB.
- Existing inputs: signing frames, gloss and written-language annotations. Directory name `features/fullFrame-210x260px` does not mean PCAT-compatible pose features.
- Preparation: extract/obtain mapped landmarks; optionally create aligned crops; import official pipe-delimited `name, translation` records using the implemented adapter, with separate train/dev/test invocations.
- Terms/splits: retain official assignments and provider attribution/terms; verify hosted-use rights. Narrow weather-domain results do not establish ISL conversational performance.
- Compute: recommended first complete pose benchmark on Kaggle, after a measured pilot.

### A4. CSL-Daily — Chinese signing → Chinese

- Source: [official access page](https://ustc-slr.github.io/datasets/2021_csl_daily/), [original paper/statistics](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf).
- 18,401 train / 1,077 dev / 1,176 test; gloss and written Chinese annotations.
- Access: research institutions; release agreement requires a full-time staff signature. Check archive contents and hosted processing against your approved agreement.
- Preparation: mapped observations + the implemented `csldaily` annotation adapter. Use a suitable Chinese-capable text model or declared byte baseline. Score exported text with the benchmark's Chinese normalization; built-in whitespace ROUGE/13a BLEU is not an automatic equivalent.
- Compute: second-stage benchmark; pose pilot on T4, larger-memory GPU for expensive visual variants if profiling requires it.

### A5. How2Sign — ASL → English with existing keypoints

- Source: [provider downloads](https://how2sign.github.io/), [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.pdf).
- Original green-screen split: 31,128 / 1,741 / 2,322 clips, 79.12 h; separate approximately 3 h Panoptic material.
- Existing modalities: videos, body/face/hand 2D keypoints and English annotations. Frontal clip downloads total about 34.9 GB; listed keypoint archives total about 23.8 GB. Full recordings are much larger.
- Preparation: map OpenPose-style keypoints and validity to the project layout, convert targets to generic JSONL. Choose original versus manually realigned segmentation consistently; different segmentation can change clip counts.
- Terms: provider lists CC-BY-NC-4.0 and research-only use. Standard splits contain signer overlap; do not label the whole benchmark signer-independent.
- Compute: a useful pose-first ASL option. RGB expansion and repeated NPZ decode need measurement before full visual training.

### A6. WSLP 2025/2026 translation shared tasks — ISL evaluation protocol

- Sources: [2025 task page](https://exploration-lab.github.io/WSLP/task/), [2026 findings](https://aclanthology.org/2026.wslp-1.2/), [2026 translation report](https://aclanthology.org/2026.wslp-1.13/).
- Choose the **SLT** track. Recognition and word-presence are different datasets/objectives; their split sizes are not translation counts.
- Obtain the actual release, permitted training resources and scorer from the organizers. Availability of test labels, submissions and task downloads can differ after a challenge ends.
- Preparation: convert supplied pose/video observations and exact task targets; retain disjoint source-video evaluation. No dedicated WSLP adapter exists.
- Role: held-out ISL evaluation rather than another corpus to concatenate indiscriminately. The 2026 report discusses fluency versus grounding failures, so include input-dependence controls.

### A7. ISH-NEWS (2026) — smaller ISL news-domain study

- Source: [original paper and data-availability section](https://www.nature.com/articles/s41598-026-60893-0).
- Reports 4,222 sentence clips from 162 source recordings. Annotations, splits, timestamps and preprocessing are linked; raw videos are not redistributed for copyright reasons.
- Preparation: obtain permissible source recordings, reconstruct exact intervals, map landmarks and targets. No dedicated adapter exists.
- Role: optional extra-domain experiment after auditing overlap with ISH-sourced iSign, ISLTranslate, OpenHands and fingerspelling material. Publication open access is not blanket permission for the original videos.

### A8. Public DGS Corpus — German signing beyond weather

- Source: [Hamburg project](https://www.sign-lang.uni-hamburg.de/dgs-korpus/public-dgs-corpus.html).
- Public portals offer more than 50 h of DGS material with translations/annotations, including downloadable files. The full approximately 560 h parent corpus is not the same public release.
- Preparation: convert linguistic annotation time spans to stable translation examples, handle conversational participants/views explicitly, extract landmarks and create your own documented source/signer split unless a chosen published protocol supplies one.
- Access/terms: public portal access does not replace the release-specific reuse conditions. No ready PCAT adapter or official PCAT benchmark is bundled.
- Role/compute: later domain-generalization research; preparation is more involved than PHOENIX.

### A9–A13. Larger translation/pretraining sources

| ID / resource | Existing material / scale | Integration, access and sensible role |
|---|---|---|
| A9 [OpenASL](https://github.com/chevalierNoir/OpenASL), [paper](https://arxiv.org/abs/2205.12870) | About 288 h ASL, over 200 signers, translations/metadata | ASL expansion; repository CC-BY-NC-ND-4.0. Acquire observations, audit surviving sources, convert annotations/landmarks and deduplicate external pretraining. No PCAT adapter |
| A10 [BOBSL v1.4](https://www.robots.ox.ac.uk/~vgg/data/bobsl/) | About 1.2M BSL subtitle sentences; 1,447 h excluding challenge; 331 GB listed MP4 / 1.5 TB frame LMDB | BBC agreement/eligibility restrictions. Interpret subtitles as potentially weak alignment and use task-specific aligned evaluation. Approved persistent compute, not the first Kaggle run |
| A11 [YouTube-ASL](https://github.com/google-research/google-research/blob/master/youtube_asl/README.md) | 984 h, 610,193 captions in the release description | Later ASL pretraining; source retrieval/rights, temporal alignment, missing URLs and evaluation contamination need checks. No downloader/converter/objective-specific adapter supplied |
| A12 [YouTube-SL-25](https://github.com/google-research/google-research/tree/master/youtube_sl_25), [paper](https://arxiv.org/abs/2407.11144) | More than 3,000 h across more than 25 sign languages | Metadata access does not make all raw videos freely redistributable. Language-conditioned sampling/model design is needed; do not flatten different languages into an unlabeled mixture |
| A13 [CSL-News pose release](https://huggingface.co/datasets/ZechengLi19/CSL-News_pose) | Current card shows 722,711 rows and 256 GB repository files; pose pickle dictionaries contain `keypoints`, `scores`, `w_h`; written Chinese labels supplied | CC-BY-NC-4.0. Existing poses save detection work, but layout/timing/safe format conversion remains. Convert trusted-source object-bearing files in a separate preparation environment; runtime NPZ loader rejects pickles. Large-scale Chinese pretraining, not an ISL substitute |

Headline counts from papers and newer cards can differ. For CSL-News use the card's exact version/hash rather than mixing its counts with a third-party repack's split policy. No single download-size figure predicts prepared RGB volume or training memory.

### A14. AfriSign — multilingual African signing

[Original research abstract/paper record](https://mlanthology.org/iclrw/2023/gueuwou2023iclrw-afrisign/), [OpenReview](https://openreview.net/forum?id=EHldk3J2xk). Bible-verse translations in six African sign languages provide a possible cross-language/domain research resource. This audit verified the research description, not unrestricted bulk archive download or hosted processing rights. Obtain the authors' current data instructions and terms first. New annotation/pose conversion and explicit language design are required; it is not a priority for the first ISL study.

## B. External evaluation and synthetic robustness

| Resource | Available resource / role | Important boundary |
|---|---|---|
| B1 [FLEURS-ASL](https://aclanthology.org/2025.naacl-long.314/) | ASL translated by five Certified Deaf Interpreters; multilingual sentence/discourse evaluation | Reserve held-out evaluation; convert observations according to exact task protocol. It is not a large supervised training corpus |
| B2 [SignNet-1M](https://github.com/openhe-hub/SignNet-1M) | 2026 generated viewpoint/background/identity variants; release table lists 350,530 How2Sign variants, 134 GiB | Full approximately 1M release is not all marked downloadable. Keep derivatives with original source splits; synthetic identities are not independent human signers. Generated views do not prove physical 3D invariance |
| B3 Included synthetic **pose order** | `pcat-demo-data`: reproducible nonlinguistic hand/face event-order codes | Fully generated offline, native NPZ/JSONL; ideal software check, never a real-language result |
| B4 Included synthetic **pixel patterns** | `pcat-demo-visual`: actual observed pixel inputs with artificial labels | Tests visual learning/input dependence only; not mouthings, gaze or spelling accuracy |

## C. Isolated recognition: optional, requires a separate experiment

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

### C9. OpenHands labeled pose collection: useful existing preprocessing

[Authors' Zenodo record](https://zenodo.org/records/6674324) lists approximately 6.0 GB total, including **INCLUDE 662.9 MB**, **WLASL 1.0 GB**, **MS-ASL 1.3 GB**, **AUTSL 1.5 GB**, **ASLLVD skeleton 264.3 MB**, **GSL 725.8 MB**, and **LSA64 cut/full 215.0/226.6 MB**. These are archive labels, not PCAT-converted sizes. The small `RWTH-PHOENIX-Weather-Signer03-cutout` archive is not the full PHOENIX-2014T translation benchmark.

Useful because detection has already been done; still inspect layout, coordinates, scores, timing and upstream terms. A derivative record does not erase original conditions. GSL in this collection must be interpreted using that derivative's recognition metadata, not assumed to provide an arbitrary Greek translation split. No OpenHands loader is bundled here.

### C10. OpenHands unlabeled Indian-SL pose collection

[Authors' record](https://zenodo.org/records/5558897), [pretraining documentation](https://github.com/AI4Bharat/OpenHands/blob/main/docs/instructions/self_supervised.rst): unlabeled pose material from ISH News, MBM Vadodara, NewzHook, NIOS and Sign Library; record lists **114.5 GB**. Documentation describes HDF5 keypoints/visibility for its pretraining pipeline.

There are no sentence targets supplied merely by having unlabeled poses. Current PCAT supervised trainer cannot consume it without a new self-supervised objective/data path. Audit source overlap before pretraining. Inspect derivative and upstream permissions before cloud copies. This is optional future work, not a faster substitute for iSign translation labels.

## D. Fingerspelling and entity accuracy

| Resource | Existing material | How to use responsibly |
|---|---|---|
| D1 [ISL-FS](https://kirandevraj.github.io/ISL-Fingerspelling/) | 1,308 ISL spelling segments from 499 ISH News videos, aligned text | Specialized spelling experiment; verify source-disjoint splits, displayed-text shortcuts and access terms |
| D2 [ISL-FS-Dense](https://kirandevraj.github.io/ISL-Fingerspelling/) | Same project's dense letter/frame labels and 11,700+ automatically discovered segments announced for 2026 | Distinguish manually verified base annotations from discovered segments; release availability/terms must be checked. Not independent of ISL-FS/ISH sources |
| D3 [FSboard](https://openaccess.thecvf.com/content/CVPR2025/html/Georg_FSboard_Over_3_Million_Characters_of_ASL_Fingerspelling_Collected_via_CVPR_2025_paper.html) | More than 3M ASL characters, more than 250 h, 147 signers | Smartphone fingerspelling; requires dedicated task protocol, observation conversion and character-error evaluation. Not ISL lexical supervision |
| D4 [ChicagoFSWild](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) | 7,304 ASL spelling sequences, 160 signers; 14 GB listed archive | Frames, processed spelling labels, partitions/signer metadata; hand-box annotations for a subset. Convert observations and preserve documented splits |
| D5 [ChicagoFSWild+](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) | 55,232 ASL spelling sequences, 260 signers; 82 GB listed archive | Larger related release; do not assume it is independent of the smaller corpus. Review release terms and original-video permissions |

Spelling can use an explicitly evaluated sequence-to-character decoder after conversion, but this repository does not supply the official specialized adapters, CTC head or CER reporting pipeline. Generic sentence BLEU is not adequate evidence of fingerspelling success.

## E. Geometry/gesture resources: auxiliary research only

| Resource | Available supervision | Role / boundary |
|---|---|---|
| E1 [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/), [official repository/license](https://github.com/facebookresearch/InterHand2.6M/blob/main/README.md) | Interacting-hand pose images/3D annotations; CC-BY-NC-4.0 | Separate hand-tracker/geometry pretraining. No sign-language sentence targets or full torso/face observations; cannot be passed directly to the SLT trainer |
| E2 [HaGRID / HaGRIDv2](https://github.com/hukenovs/hagrid/blob/master/README.md) | Gesture images/boxes and related recognition resources | Optional detector/gesture work, not sign-language translation. Provider specifies a variant of CC-BY-SA-4.0; read the actual license. No corresponding auxiliary objective is bundled |

## F. Exact compatibility and acquisition checklist

For **every** selected release, record:

1. Provider URL, version/date, access agreement, actual downloaded hashes/bytes and any approval covering hosted processing. A public Kaggle mirror is not evidence of provider authorization.
2. Task identity: sentence translation, isolated label, spelling, unlabeled pose or geometry. Use written targets for SLT, never silently replace them with gloss/class names.
3. Source video/episode, interval, signer, session, camera and augmentation lineage. Prevent cross-split source leakage before normalization/extraction; content hashes alone miss near duplicates.
4. Joint definitions, handedness, image mirroring, coordinate units/origin, confidence meaning, frame rate and timestamps. Read file headers rather than guessing indices from tensor dimensions.
5. Conversion to `pose[T,J,3]`, `confidence[T,J]`, `timestamps[T]` and JSONL `id,pose,text`; for visual mode also aligned `rgb` and `rgb_mask` arrays. Use [DATA.md](DATA.md) and [REAL_DATA.md](REAL_DATA.md).
6. Whether the full body anchors, hands and face are actually observed. Do not fabricate missing joints with positive confidence. If the skeleton lacks required anchors, choose a defensible compatible layout/observation mode instead of claiming a complete conversion.
7. Train/dev/test usable counts, excluded IDs/reasons, token/frame percentiles, missing-observation distributions and sample visual inspection. Respect strict no-silent-truncation behavior.
8. Actual compressed prepared-array size, per-epoch read/decode time, optimizer/checkpoint size, longest-clip memory and resume peak. Dataset hours alone cannot establish training ETA.

**Existing pose archives remove some extraction work; they do not remove conversion, split auditing or validation.** Use a small source-disjoint subset for preparation checks before downloading/converting every optional resource.
