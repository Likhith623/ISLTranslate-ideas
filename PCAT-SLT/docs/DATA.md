# Data preparation and input format

The model consumes landmarks and optional actual RGB observations to written text. It does not extract pose from video. `depth_policy: use` requires a shared Euclidean 3D frame; `depth_policy: ignore` explicitly uses consistent image-plane x/y coordinates and discards depth. All observations must share a common time base. This file specifies observation arrays, crop preparation and corpus annotation import. Do not combine local MediaPipe hand coordinates, normalized face coordinates and unrelated torso depths as though they were calibrated geometry.

## Manifest

One JSON object per UTF-8 line:

```json
{"id":"clip-001","pose":"poses/clip-001.npz","rgb":"rgb/clip-001.npz","text":"The target sentence.","signer":"signer-07","source_video":"recording-04","sign_language":"ISL","split_group":"session-04"}
```

`id`, `pose` and `text` are required. Paths are relative to the manifest, or absolute. Extra provenance fields are preserved in input fingerprints. `signer` is required when running validation with `--signer-independent`. IDs and resolved pose paths must be unique across splits. Do not construct train/test splits by randomly cutting clips from the same source recording. Dataset license, permission, subject consent and split provenance are the dataset curator's responsibility; the archive redistributes no third-party corpus.

Manifest paths in YAML resolve from the command's working directory. Use the project root for the documented commands. NPZ paths resolve from the manifest location. Target strings are preserved exactly: no silent lowercasing, normalization, truncation, tokenization or translation is applied. Byte-mode targets longer than `max_bytes` are rejected; adjust both that limit and `max_target_length`. A byte target of U bytes requires U+1 decoder positions. Pretrained mode instead uses the matching tokenizer and checks the resulting decoder-token length. No target is silently truncated.

## NPZ arrays

| Key | Shape | Contract |
|---|---|---|
| `pose` | `[T,J,3]` | Floating coordinates in a single Euclidean frame; observed coordinates must be finite |
| `confidence` | `[T,J]` | Finite values in `[0,1]`; zero means missing |
| `timestamps` | `[T]` | Strictly increasing finite times in seconds; loader subtracts the first timestamp |

At least two frames are required. Excess frames cause an explicit error rather than temporal subsampling. Archives are opened with `allow_pickle=False`. Poses and confidence are cast to FP32; timestamps are first processed in FP64 then represented as relative FP32 seconds. Timestamp intervals that collapse at FP32 precision are rejected.

The loader interpolates missing coordinates in physical time and uses nearest observed values beyond the observed range; confidence remains zero. An entirely missing joint is placed at the selected reference's shoulder midpoint and marked zero confidence. `anchor_policy: first_reliable` searches valid frames for a reliable nondegenerate triplet; `anchor_policy: first` requires the first frame. The same rule is used by the loader and model, and no padded frame is eligible. Observed NaNs are rejected; unobserved NaNs are handled by imputation. Interpolation is a pragmatic missing-data policy, not a reconstruction guarantee. Train and evaluate pose-missingness stress tests separately.

## Default 66-joint layout

This is a **project-specific interchange layout**, not an asserted MediaPipe or OpenPose ordering:

| Indices (inclusive) | Identity |
|---|---|
| 0 | Anatomical left shoulder |
| 1 | Anatomical right shoulder |
| 2 | Neck/upper torso point not collinear with the shoulders |
| 3–23 | Left hand, 21 joints in a consistent order |
| 24–44 | Right hand, 21 joints in the same hand convention |
| 45–62 | Face/mouth, 18 consistently selected joints |
| 63–65 | Three additional torso joints |

Four regions are left hand, right hand, face and torso `[0,1,2,63,64,65]`. Set explicit `model.joints`, `model.anchors` and `model.regions` in YAML for an existing skeleton layout. Each region uses an affine map sized to its own number of joints. Regional ordering is significant. Calibrated joint extraction and a dataset-specific skeleton mapping are external data preparation tasks, not missing implementations of a claimed raw-video input path.

## Validation and leakage checks

```bash
pcat-validate --config configs/base.yaml --signer-independent
```

This reads every record, validates shapes/timestamps/confidence, checks body geometry, rejects ID/path duplication and exact duplicates of decoded arrays, and enforces signer separation when requested. When `source_video` or `split_group` is supplied, cross-split reuse of those exact group IDs is rejected. Missing provenance and near duplicates still require curation. Training independently checks manifest overlap and fingerprints every referenced training/validation NPZ, rejecting resume when those bytes or manifest semantics change. Hashing requires an initial full read of source files; it is intentional reproducibility overhead.

## Collation and padding

`collate_signs` pads time and target axes separately and returns pose, confidence, timestamps, source lengths, decoder input, shifted labels, IDs and reference texts. Labels use PAD=0. All padded source vertices are removed from the clock before integration; no padded transition can contribute area. The decoder uses causal/target masks and, for enhanced detail memory, a source padding mask. `encode_with_mask` returns that memory and its mask. Every admitted original frame is retained in detail mode. The coarse operator returns the sum of the configured progress-cell counts, including for stationary sequences; static shape is preserved through midpoint features, so there is no all-masked source sequence.

## Synthetic data

`pcat-demo-data` generates geometric hand/face event order with text codes `hf` and `fh`. These are not signs, words, glosses, a sign-language corpus or natural-language translation. Distinct output folders are required to avoid overwriting existing data. The generator supplies proper rotations/scales/translations, variable durations and frame counts for exercising the full pipeline.

## RGB observations

For each clip, prepare the documented pose NPZ (`pose`, `confidence`, `timestamps`) and, in visual mode, an RGB NPZ:

| Key | Shape / contract |
|---|---|
| `rgb` | `[T,R,3,H,W]`, float RGB in [0,1], H/W at least 8 and at least one DINO patch when using DINOv2 |
| `rgb_mask` | `[T,R]`, boolean observation validity |
| `timestamps` | `[T]`, seconds in the **same clip origin** as pose timestamps |

The loader checks the original first timestamp as well as every relative sample time. Rebasing away a constant offset is not accepted. Pose and RGB must have exactly matching frame counts; cropping is not a license to guess alignment. Masked/padded pixel values are removed before encoding. Valid nonfinite/out-of-range pixels are rejected.

`visual_to_articulator` maps crop regions to configured pose channels. The example order is left hand, right hand, mouth, eyes, routed `[0,1,2,2]`. Region semantics are part of the data contract. Actual gaze/mouthing recognition depends on the observations and learning, not that routing alone.

## Crop preprocessing

`prepare_rgb.py` consumes a frame folder and a box NPZ containing:

- `boxes`: `[T,R,4]` normalized xyxy coordinates.
- `rgb_mask`: `[T,R]` boolean observations.
- `timestamps`: strictly increasing seconds.
- Optional `frame_names`: explicit ordered basenames; use this when alignment cannot safely be inferred from filenames.

Without `frame_names`, image files are sorted naturally by numbered filename. Every observed box must be finite, in [0,1] and have positive extent. Crops preserve aspect ratio through square padding and resizing. No 3D landmark is silently treated as an image coordinate. Missing boxes remain marked absent. DINOv2 normalization is applied once in its encoder; the stored RGB remains unnormalized [0,1].

```bash
python scripts/prepare_rgb.py --frames data/frames/clip --boxes data/boxes/clip.npz \
  --output data/rgb/clip.npz --size 112
```

The model does not bundle a detector or a 3D reconstruction system. Existing image-plane landmarks can be supplied with `depth_policy: ignore`; all x/y coordinates must share the original image frame. If no reliable body reference exists and `allow_unanchored_rgb` is enabled, geometry is marked absent and observed RGB is required. This fallback does not invent pose.

## PHOENIX-2014T

[Provider page](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and [original paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1910.pdf).

Use the official pipe-separated annotation records with `name` and written `translation`. The `orth`/gloss field must not be substituted for translation. For one official annotation file:

```bash
pcat-import-corpus --dataset phoenix2014t --annotations data/PHOENIX-2014-T.train.corpus.csv \
  --split train --pose-root data/phoenix_poses --rgb-root data/phoenix_rgb --output data/manifests_train
```

Run separate supplied annotation files into separate output folders with their explicit splits; reference those generated manifests in YAML. `dev` is normalized to `val`. `--pose-template '{split}/{id}.npz'` supports a preprocessed split directory structure. Other header variants require explicit `--id-column` and `--text-column`.

## ISLTranslate

The [official CSV](https://raw.githubusercontent.com/Exploration-Lab/ISLTranslate/main/data/ISLTranslate.csv) has `uid,text`. The [paper](https://aclanthology.org/2023.findings-acl.665/) describes approximately 31k pairs. The CSV is not a guarantee of signer/source-independent splits or that every caption exactly describes all visible signing.

```bash
pcat-import-corpus --dataset isltranslate --annotations data/ISLTranslate.csv \
  --split-map data/isl_splits.json --pose-root data/isl_poses --rgb-root data/isl_rgb --output data/real
```

The assignment map is JSON mapping each uid to `train`, `val`/`dev` or `test`. No random split is created. When uid follows `source-number`, the prefix is retained as a source-video group for leakage checks; verify that convention for your release. If a published assignment overlaps source groups, document and resolve the evaluation policy rather than silently relabeling an altered split as the official benchmark.

## CSL-Daily

[Official release/access instructions](https://ustc-slr.github.io/datasets/2021_csl_daily/). The provider's agreement controls access; raw annotations were not available in this workspace for corpus-level validation.

The adapter accepts primitive metadata records under `info`, or normalized JSON/JSONL/CSV with `name`, written `text` and `split`. A character list is joined into the written Chinese string, not into gloss labels. Primitive pickle metadata is read by an unpickler that rejects all global/object reconstruction; object-bearing variants must be exported to plain JSON in a separately trusted preparation environment.

```bash
pcat-import-corpus --dataset csldaily --annotations data/csl_plain_metadata.json \
  --pose-root data/csl_poses --rgb-root data/csl_rgb --output data/csl_manifests
```

Header variants can use `--id-column`, `--text-column`, `--split-column`, `--delimiter`, or an explicit split map. This is a declared adapter schema with contract tests, not a claim that every restricted release variant was tested.

## CISLR and missing inputs

[CISLR's authors](https://github.com/Exploration-Lab/CISLR) identify it as word-level recognition. It must not be passed off as continuous translation. The importer rejects `--dataset cislr` for this task. The synthetic datasets are also diagnostic tasks, not replacement SLT benchmarks.

Prepared pose/RGB files must exist before import. Import does not silently skip missing records. Every manifest is then checked by `pcat-validate`, which reads arrays, checks geometry/observations, rejects exact leakage and optionally enforces signer separation. Known source groups, missing pixels, unanchored fallback and file fingerprints should accompany any experiment report.

## Release and conversion records

Record provider/version/access terms, downloaded hashes, task/target type, source recordings/intervals/signers/views, and the exact coordinate/layout/timestamp conversion. Report usable/excluded IDs, frame/target length percentiles, missing observations and manual alignment checks. Do not convert isolated labels into alleged sentence targets or fabricate unobserved anchors with positive confidence. The [dataset catalog and budgets](../TRAINING_AND_DATASETS.md) describe acquisition options and resource planning.
