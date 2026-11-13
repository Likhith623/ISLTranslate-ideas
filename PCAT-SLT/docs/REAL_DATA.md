# Real-corpus integration and observation preparation

The importers preserve annotation targets and split assignments. They do not redistribute licensed videos, reconstruct unknown geometry or invent missing features.

## Observation files

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
