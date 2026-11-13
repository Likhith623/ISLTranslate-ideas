# Data contract

The model consumes landmarks and optional actual RGB observations to written text. It does not extract pose from video. `depth_policy: use` requires a shared Euclidean 3D frame; `depth_policy: ignore` explicitly uses consistent image-plane x/y coordinates and discards depth. All observations must share a common time base. Visual input and corpus adapters are specified in [REAL_DATA.md](REAL_DATA.md). Do not combine local MediaPipe hand coordinates, normalized face coordinates and unrelated torso depths as though they were calibrated geometry.

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

## V0.3 modality and checkpoint additions

Visual mode requires an `rgb` archive path on every row, with explicit observation masks. RGB and pose clip origins and relative timestamps must match; constant offsets are rejected. RGB file contents participate in training/validation fingerprints, and duplicate RGB paths/content are checked across splits. With opt-in unanchored RGB fallback, canonical geometry/confidence is marked absent when no reference exists, and observed pixels are required. Source snapshots, active pretrained weights and tokenizer files are included in format-3 checkpoints, so checkpoint inference/resume does not redownload original models.
