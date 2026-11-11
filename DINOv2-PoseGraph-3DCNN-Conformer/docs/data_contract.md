# ISL-MEGA data contract

## 1. NPZ record schema

One NPZ stores one sentence/phrase clip or one explicitly defined self-supervised clip.

| Key | Required | Shape | Meaning |
|---|---|---|---|
| `frames` | Yes | `[Tv,H,W,3]` or `[Tv,3,H,W]` | RGB frames, `uint8` or float `[0,1]` |
| `video_timestamps` | Yes | `[Tv]` | Seconds in a shared time base; strictly increasing |
| `pose_xy` | Yes | `[Tp,66,2]` | Normalized original-frame `(x,y)` landmark positions |
| `pose_timestamps` | Yes | `[Tp]` | Seconds, same time base as video timestamps |
| `pose_valid` | No | `[Tp,66]` | Whether each landmark is a valid observation; defaults to finite coordinates |
| `pose_confidence` | No | `[Tp,66]` | Detector confidence in `[0,1]`; omitted means confidence is unknown |
| `boxes` | No | `[Tv,4,4]` | Normalized original-frame `xyxy` ROIs in fixed region order |
| `box_valid` | No | `[Tv,4]` | Per-ROI validity |
| `box_scores` | No | `[Tv,4]` | Detector score or explicitly uncalibrated availability proxy |

Do not include Python objects/pickled arrays. The loader uses `allow_pickle=False`. Integer frames must be `uint8`; floating-point frames must be finite and within `[0,1]`. Unsupported integer types such as `uint16` are rejected instead of silently rescaled.

## 2. Coordinate convention

All pose points and boxes are expressed relative to the original decoded frame before spatial resizing:

- Origin: top-left.
- `x`: increases to the right.
- `y`: increases downward.
- Coordinates/box edges are normalized to `[0,1]` using frame width/height.
- `boxes[..., :]` order is `[x_min, y_min, x_max, y_max]`.

The loader applies aspect-preserving letterboxing to RGB frames and applies the identical affine transform to pose points and boxes. Do not pre-letterbox coordinates and then let the loader letterbox a second time.

## 3. Canonical 66-landmark order

The model requires this exact concatenated order:

### 3.1 Left hand: indices 0–20

MediaPipe Hands 21-point order, expressed in the **signer's anatomical left hand**, not screen-left. Standard sequence: wrist; thumb CMC/MCP/IP/tip; index MCP/PIP/DIP/tip; middle MCP/PIP/DIP/tip; ring MCP/PIP/DIP/tip; pinky MCP/PIP/DIP/tip. The graph connects wrist-to-finger chains.

### 3.2 Right hand: indices 21–41

The same MediaPipe Hands order for the signer's anatomical right hand.

### 3.3 Upper body: indices 42–47

Exactly:

1. Left shoulder.
2. Right shoulder.
3. Left elbow.
4. Right elbow.
5. Left wrist.
6. Right wrist.

The graph edges are shoulder-left ↔ shoulder-right, left shoulder ↔ left elbow ↔ left wrist, and right shoulder ↔ right elbow ↔ right wrist.

### 3.4 Sparse face: indices 48–65

The model treats these 18 ordered points as a fully connected sparse face graph. The adapter must map detector landmarks to this stable order:

1. Left eyebrow outer.
2. Left eyebrow inner.
3. Right eyebrow inner.
4. Right eyebrow outer.
5. Left eye outer canthus.
6. Left eye inner canthus.
7. Right eye inner canthus.
8. Right eye outer canthus.
9. Nose bridge.
10. Nose tip.
11. Glabella.
12. Left cheek.
13. Right cheek.
14. Left mouth corner.
15. Upper-lip centre.
16. Right mouth corner.
17. Lower-lip centre.
18. Chin.

This is an ISL-MEGA adapter contract, not a claim that all face detectors expose those exact points. Build and verify an explicit detector-to-contract mapping. Do not assume MediaPipe Face Mesh's first 18 points have this semantic order.

## 4. Timestamps and alignment

`video_timestamps` and `pose_timestamps` must share a common clock/time origin and be finite, nonempty and strictly increasing. If one clock is wall time/source PTS and the other is clip-relative time, convert them to the same reference before export. The supplied video/pose converter preserves the video's PTS and offers `--pose-time-offset-seconds` when pose timestamps require an explicit offset.

For each video timestamp:

- An exactly matching pose timestamp is copied.
- Otherwise, the loader linearly interpolates between the adjacent pose samples only when their time gap is no larger than `pose.max_interpolation_gap_seconds`.
- A landmark is interpolated only when it is valid at both endpoints.
- There is no extrapolation beyond the first/last pose timestamp.
- Large gaps result in invalid observations, not invented points.

Video frames are not temporally downsampled. The loader only spatially resizes/letterboxes them.

## 5. ROI order

If external detector boxes are supplied, their fixed axis-1 order is:

0. Face.
1. Left hand.
2. Right hand.
3. Upper-body/global signer view.

`box_scores` should be calibrated detector confidence if available. If unavailable, the loader uses a presence proxy (and uses 0.25 for the derived full-frame fallback body ROI). Never report these proxies as detector confidence.

If boxes are absent, face/hand boxes are derived conservatively from visible keypoints; body is derived from hands and upper-body points. When all those points are missing, the body stream receives a full-frame fallback marked with a low quality proxy. That fallback is a global video view, not a successful body detection.

## 6. Manifest JSONL

Every line is one JSON object. Relative NPZ paths are resolved relative to the manifest file's directory.

Supervised example:

```json
{"uid":"clip-0001","npz":"npz/clip-0001.npz","translation":"The student is writing in a book.","source_id":"original-video-014","signer_id":"signer-03","fingerspelling_text":""}
```

Self-supervised example (no `translation` required):

```json
{"uid":"unlabeled-0001","npz":"npz/unlabeled-0001.npz","source_id":"original-video-014","signer_id":"signer-03"}
```

`uid` must be a nonempty string and globally unique across all split manifests. The split checker rejects repeated UIDs across train/validation/test. `source_id` must identify the original source video or parent recording, not a derived sentence fragment. `signer_id` should be a stable pseudonymous ID where permitted. Do not infer or expose real signer identities in published manifests.

`translation` is required for supervised training and evaluation. The loader allows it to be absent only when `require_translation=False` (the self-supervised pretraining path).

## 7. Fingerspelling labels

`fingerspelling_text` is an optional complete-clip character string. Current codec characters are lowercase English a–z, digits 0–9, hyphen, apostrophe and space. Unsupported characters raise an error rather than being silently dropped. The current model does not support timestamped local spans within a longer sentence; create a properly annotated segment record if using this objective.

## 8. Data rights and quality

The code validates file structure, numeric validity, basic coverage and split group metadata. It does not validate that the translation is correct, that the pose extractor mapped anatomy correctly, that the signer metadata is reliable, or that all videos are licensed for research. Those require independent provenance, annotation and ethics checks.
