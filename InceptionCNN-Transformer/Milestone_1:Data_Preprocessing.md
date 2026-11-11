# Milestone 1: Data Preprocessing (MediaPipe Keypoint Extraction)

We are converting 3,544 raw ASL Citizen videos into lightweight, fixed-shape `.npy`
keypoint arrays ready to feed into our 1D-CNN + Transformer model.

## Dataset: ASL Citizen

| Property | Value |
|---|---|
| Source | ASL Citizen (Microsoft) |
| Total videos in top-100 subset | **3,544** |
| Unique words (classes) | **100** |
| Average videos per word | **35.4** |
| Missing / dead videos | **0** (all videos confirmed on disk) |
| Train / Val / Test split | **1,727 / 381 / 1,436** (preserves original signer splits) |

The original signer-based splits are preserved so that no signer's style appears in both
training and testing — this ensures **signer-independent evaluation**.

---

## Directory Layout

```
Sing_AR/
├── archive/
│   ├── splits/
│   │   ├── train.csv             ← original ASL Citizen splits (Participant ID, Video file, Gloss, ASL-LEX Code)
│   │   ├── val.csv
│   │   └── test.csv
│   ├── videos/                   ← all 83,399 .mp4 files (flat, named as referenced in CSVs)
│   └── asl_citizen_top100.json   ← ✅ ALREADY GENERATED — top-100 mapping with splits
│
└── preprocessing/
    ├── models/
    │   ├── face_landmarker.task
    │   ├── hand_landmarker.task
    │   └── pose_landmarker_full.task
    ├── requirements.txt
    ├── extract_keypoints.py      ← ⚠️ NEEDS REWRITE for ASL Citizen
    └── output/                   ← will be created by the script
        ├── train/{gloss}/{video_id}.npy
        ├── val/{gloss}/{video_id}.npy
        └── test/{gloss}/{video_id}.npy
```

---

## Keypoint Subset (85 total per frame)

Every frame produces a `(85, 3)` array of `[x, y, z]` coordinates:

| Landmark Group | Count | Indices |
|---|---|---|
| **Face — Eyes** | 4 | 33, 133, 263, 362 |
| **Face — Left Eyebrow** | 7 | 70, 63, 105, 66, 107, 55, 65 |
| **Face — Right Eyebrow** | 7 | 300, 293, 334, 296, 336, 285, 295 |
| **Face — Outer Upper Lip** | 11 | 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291 |
| **Face — Outer Lower Lip** | 9 | 146, 91, 181, 84, 17, 314, 405, 321, 375 |
| **Face — Inner Lip Center** | 1 | 14 |
| **Pose — Shoulders & Elbows** | 4 | 11 (L-shoulder), 12 (R-shoulder), 13 (L-elbow), 14 (R-elbow) |
| **Left Hand** | 21 | All 21 MediaPipe hand landmarks |
| **Right Hand** | 21 | All 21 MediaPipe hand landmarks |
| **Total** | **85** | |

---

## Per-Video Processing Pipeline

Each video goes through 5 steps before being saved.

> [!IMPORTANT]
> **The NaN Approach (critical fix):** When MediaPipe fails to detect a hand or face,
> we fill those keypoints with `np.nan` — NOT zeros. Filling with zeros would place the
> hand at the top-left origin `[0,0,0]`, causing the cubic spline resampler to violently
> swing the trajectory to the corner and back, corrupting the entire gesture. NaNs are
> invisible to the interpolator and get bridged smoothly over.

```
Raw .mp4
    │
    ▼  Step 1: EXTRACT  (NaN Approach)
    │  OpenCV reads each frame. MediaPipe Task API (Face + Hand + Pose
    │  landmarkers) extracts the 85 keypoints.
    │  ✅ Missing detections → np.nan  (NOT zeros)
    │  Result: (T_raw, 85, 3) float32 with NaNs where detection failed
    │
    ▼  Step 2: TRIM DEAD FRAMES
    │  Check the Pose shoulder landmarks (rows 39 & 40 in the 85-pt array).
    │  Slice off any leading or trailing frames where BOTH shoulders are NaN
    │  (blank black screen). If < 4 active frames remain → discard video.
    │  Result: (T_active, 85, 3)  — trimmed, still contains mid-sequence NaNs
    │
    ▼  Step 3: NORMALIZE
    │  For each frame, if shoulders are detected, center on the shoulder
    │  midpoint and scale by inter-shoulder distance. Frames where shoulders
    │  are NaN are left as NaN (will be handled in Phase A below).
    │  Result: (T_active, 85, 3)  — normalized
    │
    ▼  Step 4A: NaN BRIDGING  (Phase A — fix mid-sequence dropouts)
    │  Use pandas DataFrame.interpolate(method='linear', axis=0) to draw
    │  a straight line across any NaN gaps mid-sequence.
    │  Example: hand drops at frame 10, reappears at frame 13 →
    │  frames 11 & 12 get linearly interpolated between the real values.
    │  Any remaining edge NaNs (start/end of sequence) are forward/back-filled.
    │  Result: (T_active, 85, 3)  — fully dense, NO NaNs remaining
    │
    ▼  Step 4B: SEQUENCE RESAMPLING  (Phase B — fix variable length)
    │  Now that the data is clean and unbroken, use
    │  scipy.interpolate.interp1d (cubic spline) to mathematically
    │  stretch or compress to exactly TARGET = 60 frames.
    │  Falls back to linear if T_active < 4.
    │  Result: (60, 85, 3)  — fixed shape, every single video
    │
    ▼  Step 5: SAVE
       np.save(output/{split}/{gloss}/{video_id}.npy)
       Final shape: (60, 85, 3), dtype float32
```

---

## Progress Bar (CLI Output)

The script uses `tqdm` to show a live progress bar during the extraction run:

```
Extracting [train]: 100%|████████████| 1727/1727 [14:22<00:00,  2.01video/s]
Extracting [val]:   100%|████████████|  381/381  [03:09<00:00,  2.01video/s]
Extracting [test]:  100%|████████████| 1436/1436 [11:58<00:00,  2.01video/s]
```

Each bar shows:
- Which split is being processed (`train` / `val` / `test`)
- Videos completed vs. total
- Elapsed time and estimated time remaining (ETA)
- Processing speed in videos/second

Errors and warnings are printed above the bar using `tqdm.write()` so they don't corrupt the progress display.

---

## Changes to `extract_keypoints.py`

#### [MODIFY] `extract_keypoints.py`

The script will be rewritten with the following changes vs. the WLASL version:

1. **New data source:** Reads from `asl_citizen_top100.json` instead of `nslt_100.json` + `WLASL_v0.3.json`.
2. **Split-aware output:** Saves files into `output/train/`, `output/val/`, `output/test/` subdirectories, mirroring the original splits.
3. **Dead-frame trimming:** New `trim_dead_frames()` function runs before normalization and resampling.
4. **Resume-safe:** Already-processed `.npy` files are skipped so the job can be interrupted and restarted.

---

## Verification Plan

### Automated Tests
- Run on 3 videos (one from each split) as a smoke test before the full run.
- Assert output shape is exactly `(60, 85, 3)` and dtype is `float32` for all.
- Assert no file is entirely zeros (catches videos where all landmarks failed).

### Manual Verification
- Confirm `output/train/`, `output/val/`, `output/test/` are all populated.
- Spot-check that the same gloss appears in all three splits (e.g., `APPLE` exists in train, val, and test — with different signers).
- Confirm total file count ≈ 3,544 (minus any discarded videos with < 4 active frames).
