"""
Milestone 1: MediaPipe Keypoint Extraction Pipeline (ASL Citizen)
=================================================================
Converts raw ASL Citizen videos into clean, fixed-shape keypoint arrays.

Pipeline per video:
  1. EXTRACT        — MediaPipe Task API → (T_raw, 85, 3), NaN for misses
  2. TRIM           — Strip leading/trailing frames where both shoulders are NaN
  3. NORMALIZE      — Center on shoulder midpoint, scale by inter-shoulder dist
  4A. NaN BRIDGE    — pandas linear interpolation fills mid-sequence NaN gaps
  4B. RESAMPLE      — scipy cubic spline stretches/squishes to TARGET_FRAMES
  5. SAVE           — output/{split}/{gloss}/{video_id}.npy  shape (60, 85, 3)

Output shape per file: (60, 85, 3) float32
"""

import json
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from scipy.interpolate import interp1d
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TARGET_FRAMES   = 60        # Every .npy will be exactly (60, 85, 3)
MIN_ACTIVE_FRAMES = 4       # Discard videos shorter than this after trimming

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)
ARCHIVE_DIR   = os.path.join(PROJECT_ROOT, "archive")
VIDEOS_DIR    = os.path.join(ARCHIVE_DIR, "videos")
MAPPING_PATH  = os.path.join(ARCHIVE_DIR, "asl_citizen_top100.json")
MODELS_DIR    = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "output")

FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker_full.task")

# ─────────────────────────────────────────────────────────────────────────────
# KEYPOINT SUBSET DEFINITIONS  (85 total)
# ─────────────────────────────────────────────────────────────────────────────

FACE_INDICES = [
    # Eyes (4)
    33, 133, 263, 362,
    # Left Eyebrow (7)
    70, 63, 105, 66, 107, 55, 65,
    # Right Eyebrow (7)
    300, 293, 334, 296, 336, 285, 295,
    # Outer Upper Lip (11)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # Outer Lower Lip (9)
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    # Inner Lip Center (1)
    14,
]  # 39 total

POSE_INDICES   = [11, 12, 13, 14]   # L-shoulder, R-shoulder, L-elbow, R-elbow
POSE_START     = len(FACE_INDICES)  # 39
L_SHOULDER_ROW = POSE_START + 0     # index into the 85-pt array
R_SHOULDER_ROW = POSE_START + 1

NUM_KEYPOINTS  = len(FACE_INDICES) + len(POSE_INDICES) + 21 + 21  # 85
NUM_COORDS     = 3  # x, y, z

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARKERS
# ─────────────────────────────────────────────────────────────────────────────

def create_landmarkers():
    """Initialise the three MediaPipe Task-API landmarkers (IMAGE mode)."""
    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
    )
    hand_lm = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            num_hands=2,
        )
    )
    pose_lm = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            num_poses=1,
        )
    )
    return face_lm, hand_lm, pose_lm

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EXTRACT (NaN Approach)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame_keypoints(frame_rgb, face_lm, hand_lm, pose_lm):
    """
    Run all three landmarkers on one RGB frame.
    Returns (85, 3) float32.  Missing detections → np.nan (NOT zeros).
    """
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    kps    = np.full((NUM_KEYPOINTS, NUM_COORDS), np.nan, dtype=np.float32)
    cursor = 0

    # ── Face (39 pts) ──────────────────────────────────────────────────────
    face_res = face_lm.detect(mp_img)
    if face_res.face_landmarks:
        lms = face_res.face_landmarks[0]
        for idx in FACE_INDICES:
            if idx < len(lms):
                lm = lms[idx]
                kps[cursor] = [lm.x, lm.y, lm.z]
            cursor += 1
    else:
        cursor += len(FACE_INDICES)   # leave as NaN

    # ── Pose (4 pts) ───────────────────────────────────────────────────────
    pose_res = pose_lm.detect(mp_img)
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        for i, idx in enumerate(POSE_INDICES):
            if idx < len(lms):
                lm = lms[idx]
                kps[cursor + i] = [lm.x, lm.y, lm.z]
    cursor += len(POSE_INDICES)

    # ── Hands (21 + 21 pts) ────────────────────────────────────────────────
    # Leave as NaN if hand not detected (correct — not zeros)
    hand_res = hand_lm.detect(mp_img)
    if hand_res.hand_landmarks:
        for i, handedness in enumerate(hand_res.handedness):
            label = handedness[0].category_name
            arr   = np.array([[lm.x, lm.y, lm.z]
                               for lm in hand_res.hand_landmarks[i]],
                              dtype=np.float32)
            if label == "Left":
                kps[cursor: cursor + 21] = arr
            else:
                kps[cursor + 21: cursor + 42] = arr

    return kps  # (85, 3), NaN where detection failed

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRIM DEAD FRAMES
# ─────────────────────────────────────────────────────────────────────────────

def trim_dead_frames(seq):
    """
    Remove leading and trailing frames where BOTH shoulders are NaN.
    These correspond to blank black screens at the start/end of the video.
    Returns None if fewer than MIN_ACTIVE_FRAMES remain.
    """
    l_sh = seq[:, L_SHOULDER_ROW, :]  # (T, 3)
    r_sh = seq[:, R_SHOULDER_ROW, :]

    # A frame is "active" if at least one shoulder has a real value
    active = ~(np.all(np.isnan(l_sh), axis=1) & np.all(np.isnan(r_sh), axis=1))

    active_indices = np.where(active)[0]
    if len(active_indices) < MIN_ACTIVE_FRAMES:
        return None

    first, last = active_indices[0], active_indices[-1] + 1
    return seq[first:last]  # (T_active, 85, 3)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — NORMALIZE
# ─────────────────────────────────────────────────────────────────────────────

def normalize_sequence(seq):
    """
    Per-frame: center on shoulder midpoint, scale by inter-shoulder distance.
    Frames where shoulders are NaN are left as NaN (bridged in Step 4A).
    """
    out = seq.copy()
    for t in range(len(out)):
        l_sh = out[t, L_SHOULDER_ROW]
        r_sh = out[t, R_SHOULDER_ROW]
        if np.any(np.isnan(l_sh)) or np.any(np.isnan(r_sh)):
            continue  # leave as NaN, will be bridged later
        midpoint = (l_sh + r_sh) / 2.0
        distance = np.linalg.norm(r_sh - l_sh)
        out[t] -= midpoint
        if distance > 1e-6:
            out[t] /= distance
    return out

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4A — NaN BRIDGING  (fix mid-sequence dropouts)
# ─────────────────────────────────────────────────────────────────────────────

def bridge_nans(seq):
    """
    Use pandas linear interpolation along the time axis to fill NaN gaps.
    Example: hand drops at frame 10, reappears at frame 13 →
             frames 11 & 12 get a straight line between the real values.
    Edge NaNs (start/end of sequence) are forward/back-filled.

    Input:  (T, 85, 3) with NaNs
    Output: (T, 85, 3) fully dense, no NaNs
    """
    T = len(seq)
    flat = seq.reshape(T, -1)  # (T, 255)

    df = pd.DataFrame(flat)
    df = df.interpolate(method="linear", axis=0, limit_direction="both")

    # If any NaNs remain (e.g., a feature was NEVER detected), fill with 0
    df = df.fillna(0.0)

    return df.values.reshape(T, NUM_KEYPOINTS, NUM_COORDS).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4B — SEQUENCE RESAMPLING  (fix variable length)
# ─────────────────────────────────────────────────────────────────────────────

def resample_sequence(seq, target_len=TARGET_FRAMES):
    """
    Stretch or compress a (T_active, 85, 3) sequence to (target_len, 85, 3)
    using cubic spline interpolation along the time axis.
    Falls back to linear if T_active < 4.
    """
    T = len(seq)
    if T == target_len:
        return seq

    old_times = np.linspace(0, 1, T)
    new_times = np.linspace(0, 1, target_len)
    flat      = seq.reshape(T, -1)  # (T, 255)

    kind         = "cubic" if T >= 4 else "linear"
    interpolator = interp1d(old_times, flat, kind=kind, axis=0)

    return interpolator(new_times).reshape(
        target_len, NUM_KEYPOINTS, NUM_COORDS
    ).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# FULL VIDEO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_video(path, face_lm, hand_lm, pose_lm):
    """
    Run the complete 5-step pipeline for one video.
    Returns (60, 85, 3) float32 or None if the video should be discarded.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(extract_frame_keypoints(rgb, face_lm, hand_lm, pose_lm))
    cap.release()

    if not frames:
        return None

    seq = np.stack(frames, axis=0)    # Step 1 → (T_raw, 85, 3)
    seq = trim_dead_frames(seq)        # Step 2 → (T_active, 85, 3) or None
    if seq is None:
        return None
    seq = normalize_sequence(seq)      # Step 3 → (T_active, 85, 3) normalized
    seq = bridge_nans(seq)             # Step 4A → (T_active, 85, 3) no NaNs
    seq = resample_sequence(seq)       # Step 4B → (60, 85, 3)
    return seq                         # Step 5 → saved by caller

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split_name, split_data, face_lm, hand_lm, pose_lm):
    """Process all videos for one split, with a live tqdm progress bar."""
    processed = skipped = errors = discarded = 0

    bar = tqdm(
        split_data.items(),
        total=len(split_data),
        desc=f"Extracting [{split_name:<5}]",
        unit="video",
        dynamic_ncols=True,
    )

    for video_file, meta in bar:
        gloss    = meta["gloss"]
        vid_path = os.path.join(VIDEOS_DIR, video_file)
        out_dir  = os.path.join(OUTPUT_DIR, split_name, gloss)
        os.makedirs(out_dir, exist_ok=True)

        # Use the filename stem as the unique ID
        vid_id   = os.path.splitext(video_file)[0]
        out_path = os.path.join(out_dir, f"{vid_id}.npy")

        # Resume-safe: skip already-processed files
        if os.path.exists(out_path):
            processed += 1
            continue

        if not os.path.exists(vid_path):
            skipped += 1
            tqdm.write(f"  [MISSING] {video_file}")
            continue

        try:
            seq = process_video(vid_path, face_lm, hand_lm, pose_lm)
            if seq is None:
                discarded += 1
                tqdm.write(f"  [DISCARD] {video_file} — too few active frames")
                continue
            np.save(out_path, seq)
            processed += 1
        except Exception as e:
            errors += 1
            tqdm.write(f"  [ERROR  ] {video_file}: {e}")

        # Live stats in the bar suffix
        bar.set_postfix(ok=processed, miss=skipped, disc=discarded, err=errors)

    return processed, skipped, discarded, errors


def main():
    print("=" * 66)
    print("  ASL Citizen Keypoint Extraction — Top-100 Words")
    print(f"  Target shape per video : ({TARGET_FRAMES}, {NUM_KEYPOINTS}, {NUM_COORDS})")
    print("=" * 66)

    # Load mapping
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    splits_data = mapping["splits"]  # {"train": {...}, "val": {...}, "test": {...}}

    total_vids = sum(len(v) for v in splits_data.values())
    print(f"  Train : {len(splits_data['train']):>5} videos")
    print(f"  Val   : {len(splits_data['val']):>5} videos")
    print(f"  Test  : {len(splits_data['test']):>5} videos")
    print(f"  Total : {total_vids:>5} videos\n")

    print("[MediaPipe] Initialising landmarkers …")
    face_lm, hand_lm, pose_lm = create_landmarkers()
    print("[MediaPipe] All three landmarkers ready.\n")

    grand = {"processed": 0, "skipped": 0, "discarded": 0, "errors": 0}

    for split_name in ["train", "val", "test"]:
        p, s, d, e = process_split(
            split_name, splits_data[split_name], face_lm, hand_lm, pose_lm
        )
        grand["processed"]  += p
        grand["skipped"]    += s
        grand["discarded"]  += d
        grand["errors"]     += e
        print()  # blank line between splits

    # ── Final summary ─────────────────────────────────────────────────────
    print("=" * 66)
    print("  EXTRACTION COMPLETE")
    print("=" * 66)
    print(f"  Processed  (saved)   : {grand['processed']}")
    print(f"  Missing    (no file) : {grand['skipped']}")
    print(f"  Discarded  (< {MIN_ACTIVE_FRAMES} frames): {grand['discarded']}")
    print(f"  Errors               : {grand['errors']}")
    print(f"  Output directory     : {OUTPUT_DIR}")
    print("=" * 66)

    # ── Sanity check on first file from each split ────────────────────────
    print("\n[Sanity Checks]")
    all_ok = True
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        if not os.path.isdir(split_dir):
            print(f"  [{split_name}] ❌  output directory missing!")
            all_ok = False
            continue
        found = False
        for root, _, files in os.walk(split_dir):
            for fname in sorted(files):
                if fname.endswith(".npy"):
                    sample = np.load(os.path.join(root, fname))
                    exp    = (TARGET_FRAMES, NUM_KEYPOINTS, NUM_COORDS)
                    status = "✅" if sample.shape == exp else "❌"
                    gloss  = os.path.basename(root)
                    print(f"  [{split_name}] {status}  {gloss}/{fname} → {sample.shape}")
                    if sample.shape != exp:
                        all_ok = False
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"  [{split_name}] ⚠️   no .npy files found")

    print("\n✅ All sanity checks passed!" if all_ok else "\n❌ Some checks failed — review errors above.")


if __name__ == "__main__":
    main()
