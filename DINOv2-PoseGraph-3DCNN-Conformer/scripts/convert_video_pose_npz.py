#!/usr/bin/env python3
"""Combine an RGB video and exported 66-point pose NPZ into the ISL-MEGA record schema.

Pose input NPZ must contain pose_xy [Tp,66,2], pose_timestamps [Tp], and may
contain pose_confidence, pose_valid, boxes, box_valid, box_scores.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--pose-npz", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pose-time-offset-seconds", type=float, default=0.0,
                   help="Add this offset to pose timestamps when they are clip-relative but video PTS is not")
    args = p.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install `pip install -e '.[video]'` to decode videos") from exc
    with np.load(args.pose_npz, allow_pickle=False) as pose:
        required = {"pose_xy", "pose_timestamps"}
        missing = required - set(pose.files)
        if missing:
            raise ValueError(f"Pose NPZ missing: {sorted(missing)}")
        pose_xy = np.asarray(pose["pose_xy"], dtype=np.float32)
        pose_ts = np.asarray(pose["pose_timestamps"], dtype=np.float64) + args.pose_time_offset_seconds
        if pose_ts.ndim != 1 or len(pose_ts) < 1 or not np.isfinite(pose_ts).all():
            raise ValueError("pose_timestamps must be a nonempty finite 1-D vector")
        if len(pose_ts) > 1 and np.any(np.diff(pose_ts) <= 0):
            raise ValueError("pose_timestamps must be strictly increasing")
        if pose_xy.shape != (len(pose_ts), 66, 2):
            raise ValueError(f"pose_xy must have shape [Tp,66,2], got {pose_xy.shape}")
        pose_payload = {
            "pose_xy": pose_xy,
            "pose_timestamps": pose_ts,
        }
        for key in ("pose_confidence", "pose_valid", "boxes", "box_valid", "box_scores", "fingerspelling_spans"):
            if key in pose:
                pose_payload[key] = np.asarray(pose[key])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    frames, times = [], []
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    first_ms = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not np.isfinite(ms) or (ms == 0.0 and len(times) > 0):
            ms = len(times) * 1000.0 / fps
        if first_ms is None:
            first_ms = ms
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Preserve the video's presentation timestamp time base. Dataset loading
        # re-bases times only after RGB/pose alignment is complete.
        times.append(ms / 1000.0)
    cap.release()
    if not frames:
        raise RuntimeError("Video contained no decodable frames")
    video_times = np.asarray(times, dtype=np.float64)
    # Timestamp repair is deliberately not silent: duplicate/nonmonotonic clock
    # values are replaced only by the documented nominal-FPS fallback.
    if len(video_times) > 1 and np.any(np.diff(video_times) <= 0):
        origin = (first_ms / 1000.0) if first_ms is not None else 0.0
        video_times = origin + np.arange(len(frames), dtype=np.float64) / fps
    payload = {
        "frames": np.stack(frames), "video_timestamps": video_times.astype(np.float64),
        **pose_payload,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"Saved {len(frames)} video frames and {len(payload['pose_xy'])} pose frames to {args.out}")

if __name__ == "__main__":
    main()
