"""
Dataset and Collation Utilities for AB-ST.

Provides robust sequence collation, temporal mask creation,
and landmark adapter integration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    class Dataset:  # type: ignore
        pass

from .adapter_543_to_66 import map_543_to_66, validate_canonical_66_structure


class SignTranslationDataset(Dataset):
    """
    Dataset representing multimodal sign language pairs (video/patches, pose, text).
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        max_frames: int = 256,
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.max_frames = max_frames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        
        # Load landmarks
        landmarks = item.get("landmarks")
        if isinstance(landmarks, str) or isinstance(landmarks, Path):
            path = Path(landmarks)
            if path.suffix == ".npy":
                import numpy as np
                landmarks = np.load(path)
            elif path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    landmarks = json.load(f)
                    
        # Map 543 -> 66 if needed
        landmarks_66 = map_543_to_66(landmarks)
        
        # Truncate to max_frames if necessary
        if hasattr(landmarks_66, "shape"):
            T = landmarks_66.shape[0]
            if T > self.max_frames:
                landmarks_66 = landmarks_66[:self.max_frames]
        elif isinstance(landmarks_66, list):
            if len(landmarks_66) > self.max_frames:
                landmarks_66 = landmarks_66[:self.max_frames]
                
        out = {
            "video_id": item.get("video_id", f"sample_{idx}"),
            "landmarks_66": landmarks_66,
            "text": item.get("text", ""),
        }
        
        if "patches" in item:
            out["patches"] = item["patches"]
        elif "frames" in item:
            out["frames"] = item["frames"]
            
        if "target_ids" in item:
            out["target_ids"] = item["target_ids"]
            
        if "ctc_targets" in item:
            out["ctc_targets"] = item["ctc_targets"]

        return out


def collate_abst_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate variable-length sign sequences with padding and temporal masks.
    """
    if not _HAS_TORCH:
        return {"raw_batch": batch}

    B = len(batch)
    lengths = [
        sample["landmarks_66"].shape[0]
        if hasattr(sample["landmarks_66"], "shape")
        else len(sample["landmarks_66"])
        for sample in batch
    ]
    max_T = max(lengths) if lengths else 1

    # Prepare padded landmarks: (B, max_T, 66, 3)
    padded_landmarks = torch.zeros((B, max_T, 66, 3), dtype=torch.float32)
    temporal_mask = torch.zeros((B, max_T), dtype=torch.bool)

    for i, sample in enumerate(batch):
        lms = sample["landmarks_66"]
        if not isinstance(lms, torch.Tensor):
            lms = torch.tensor(lms, dtype=torch.float32)
        T_curr = lms.shape[0]
        padded_landmarks[i, :T_curr, :, :lms.shape[-1]] = lms[:, :, :3]
        temporal_mask[i, :T_curr] = True

    collated = {
        "video_ids": [s["video_id"] for s in batch],
        "landmarks_66": padded_landmarks,
        "temporal_mask": temporal_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "texts": [s["text"] for s in batch],
    }

    # Handle patches if available: (B, max_T, 16, 16, C)
    if "patches" in batch[0]:
        first_patch = batch[0]["patches"]
        if isinstance(first_patch, torch.Tensor):
            Hg, Wg, C = first_patch.shape[1], first_patch.shape[2], first_patch.shape[3]
        else:
            first_patch = torch.tensor(first_patch, dtype=torch.float32)
            Hg, Wg, C = first_patch.shape[1], first_patch.shape[2], first_patch.shape[3]
            
        padded_patches = torch.zeros((B, max_T, Hg, Wg, C), dtype=first_patch.dtype)
        for i, sample in enumerate(batch):
            p = sample["patches"]
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float32)
            padded_patches[i, :p.shape[0]] = p
        collated["patches"] = padded_patches

    # Handle target_ids: (B, max_L) with -100 padding
    if "target_ids" in batch[0] and batch[0]["target_ids"] is not None:
        tgt_list = [torch.as_tensor(s["target_ids"], dtype=torch.long) for s in batch]
        max_L = max(len(t) for t in tgt_list)
        padded_tgt = torch.full((B, max_L), -100, dtype=torch.long)
        for i, t in enumerate(tgt_list):
            padded_tgt[i, :len(t)] = t
        collated["target_ids"] = padded_tgt

    return collated

