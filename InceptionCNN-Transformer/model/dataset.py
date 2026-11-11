"""
ASL Citizen Dataset — PyTorch Dataset and DataLoader factory.

Loads .npy keypoint files from output/{split}/{gloss}/
Flattens (60, 85, 3) → (60, 255) per sample.
Builds and saves an alphabetical gloss→label_id mapping to class_map.json.
Applies augmentations on the train split only.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (resolved relative to this file so it works both locally and in Colab)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = os.path.join(os.path.dirname(SCRIPT_DIR), "output")
CLASS_MAP_PATH = os.path.join(SCRIPT_DIR, "class_map.json")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD / LOAD LABEL MAP
# ─────────────────────────────────────────────────────────────────────────────

def build_class_map(output_dir: str = None) -> dict:
    """
    Scan all split directories to collect every unique gloss, sort
    alphabetically, assign an integer label, and save to class_map.json.
    Returns {gloss: label_id}.
    """
    if output_dir is None: output_dir = OUTPUT_DIR
    
    glosses = set()
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            if os.path.isdir(os.path.join(split_dir, name)):
                glosses.add(name)

    class_map = {gloss: idx for idx, gloss in enumerate(sorted(glosses))}
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"[ClassMap] {len(class_map)} classes saved to {CLASS_MAP_PATH}")
    return class_map


def load_class_map(output_dir: str = None) -> dict:
    """Load existing class_map.json, or build it from scratch."""
    if output_dir is None: output_dir = OUTPUT_DIR
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH) as f:
            return json.load(f)
    return build_class_map(output_dir)



# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATIONS  (train split only)
# ─────────────────────────────────────────────────────────────────────────────

def augment(x: np.ndarray) -> np.ndarray:
    """
    x shape: (60, 255)

    1. Gaussian noise jitter (sigma=0.005)  — simulates camera noise
    2. Random temporal shift (±3 frames, circular wrap)  — simulates
       variation in signing speed / recording start offset
    """
    x = x + np.random.normal(0, 0.005, x.shape).astype(np.float32)
    shift = np.random.randint(-3, 4)   # -3 to +3 inclusive
    x = np.roll(x, shift, axis=0)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ASLDataset(Dataset):
    """
    Lazily loads .npy keypoint files for one split.

    Args:
        split      : 'train', 'val', or 'test'
        class_map  : dict mapping gloss folder name → integer label
        output_dir : root directory containing train/val/test folders
        augment_fn : optional callable applied to numpy array before tensorising
    """

    def __init__(self, split: str, class_map: dict,
                 output_dir: str = None, augment_fn=None):
        if output_dir is None: output_dir = OUTPUT_DIR
        self.augment_fn = augment_fn
        self.samples: list = []   # (file_path, label)

        split_dir = os.path.join(output_dir, split)
        for gloss_folder in sorted(os.listdir(split_dir)):
            gloss_path = os.path.join(split_dir, gloss_folder)
            if not os.path.isdir(gloss_path):
                continue
            label = class_map.get(gloss_folder)
            if label is None:
                continue
            for fname in os.listdir(gloss_path):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(gloss_path, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = np.load(path)                           # (60, 85, 3)
        x = x.reshape(60, -1).astype(np.float32)   # (60, 255)
        if self.augment_fn is not None:
            x = self.augment_fn(x)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 64, num_workers: int = 2,
                    output_dir: str = None):
    """Build and return (train_loader, val_loader, test_loader)."""
    if output_dir is None: output_dir = OUTPUT_DIR
    class_map = load_class_map(output_dir)

    train_ds = ASLDataset("train", class_map, output_dir, augment_fn=augment)
    val_ds   = ASLDataset("val",   class_map, output_dir, augment_fn=None)
    test_ds  = ASLDataset("test",  class_map, output_dir, augment_fn=None)

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
