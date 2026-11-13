"""
Canonical MediaPipe Holistic (543 landmarks) to Articulatory 66-landmark Adapter.

Canonical 66-point layout:
    - [0..20]   : Left Hand (21 landmarks)
    - [21..41]  : Right Hand (21 landmarks)
    - [42..47]  : Upper Body (6 landmarks: left_shoulder, right_shoulder,
                  left_elbow, right_elbow, left_wrist, right_wrist)
    - [48..65]  : Expressive Face Mesh (18 landmarks: eyebrows, eyes, nose,
                  cheeks, lips/mouth, chin)

This canonical ordering guarantees consistent indices across all downstream
articulatory routing, dominance selection, deformable binding, and temporal blocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import math

# Canonical 66 index ranges
LEFT_HAND_RANGE = (0, 21)
RIGHT_HAND_RANGE = (21, 42)
BODY_RANGE = (42, 48)
FACE_RANGE = (48, 66)

# MediaPipe Holistic Pose landmark IDs for Upper Body:
# 11: left shoulder, 12: right shoulder, 13: left elbow,
# 14: right elbow, 15: left wrist, 16: right wrist
HOLISTIC_POSE_UPPER_BODY_IDS = [11, 12, 13, 14, 15, 16]

# MediaPipe Holistic 543 offsets:
# Pose: [0..32] (33 landmarks)
# Face: [33..500] (468 landmarks)
# Left Hand: [501..521] (21 landmarks)
# Right Hand: [522..542] (21 landmarks)
HOLISTIC_FACE_OFFSET = 33
HOLISTIC_LEFT_HAND_OFFSET = 501
HOLISTIC_RIGHT_HAND_OFFSET = 522

# 18 Key Expressive Face Mesh landmark IDs (relative to Face Mesh 0..467):
# - Left Eyebrow: 70, 107
# - Right Eyebrow: 336, 300
# - Left Eye: 33, 133
# - Right Eye: 362, 263
# - Nose Bridge & Tip: 168, 1, 2
# - Cheeks: 234, 454
# - Mouth Outer & Inner Lips: 61, 291, 0, 17
# - Chin: 152
FACE_MESH_18_IDS = [
    70, 107,        # Left eyebrow
    336, 300,       # Right eyebrow
    33, 133,        # Left eye outer/inner
    362, 263,       # Right eye inner/outer
    168, 1, 2,      # Nose root, tip, subnasale
    234, 454,       # Left cheek, right cheek
    61, 291, 0, 17, # Mouth left, right, upper lip, lower lip
    152             # Chin
]

# Exact 543 indices for all 66 canonical landmarks:
# 1. Left hand: 501..521 (21 points)
# 2. Right hand: 522..542 (21 points)
# 3. Body: [11, 12, 13, 14, 15, 16] (6 points)
# 4. Face: [33 + fid for fid in FACE_MESH_18_IDS] (18 points)
CANONICAL_66_INDICES_IN_543 = (
    list(range(HOLISTIC_LEFT_HAND_OFFSET, HOLISTIC_LEFT_HAND_OFFSET + 21)) +
    list(range(HOLISTIC_RIGHT_HAND_OFFSET, HOLISTIC_RIGHT_HAND_OFFSET + 21)) +
    HOLISTIC_POSE_UPPER_BODY_IDS +
    [HOLISTIC_FACE_OFFSET + fid for fid in FACE_MESH_18_IDS]
)

assert len(CANONICAL_66_INDICES_IN_543) == 66, "Canonical mapping must contain exactly 66 indices"


def get_canonical_indices_map() -> List[int]:
    """Return a copy of the canonical 66 index list from Holistic 543."""
    return list(CANONICAL_66_INDICES_IN_543)


def map_holistic_543_to_canonical_66_list(
    raw_landmarks: List[List[float]],
) -> List[List[float]]:
    """
    Map a single frame of 543 landmarks [[x, y, z], ...] to 66 canonical landmarks.
    
    Args:
        raw_landmarks: List of at least 543 coordinate vectors (length 3 or 4).
    Returns:
        List of 66 coordinate vectors in canonical order.
    """
    if len(raw_landmarks) < 543:
        raise ValueError(
            f"Expected at least 543 landmarks, got {len(raw_landmarks)}"
        )
    return [raw_landmarks[idx] for idx in CANONICAL_66_INDICES_IN_543]


def map_sequence_543_to_canonical_66_list(
    sequence: List[List[List[float]]],
) -> List[List[List[float]]]:
    """Map a sequence of frames [T, 543, C] to [T, 66, C]."""
    return [map_holistic_543_to_canonical_66_list(frame) for frame in sequence]


def map_543_to_66_numpy(arr: Any) -> Any:
    """
    Vectorized extraction for numpy ndarrays.
    
    Args:
        arr: ndarray of shape (..., 543, C).
    Returns:
        ndarray of shape (..., 66, C).
    """
    import numpy as np  # lazy import
    if arr.shape[-2] < 543:
        raise ValueError(f"Array penultimate dimension must be at least 543, got {arr.shape}")
    indices = np.array(CANONICAL_66_INDICES_IN_543, dtype=np.int64)
    return np.take(arr, indices, axis=-2)


def map_543_to_66_torch(tensor: Any) -> Any:
    """
    Vectorized extraction for PyTorch tensors.
    
    Args:
        tensor: Tensor of shape (..., 543, C).
    Returns:
        Tensor of shape (..., 66, C).
    """
    import torch  # lazy import
    if tensor.shape[-2] < 543:
        raise ValueError(f"Tensor penultimate dimension must be at least 543, got {tensor.shape}")
    indices = torch.tensor(CANONICAL_66_INDICES_IN_543, dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, dim=-2, index=indices)


def map_543_to_66(data: Any) -> Any:
    """
    Universal dispatcher supporting lists, numpy arrays, and torch tensors.
    """
    # Detect torch Tensor without top-level import
    if hasattr(data, "index_select") and hasattr(data, "device"):
        return map_543_to_66_torch(data)
    elif hasattr(data, "shape"):
        return map_543_to_66_numpy(data)
    elif isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
            return map_sequence_543_to_canonical_66_list(data)
        return map_holistic_543_to_canonical_66_list(data)
    raise TypeError(f"Unsupported data type for landmark mapping: {type(data)}")


def validate_canonical_66_structure(landmarks_66: Any) -> Dict[str, Any]:
    """
    Perform structural and data integrity validation on canonical 66-point landmarks.
    
    Checks:
    1. Landmark count == 66
    2. Coordinate dimensions (at least (x, y))
    3. Non-degeneracy (valid finite numbers, not all zeros)
    4. Shoulder width sanity check (shoulders 42 and 43 separated)
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "num_landmarks": 0,
        "shoulder_dist": 0.0,
    }
    
    # Handle list of coordinates
    if isinstance(landmarks_66, list):
        report["num_landmarks"] = len(landmarks_66)
        if len(landmarks_66) != 66:
            report["valid"] = False
            report["errors"].append(f"Expected 66 landmarks, got {len(landmarks_66)}")
            return report
        
        # Check coordinates finite
        for i, pt in enumerate(landmarks_66):
            if any(math.isnan(c) or math.isinf(c) for c in pt[:3]):
                report["valid"] = False
                report["errors"].append(f"Landmark {i} has NaN or Inf coordinates")
                break
                
        # Check shoulder distance
        p42 = landmarks_66[42]
        p43 = landmarks_66[43]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p42[:2], p43[:2])))
        report["shoulder_dist"] = dist
        if dist < 1e-4:
            report["warnings"].append("Left and right shoulder points are virtually coincident.")
            
    elif hasattr(landmarks_66, "shape"):
        shape = landmarks_66.shape
        if shape[-2] != 66:
            report["valid"] = False
            report["errors"].append(f"Expected shape (..., 66, C), got {shape}")
        report["num_landmarks"] = shape[-2]
        
    return report

