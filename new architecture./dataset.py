"""Dataset Loading, Feature Extraction, and Structured Collation for RA-MAGC.

Extracts structured hand, body, and facial streams with physical bone edges,
preserving all canonical dataset paths and source-held-out split protocols.
Matches every specification in ideas.md Section 2.
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig

# 6 Upper Body Landmarks (Left/Right Shoulders, Elbows, Wrists)
UPPER_JOINTS = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST")

# 21 Hand Landmarks
HAND_JOINTS = (
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
)

# 18 Sparse Facial Landmarks (Brows, Eyes, Mouth)
FACE_POINTS = (
    "33", "133", "159", "145", "362", "263", "386", "374", "70", "63", "105", 
    "336", "296", "334", "61", "291", "13", "14"
)

# 20 physical bone edges per hand (ideas.md Section 2.3)
HAND_EDGES = tuple((start, end) for finger in (
    (0, 1, 2, 3, 4),      # Thumb
    (0, 5, 6, 7, 8),      # Index
    (0, 9, 10, 11, 12),   # Middle
    (0, 13, 14, 15, 16),  # Ring
    (0, 17, 18, 19, 20)   # Pinky
) for start, end in zip(finger, finger[1:]))

# 5 physical bone edges for upper body
BODY_EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5))


def compute_torso_reference(
    body_xy: np.ndarray, 
    body_valid: np.ndarray, 
    uid: str = "sample", 
    epsilon: float = 1e-4
) -> tuple[np.ndarray, float]:
    """
    Computes a single unified torso reference for the entire clip (AUDIT_REPORT.md Section 3.3).
    - center: left/right shoulder midpoint when observed [T, 2];
      fallback: median valid midpoint within the continuous clip.
    - scale: median valid shoulder distance within the clip.
    Raises ValueError if the clip has no valid left/right shoulder pair.
    """
    T = len(body_xy)
    if body_xy.shape[1] < 2 or body_valid.shape[1] < 2:
        raise ValueError(f"{uid}: body landmarks must contain at least left/right shoulders (indices 0 and 1)")
        
    left_valid = body_valid[:, 0]
    right_valid = body_valid[:, 1]
    shoulder_anchors = left_valid & right_valid
    
    shoulder_diffs = body_xy[:, 1] - body_xy[:, 0]  # right - left
    widths = np.linalg.norm(shoulder_diffs, axis=-1)
    good_anchors = shoulder_anchors & np.isfinite(widths) & (widths > epsilon)
    
    if not good_anchors.any():
        raise ValueError(f"{uid}: clip has no valid left/right shoulder pair for torso normalization")
        
    midpoints = (body_xy[:, 0] + body_xy[:, 1]) / 2.0  # [T, 2]
    fallback_center = np.median(midpoints[good_anchors], axis=0)  # [2]
    centers = np.where(good_anchors[:, None], midpoints, fallback_center[None, :])  # [T, 2]
    scale = max(float(np.median(widths[good_anchors])), epsilon)
    
    return centers, scale


def normalize_with_torso(
    coords: np.ndarray, 
    valid: np.ndarray, 
    centers: np.ndarray, 
    scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalizes coordinates of any anatomical region (body, hands, face) using the unified torso reference.
    coords: [T, N, 2]
    valid: [T, N]
    centers: [T, 2]
    scale: float
    """
    usable = valid & np.all(np.isfinite(coords), axis=-1)
    normalized = (coords - centers[:, None, :]) / scale
    normalized = np.where(usable[..., None], normalized, 0.0)
    return normalized, usable


def compute_velocity(positions: np.ndarray, valid: np.ndarray, times: np.ndarray):
    """
    Computes motion velocity in units per physical elapsed second.
    Motion is strictly masked when consecutive frames are invalid or dt <= 0.
    """
    velocity = np.zeros_like(positions)
    v_valid = np.zeros_like(valid)
    dt = np.diff(times)
    
    # Motion valid only if consecutive frames are both valid and dt > 0
    mask = valid[1:] & valid[:-1] & (dt[:, None] > 0)
    diff = (positions[1:] - positions[:-1]) / np.maximum(dt[:, None, None], 1e-6)
    
    velocity[1:] = np.where(mask[..., None], diff, 0.0)
    v_valid[1:] = mask
    return velocity, v_valid


def build_edge_tensors(
    positions: np.ndarray, 
    valid_mask: np.ndarray, 
    num_nodes: int, 
    edges: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs edge attributes [T, N, N, 4] and adjacency mask [T, N, N]
    where e_ij = [dx, dy, length, valid] (ideas.md Section 2.3).
    """
    T = len(positions)
    edge_features = np.zeros((T, num_nodes, num_nodes, 4), dtype=np.float32)
    adj_mask = np.zeros((T, num_nodes, num_nodes), dtype=bool)
    
    # Self-loops
    for i in range(num_nodes):
        adj_mask[:, i, i] = valid_mask[:, i]
        edge_features[:, i, i, 3] = valid_mask[:, i].astype(np.float32)
        
    for i, j in edges:
        e_valid = valid_mask[:, i] & valid_mask[:, j]
        diff = positions[:, j] - positions[:, i]
        length = np.linalg.norm(diff, axis=-1, keepdims=True)
        
        # Symmetrical undirected edge vectors
        edge_vec = np.concatenate([diff, length, e_valid[:, None].astype(np.float32)], axis=-1)
        rev_vec = np.concatenate([-diff, length, e_valid[:, None].astype(np.float32)], axis=-1)
        
        edge_features[:, i, j] = np.where(e_valid[:, None], edge_vec, 0.0)
        edge_features[:, j, i] = np.where(e_valid[:, None], rev_vec, 0.0)
        adj_mask[:, i, j] = e_valid
        adj_mask[:, j, i] = e_valid
        
    return edge_features, adj_mask


@dataclass
class StructuredPoseSample:
    """A single continuous sign example with structured anatomical streams."""
    uid: str
    text: str
    times: np.ndarray
    left_hand_nodes: np.ndarray   # [T, 21, 13]
    left_hand_edges: np.ndarray   # [T, 21, 21, 4]
    left_hand_adj: np.ndarray     # [T, 21, 21]
    left_hand_valid: np.ndarray   # [T, 21]
    right_hand_nodes: np.ndarray  # [T, 21, 13]
    right_hand_edges: np.ndarray  # [T, 21, 21, 4]
    right_hand_adj: np.ndarray    # [T, 21, 21]
    right_hand_valid: np.ndarray  # [T, 21]
    body_nodes: np.ndarray        # [T, 6, 10]
    body_edges: np.ndarray        # [T, 6, 6, 4]
    body_adj: np.ndarray          # [T, 6, 6]
    body_valid: np.ndarray        # [T, 6]
    face_nodes: np.ndarray | None = None  # [T, 18, 13]
    face_valid: np.ndarray | None = None  # [T, 18]


def extract_structured_sample(
    uid: str,
    text: str,
    times: np.ndarray,
    segments: np.ndarray | None,
    raw_coords: dict[str, np.ndarray],
    raw_valid: dict[str, np.ndarray],
    raw_conf: dict[str, np.ndarray] | None = None,
    include_face: bool = False
) -> StructuredPoseSample:
    """
    Constructs the exact 13-D hand node, 10-D body node, and 13-D face node feature tensors
    matching AUDIT_REPORT.md Section 3 and README.md Section 2.
    """
    times = np.asarray(times, dtype=np.float64)
    T = len(times)
    if T < 1:
        raise ValueError(f"{uid}: sequence must contain at least 1 frame")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{uid}: timestamps contain non-finite values (NaN or Inf)")
    if T > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{uid}: timestamps must be strictly increasing with no duplicates or backwards steps")
    if segments is not None and len(np.unique(segments)) > 1:
        raise ValueError(f"{uid}: sample contains multiple segment IDs; must represent one continuous segment")

    raw_conf = raw_conf or {}

    def get_confidence(region_key: str, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        if region_key in raw_conf and raw_conf[region_key] is not None:
            c = np.asarray(raw_conf[region_key][:, :num_nodes], dtype=np.float32)
            c_val = np.isfinite(c) & (c >= 0.0)
            c_clean = np.where(c_val, c, 0.0).astype(np.float32)
            return c_clean, c_val.astype(np.float32)
        else:
            return np.zeros((T, num_nodes), dtype=np.float32), np.zeros((T, num_nodes), dtype=np.float32)

    # 1. Establish common torso reference from body shoulders (AUDIT_REPORT.md Section 3.3)
    body_xy = np.asarray(raw_coords["body"][:, :6, :2], dtype=np.float32)
    body_obs = np.asarray(raw_valid["body"][:, :6], dtype=bool)
    centers, scale = compute_torso_reference(body_xy, body_obs, uid=uid)

    # Normalize Upper Body Positions (Shoulders, Elbows, Wrists)
    norm_body_xy, body_usable = normalize_with_torso(body_xy, body_obs, centers, scale)
    body_vel, body_vel_valid = compute_velocity(norm_body_xy, body_usable, times)
    body_conf, body_conf_val = get_confidence("body", 6)

    # Upper Body Nodes: [T, 6, 10]
    # (x, y, vx, vy, det_conf, conf_valid, observed, imputed, pos_valid, vel_valid)
    body_nodes = np.concatenate([
        norm_body_xy, body_vel,
        body_conf[..., None],
        body_conf_val[..., None],
        body_obs[..., None].astype(np.float32),
        np.zeros((T, 6, 1), dtype=np.float32),  # imputed placeholder
        body_usable[..., None].astype(np.float32),
        body_vel_valid[..., None].astype(np.float32)
    ], axis=-1)
    body_edges, body_adj = build_edge_tensors(norm_body_xy, body_usable, 6, BODY_EDGES)

    # 2. Hands (Left and Right)
    hand_samples = {}
    for side in ("left", "right"):
        hand_key = f"{side}_hand"
        hand_xy = np.asarray(raw_coords[hand_key][:, :21, :2], dtype=np.float32)
        hand_obs = np.asarray(raw_valid[hand_key][:, :21], dtype=bool)
        
        # Use common body shoulder center/scale for global hand normalization
        norm_hand_xy, hand_usable = normalize_with_torso(hand_xy, hand_obs, centers, scale)
        hand_vel, hand_vel_valid = compute_velocity(norm_hand_xy, hand_usable, times)
        hand_conf, hand_conf_val = get_confidence(hand_key, 21)
        
        # Wrist-relative coordinates (x_local, y_local)
        wrist_pos = norm_hand_xy[:, 0:1, :]  # Joint 0 is WRIST
        wrist_valid = hand_usable[:, 0:1]
        local_hand_xy = norm_hand_xy - wrist_pos
        local_valid = hand_usable & wrist_valid
        local_hand_xy = np.where(local_valid[..., None], local_hand_xy, 0.0)
        
        # Hand Nodes: [T, 21, 13]
        # (x, y, vx, vy, x_local, y_local, det_conf, conf_valid, observed, imputed, pos_valid, vel_valid, local_valid)
        hand_nodes = np.concatenate([
            norm_hand_xy, hand_vel, local_hand_xy,
            hand_conf[..., None],
            hand_conf_val[..., None],
            hand_obs[..., None].astype(np.float32),
            np.zeros((T, 21, 1), dtype=np.float32),
            hand_usable[..., None].astype(np.float32),
            hand_vel_valid[..., None].astype(np.float32),
            local_valid[..., None].astype(np.float32)
        ], axis=-1)
        
        hand_edges, hand_adj = build_edge_tensors(norm_hand_xy, hand_usable, 21, HAND_EDGES)
        hand_samples[side] = (hand_nodes, hand_edges, hand_adj, hand_usable)

    # 3. Optional Facial Landmarks (18 points: AUDIT_REPORT.md Section 3.1 & 3.2)
    face_nodes = None
    face_usable = None
    if include_face and "face" in raw_coords and "face" in raw_valid:
        face_xy = np.asarray(raw_coords["face"][:, :18, :2], dtype=np.float32)
        face_obs = np.asarray(raw_valid["face"][:, :18], dtype=bool)
        norm_face_xy, face_usable = normalize_with_torso(face_xy, face_obs, centers, scale)
        face_vel, face_vel_valid = compute_velocity(norm_face_xy, face_usable, times)
        face_conf, face_conf_val = get_confidence("face", 18)
        
        # Inner eye anchors (point index 0 = 33, point index 4 = 362 in FACE_POINTS)
        eye_center = (norm_face_xy[:, 0:1, :] + norm_face_xy[:, 4:5, :]) / 2.0  # [T, 1, 2]
        eye_diff = norm_face_xy[:, 4:5, :] - norm_face_xy[:, 0:1, :]
        eye_dist = np.linalg.norm(eye_diff, axis=-1, keepdims=True)  # [T, 1, 1]
        
        eye_valid = (face_usable[:, 0:1] & face_usable[:, 4:5] & (eye_dist.squeeze(-1) > 1e-4))  # [T, 1]
        
        # Explicit broadcasting across landmark and coordinate axes [T, 18, 2]
        local_face_xy = np.where(
            eye_valid[:, None, :], 
            (norm_face_xy - eye_center) / np.maximum(eye_dist, 1e-4), 
            0.0
        )
        local_face_valid = face_usable & eye_valid  # [T, 18]
        
        # Face Nodes: [T, 18, 13]
        # (x, y, vx, vy, x_local, y_local, det_conf, conf_valid, observed, imputed, pos_valid, vel_valid, local_valid)
        face_nodes = np.concatenate([
            norm_face_xy, face_vel, local_face_xy,
            face_conf[..., None],
            face_conf_val[..., None],
            face_obs[..., None].astype(np.float32),
            np.zeros((T, 18, 1), dtype=np.float32),
            face_usable[..., None].astype(np.float32),
            face_vel_valid[..., None].astype(np.float32),
            local_face_valid[..., None].astype(np.float32)
        ], axis=-1)

    return StructuredPoseSample(
        uid=uid,
        text=text,
        times=times,
        left_hand_nodes=hand_samples["left"][0],
        left_hand_edges=hand_samples["left"][1],
        left_hand_adj=hand_samples["left"][2],
        left_hand_valid=hand_samples["left"][3],
        right_hand_nodes=hand_samples["right"][0],
        right_hand_edges=hand_samples["right"][1],
        right_hand_adj=hand_samples["right"][2],
        right_hand_valid=hand_samples["right"][3],
        body_nodes=body_nodes,
        body_edges=body_edges,
        body_adj=body_adj,
        body_valid=body_usable,
        face_nodes=face_nodes,
        face_valid=face_usable
    )


def pad_time_tensor(tensors: list[torch.Tensor], value: float = 0.0) -> torch.Tensor:
    """Pad variable-length sequences to the batch maximum."""
    max_len = max(t.shape[0] for t in tensors)
    out = tensors[0].new_full((len(tensors), max_len, *tensors[0].shape[1:]), value)
    for i, t in enumerate(tensors):
        out[i, :len(t)] = t
    return out


class RAMAGCCollator:
    """Batch collator preserving all multi-stream graph tensors and timing."""
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, samples: list[StructuredPoseSample]) -> dict:
        lengths = torch.tensor([len(s.times) for s in samples], dtype=torch.long)
        
        batch = {
            "lengths": lengths,
            "timestamps": pad_time_tensor([torch.from_numpy(s.times).float() for s in samples]),
            "left_hand_nodes": pad_time_tensor([torch.from_numpy(s.left_hand_nodes).float() for s in samples]),
            "left_hand_edges": pad_time_tensor([torch.from_numpy(s.left_hand_edges).float() for s in samples]),
            "left_hand_adj": pad_time_tensor([torch.from_numpy(s.left_hand_adj).bool() for s in samples]),
            "left_hand_valid": pad_time_tensor([torch.from_numpy(s.left_hand_valid).bool() for s in samples]),
            "right_hand_nodes": pad_time_tensor([torch.from_numpy(s.right_hand_nodes).float() for s in samples]),
            "right_hand_edges": pad_time_tensor([torch.from_numpy(s.right_hand_edges).float() for s in samples]),
            "right_hand_adj": pad_time_tensor([torch.from_numpy(s.right_hand_adj).bool() for s in samples]),
            "right_hand_valid": pad_time_tensor([torch.from_numpy(s.right_hand_valid).bool() for s in samples]),
            "body_nodes": pad_time_tensor([torch.from_numpy(s.body_nodes).float() for s in samples]),
            "body_edges": pad_time_tensor([torch.from_numpy(s.body_edges).float() for s in samples]),
            "body_adj": pad_time_tensor([torch.from_numpy(s.body_adj).bool() for s in samples]),
            "body_valid": pad_time_tensor([torch.from_numpy(s.body_valid).bool() for s in samples]),
            "uids": [s.uid for s in samples],
            "texts": [s.text for s in samples]
        }
        
        if any(s.face_nodes is not None for s in samples):
            batch["face_nodes"] = pad_time_tensor([
                torch.from_numpy(s.face_nodes).float() if s.face_nodes is not None
                else torch.zeros((len(s.times), 18, 13), dtype=torch.float32)
                for s in samples
            ])
            batch["face_valid"] = pad_time_tensor([
                torch.from_numpy(s.face_valid).bool() if s.face_valid is not None
                else torch.zeros((len(s.times), 18), dtype=torch.bool)
                for s in samples
            ])
        else:
            batch["face_nodes"] = None
            batch["face_valid"] = None
            
        if self.tokenizer is not None:
            encoded = self.tokenizer([s.text for s in samples], padding=True, truncation=False, return_tensors="pt")
            labels = encoded["input_ids"].masked_fill(~encoded["attention_mask"].bool(), -100)
            batch["labels"] = labels
            
        return batch


class RAMAGCDataset(Dataset):
    """
    Standard PyTorch Dataset for RA-MAGC structured samples.
    Can be initialized with pre-extracted StructuredPoseSample items,
    or can extract samples on-the-fly from pose records.
    """
    def __init__(self, samples: list[StructuredPoseSample]):
        self.samples = samples
        self.lengths = [len(s.times) for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> StructuredPoseSample:
        return self.samples[index]
