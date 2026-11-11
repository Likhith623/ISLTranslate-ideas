# Milestone 2: Model Training (1D-CNN + Shallow Transformer)

Train a word-level ASL sign classifier on the extracted keypoint embeddings.
Target: highest possible accuracy on Mac hardware first. A smaller Android
model will be derived separately via knowledge distillation later.

---

## Dataset Summary (from Milestone 1)

| Split | Videos | Classes | Avg/class |
|-------|--------|---------|-----------|
| Train | 2,838  | 100     | 28.4      |
| Val   | 359    | 100     | 3.6       |
| Test  | 346    | 100     | 3.5       |

**Input shape per sample:** `(60, 255)` — 60 frames × (85 keypoints × 3 coords flattened)
**Output:** 1 class label from 100 possible ASL glosses

---

## Architecture: Hybrid 1D-CNN + Transformer

> [!IMPORTANT]
> **The PyTorch Conv1d Shape Gotcha**
> `nn.Linear` and `nn.TransformerEncoder` expect tensors in the format
> `(batch, sequence_length, features)`. But `nn.Conv1d` expects
> `(batch, channels, sequence_length)` — the feature and sequence axes are **swapped**.
>
> Passing `(batch, 60, 256)` into `Conv1d` would make PyTorch think you have
> 60 channels and a sequence of length 256 — completely wrong.
>
> **Fix:** explicitly `.permute(0, 2, 1)` before every CNN block and
> `.permute(0, 2, 1)` again to restore the original shape before the Transformer.

```
Input: (batch, 60, 255)    ← 60 time-steps, 255 features per step
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  BLOCK 1: Input Projection                                 │
│  Linear(255 → 256)  +  LayerNorm(256)  +  GELU            │
│  Output: (batch, 60, 256)    ← (batch, seq, features)     │
└────────────────────────────────────────────────────────────┘
         │
         │  .permute(0, 2, 1)          ← REQUIRED before Conv1d
         ▼  Shape becomes: (batch, 256, 60)   (batch, channels, seq)
         │
┌────────────────────────────────────────────────────────────┐
│  BLOCK 2: Inception-Style Multi-Scale 1D-CNN               │
│                                                            │
│  Same input (batch, 256, 60) fed into THREE branches       │
│  simultaneously (in parallel, NOT sequentially):           │
│                                                            │
│  ┌─ Branch A: Conv1d(256→256, k=3, p=1) + BN + GELU ─┐   │
│  │  Detects: sharp rapid flicks (3-frame window)        │   │
│  │  Output:  (batch, 256, 60)                          │   │
│  │                                                     │   │
│  ├─ Branch B: Conv1d(256→256, k=5, p=2) + BN + GELU ─┤   │
│  │  Detects: medium-speed arm arcs (5-frame window)    │   │
│  │  Output:  (batch, 256, 60)                          │   │
│  │                                                     │   │
│  └─ Branch C: Conv1d(256→256, k=7, p=3) + BN + GELU ─┘   │
│     Detects: slow shoulder rolls (7-frame window)          │
│     Output:  (batch, 256, 60)                              │
│                                                            │
│  Element-wise Sum (A + B + C) → (batch, 256, 60)          │
│  Residual Add  →  input + sum_output → Dropout(0.1)       │
│                                                            │
│  ⚠ All three branches see the SAME input.                 │
│    They learn DIFFERENT motion patterns independently.     │
│    Summing (not stacking) keeps the channel dim at 256     │
│    so the residual add works cleanly.                      │
│  Output: (batch, 256, 60)    ← (batch, channels, seq)     │
└────────────────────────────────────────────────────────────┘
         │
         │  .permute(0, 2, 1)          ← REQUIRED before Transformer
         ▼  Shape becomes: (batch, 60, 256)   (batch, seq, features)
         │
┌────────────────────────────────────────────────────────────┐
│  BLOCK 3: Shallow Transformer Encoder (4 layers)           │
│                                                            │
│  Positional Encoding (sinusoidal, fixed — no extra params) │
│  ×4 TransformerEncoderLayer:                               │
│    - d_model          = 256                                │
│    - nhead            = 8   (8 attention heads)            │
│    - dim_feedforward  = 512                                │
│    - dropout          = 0.1                                │
│    - activation       = 'gelu'                             │
│    - norm_first       = True  (Pre-LN, more stable)       │
│                                                            │
│  Output: (batch, 60, 256)    ← (batch, seq, features)     │
└────────────────────────────────────────────────────────────┘
         │
         ▼  Global Average Pool over the time (seq) dimension
         │  torch.mean(x, dim=1)
         │  Shape: (batch, 256)
         ▼
┌────────────────────────────────────────────────────────────┐
│  CLASSIFIER HEAD                                           │
│  LayerNorm(256)                                            │
│  Dropout(0.3)                                              │
│  Linear(256 → 512)  +  GELU                               │
│  Dropout(0.2)                                              │
│  Linear(512 → 100)                                         │
│  Output: (batch, 100)  ← raw logits, one per gloss        │
└────────────────────────────────────────────────────────────┘
```

**Estimated model size:** ~5M parameters (~20 MB) — optimised for Mac accuracy,
not Android size. A distilled smaller model will be built in a later milestone.

---

## Parameter Count Breakdown

| Component | Parameters |
|---|---|
| Input projection Linear(255→256) | ~65K |
| 3× Conv1d blocks (k=3,5,7) | ~1.97M |
| 4× Transformer layers (d=256, ffn=512, h=8) | ~2.1M |
| Classifier head (256→512→100) | ~183K |
| **Total** | **~4.3M** |

---

## New Files

```
Sing_AR/
└── model/
    ├── dataset.py      ← PyTorch Dataset + DataLoader factory
    ├── model.py        ← ASLTransformer nn.Module
    ├── train.py        ← Training loop
    ├── evaluate.py     ← Test-set evaluation + confusion matrix
    ├── class_map.json  ← Auto-generated: gloss → label_id
    └── checkpoints/
        └── best_model.pt
```

---

### [NEW] `model/dataset.py`

- Walks `preprocessing/output/{split}/{gloss}/` to find all `.npy` files
- Flattens `(60, 85, 3)` → `(60, 255)` at load time
- Builds a **sorted, alphabetical `gloss → label_id`** map, saves to `class_map.json`
- **Train-only augmentations:**
  - Gaussian noise jitter `(σ=0.005)` — simulates camera noise
  - Random temporal shift `(±3 frames, wrap-around)` — simulates signing speed variation
- Returns `(tensor, label)` pairs

---

### [NEW] `model/model.py`

The `ASLTransformer` `nn.Module` implementing the architecture above.

Key implementation details in the forward pass:

```python
# Block 1 → Block 2: permute before CNN
x = x.permute(0, 2, 1)           # (batch, 60, 256) → (batch, 256, 60)

# Block 2: Inception-style — parallel branches on the SAME input
residual = x
branch_a = self.conv3(x)         # (batch, 256, 60) — kernel=3
branch_b = self.conv5(x)         # (batch, 256, 60) — kernel=5
branch_c = self.conv7(x)         # (batch, 256, 60) — kernel=7
x = branch_a + branch_b + branch_c   # element-wise sum
x = self.dropout(x + residual)   # residual add

# Block 2 → Block 3: permute back before Transformer
x = x.permute(0, 2, 1)           # (batch, 256, 60) → (batch, 60, 256)
x = self.pos_encoding(x)
x = self.transformer(x)
```

---

### [NEW] `model/train.py`

| Setting | Value | Reason |
|---|---|---|
| Batch size | 64 | Larger batch = more stable gradients |
| Max epochs | 150 | Early stopping will trigger before this |
| Optimizer | AdamW | Best for Transformers |
| Learning rate | 3e-4 | Standard for AdamW + Transformer |
| Weight decay | 1e-4 | L2 regularisation |
| Scheduler | CosineAnnealingWarmRestarts | Warm restarts escape local minima |
| Loss | CrossEntropyLoss + label_smoothing=0.1 | Prevents overconfidence on small data |
| Early stopping patience | 20 epochs | Generous, avoids stopping too soon |

**Console output per epoch:**
```
Epoch  12/150 │ Loss: 2.3147 │ Val Acc: 63.2% │ LR: 2.8e-4 │ ETA: 4m32s  ⬆ best
Epoch  13/150 │ Loss: 2.2891 │ Val Acc: 61.7% │ LR: 2.6e-4 │ ETA: 4m18s
```

---

### [NEW] `model/evaluate.py`

Runs after training on the locked **test split**. Reports:
- **Top-1 Accuracy** — exact correct word
- **Top-5 Accuracy** — correct word in top 5 predictions
- **Per-class accuracy table** — all 100 words sorted by accuracy
- **Confusion matrix** — saved as a `.png` heatmap

---

## Verification Plan

### Automated
- Shape assertion after every block in a single forward pass before training starts
- Assert `class_map.json` has exactly 100 entries

### Training targets
- Val accuracy > 60% by epoch 30 (sanity signal training is converging)
- **Final test Top-1 accuracy target: > 75%**
- **Final test Top-5 accuracy target: > 92%**

### Post-training
- Load `best_model.pt` and run inference on one sample from each split
- Confirm output shape `(1, 100)` and `argmax` maps to a known gloss
