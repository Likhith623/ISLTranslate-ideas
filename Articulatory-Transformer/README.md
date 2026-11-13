# Articulatory-Bound Spatiotemporal Transformer (AB-ST)

> **A World-Class, Q1-Standard Architecture for Continuous Gloss-Free Indian Sign Language Translation (ISLT)**

---

## 1. Architectural Highlights

AB-ST resolves the critical structural limitations of prior sign language translation models (such as ISL-MEGA / DINOv2-PoseGraph-3DCNN-Conformer) by grounding sequence modeling directly into the phonetic and linguistic principles of Sign Language Phonology (Brentari 1998, Sandler & Lillo-Martin 2006, Cormier et al. 2012):

1. **Torso-Midline Normalization & Dynamic Dominance Routing (`dominance.py`):**
   - Normalizes 3D coordinates using the dynamic shoulder midpoint $p_{\text{mid}}(t) = (p_{42}(t) + p_{43}(t))/2$ and EMA shoulder width $\bar{w}(t)$.
   - Computes kinetic energy $E = \sum_t \|\Delta p\|^2$ across hands to route articulators into **Dominant (DH)** vs. **Base (BH)** channels with learned role embeddings, resolving left/right-handed and asymmetric signing ambiguity.
2. **Deformable Keypoint-Guided Patch Binding (`binding.py`):**
   - Replaces the 1D global CLS bottleneck with a spatial feature grid ($16 \times 16 \times C$) from DINOv2.
   - Bilinear grid sampling at deformable landmark offsets $\Delta p_j$ binds local visual texture (handshape morphology, eye gaze, lip contours) directly to keypoint kinematics.
3. **Factored 4-Articulator Token Representation (`articulators.py`):**
   - Partitions 66 canonical landmarks into 4 distinct semantic channels per frame: **Dominant Hand (21)**, **Base Hand (21)**, **Face/Mouthing (18)**, and **Torso Anchor (6)**.
   - Produces $[B, T, 4, D]$ without premature spatio-temporal pooling.
4. **Axial Spatio-Temporal Conformer (`temporal.py`):**
   - Interleaved $4 \times 4$ spatial self-attention (modeling bilaterial contact and signing space) with full-resolution temporal Conformer blocks ($T$ frames).
   - Dedicated fingerspelling CTC auxiliary head branching from the Dominant Hand stream.
5. **Articulatory Q-Former Resampler (`resampler.py`):**
   - $M = 32$ learned linguistic query tokens cross-attend to the $4T$ spatio-temporal tokens, condensing continuous sign videos into fixed-length semantic prompt embeddings feeding FLAN-T5 or modern LLMs.
6. **Canonical MediaPipe 543 -> 66 Adapter (`adapter_543_to_66.py`):**
   - Mathematically verified mapping from MediaPipe Holistic 543 landmarks to 66 canonical joints with zero slice-shift bugs.

---

## 2. Directory Structure

```
Articulatory-Transformer/
├── configs/
│   ├── base.yaml              # Default full training configuration
│   └── smoke.yaml             # Fast verification / testing configuration
├── docs/
│   └── ARCHITECTURE.md        # Mathematical specification & proofs
├── scripts/
│   ├── convert_pose_543_to_66.py # Offline pose converter & validator
│   ├── train.py               # Production training script
│   └── evaluate.py            # Evaluation & metric calculation (BLEU-1..4, ROUGE-L)
├── src/
│   └── articulatory_transformer/
│       ├── __init__.py
│       ├── adapter_543_to_66.py   # Canonical 543 -> 66 keypoint adapter
│       ├── articulators.py        # 4-Articulator token grouper
│       ├── binding.py             # Deformable keypoint-patch cross-attention
│       ├── config.py              # Strongly-typed configuration schema
│       ├── data.py                # Dataset loaders & preprocessing
│       ├── decoder.py             # LLM / T5 cross-attention adapter
│       ├── dominance.py           # Kinematic dominance routing & torso centering
│       ├── losses.py              # Translation CE + CTC + Modality Dropout
│       ├── model.py               # Full end-to-end AB-ST model wrapper
│       ├── resampler.py           # Articulatory Q-Former resampler
│       ├── temporal.py            # Axial Spatio-Temporal Conformer
│       └── vision.py              # DINOv2 patch extractor & feature cache
├── tests/
│   ├── test_adapter_543_to_66.py
│   ├── test_dominance.py
│   └── test_model_structure.py
├── pyproject.toml
└── README.md
```

---

## 3. Quick Start

### Pose Mapping & Validation
```bash
python scripts/convert_pose_543_to_66.py --input-dir /path/to/holistic_543 --output-dir /path/to/canonical_66
```

### Running Verification Tests
```bash
python -m unittest discover tests/
```

