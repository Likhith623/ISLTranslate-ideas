# RA-MAGC Architecture Implementation

**Model Name:** Reliability-Aware Multi-Rate Articulatory Graph-Conformer (RA-MAGC)  
**Task:** Continuous Gloss-Free Indian Sign Language to English Translation  
**Reference Document:** `ideas.md`

---

## 1. Why We Do NOT Use Stratified K-Fold

In this project and in peer-reviewed Sign Language Translation literature, **Stratified K-Fold is strictly avoided for three critical reasons**:

1. **Task Mismatch (Continuous Generation vs. Discrete Classification)**:
   - Stratified K-Fold is designed for single-label discrete classification (e.g., predicting 1 of $K$ classes), where every sample has an atomic categorical label $y \in \{0, \dots, C-1\}$.
   - Sign Language Translation (SLT) is **free-form sequence-to-sequence text generation**. Every target is an open-vocabulary English sentence of variable length. There is no categorical class label to stratify over.

2. **The Signer / Background Contamination Trap (Catastrophic Data Leakage)**:
   - In continuous sign datasets like ISLTranslate, dozens of clips originate from the same signer and recording session.
   - If you use random or stratified K-Fold across clips, clips from the **exact same signer in the exact same room with the exact same clothes** end up in both the training set and the test set.
   - The neural network simply memorizes the signer's face, body proportions, and idiosyncratic signing style rather than learning sign grammar. This yields artificially inflated test scores that completely collapse when tested on a new signer.

3. **Temporal Context Leakage**:
   - Continuous sign clips often represent consecutive 5-second windows of a single continuous lecture. Shuffling them randomly across folds leaks temporal context into the test set.

### What We Use Instead: `source-held-out` Partitioning
We enforce **Grouped Source-Disjoint Splitting**:
- All clips from any given parent video / signer (`source_id`) are assigned **exclusively to Train, Validation, or Test**.
- The Test set tests translation generalization on **completely unseen signers**.
- The codebase enforces this through cryptographic manifest hashes (`splits/source-v1/leakage.json`).

---

## 2. Dataset Paths Contract

All paths match the canonical repository layout in `configs/baseline.yaml`:

| Resource | Path |
| :--- | :--- |
| **Target Annotations** | `data/raw/ISLTranslate.csv` |
| **Pose Directory** | `data/raw/poses/` (`{uid}.pose`) |
| **Source Mapping** | `data/metadata/source_mapping.csv` |
| **Manifests** | `data/manifests/target.jsonl` |
| **Splits Directory** | `splits/source-v1/` (`splits.json`, `leakage.json`) |
| **Pretrained T5 Model** | `artifacts/models/flan-t5-small/` |
| **Feature Statistics** | `artifacts/stats/pose.json` |

---

## 3. Architecture Overview & Tensor Dimensions

```
                    Raw Keypoint Streams
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
Left Hand [B,T,21,11]   Right Hand [B,T,21,11]  Upper Body [B,T,6,8]
Left Edges [B,T,21,21,4] Right Edges [B,T,21,21,4] Body Edges [B,T,6,6,4]
     │                       │                       │
     └───────────┬───────────┘                       │
                 ▼                                   ▼
      Shared Hand Graph Encoder              Body Graph Encoder
      + Left/Right Laterality Embed           (Edge-Aware GAT)
                 │                                   │
                 ▼                                   ▼
          Hand Tokens [B,T,128]              Body Tokens [B,T,128]
                 └───────────────────┬───────────────┘
                                     ▼
                    Cross-Region Interaction Transformer
                                     │
                                     ▼
                      Reliability-Aware Gated Fusion
                                     │
                                     ▼
                  Full-Resolution Features [B, T, 256]
                                     │
                                     ▼
                 2x Multi-Scale Temporal CNNs (stride 1)
                     (Kernels: 3, 5, 5 d=2, 9 d=2)
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
        Fine Memory Z [B, T, 256]           Pairwise Aggregator (S=2)
         (100% frame resolution)           Coarse Tokens [B, ceil(T/2), 256]
                 │                                       │
                 │                                       ▼
                 │                         4-Block Conformer Backbone
                 │                        (Relative Elapsed-Time Bias)
                 │                                       │
                 └────── Local Detail Retrieval ─────────┘
                                     │
                                     ▼
                       Fused Sequence [B, ceil(T/2), 256]
                                     │
                                     ▼
                        Visual Bridge (LayerNorm + Linear)
                                     │
                                     ▼
                      FLAN-T5-small Encoder-Decoder
                                     │
                                     ▼
                             English Translation
```

---

## 4. Package Modules

- `config.py`: Complete hyperparameters and dataset path mappings.
- `graph_layers.py`: Edge-aware graph attention (`EdgeAwareGraphAttention`), masked attention pooling (`MaskedAttentionPooling`), shared hand encoder (`HandGraphEncoder`), body encoder (`UpperBodyGraphEncoder`), and facial encoder (`FaceSetEncoder`).
- `fusion.py`: Cross-region transformer (`RegionInteraction`) and reliability gating (`ReliabilityGate`).
- `temporal.py`: Multi-scale depthwise 1D temporal CNN (`MultiScaleTemporalBlock`), pairwise aggregator (`PairwiseAggregator`), Conformer (`ConformerBlock`), and fine detail retrieval (`FineDetailRetrieval`).
- `model.py`: Top-level `RAMAGCVisualEncoder` and `RAMAGCTranslator`.
- `dataset.py`: Structured dataset collation with bone edges, temporal padding masks, and `RAMAGCDataset`.

