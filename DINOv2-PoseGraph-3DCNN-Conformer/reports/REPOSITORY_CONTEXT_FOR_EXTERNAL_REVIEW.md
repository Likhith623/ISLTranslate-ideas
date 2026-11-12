# Repository Context for External Review & Continuity

**Project:** ISL-MEGA / CARP-S2T Continuous Sign Language Translation  
**Repository Path:** `DINOv2-PoseGraph-3DCNN-Conformer`  
**Current Version:** `0.2.0`  
**Date:** November 2026  
**Auditor:** Principal AI/ML Research Scientist & Systems Architect  

---

## 1. Executive Orientation

This document provides a complete, self-contained orientation for external research reviewers, collaborating scientists, or AI coding agents inspecting or extending this repository.

### What This Project Is:
An advanced multimodal Continuous Sign Language Translation (CSLT) research system designed for low-resource sign languages, specifically Indian Sign Language (ISL). The project was initiated as **ISL-MEGA** (a 10-stream research prototype combining DINOv2 visual features, a 66-point anatomical pose graph, 3D CNNs, Conformer temporal backbones, and FLAN-T5 text decoding). Following an exhaustive scientific audit, it is currently undergoing formal transition to **CARP-S2T** (*Confidence-Aware Anatomical Reliability-Preserving Sign-to-Text Network*).

---

## 2. Directory Architecture & File Map

```
DINOv2-PoseGraph-3DCNN-Conformer/
├── configs/
│   ├── ablation_v1_baseline.yaml      # Baseline ablation config
│   ├── research.yaml                  # Full 10-stream prototype (computationally unviable on local machines)
│   ├── research_v3_compact.yaml       # Recommended compact config (frozen DINOv2-Base + 66-pt pose)
│   ├── smoke.yaml                     # Fast synthetic smoke-test configuration
│   └── smoke_v2.yaml                  # Smoke test validating v2 cross-articulator graphs
├── docs/
│   ├── architecture.md                # Mathematical specification of prototype v0.2.0
│   ├── data_contract.md               # Strict schema for NPZ tensors, bounding boxes, landmarks
│   ├── implementation_status.md       # Implementation vs. theoretical progress tracker
│   ├── references.md                  # Core bibliography (DINOv2, Conformer, T5, MediaPipe)
│   └── research_protocol.md           # Formal hypothesis and experimental protocols
├── reports/
│   ├── 01_REPOSITORY_INTELLIGENCE_REPORT.md      # Complete codebase inventory & tensor lifecycle
│   ├── 02_DATASET_AND_TRAINING_AUDIT.md          # Dataset provenance, 66-pt mapping, DSU leakage
│   ├── 03_SCIENTIFIC_NOVELTY_INVESTIGATION.md    # 2020–2026 SOTA, critique of EvoHARE-Net
│   ├── 04_FINAL_ARCHITECTURE_SPECIFICATION.md    # Definitive CARP-S2T mathematical architecture
│   ├── 05_IMPLEMENTATION_BLUEPRINT.md            # File-by-file refactoring and migration blueprint
│   ├── 06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md   # Baselines, metrics, A0–A12 ablations, deaf rubric
│   └── 07_EXECUTIVE_DECISION_REPORT.md           # Executive verdicts and minimum viable path
├── scripts/
│   ├── convert_video_pose_npz.py      # Preprocesses MP4 videos and MediaPipe poses to unified NPZ
│   ├── create_synthetic_dataset.py    # Generates synthetic data fixtures for testing
│   ├── evaluate.py                    # Evaluates BLEU-1..4 and ROUGE-L on test set
│   ├── make_splits.py                 # Generates source-held-out and signer-disjoint splits via DSU
│   ├── pretrain_visual.py             # Pretraining script for visual adapters
│   ├── train.py                       # Master end-to-end training script (AMP, DDP, validation)
│   └── validate_data.py               # Checks dataset integrity and detects corrupted frames
├── src/isl_mega/
│   ├── __init__.py                    # Public API exports
│   ├── checkpoint.py                  # Atomic checkpoint saving and EMA weights
│   ├── cli.py                         # CLI wrapper for training and evaluation
│   ├── config.py                      # Dataclass configuration schemas and validation
│   ├── data.py                        # Dataset classes, collate functions, and DSU splitting
│   ├── decoder.py                     # FLAN-T5 cross-attention bridge and beam search
│   ├── fingerspelling.py              # BiGRU + CTC head for rapid fingerspelling sequences
│   ├── fusion.py                      # Multimodal reliability gating and lag bias
│   ├── losses.py                      # Composite multi-task loss functions
│   ├── metrics.py                     # BLEU, ROUGE, and sacrebleu metric wrappers
│   ├── model.py                       # Master neural orchestrator
│   ├── motion.py                      # 3D CNN motion pathway (disabled by default)
│   ├── planning.py                    # Latent Goal Planner with learnable queries
│   ├── pose.py                        # 66-point anatomical ST-GCN + cross-articulator gate
│   ├── pretraining.py                 # Masked sequence modelling pretraining heads
│   ├── temporal.py                    # Multi-scale temporal stem + Conformer backbone
│   └── vision.py                      # DINOv2 appearance backbone and region pooling
├── tests/
│   ├── test_architecture.py           # 55 unit tests covering individual modules
│   └── test_novel_v2.py               # Regression tests for pose gating, torso norm, planner
├── pyproject.toml                     # Build configuration and dependencies
├── AUDIT_REPORT.md                    # Historical audit record
└── README.md                          # Main project documentation
```

---

## 3. Key Findings & Critical Fixes Applied (v0.2.0)

1. **The Cross-Articulator Gate (Previously P0 Bug):**  
   In v0.1.0, `CrossArticulatorGraph.gate` was initialized to $0.0$, causing $\tanh(\text{gate}) = 0.0$. This completely multiplied out the cross-articulator residual connection and prevented gradient flow on step 1. In v0.2.0, this was fixed to `gate_init=0.1`, passing all regression tests.
2. **Dead 3D Motion Pathway:**  
   `motion.py` is present and functional, but set to `motion.enabled: false` across all active training configs due to extreme VRAM consumption.
3. **Hardware Infeasibility of `research.yaml`:**  
   `configs/research.yaml` specifies 4 $\times$ DINOv2-Large crops ($518\times518$) with FLAN-T5-XL, consuming over $38\text{ GB}$ VRAM per sample. It cannot execute on standard academic hardware or the local Apple M2 Air ($8\text{ GB}$).
4. **DSU Graph Percolation Hazard:**  
   Joint grouping by `source_id` and `signer_id` via Disjoint-Set Union can cause transitive percolation into a giant connected component ($> 60\%$ of data), starving validation/test sets. Two explicit evaluation protocols are established: Protocol 1 (Source-Disjoint) and Protocol 2 (Signer-Independent).

---

## 4. Canonical Dataset Registry and Anatomical Mapping

### Datasets:
- **ISLTranslate (ACL Findings 2023):** 31,107 clips, 291 gold human-validated test utterances, source-held-out splits.
- **iSign v1.1 (ACL Findings 2024):** 84,204 utterances (~112 hrs), rich semantic labels, CC-BY-NC-SA-4.0.
- **GloSL-ISL (LREC / PyPI 2024–2025):** 126,846 samples (269.03 hrs), pose-first annotations, CC-BY-4.0.
- **ISL-FS-Dense (ACL WSLP 2025):** 1,308 gold manual and 11,700 auto segments with character CTC timestamps.
- **ISH-NEWS (Nature Sci Rep 2026):** 4,222 continuous news broadcast videos (~48 hrs).

### The Anatomical 66-Landmark Layout (MediaPipe Holistic Subset):
- **Upper Body (6 pts):** Left/Right Shoulder (11, 12), Left/Right Elbow (13, 14), Left/Right Wrist (15, 16).
- **Left Hand (21 pts):** Complete MediaPipe hand kinematic chain (indices 0 to 20).
- **Right Hand (21 pts):** Complete MediaPipe hand kinematic chain (indices 0 to 20).
- **Face Contours (18 pts):** Eyes, eyelids, eyebrows, and lip outer/inner boundaries for mouthing detection.

---

## 5. The Winning Architecture: CARP-S2T

The scientific consensus of this audit is to replace component stacking with **CARP-S2T**:
- **Evidential Reliability Fusion (ERF):** Parameterizes modality confidence as a Dirichlet distribution, outputting closed-form belief masses and epistemic uncertainty $u(t) = M / S(t)$.
- **Anatomical Symmetry & Torso Normalization:** Bilateral mirror-canonicalization reflects the left hand into the right hand coordinate space, enabling full parameter sharing.
- **Linear Temporal Modeling $\mathcal{O}(T)$:** Replaces quadratic Conformer self-attention with a Bidirectional Selective State-Space Backbone (Bi-S6 / Sign-Mamba).
- **Region-LoRA Appearance:** Uses a single shared frozen DINOv2-Base with lightweight low-rank adapters ($r=16$) for subtle hand gestures and mouthing.

---

## 6. Guide to Comprehensive Audit Reports

Detailed documentation is available in the `reports/` directory:
- [01_REPOSITORY_INTELLIGENCE_REPORT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/01_REPOSITORY_INTELLIGENCE_REPORT.md)
- [02_DATASET_AND_TRAINING_AUDIT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/02_DATASET_AND_TRAINING_AUDIT.md)
- [03_SCIENTIFIC_NOVELTY_INVESTIGATION.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md)
- [04_FINAL_ARCHITECTURE_SPECIFICATION.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/04_FINAL_ARCHITECTURE_SPECIFICATION.md)
- [05_IMPLEMENTATION_BLUEPRINT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/05_IMPLEMENTATION_BLUEPRINT.md)
- [06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md)
- [07_EXECUTIVE_DECISION_REPORT.md](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/07_EXECUTIVE_DECISION_REPORT.md)

---

## 7. How to Execute Tests & Verify Local Environment

```bash
# 1. Verify code syntax across all files
python -m compileall -q src scripts tests configs

# 2. Run unit and regression tests (in environment with torch & pytest)
pytest -q

# 3. Test synthetic dataset generation
python scripts/create_synthetic_dataset.py --output data/synthetic_fixture.npz

# 4. Run compact smoke test forward pass
python -m isl_mega.cli train --config configs/smoke.yaml --dry-run
```
