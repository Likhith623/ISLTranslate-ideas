# Phase 1 Verification: Critical Mathematical and Architectural Issues

**Repository:** `DINOv2-PoseGraph-3DCNN-Conformer` (ISL-MEGA v0.2.0)  
**Report Type:** Mathematical and Theoretical Verification  
**Audit Date:** November 2026  
**Auditor:** Principal Research Scientist & Deep Learning Systems Architect  

---

## 1. Deep Mathematical Investigation of Dirichlet Evidential Fusion

Previous reports proposed replacing heuristic gating with **Dirichlet Evidential Uncertainty Learning** over modality reliability. We independently verify the mathematical coherence of this proposal against formal evidential learning theory (Sensoy et al., NeurIPS 2018; Ulmer et al., ICLR 2023).

### Independent Analysis of the 7 Foundational Questions:

#### Q1: What exactly does each Dirichlet variable represent?
- **Analysis:** In Dirichlet Evidential Deep Learning, the network predicts concentration parameters $\alpha = (\alpha_1, \dots, \alpha_K)$ defining a Dirichlet density $\text{Dir}(p \mid \alpha)$ over a categorical distribution $p$ on the unit probability simplex $\Delta^{K-1}$.
- **Mathematical Incoherence:** Applying this to multimodal sign fusion assumes that at each video frame $t$, the modalities (vision, pose, fingerspelling) form mutually exclusive categories where only *one* modality is "correct" or "active". This is fundamentally false for sign language: visual appearance and pose kinematics are **complementary simultaneous signals**, not mutually exclusive classes.

#### Q2: Is uncertainty representing class uncertainty or modality reliability?
- **Analysis:** Standard evidential deep learning models epistemic uncertainty regarding *label classification* (e.g. "is this sign APPLE or ORANGE?"). In the proposed blueprint, the Dirichlet distribution was repurposed to estimate *sensor failure* (e.g. "is the pose detector corrupted?"). Repurposing classification evidence to sensor reliability without an underlying generative sensor-noise model conflates predictive ambiguity with physical sensor degradation.

#### Q3: What supervision target is used?
- **The Core Flaw:** The proposed blueprint specified:
  $$\mathcal{L}_{\text{evidential}} = \sum_{m=1}^M \left( y_m - \frac{\alpha_m}{S} \right)^2 + \frac{\alpha_m (S - \alpha_m)}{S^2 (S + 1)} + \lambda_{\text{KL}} \text{KL}\left[\text{Dir}(\alpha) \,\|\, \text{Dir}(\mathbf{1})\right]$$
  Where $y_m$ was denoted as `ground_truth_reliability`.
- **Reality:** **No continuous sign language dataset (ISLTranslate, iSign, GloSL-ISL) contains ground-truth reliability labels $y_m$.** There is no human or automated annotation indicating what fraction of evidence belongs to RGB versus pose. Without ground-truth targets $y_m$, this objective cannot be computed during supervised training.

#### Q4: How are observations of missing modalities handled?
- **Analysis:** In `src/isl_mega/fusion.py`, missing modalities are handled via binary masking (`src_key_padding_mask`). If a hand is off-screen, its features are zeroed and masked. In an evidential formulation, an absent modality should yield zero evidence ($e_m = 0 \implies \alpha_m = 1$). However, if all modalities have missing frames, the Dirichlet prior collapses to uniform $\text{Dir}(\mathbf{1})$, yielding maximum uncertainty $u=1$. How the text decoder conditions on an all-uncertain latent state was left mathematically undefined.

#### Q5: Does the formulation guarantee that unreliable modalities are downweighted?
- **Analysis:** **No.** Because the evidence network is trained end-to-end without direct supervision on $e_m$, gradient descent will simply adjust $e_m$ to minimize text translation cross-entropy. If the pose graph produces convenient spurious correlations (e.g. signer clothing artifacts or height idiosyncrasies), the evidence network will allocate *high* evidence to corrupted pose signals. It does not possess an intrinsic guarantee of downweighting corrupted sensors.

#### Q6: Is the uncertainty properly calibrated?
- **Analysis:** Uncalibrated. Evidential neural networks are notorious for suffering from severe overconfidence on out-of-distribution inputs unless regularized with extensive negative sampling or synthetic corruption augmentation during training.

#### Q7: Is the claimed degradation bound mathematically established?
- **Verdict:** **UNSUBSTANTIATED.** The earlier report asserted a *"mathematically provable degradation bound under landmark corruption"*. No proof, theorem, or mathematical derivation exists in the repository or the literature. It was an unsubstantiated theoretical hypothesis.

---

## 2. Coordinate Systems, Torso Normalization & Anatomical Symmetry

### A. Coordinate System Discrepancies and Incompatible Spaces
The repository handles multiple distinct coordinate spaces that must remain strictly registered:
1. **Raw Video Pixel Space:** $[0, W] \times [0, H]$ in variable original video resolutions ($720\text{p}, 1080\text{p}$).
2. **Letterboxed Canvas Space:** $[0, 1] \times [0, 1]$ on a square canvas ($S \times S$). Handled correctly in `data.py` via `letterbox_frames` and affine transformations `transform_points_to_letterbox`.
3. **Torso-Relative Coordinate Space:** 
   In `src/isl_mega/pose.py`:
   $$\tilde{p}_i = \frac{p_i - p_{\text{anchor}}}{\text{scale}}$$
   Where $p_{\text{anchor}}$ is shoulder midpoint for body, wrist for hands, and centroid for face.
   **Critical Issue:** When `CrossArticulatorGraph` computes relative distances between hand tips and facial lips:
   $$rel = \frac{p_{\text{hand}} - p_{\text{face}}}{\text{scale}}$$
   It uses the *letterboxed global coordinates* $xy$, NOT the local centered coordinates! This is kinematically valid, but the scale normalizer `estimate_torso_scale` defaults to $1.0$ when shoulders are missing, causing sudden scale jumps by a factor of $2.5\times$ between consecutive frames.

### B. The Bilateral Mirror-Canonicalization Paradox in Sign Language
Earlier reports proposed reflecting the left hand across the sagittal plane:
$$\tilde{x}_{\text{left, canonical}} = -\tilde{x}_{\text{left}}$$
to allow complete parameter sharing between left and right hands.

#### Linguistic and Semantic Falsification:
In sign language linguistics (Stokoe, 1960; Sandler & Lillo-Martin, 2006; ISL Grammar Principles):
1. **Symmetric Signs (Type 1):** Both hands move with identical configurations and symmetrical kinematics (e.g. "BOOK", "CHANGE"). Mirror reflection is valid here.
2. **Asymmetric Signs (Type 2 & Type 3):** One hand is the **dominant articulator** (producing lexical movement and shape), while the other hand is the **base / passive anchor** (providing spatial reference, e.g. "WRITE", "KNIFE", "RULE", "STOP").
- **The Paradox:** If left-hand coordinates are mirrored and passed into a weight-shared encoder without preserving active vs. base role indicators, the network **cannot distinguish between an active left hand and an active right hand acting upon a passive base hand**. 
- In `src/isl_mega/pose.py` lines 393–396, the codebase partially mitigates this by adding a learned `hand_side_emb` ($d=512$). However, complete geometric mirroring without dominant-hand conditioning degrades semantic translation on asymmetric two-handed signs.

---

## 3. Temporal Sequence Modeling: Conformer vs. State-Space (Mamba)

### Theoretical Scaling Comparison:

| Property | Current Conformer Backbone (`temporal.py`) | Proposed Selective State-Space (Sign-Mamba) |
| :--- | :--- | :--- |
| **Computational Complexity** | $\mathcal{O}(T^2 \cdot D + T \cdot D \cdot K_{\text{conv}})$ | $\mathcal{O}(T \cdot D \cdot N_{\text{state}})$ (Strictly Linear) |
| **KV-Cache / Recurrent Memory** | Quadratic activation footprint in $T$ | Constant state memory $\mathcal{O}(D \cdot N_{\text{state}})$ |
| **Temporal Inductive Bias** | Explicit relative-time bias (`RelativeTimeBias`) + depthwise conv | Continuous-time ODE discretisation ($\Delta, A, B, C$) |
| **Hardware Compatibility** | Native PyTorch, runs on CPU, MPS, CUDA | Requires compiled CUDA kernels (`mamba-ssm`); slow on CPU/MPS without fallback |
| **Empirical Status in Codebase**| **Implemented, active, passes synthetic tests** | **Zero lines of code implemented in repository** |

### Why Mamba Cannot Be Assumed Superior Without Experimentation:
In continuous sign language, fine fingerspelled characters and rapid transitions occur within 2–3 frames ($80\text{–}120\text{ ms}$). While selective state-space models excel at compressing long histories, state-space compression can act as a low-pass filter, potentially smoothing out subtle high-frequency finger configurations that Conformer self-attention preserves via direct pair-wise queries. A controlled empirical comparison is mandatory before replacing the Conformer.

---

## 4. Decoder Behavior and the 9-Loss Multi-Task Dilemma

### A. T5 Conditioning Architecture: Encoder Input vs. Cross-Attention
In `src/isl_mega/decoder.py`:
- The model projects visual memory `(B, T, D)` and latent plan `(B, K, D)` to `t5_dim` and passes them via `inputs_embeds` into **T5's Encoder**.
- T5's Encoder runs full bidirectional self-attention over the multimodal tokens.
- T5's Decoder then cross-attends to the output of T5's Encoder.
- **Verdict:** This is architectural dual-encoding: the video is first encoded by Conformer-12, then encoded a second time by T5-Encoder-12, before entering T5-Decoder-12! This incurs massive parameter and computational redundancy ($24\text{ layers of encoder processing}$).

### B. The 9-Loss Multi-Task Instability
`scripts/train.py` computes:
$$\mathcal{L} = \mathcal{L}_{\text{trans}} + w_1 \mathcal{L}_{\text{ctc}} + w_2 \mathcal{L}_{\text{spot}} + w_3 \mathcal{L}_{\text{ot}} + w_4 \mathcal{L}_{\text{cm}} + w_5 \mathcal{L}_{\text{div}} + w_6 \mathcal{L}_{\text{ord}} + w_7 \mathcal{L}_{\text{ent}} + w_8 \mathcal{L}_{\text{cf}}$$
1. **Gradient Interference:** $\mathcal{L}_{\text{ent}}$ forces stream weights toward uniform entropy, while $\mathcal{L}_{\text{trans}}$ forces stream weights toward the single most informative modality. These gradients point in opposing directions.
2. **Sinkhorn OT Instability:** In `entropic_semantic_ot`, when sequence lengths vary widely or embeddings have small norm, $30$ Sinkhorn iterations produce numerical underflow in FP16/BF16, requiring artificial clamping (`clamp_min(0)`).
3. **Counterfactual Margin Regularizer:** Computes a full forward pass on batch-rolled inputs to calculate KL divergence. This doubles the decoder memory footprint per training step.

