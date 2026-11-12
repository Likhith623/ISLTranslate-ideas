# Final Architecture Specification: CARP-S2T (Confidence-Aware Anatomical Reliability-Preserving Sign-to-Text Network)

**Repository:** `DINOv2-PoseGraph-3DCNN-Conformer`  
**Document Code:** `04_FINAL_ARCHITECTURE_SPECIFICATION`  
**Status:** Canonical Architectural Recommendation for Q1 Submission  
**Authors:** ISL-MEGA Research & Engineering Group

---

## 1. Architectural Identity and Core Scientific Rationale

We formally specify **CARP-S2T** (*Confidence-Aware Anatomical Reliability-Preserving Sign-to-Text Network*), a compact, mathematically rigorous neural architecture engineered for continuous sign language translation under realistic visual corruptions, landmark occlusions, and signer variations.

### Why CARP-S2T Replaces the Prototype and Rejected Candidates:
1. **Elimination of Parameter Bloat:** The prototype stacked 4 unconstrained DINOv2-Large crops, a 3D CNN, a 12-layer Conformer, and FLAN-T5-XL ($\approx 3.5\text{B parameters}$, $> 38\text{ GB}$ VRAM per sample). CARP-S2T condenses this into a shared frozen DINOv2-Base with parameter-efficient Region-LoRA adapters ($r=16$), an anatomical 66-point graph, and a linear-time selective state-space backbone ($\approx 185\text{M trainable parameters}$, runnable on a single 16–24 GB GPU).
2. **Rejection of Biological Plasticity (EvoHARE-Net):** Rather than using non-falsifiable "evolutionary module growth" that causes catastrophic forgetting and non-deterministic computation graphs, CARP-S2T introduces **Dirichlet Evidential Uncertainty Fusion**, a closed-form, mathematically provable mechanism that dynamically reweights modalities based on observed sensor degradation.
3. **Linear Temporal Complexity $\mathcal{O}(T)$:** Replaces quadratic Conformer self-attention with a **Bidirectional Selective State-Space Model (Bi-S6 / Sign-Mamba)**, maintaining global temporal context with constant memory per step.

---

## 2. Complete System Architecture and Information Flow

```mermaid
flowchart TD
    subgraph Inputs ["1. Multimodal Observations"]
        V_frame["Raw RGB Video Frame: (B, T, 3, 224, 224)"]
        V_hand["Dominant Hand Crop: (B, T, 3, 112, 112)"]
        P_raw["MediaPipe Pose Landmarks: (B, T, 66, 3)"]
    end

    subgraph Preprocessing ["2. Geometric Normalization"]
        P_raw --> TorsoNorm["Torso Scale & Origin Normalization"]
        TorsoNorm --> MirrorCanon["Bilateral Mirror-Canonicalization\n(Left Hand mapped to Right Hand space)"]
        MirrorCanon --> P_norm["Normalized Landmarks: (B, T, 66, 3)"]
    end

    subgraph Feature_Extractors ["3. Specialized Structural Encoders"]
        V_frame & V_hand --> DINO_LoRA["Shared Frozen DINOv2-Base (768d)\n+ Region-LoRA Adapters (r=16)"]
        P_norm --> STGCN["Anatomical 4-Region ST-GCN\n(Upper Body, Shared Hands, Face)"]
        STGCN --> CrossArt["Cross-Articulator Message Passing\n(Learned Gate init=0.1)"]
        P_norm --> FSEncoder["Fingerspelling Lexical BiGRU (256d)"]
        
        DINO_LoRA --> F_vis["Visual Tokens: (B, T, D_m)"]
        CrossArt --> F_pose["Pose Tokens: (B, T, D_m)"]
        FSEncoder --> F_fs["Fingerspelling Tokens: (B, T, D_m)"]
        FSEncoder -.-> CTC_Head["Auxiliary Character CTC Head"]
    end

    subgraph Evidential_Fusion ["4. Evidential Reliability Fusion (ERF)"]
        P_norm & V_frame --> QualExtract["Local Quality Estimator\n[Confidence, Jitter, Blur, Illumination]"]
        QualExtract --> DirichletHead["Dirichlet Evidential Density Network\nOutputs alpha_m > 1"]
        DirichletHead --> ExpectedWeights["Belief Masses: b_m & Uncertainty: u"]
        F_vis & F_pose & F_fs & ExpectedWeights --> ModalityWeighing["Dynamic Evidential Gating"]
        ModalityWeighing --> FusedTokens["Fused Multimodal Sequence: (B, T, D_m)"]
    end

    subgraph Temporal_Core ["5. Bidirectional Selective State-Space Backbone"]
        FusedTokens --> BiMamba["6-Layer Bidirectional S6 (Sign-Mamba)\nLinear Complexity O(T)"]
        BiMamba --> SaliencyGate["Monotonic Latent Saliency Bottleneck\nDownsampling factor: 2x"]
        SaliencyGate --> LatentContext["Latent Semantic State: (B, T_sub, D_m)"]
    end

    subgraph Text_Generation ["6. Language Generation"]
        LatentContext --> CrossAttnProj["Linear Cross-Attention Bridge"]
        CrossAttnProj --> T5Decoder["FLAN-T5-base Autoregressive Decoder"]
        T5Decoder --> TargetText["Translated English Text: (B, L_tgt)"]
    end
```

---

## 3. Mathematical Specification of Core Components

### A. Torso Normalization and Bilateral Mirror-Canonicalization
Given raw 3D landmark coordinates $p_i(t) = (x_i, y_i, z_i) \in \mathbb{R}^3$ for landmark $i \in \{0, \dots, 65\}$:
1. **Reference Origin:** Midpoint of left shoulder ($i=0$) and right shoulder ($i=1$):
   $$p_{\text{mid}}(t) = \frac{p_0(t) + p_1(t)}{2}$$
2. **Kinematic Scale Normalizer:** Euclidean distance between shoulders:
   $$d_{\text{torso}}(t) = \|p_0(t) - p_1(t)\|_2 + \epsilon$$
3. **Normalized Coordinates:**
   $$\tilde{p}_i(t) = \frac{p_i(t) - p_{\text{mid}}(t)}{d_{\text{torso}}(t)}$$
4. **Bilateral Mirror-Canonicalization:** Hand shape and fingerspelling configurations are anatomically symmetric. To allow complete parameter sharing between left hand (indices $6\dots26$) and right hand (indices $27\dots47$), left-hand coordinates are reflected across the sagittal plane:
   $$\tilde{p}_{i, \text{canonical}}(t) = \begin{cases} (-\tilde{x}_i, \tilde{y}_i, \tilde{z}_i) & \text{if } i \in \text{Left Hand} \\ (\tilde{x}_i, \tilde{y}_i, \tilde{z}_i) & \text{otherwise} \end{cases}$$

### B. Anatomical 4-Region ST-GCN with Active Cross-Articulator Gating
The 66 landmarks are partitioned into four subgraphs $\mathcal{V}_k$ ($k \in \{\text{body}, \text{left\_hand}, \text{right\_hand}, \text{face}\}$).
1. **Spatial Graph Convolution:** Within subgraph $k$, feature propagation follows:
   $$H_k^{(l+1)} = \sigma\left(\sum_{a \in \mathcal{A}} \tilde{D}_a^{-\frac{1}{2}} \tilde{A}_a \tilde{D}_a^{-\frac{1}{2}} H_k^{(l)} W_{k, a}^{(l)}\right)$$
   Where $\mathcal{A} = \{\text{self}, \text{inward}, \text{outward}\}$ partitions skeletal kinematic bones. Crucially, $W_{\text{left\_hand}, a} \equiv W_{\text{right\_hand}, a}$, forcing identical physical manifold learning across both hands.
2. **Active Cross-Articulator Message Passing:**
   Relational interactions between hands and face (e.g. touching cheek or chin) are modeled via multi-head cross-attention across subgraph pooled representations:
   $$M_{\text{cross}} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
   The output incorporates an active learned gate initialized strictly to $\gamma_0 = 0.1$:
   $$H_{\text{pose}} = H_{\text{spatial}} + \tanh(\gamma) \cdot \text{Dropout}(W_c M_{\text{cross}}), \quad \gamma \sim \mathcal{N}(0.1, 0.01)$$

### C. Evidential Reliability Fusion (ERF)
Instead of static or heuristic soft-gating, CARP-S2T models modality confidence as a **Subjective Logic Dirichlet distribution**.
1. For each modality stream $m \in \{\text{vision}, \text{pose}, \text{fingerspelling}\}$ at time step $t$, a lightweight projection head predicts positive evidence parameters $e_m(t) \in \mathbb{R}_+^K$:
   $$e_m(t) = \text{Softplus}\left(W_e q_m(t) + b_e\right)$$
   Where $q_m(t)$ is the observable quality vector (confidence scores, tracking jitter, spatial velocity).
2. The Dirichlet concentration parameters are $\alpha_m(t) = e_m(t) + 1$. The total Dirichlet strength is:
   $$S(t) = \sum_{m=1}^M \alpha_m(t)$$
3. The expected dynamic modality reliability weight $\hat{w}_m(t)$ and overall epistemic uncertainty $u(t)$ are given in closed form by:
   $$\hat{w}_m(t) = \frac{\alpha_m(t)}{S(t)}, \quad u(t) = \frac{M}{S(t)}$$
   When landmark tracking fails (e.g. occlusion causes landmark jitter and confidence $\to 0$), $e_{\text{pose}} \to 0 \implies \alpha_{\text{pose}} \to 1$. The network dynamically shifts its weight to the robust visual stream $F_{\text{vis}}$ while elevating epistemic uncertainty $u(t)$.

### D. Bidirectional Selective State-Space Temporal Backbone (Bi-S6)
Continuous sign translation requires linear temporal scaling over long video sequences ($T \approx 100\dots300$ frames). 
The continuous-time linear state-space system maps input $x(t) \in \mathbb{R}$ to latent state $h(t) \in \mathbb{R}^N$ and output $y(t) \in \mathbb{R}$:
$$h'(t) = A h(t) + B x(t), \quad y(t) = C h(t) + D x(t)$$
1. **Input-Dependent Discretization:** The parameters $B(t), C(t)$, and step size $\Delta(t)$ are continuous functions of the input token $x(t)$:
   $$\Delta(t) = \text{Softplus}\left(\text{Linear}_\Delta(x(t))\right), \quad B(t) = \text{Linear}_B(x(t)), \quad C(t) = \text{Linear}_C(x(t))$$
   Discretizing via Zero-Order Hold (ZOH):
   $$\bar{A}(t) = \exp\left(\Delta(t) A\right), \quad \bar{B}(t) = \left(\Delta(t) A\right)^{-1} \left(\exp\left(\Delta(t) A\right) - I\right) \cdot \Delta(t) B(t)$$
2. **Bidirectional Temporal Scanning:**
   Because sign translation is an offline sentence-level task, future frames provide crucial phonetic disambiguation:
   $$h_{\text{fwd}}(t) = \bar{A}_{\text{fwd}}(t) h_{\text{fwd}}(t-1) + \bar{B}_{\text{fwd}}(t) x(t)$$
   $$h_{\text{bwd}}(t) = \bar{A}_{\text{bwd}}(t) h_{\text{bwd}}(t+1) + \bar{B}_{\text{bwd}}(t) x(t)$$
   $$y(t) = W_o \left[ C_{\text{fwd}}(t) h_{\text{fwd}}(t) \,\|\, C_{\text{bwd}}(t) h_{\text{bwd}}(t) \right]$$
   Temporal complexity is strictly **$\mathcal{O}(T)$**, compared to $\mathcal{O}(T^2)$ in standard Transformers.

---

## 4. Complete Tensor Shape Progression Table

| Processing Stage | Module Name | Input Tensor Dimensions | Output Tensor Dimensions | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **RGB Extraction** | `VisionBackbone` | `(B, T, 3, 224, 224)` | `(B, T, 512)` | Frozen DINOv2-B + Hand-LoRA Adapter |
| **Pose Normalization** | `TorsoNormalizer` | `(B, T, 66, 3)` | `(B, T, 66, 3)` | Zero-centered, scale invariant, mirrored |
| **Skeletal GCN** | `AnatomicalSTGCN` | `(B, T, 66, 3)` | `(B, T, 4, 256)` | 4 regions: body, 2 hands (shared), face |
| **Cross-Articulator**| `CrossArticulator` | `(B, T, 4, 256)` | `(B, T, 512)` | Attention across regions; active gate $0.1$ |
| **Fingerspelling** | `FingerspellingGRU` | `(B, T, 21, 3)` | `(B, T, 256)` | Dominant hand sequence; auxiliary CTC |
| **Evidential Fusion** | `EvidentialFusion` | `[(B,T,512), (B,T,512), (B,T,256)]`| `(B, T, 512)` | Weighted by Dirichlet belief $\hat{w}_m(t)$ |
| **Bi-S6 Backbone** | `SignMambaBackbone` | `(B, T, 512)` | `(B, T, 512)` | 6 bidirectional S6 blocks; $\mathcal{O}(T)$ compute |
| **Saliency Bottleneck**| `MonotonicSampler` | `(B, T, 512)` | `(B, T // 2, 512)` | Downsampled temporal representation |
| **Cross-Attention** | `DecoderBridge` | `(B, T // 2, 512)` | `(B, T // 2, 768)` | Projected to FLAN-T5-base dimension |
| **Language Decoder** | `FLAN-T5-base` | `(B, T // 2, 768)`, `(B, L_tgt)` | `(B, L_tgt, 32128)` | Cross-entropy text loss / Beam search |

---

## 5. Failure Mode Analysis and Robustness Mitigations

| Failure Mode / Edge Case | Physical Cause | Observed Phenomenon | CARP-S2T Structural Mitigation |
| :--- | :--- | :--- | :--- |
| **Total Landmark Loss** | Signer hands exit camera frame or motion blur degrades keypoint extraction. | Skeletal landmarks drop to zero confidence or NaN. | Evidential Dirichlet estimator drives $\alpha_{\text{pose}} \to 1$ ($e_{\text{pose}} \to 0$); fusion weight dynamically shifts 100% of gradient to the visual DINOv2 stream. |
| **Signer Handedness** | Signer is left-hand dominant. Hand shapes and dominant articulation are inverted. | Right-hand-dominant models suffer 35% BLEU drop on left-handed signers. | Bilateral Mirror-Canonicalization reflects all left-hand coordinates into canonical dominant space before the shared ST-GCN, guaranteeing handedness invariance. |
| **Rapid Fingerspelling Blurs** | Technical terms or proper nouns fingerspelled at $> 8$ handshapes per second. | Temporal downsampling skips or merges fast transitions. | Fingerspelling CTC head operates on uncompressed pre-downsampled temporal states ($T$), anchoring character boundaries before sequence downsampling. |
| **Evidential Certainty Collapse**| Model becomes overconfident on noisy training artifacts. | Dirichlet concentration $S \to \infty$; uncertainty $u \to 0$ on out-of-distribution signers. | Kullback-Leibler (KL) divergence penalty on evidence regularizes predicted Dirichlet distribution against a flat uniform prior $\text{Dir}(1, 1, 1)$. |
| **Short Utterance Vanishing** | 1-word or short 5-frame sign utterances downsampled below minimum decoder length. | Temporal stem downsamples sequence to $T' < 2$ tokens. | Adaptive pooling enforces dynamic minimum length $T_{\min} = 4$ via reflection padding if $T < 8$. |

