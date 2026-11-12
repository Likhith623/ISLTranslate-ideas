# Mathematical, Anatomical, and Temporal Correctness Audit — ISL-MEGA

**Audited Codebase:** `isl-mega` v`0.2.0` ([`DINOv2-PoseGraph-3DCNN-Conformer/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer), git HEAD `2157667`)  
**Scope & Verification Basis:** Complete mathematical and representational inspection of every source module in `src/isl_mega/*.py`, CLI/dataset/training scripts (`scripts/*.py`), configs (`configs/*.yaml`), documentation (`docs/*.md`), and sibling/parent parsers (`CNN-BiLSTM`, `PoseGraph-Conformer`, `InceptionCNN-Transformer`), supplemented by pure-Python analytical checks of graph distances and bucket quantization intervals. No source files were modified.

> [!IMPORTANT]
> **Implementation vs. Unimplemented Theory Boundary:** Several design reports inside `reports/` describe mathematical guarantees that **do not exist in the `isl-mega` source code**:
> - **No Dirichlet / Evidential Fusion exists:** [`fusion.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py) implements neither Dirichlet concentration parameters ($\alpha_m = e_m + 1$), Subjective Logic uncertainty ($u = M/S$), nor KL evidence regularization (all claimed in [`04_FINAL_ARCHITECTURE_SPECIFICATION.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/04_FINAL_ARCHITECTURE_SPECIFICATION.md#L96-L106)).
> - **No Bi-Mamba (`O(T)` S6) exists:** [`temporal.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py) runs quadratic $\mathcal{O}(M^2)$ self-attention (`GlobalTemporalBlock`).
> - **No dominant/passive hand canonicalization exists:** [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L393-L405) mirrors anatomical right to left *only* inside `RegionGraphEncoder`, leaving all downstream pathways handedness-sensitive.
>
> This audit evaluates the **actual executable mathematics** implemented in `src/isl_mega/`.

---

## A. Anatomical Coordinates & Kinematic Representation

### A.1 Canonical 66-Point Mapping, Absent Adapter & Cross-Contract Collisions

Implemented slices in [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L14-L17) and [`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L144-L149):

| Canonical Slice | Count | Region | Anatomical Role Defined in [`docs/data_contract.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md#L38-L81) | Runtime Extractor Status |
|---|---:|---|---|---|
| `0:21` | `21` | `LEFT` (signer's left hand) | MediaPipe Hands 21 points (`0`=wrist, `1..4`=thumb, ..., `17..20`=pinky) | **No MediaPipe / `.pose` parser in repo** |
| `21:42` | `21` | `RIGHT` (signer's right hand) | MediaPipe Hands 21 points (`21`=wrist, ..., `41`=pinky tip) | **No MediaPipe / `.pose` parser in repo** |
| `42:48` | `6` | `BODY` (upper body) | `[42]=L_shoulder`, `[43]=R_shoulder`, `[44]=L_elbow`, `[45]=R_elbow`, `[46]=L_wrist`, `[47]=R_wrist` | Maps conceptually to MediaPipe Pose `[11..16]` |
| `48:66` | `18` | `FACE` (sparse face) | Eyebrows (`48..51`), eye canthi (`52..55`), nose/glabella/cheeks (`56..60`), mouth+chin (`61..65`) | **Contradicts sibling extractors & prior audit scripts** |

Three concrete anatomical contract hazards exist in the workspace:

1. **Silent Permutation vs. Parent Pipeline:** [`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L3-L5) instructs users to *"convert it through the parent ISLTranslate pose parser"*, yet the parent parser in [`CNN-BiLSTM/src/isl_slt/features/pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/src/isl_slt/features/pose.py#L38-L40) exports 66 points ordered `[UPPER 0:6, LEFT 6:27, RIGHT 27:48, FACE 48:66]`. Because [`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L278) only checks `pose_xy_raw.shape == (len(pose_times), 66, 2)`, directly loading parent exports silently treats shoulders/elbows/wrists as left-hand palm/thumb landmarks and mixes wrists into face/body groups without raising an error.
2. **Incompatible 18-Point Face Definitions:** [`docs/data_contract.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md#L63-L80) defines `48..65` as 4 eyebrow points, 4 eye canthi, 5 nose/glabella/cheek points, 4 mouth points, and chin (`PoseExpert` treats `FACE` as an unlabeled unordered `fully_connected=True` graph with learned `node_id` embeddings, which hides index semantics inside a single run but prevents cross-dataset mixing). Meanwhile, both parent repos ([`CNN-BiLSTM/.../pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/src/isl_slt/features/pose.py#L17-L18) and [`PoseGraph-Conformer/ramagc/dataset.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/PoseGraph-Conformer/ramagc/dataset.py#L33-L36)) select 18 MediaPipe Face Mesh indices `("33","133","159","145","362","263","386","374","70","63","105","336","296","334","61","291","13","14")` (8 eyelids/canthi first, 6 eyebrow points second, 4 lip points, **0 nose/cheek/chin points**).
3. **Broken 543-to-66 Recipe in Prior Audit Report:** Phase-1 report [`07_ACTIONABLE_IMPLEMENTATION_PLAN.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/phase_1_verification/07_ACTIONABLE_IMPLEMENTATION_PLAN.md#L34-L41) specifies `left_hand = holistic_pts[:, 468:489, :2]` and `face_idx = [33, 133, ..., 13, 14]`. In standard concatenated 543-point MediaPipe Holistic vectors (`Pose 0:33` + `Face 33:501` + `LeftHand 501:522` + `RightHand 522:543`), slice `468:489` extracts **raw facial mesh points**, and unshifted `face_idx` (`13, 14`) indexes **POSE_LANDMARKS elbows** (`LEFT_ELBOW`, `RIGHT_ELBOW`)! Because [`convert_video_pose_npz.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/convert_video_pose_npz.py#L37) cannot detect semantic misalignment, executing that recipe would silently corrupt both hands and face.

---

### A.2 Coordinate Reference Frames, Affine Transform, Torso Scale & Kinematics

#### Equation 1 — Letterbox Affine Transformation ([`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L31-L74))

Given frame dimensions $(H, W)$, square canvas $S = \text{image\_size}$, scale $c = \min(S/H, S/W)$, resized integers $H_r = \text{round}(H c)$, $W_r = \text{round}(W c)$, and pad offsets $l = \lfloor (S - W_r)/2 \rfloor$, $\tau = \lfloor (S - H_r)/2 \rfloor$, raw normalized coordinates $(u, v) \in [0,1]^2$ map to letterbox coordinates $p = (x, y) \in \mathbb{R}^2$:

$$x = \frac{W_r}{S}\, u + \frac{l}{S}, \qquad y = \frac{H_r}{S}\, v + \frac{\tau}{S}$$

- **Variables & Dimensions:** $u, v \in \mathbb{R}^{T \times 66}$ (pose) or $\mathbb{R}^{T \times 4 \times 2}$ (ROI corners); $x, y \in \mathbb{R}^{T \times 66}$ in $[0, 1]^2$ when valid.
- **Consistency & Code Match:** Exact match (`scale_x = resized_w / S`, `scale_y = resized_h / S`). Note integer rounding of $H_r, W_r$ and floor division of $(l, \tau)$ introduces at most $0.5/S$ px rounding error ($< 0.23\%$ at $S=224$). Points outside $[0,1]^2$ are masked out (`pose_valid &= ((pose_xy >= 0) & (pose_xy <= 1)).all(-1)`, `L312-313`).
- **Limitation (ROI Aspect-Ratio Distortion):** Derived boxes in [`_bbox`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L124-L135) expand `[w, h]` independently (`np.maximum(hi - lo, 0.025) * margin`, rectangular, not `max(w, h)`), and [`crop_regions`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L20-L59) resamples every rectangular box onto a square $S_{\text{crop}} \times S_{\text{crop}}$ ViT grid via bilinear `grid_sample`. Rectangular hand/body boxes therefore undergo **anisotropic shear/aspect squashing** in RGB while pose retains isotropic letterboxed $(x, y)$.

#### Equation 2 — Frame-Wise Torso Scale Cascade ([`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L61-L86))

Let $p_{b,t,i} \in [0,1]^2$ be letterbox coordinates and $m_{b,t,i} \in \{0,1\}$ be combined validity at frame $t$. Defining visible bounding-box diagonal $\text{diag}(G) = \sqrt{\max(\max_{i \in G} x_i - \min_{i \in G} x_i, 0)^2 + \max(\max_{i \in G} y_i - \min_{i \in G} y_i, 0)^2 + \varepsilon}$ ($\varepsilon = 10^{-8}$) when $|G_{\text{valid}}| \ge 2$:

$$\sigma_{\text{raw}}(t) = \begin{cases} \sqrt{\|p_{\text{body}, 0}(t) - p_{\text{body}, 1}(t)\|_2^2 + 10^{-8}} & \text{if } m_{42}(t) \land m_{43}(t) \\ 0.65 \cdot \text{diag}(\text{BODY}_{\text{valid}}(t)) & \text{else if } |\text{BODY}_{\text{valid}}(t)| \ge 2 \\ 1.25 \cdot \text{diag}(\text{FACE}_{\text{valid}}(t)) & \text{else if } |\text{FACE}_{\text{valid}}(t)| \ge 2 \\ 1.0 & \text{otherwise} \end{cases}$$

$$\sigma(t) = \operatorname{clamp}\!\big(\sigma_{\text{raw}}(t),\, 0.05,\, 2.0\big) \quad \in \mathbb{R}^{B \times T}$$

- **Consistency & Assumptions:** Matches code (`L82-86`). Implicitly assumes (i) shoulders are parallel to the image plane (ignoring yaw foreshortening, which shrinks 2D shoulder span by $\cos\theta$), (ii) empirical ratios $\sigma_{\text{shoulder}} \approx 0.65\,\text{diag}(\text{body}) \approx 1.25\,\text{diag}(\text{face})$ hold across cameras/signers, and (iii) $\sigma(t)$ varies smoothly across $t$.
- **Critical Limitation (Discrete Scale Switching Spikes):** Because $\sigma(t)$ uses zero temporal smoothing (unlike sibling [`PoseGraph-Conformer`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/PoseGraph-Conformer/ramagc/dataset.py) and [`CNN-BiLSTM`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/CNN-BiLSTM/src/isl_slt/features/pose.py#L85), which take sequence medians), a **single-frame dropout** of one shoulder (e.g., partial occlusion) switches from `shoulder_span` ($\approx 0.22$) to $0.65 \cdot \text{diag}(\text{body})$ (e.g., $\approx 0.35$ if wrists are outstretched, or $\approx 0.12$ if only shoulder+elbow remain) or $1.25 \cdot \text{diag}(\text{face}) \approx 0.10$, causing a sudden $\mathbf{0.5\times\text{--}2.2\times}$ **multiplicative jump** in both normalized local coordinates and finite-difference velocities/accelerations!

#### Equation 3 — Anchor Selection & Local Coordinates ([`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L360-L390))

For each region $g \in \{\text{left\_hand}, \text{right\_hand}, \text{body}, \text{face}\}$, valid mask $M_g(t) = \{i \in g : m_i(t) = 1\}$:

$$a_g(t) = \begin{cases} p_{g,\text{wrist}}(t) & g \in \{\text{L, R}\},\, m_{\text{wrist}}(t)=1 \\ \frac{1}{|M_g(t)|}\sum_{i \in M_g(t)} p_{g,i}(t) & g \in \{\text{L, R}\},\, m_{\text{wrist}}(t)=0 \lor g=\text{face} \end{cases}$$

$$a_{\text{body}}(t) = \begin{cases} \frac{m_{42}(t) p_{42}(t) + m_{43}(t) p_{43}(t)}{m_{42}(t) + m_{43}(t)} & \text{if } m_{42}(t) \lor m_{43}(t) \\ \frac{1}{|M_{\text{body}}(t)|}\sum_{i \in M_{\text{body}}(t)} p_i(t) & \text{otherwise} \end{cases}$$

$$r_{g,i}(t) = \begin{cases} \dfrac{p_{g,i}(t) - a_g(t)}{\sigma(t)} & \text{if } M_g(t) \neq \emptyset \text{ (and } \sigma(t) \equiv 1 \text{ when unnormalized)} \\ (0,0) & \text{otherwise} \end{cases}$$

- **Exact Code Defect in `a_body(t)`:** Line `L378` uses `has_shoulder = shoulder_valid.any(dim=2, keepdim=True)`, contradicting [`docs/architecture.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/architecture.md#L46) and comment `L372` (*"fall back to the visible body centroid only when neither shoulder is available"*). When one shoulder is valid and the other drops out (`m_42=1, m_43=0`), `a_body(t)` snaps by **$0.5\,\sigma(t)$** onto the single visible shoulder, shifting every body coordinate discontinuously by half a shoulder-width!
- **Silent Anchor Mode Switching (Hands & Face):** When `wrist` drops out in a hand, `a_hand(t)` shifts instantly to the centroid of whatever subset of the remaining 20 finger joints survived (`L367`), with **no indicator bit** telling downstream `EdgeGraphBlock` or `RegionGraphEncoder` whether `rel_xy` is wrist-centered or partial-finger-centroid-centered. Moreover, because `RegionGraphEncoder` sums `Linear(f_i)` across channels `[0..10]` while `node_id` (`L179`) marks index identity regardless of anchor change, losing wrist `0` alters `rel_xy` of all 20 other hand nodes simultaneously.

#### Equation 4 — 11-Channel Node Feature Vector ([`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L300-L324, L389))

For joint $i$ at frame $t$ ($0 \le t < T$, timestamps $\tau_t$ in seconds, $\Delta t_t = \max(\tau_t - \tau_{t-1}, 10^{-5})$, pairwise validity $\pi_i(t) = m_i(t) \land m_i(t-1) \land (\tau_t > \tau_{t-1})$):

$$v_i(t) = \begin{cases} \dfrac{p_i(t) - p_i(t-1)}{\Delta t_t \cdot \sigma(t)} & t \ge 1,\, \pi_i(t) = 1 \\ (0,0) & \text{otherwise} \end{cases} \qquad a_i(t) = \begin{cases} \dfrac{v_i(t) - v_i(t-1)}{\Delta t_t} & t \ge 2,\, \pi_i(t) \land \pi_i(t-1) \\ (0,0) & \text{otherwise} \end{cases}$$

$$f_i(t) = \Big[\underbrace{x_i(t),\, y_i(t)}_{0:2,\text{ global}},\; \underbrace{r_{x,i}(t),\, r_{y,i}(t)}_{2:4,\text{ local}},\; \underbrace{\tanh\!\left(\frac{v_i(t)}{5}\right)}_{4:6},\; \underbrace{\tanh\!\left(\frac{a_i(t)}{25}\right)}_{6:8},\; \underbrace{\tilde{c}_i(t)}_{8},\; \underbrace{m_i(t)}_{9},\; \underbrace{\kappa_i(t)}_{10}\Big] \in \mathbb{R}^{B \times T \times 66 \times 11}$$

Mathematical critique of $f_i(t)$:
1. **Channels `[0:2]` Destroy Translation & Framing Invariance:** Passing raw letterbox `(x, y)` directly into the input `Linear(11, D)` breaks signer-position invariance across camera crops, even with `normalize_by_shoulder_span=True`.
2. **Global Differentiation Divided by Single-Frame $\sigma(t)$:** $v_i(t)$ differentiates **global** letterbox coordinates $p_i(t) - p_i(t-1)$ rather than anchor-relative coordinates $r_i(t) - r_i(t-1)$. Camera pan/zoom contaminates articulator velocity, and `normalize_by_shoulder_span` scales $v_i(t)$ by $1/\sigma(t)$ *after* global differencing (`L309`)—hence $a_i(t)$ subtracts two velocities scaled by potentially different frame scales $\sigma(t)$ vs $\sigma(t-1)$.
3. **Aggressive Kinematic Saturation:** At 30 fps ($\Delta t = 0.0333\text{s}$), a natural fast signing stroke of $0.17\text{ torso/frame}$ ($\approx 5.0\text{ torso/s}$, well below fast hand whips of $8\text{--}15\text{ torso/s}$) gives $a_{\text{impulse}} = 150\text{ torso/s}^2 \implies \tanh(a/25) = \mathbf{0.999988}$ (verified numerically), flattening gradient signal $\text{sech}^2(a/25) \approx 2.4 \times 10^{-5}$.
4. **Cascading Dropout Erosion & Boundary Zeros:** `t=0` always gets zero velocity, `t=0,1` always get zero acceleration, and any 1-frame landmark miss wipes out **2 velocity frames** ($t, t+1$) and **3 acceleration frames** ($t, t+1, t+2$).

---

### A.3 Landmark Visibility, Confidence & Missing Observations

- **Combined Validity Predicate ([`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L284-L313), [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L337-L344)):**
  $$m_{b,t,i} = \text{pose\_valid}_{b,t,i} \land \text{finite}(p^{\text{raw}}) \land (p^{\text{lb}}_{b,t,i} \in [0,1]^2) \land \text{frame\_mask}_{b,t} \land \big(\neg \kappa_{b,t,i} \lor c_{b,t,i} \ge c_{\min}\big)$$
- **Per-Landmark Interpolation Asymmetry ([`align_pose_to_video`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L77-L121)):** Interpolation between bracketing pose timestamps $\tau_L \le t_v \le \tau_R$ occurs *independently per landmark* whenever $\tau_R - \tau_L \le \Delta_{\max}$ (`0.20s` in research configs). Thus within one frame, some finger joints can be linearly interpolated between $\tau_L, \tau_R$ while wrist `0` (missing at $\tau_R$) becomes invalid—triggering an abrupt wrist-to-centroid anchor jump purely from desynchronized joint interpolation. Furthermore, linear interpolation ignores temporal gap length $\alpha$ on `pose_confidence`: linearly interpolating $c = (1-\alpha)c_L + \alpha c_R$ leaves confidence unattenuated at mid-gap ($\alpha = 0.5$ across a $0.20\text{s}$ gap!).
- **Missing-Confidence Proxy Leakage:** When NPZ omits `pose_confidence`, [`data.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L260-L261) sets `pose_conf_raw = pose_valid_raw.astype(np.float32)` (`1.0` for all valid landmarks) alongside `confidence_known = False`. Although [`PoseExpert`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L340-L344) properly skips `min_confidence` thresholding where `~confidence_known`, `conf = 1.0` still enters node feature channel `[8]` AND fusion quality descriptor channel `[1]` (`conf_mean = 1.0`), forcing both modules to learn from channel `[10]`/`quality[...,2]` (`known = 0`) that `conf = 1.0` does **not** actually mean high detector confidence!
- **Missing Observations Zeroing:** Invalid nodes are explicitly zeroed (`torch.where(m[..., None], h, 0)`, [`pose.py:L184`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L184)), receive `0` attention weight via `any_key` fallback in [`EdgeGraphBlock`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L131-L137), and yield `available[name] = False` in `NodePool` when all joints in that region are invalid.

---

### A.4 Left/Right Hand Identity, Dominant vs. Non-Dominant Hands & Mirror Canonicalization

#### Equation 5 — Conditional Mirror Canonicalization ([`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L393-L405))

Only when `pose.share_hand_encoder=True` (`research.yaml`, `research_v3_compact.yaml`, `smoke_v2.yaml`), the `right_hand` feature tensor $f_{\text{right}}(t)$ (`11` channels) and edge coordinates `enc_xy` undergo x-reflection prior to shared `RegionGraphEncoder`:

$$f^{\text{canon}}_{\text{right}, i}(t) = \Big[\, \underbrace{1 - x_i(t)}_{0},\, y_i(t),\, \underbrace{-r_{x,i}(t)}_{2},\, r_{y,i}(t),\, \underbrace{-\tanh\!\tfrac{v_{x,i}(t)}{5}}_{4},\, \tanh\!\tfrac{v_{y,i}(t)}{5},\, \underbrace{-\tanh\!\tfrac{a_{x,i}(t)}{25}}_{6},\, \tanh\!\tfrac{a_{y,i}(t)}{25},\, \tilde{c}_i,\, m_i,\, \kappa_i \,\Big]$$

$$\text{enc\_xy}_{\text{right}, i}(t) = \big(1 - x_i(t),\; y_i(t)\big), \qquad h^{(0)}_{\text{hand}} = W_{\text{in}} f^{\text{canon}} + e_{\text{node}, i} + e_{\text{side}}(\text{side}), \quad \text{side} \in \{0=\text{L},\, 1=\text{R}\}$$

**Detailed Verdict on Mirror Canonicalization, Chirality & Linguistic Asymmetry:**

1. **Off-Center Reflection Error in Channel `[0]`:** Note `rel_edge = enc_xy_i - enc_xy_j = (-(x_i - x_j), y_i - y_j)` correctly removes the arbitrary constant `1.0`. However, raw node channel `[0]` maps $x \mapsto 1 - x$, reflecting about the **frame center $x = 0.5$** rather than the signer's torso midline $x_{\text{mid}}(t) = \frac{1}{2}(x_{42}(t) + x_{43}(t))$. If a signer stands slightly off-center (say shoulders centered at $x_{\text{mid}} = 0.35$), symmetric hands at $x_L = 0.25$ and $x_R = 0.45$ map to `gfeat[0] = 0.25` (L) vs `1 - 0.45 = 0.55` (R)—an artificial `+0.30` global bias in `gfeat[0]`.
2. **Intra-Encoder Suppression of Directional Asymmetry:** Within `RegionGraphEncoder`, mirroring right-hand velocity ($v_x \mapsto -v_x$) and wrist-relative $x$ ($r_x \mapsto -r_x$) means that a lateral rightward motion of the right hand (anatomically outward/ipsilateral) maps to identical `(0:8)` kinematics as a leftward motion of the left hand (also anatomically outward/ipsilateral), whereas a **screen-rightward spatial sweep across both hands** (e.g., spatial agreement/pointing toward a right-hand locus in sign space) maps to *opposite* signed kinematic vectors inside the shared hand weights! The only distinguisher inside `RegionGraphEncoder` is a single additive bias `hand_side_emb` (`[1, 1, 1, D]`) added before 3 `LayerNorm`-normalized attention+FFN blocks, which washes out additive mean offsets. *(Note: downstream in `CrossArticulatorGraph`, `Fusion`, and `FingerspellingExpert`, `left_hand` and `right_hand` return to separate unmirrored channels.)*
3. **False Handedness Invariance (The Left-Handed Signer Paradox):** Theoretical spec [`04_FINAL_ARCHITECTURE_SPECIFICATION.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/04_FINAL_ARCHITECTURE_SPECIFICATION.md#L146) claims mirror canonicalization makes the model invariant to left-handed vs right-handed signers. In reality:
   - `PoseExpert` **always** mirrors `right_hand` (`RIGHT`) and keeps `left_hand` (`LEFT`) unmirrored—regardless of which hand is active/dominant.
   - `CrossArticulatorGraph` assigns distinct `region_emb(0)` to `LEFT` and `region_emb(1)` to `RIGHT`.
   - `ReliabilityAwareFusion` places `left_hand_pose` in slot `4` (`type_embedding[4]`, `projections["left_hand_pose"]`) and `right_hand_pose` in slot `5` (`type_embedding[5]`, `projections["right_hand_pose"]`).
   - `FingerspellingExpert` (`fingerspelling.py:90`) concatenates `[left_rgb, right_rgb, left_pose, right_pose]` into an order-sensitive `Linear(2*D_v + 2*D_p, H)`!
   - Under Battison's Dominance and Symmetry Conditions in sign phonology, an asymmetric two-handed sign swaps **dominant (active) vs. non-dominant (passive base)** hand roles when performed by a left-handed signer. Because ISL-MEGA has **zero dominance detection and never swaps stream roles**, left-handed vs. right-handed signers still swap semantic features across separate learned weights in `CrossArticulatorGraph`, `Fusion`, and `FingerspellingExpert`.
4. **Vision–Pose Cross-Modal Chirality Contradiction:** [`AppearanceExpert`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L190-L203) passes **unflipped** right-hand RGB crops (`crops[:, :, 2]`) through the shared `self.hand` ViT (`L192`), whereas [`PoseExpert`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L396-L404) passes **x-flipped** right-hand coordinates through `self.left` (`share_hand_encoder=True`). Despite this opposite internal coordinate chirality in the shared encoders, [`compute_cross_modal_alignment_loss`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L161) aligns `(0 <-> 4)` (`left_hand_rgb <-> left_hand_pose`) and `(1 <-> 5)` (`right_hand_rgb <-> right_hand_pose`) via InfoNCE—while `projections["right_hand_rgb"]` and `projections["right_hand_pose"]` must learn an ad-hoc mirror reflection post-hoc just to match same-hand crops!

---

### A.5 Anatomical Graph Edges, Diameter Blindness & Duplicate Wrist Nodes

Verified shortest-path analysis (`BFS` on unweighted undirected graphs defined in [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L19-L26, L47-L58, L281-L289)):

| Subgraph | Nodes | Edges | Graph Diameter | Configured Message-Passing Depth (`layers`) | Reachability Deficit (`Diameter > Depth`) |
|---|---:|---:|---:|---:|---|
| `HAND_EDGES` (`LEFT`, `RIGHT`) | `21` | `20` (5 tree chains from wrist `0`, **no knuckle cross-links**) | **`8`** hops | **`3`** (`research*`) / **`1`** (`smoke*`) / **`2`** (default) | **Critical:** `dist(0, tips {4,8,12,16,20}) = 4 > 3`; `dist(tip_a, tip_b) = 8 > 3`. Zero fingertip-to-wrist or fingertip-to-fingertip communication in `RegionGraphEncoder`! |
| `BODY_EDGES` | `6` | `5` (`(0,1),(0,2),(2,4),(1,3),(3,5)`) | **`5`** hops | **`3`** (`research*`) / **`1`** (`smoke*`) | Wrist-to-wrist distance `dist(4, 5) = 5 > 3` inside `body` encoder (mitigated only by `CrossArticulatorGraph`). |
| `FACE` (`fully_connected=True`) | `18` | `324` (`18 x 18` dense) | **`1`** hop | **`3`** / **`1`** | None (`324` edges/frame/layer). |

> [!CAUTION]
> **Proven Intra-Hand Aperture Blindness:** Combining Stage 1 (`RegionGraphEncoder`, `layers = 3`) and Stage 2 (`CrossArticulatorGraph`, `layers = 1`, where `adj[s_l, s_l] = eye(6)` and `adj[s_r, s_r] = eye(6)` at [`pose.py:L47-L58`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L47-L58)):
> 1. In Stage 1 (`3` layers), from wrist `0`, reachable nodes are `{0..3, 5..7, 9..11, 13..15, 17..19}` (**all 5 tips `{4, 8, 12, 16, 20}` are unreachable**); from thumb tip `4`, reachable nodes are only `{1, 2, 3, 4}` (**cannot reach index tip `8`, palm knuckle `5`, or wrist `0`**).
> 2. In Stage 2 (`CrossArticulatorGraph`), intra-hand adjacency contains **only self-loops**, forbidding `left_tip_i <-> left_tip_j` or `left_wrist <-> left_tip_i` edges!
> 3. Consequently, **phonological handshape aperture** (e.g., thumb tip `4` touching index tip `8` for fingerspelled `"F"`, `"O"`, `"D"`, `"9"`, or flat-hand vs clawed handshape relations between fingertips) **never exchanges a direct or multi-hop message across graph edges anywhere in `PoseExpert`** before collapsing via permutation-invariant `NodePool`!

- **Duplicate Wrist Landmarks:** `LEFT[0]` (`0`) & `RIGHT[0]` (`21`) come from MediaPipe Hands while `BODY[4]` (`46`) & `BODY[5]` (`47`) come from MediaPipe Pose (two independent regression heads with different jitter and occlusion drop rates). Both wrist copies enter `pose_xy` AND both enter the `36`-node [`CrossArticulatorGraph`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L211-L272), creating pairs separated by $d_{ij} \approx 0$ ($\text{RBF}(0) = 1.0$) with discordant confidence/validity masks unless reconciled upstream.

---

## B. Cross-Articulator Modelling (`CrossArticulatorGraph` & `EdgeGraphBlock`)

Implemented in [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L38-L58, L89-L142, L198-L273, L425-L433) (enabled in `research`, `research_v3_compact`, `smoke_v2`).

### Equation 6 — 36-Node Bipartite Cross-Articulator Graph & RBF Edge Attention

Node subset indices per frame (`36` nodes):

$$V_{\text{cross}} = \underbrace{\text{LEFT}[0,4,8,12,16,20]}_{6,\; \text{region}=0} \;\Vert\; \underbrace{\text{RIGHT}[0,4,8,12,16,20]}_{6,\; \text{region}=1} \;\Vert\; \underbrace{\text{BODY}[0..5]}_{6,\; \text{region}=2} \;\Vert\; \underbrace{\text{FACE}[0..17]}_{18,\; \text{region}=3}$$

Binary adjacency $A \in \{0, 1\}^{36 \times 36}$ (`pose.py:L47-L58`):

$$A = \begin{pmatrix} I_6 & \mathbf{1}_{6 \times 6} & \mathbf{1}_{6 \times 6} & \mathbf{1}_{6 \times 18} \\ \mathbf{1}_{6 \times 6} & I_6 & \mathbf{1}_{6 \times 6} & \mathbf{1}_{6 \times 18} \\ \mathbf{1}_{6 \times 6} & \mathbf{1}_{6 \times 6} & I_6 & \mathbf{0}_{6 \times 18} \\ \mathbf{1}_{18 \times 6} & \mathbf{1}_{18 \times 6} & \mathbf{0}_{18 \times 6} & I_{18} \end{pmatrix}$$

Let $p_i \in \mathbb{R}^2$ be **unmirrored** global letterbox `xy` (`pose.py:L427`) and $\sigma = \sigma(t)$ be frame torso scale. Pairwise displacement & 8-d edge vector $e_{ij} \in \mathbb{R}^8$ (`pose.py:L254-L257`):

$$\delta_{ij} = \frac{p_i - p_j}{\sigma(t)} \in \mathbb{R}^2, \qquad d_{ij} = \sqrt{\|\delta_{ij}\|_2^2 + 10^{-8}} \in \mathbb{R}_+, \qquad e_{ij} = \Big[\delta_{ij},\; d_{ij},\; \exp\!\left(-\frac{d_{ij}^2}{2\sigma_r^2}\right)_{r=1}^4,\; 1\Big]$$

with fixed $\sigma_r \in (0.10,\, 0.25,\, 0.60,\, 1.50)$. In each `EdgeGraphBlock` ($H$ heads, $d_h = D/H$), using $h^{(0)} = \text{concat}(h_{\text{sub}}) + e_{\text{region}}(\text{region\_id}) \in \mathbb{R}^{(BT) \times 36 \times D}$:

$$Q, K, V = \operatorname{LN}_1(h)\, W_{qkv}, \qquad B_{ij} = W_{e2} \tanh(W_{e1} e_{ij} + b_{e1}) + b_{e2} \in \mathbb{R}^{D}$$

$$S^{(h)}_{ij} = \frac{\big\langle q^{(h)}_i,\; k^{(h)}_j + b^{(h)}_{ij} \big\rangle}{\sqrt{d_h}}, \qquad \text{allowed}_{ij} = A_{ij} \land m_i \land m_j$$

$$\alpha^{(h)}_{ij} = \begin{cases} \operatorname{softmax}_{j:\,\text{allowed}_{ij}}\!\big(S^{(h)}_{ij}\big) & \text{if } \exists j:\, \text{allowed}_{ij} \\ 0 & \text{otherwise} \end{cases}, \qquad h_{\text{out}} = h_{\text{in}} + \operatorname{Res}_{\text{attn+FFN}}\!\Big(\sum_j \alpha_{ij} v_j\Big)\odot m_i$$

And final cross-articulator residual writeback with active scalar parameter $\gamma \in \mathbb{R}^1$ ($\gamma_{\text{init}} = 0.1$):

$$\Delta = \tanh(\gamma) \cdot \big(h^{(L)}_{\text{cross}} - h^{(0)}\big) \odot m_{\text{cross}}, \qquad \tilde{h}_{\text{hand}}[k] = \begin{cases} h_{\text{hand}}[k] + \Delta[k] & k \in \{0,4,8,12,16,20\} \\ h_{\text{hand}}[k] & \text{otherwise (15 joints untouched)} \end{cases}$$

**Verification Summary:**
- **Correct Unmirrored Cross-Articulator Geometry:** `CrossArticulatorGraph` uses **unmirrored** `xy` (`pose.py:L239-L242, L427`), so left-hand-to-left-cheek vs. right-hand-to-left-cheek Euclidean separations $\delta_{ij}, d_{ij}$ are physically accurate—unlike the intra-hand `1 - x` projection.
- **Exact Mask Propagation Guarantee:** Because `allowed_{ij} = A_{ij} & m_i & m_j`, **zero information leaks from invalid keypoints** (`m_j = 0`) into attention weights, LayerNorm (`h` masked to zero at invalid rows after both `out` and `ffn` residuals, `L139, L141`), or `NodePool`. Because `A_{ii} = 1`, any single valid isolated node keeps $\text{allowed}_{ii} = 1$, giving $\alpha_{ii} = 1$ (no NaN softmax). Cloned output dict `{k: v.clone()}` (`L266`) preserves autograd correctness on indexed writeback.
- **Residual Gate & Topological Limitations:**
  1. `RegionGraphEncoder` (`layers = 3`) + `CrossArticulatorGraph` leaves **intra-hand edges (`L<->L`, `R<->R`) absent**, preventing direct finger-tip interactions.
  2. `adj[body, face] = 0` prevents head-tilt / neck-shoulder postural interaction directly between body and face nodes.
  3. **15 Non-Key Hand Nodes Bypassed:** Only 6 of 21 hand nodes receive $\Delta$, after which `NodePool` weights all 21 nodes (`L430`)—forcing `NodePool` to synthesize cross-articulator awareness mostly through 6 key indices.
  4. **Redundant Constant Bias in Edge Features:** `e_{ij}` appends `torch.ones_like(dist)` (`L257`, `L125`), linearly redundant with `nn.Linear(8, D)`'s learned bias $b_{e1}$.
  5. **Global Spatial Interaction Blindness to Orientation:** Note `B_{ij}` modifies attention **keys/scores only** (`q_i^T (k_j + b_ij)`), **not value messages $v_j$**! Edge vectors $(\delta_{ij}, \text{RBF})$ modulate *who attends to whom*, but never inject relative 2D displacement directly into aggregated Value vectors $v_j$ (unless inferred indirectly from global coordinates).

---

## C. Reliability & Uncertainty Modelling

### C.1 Absence of Evidential / Dirichlet Mathematics

Direct inspection of all 13 Python files under `src/isl_mega/` confirms:
- **Zero Dirichlet variables:** No `alpha`, `evidence`, `Softplus(evidence)`, Subjective Logic belief masses $b_m = e_m / S$, uncertainty mass $u = M/S$, or KL-against-$\text{Dir}(1)$ regularizer exists anywhere in code.
- **Zero reliability targets:** `fusion_aux["reliability"]` is emitted by [`ReliabilityAwareFusion.forward_with_aux`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L200-L205) and retained in [`ISLMEGA.encode`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/model.py#L167), yet has **zero consumers** in [`losses.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py), [`train.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py), or [`evaluate.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/evaluate.py).

### C.2 Exact Fusion, Quality Vector & Lag Imputation Equations

Let $\mathcal{S}$ be the 10 streams ([`STREAM_NAMES`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L13-L17)):
`(0) left_hand_rgb, (1) right_hand_rgb, (2) face_rgb, (3) body_rgb, (4) left_hand_pose, (5) right_hand_pose, (6) body_pose, (7) face_pose, (8) motion, (9) fingerspelling`.

#### Equation 7 — Gated Stream Mixture & Log-Reliability Bias ([`fusion.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L124-L155))

Per stream $s \in \{0..9\}$ at frame $t$, let $z_{b,t,s} \in \mathbb{R}^{D_f}$ be the Transformer-interacted feature (`self.interaction` with null fallback row when `all_missing`), $q_{b,t,s} \in \mathbb{R}^5$ the quality vector, and $m_{b,t,s} \in \{0,1\}$ stream availability (`available`). When `reliability_uncertainty_gate=True`:

$$u_{b,t,s} = \operatorname{MLP}_{\text{gate}}\!\big([z_{b,t,s},\, q_{b,t,s}]\big) \in \mathbb{R}, \qquad r_{b,t,s} = m_{b,t,s} \cdot \sigma\!\big(\operatorname{MLP}_{\text{rel}}(q_{b,t,s})\big) \in [0, 1)$$

$$\ell_{b,t,s} = u_{b,t,s} + \log\big(\max(r_{b,t,s},\; 10^{-4})\big), \qquad w_{b,t,s} = \begin{cases} \dfrac{m_{b,t,s} \exp(\ell_{b,t,s})}{\sum_{j} m_{b,t,j} \exp(\ell_{b,t,j})} & \text{if } \sum_j m_{b,t,j} \ge 1 \\ 0 & \text{otherwise} \end{cases}$$

$$h^{\text{pool}}_{b,t} = \begin{cases} \sum_{s=0}^{9} w_{b,t,s}\, z_{b,t,s} & \text{if } \sum_j m_{b,t,j} \ge 1 \\ \theta_{\text{null}} & \text{otherwise} \end{cases} \quad \in \mathbb{R}^{B \times T \times D_f}$$

- **Exact 5-D Quality Vector Semantics Across All 10 Streams ([`vision.py:L243`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L243), [`pose.py:L423`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L423), [`model.py:L107-L136`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/model.py#L107-L136)):**

| Stream Group | `q[..., 0]` | `q[..., 1]` | `q[..., 2]` | `q[..., 3]` | `q[..., 4]` |
|---|---|---|---|---|---|
| **4 RGB Streams** | `box_valid` ($\{0,1\}$) | `box_area` ($\in [0,1]$, $\sim 0.02$ hands vs $\sim 0.8$ body) | `crop_sharpness` ($\frac{1}{2}\overline{\lvert \nabla I \rvert} \in [0,1]$, typ. $0.01\text{--}0.08$) | `box_score` (or `1.0`/`0.25` uncalibrated proxy!) | `frame_mask` |
| **4 Pose Streams** | `joint_coverage` ($\frac{1}{N}\sum m_i$) | `mean_conf` ($\in [0,1]$, or **`1.0` when `known=0`**!) | `mean(conf_known)` ($\{0,1\}$) | `region_avail` ($\{0,1\}$, set identical to mask `L433`) | `frame_mask` |
| **`motion` (8)** | `body_avail` ($\{0,1\}$) | `diff_energy` ($\overline{\lvert I_t - I_{t-1} \rvert} \sim 0.01\text{--}0.05$) | `body_box_score` ($\in [0,1]$, `0.25` fallback) | `body_avail` (duplicate of `0`) | `frame_mask` |
| **`fingerspelling` (9)** | `fs_avail` (`L_ok or R_ok`) | `L_ok or R_ok` (identical to `0`!) | `spot_prob` (`span_routing=T`) or `gate` | $\max_c \operatorname{softmax}(\text{ctc})_c$ (**$\approx 0.99$ when `<blank>`!**) | `frame_mask` |

### C.3 Proof: Uncertainty Fusion Is Uncalibrated Learned Gating (Not Calibrated Uncertainty)

Four independent mathematical facts establish that `ReliabilityAwareFusion` is **strictly an uncalibrated learned softmax gating mechanism**, not a calibrated probabilistic reliability or uncertainty estimator:

1. **Zero Calibration Supervision / Identifiability Target:** No loss term compares $r_{b,t,s}$ to empirical modality accuracy, noise variance, corruption rate, or a proper scoring rule (Brier / NLL / ECE). Gradient updates to `reliability_net` flow exclusively through the downstream sequence CE / planning losses via $\partial w / \partial \ell$.
2. **Exact Additive Logit Non-Identifiability Gauge:** Because $\text{MLP}_{\text{gate}}([z_{b,t,s}, q_{b,t,s}])$ also receives $q_{b,t,s}$ (`fusion.py:L142`) and combines linearly with $\log r_{b,t,s}$:
   $$\ell_{b,t,s} = \Big(u_{b,t,s}(z, q) + \delta(q)\Big) + \Big(\log r_{b,t,s}(q) - \delta(q)\Big)$$
   for any arbitrary scalar function $\delta(q)$ such that $r' = e^{-\delta(q)} r \in [10^{-4}, 1)$ leaves attention weights $w_{b,t,s}$ and **every loss in `ISLMEGA` mathematically identical**. Consequently, the model can set $r_{b,t,s} \approx 1$ everywhere ($\log r \approx 0$) and route all gating through `quality_gate`, or invert $r_{b,t,s}$ while compensating in `quality_gate`. Nothing constrains $r_{b,t,s}$ to reflect true stream reliability or uncertainty!
3. **Catastrophic Dimensional Collision Across One Shared `71`-Parameter MLP:** [`fusion.py:L79-L83`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L79-L83) instantiates a **single** `Linear(5, 10) -> GELU -> Linear(10, 1)` (`71` weights total) applied identically to all 10 streams **without a stream-ID embedding** (`self.reliability_net(q)` at `L145`). Look at the table above:
   - On channel `3`, high score for `pose` (`1.0` = valid keypoint), `rgb` (`0.95` = strong detector), and `fingerspelling` (`0.99` = $\max_c \text{softmax}(\text{CTC})$ predicting **`<blank>` with 99% certainty during non-fingerspelling lexical signs!**) are all treated by the **identical weight** $W_{\text{rel}}[3]$! When fingerspelling is *inactive* (`<blank>` probability $0.99$), channel 3 maximizes—sending a false *"high confidence"* signal into the shared MLP.
   - On channel `2`, `rgb` sharpness ($\approx 0.03$), `pose` confidence-known flag ($0$ or $1$), `motion` body box score ($0.25$ or $1.0$), and `fs` spotting probability (`spot_prob`) share one scalar neuron!
   - When real detector confidence is absent (`confidence_known=False`), `pose` reports `conf_mean = 1.0` (`q[...,1] = 1.0`) and `rgb` reports `box_score = 1.0` (or `0.25`).
4. **Entropy Regularizer Contradiction (`v2` vs `v3`):** When `fusion_entropy_weight > 0` (`configs/research.yaml:L111`), [`compute_fusion_entropy_loss`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L105-L125) minimizes $1 - H(w)/\log S_{\text{avail}}$, pushing $w \to 1 / S_{\text{avail}}$ (uniform stream weights), directly opposing suppression of noisy or degraded streams. (`research_v3_compact.yaml:L123` rightly disables this weight at `0.0`.)

**All-Missing-Frame Imputation Behavior ([`fusion.py:L184-L196`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L184-L196)):** If frame $t$ has all 10 streams unavailable (`all_missing=True`), `pooled[b, t]` starts as `self.null` (`L155`). However, **Lag Attention still queries all neighbor frames within offset $\pm r$ and $|\Delta t| \le W$** (`L167-L195`). Therefore, an entirely blank/dropped frame surrounded by valid neighbors receives `aligned != 0` interpolated cross-stream features (`fused = output([null, lag_out(aligned)])`), whereas a frame in a completely empty clip gets `any_keys = False => aligned = 0`. Meanwhile, [`compute_cross_modal_alignment_loss`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L161-L188) is unweighted by confidence (`q_s`), treating `conf = 0.01` valid landmarks equally to `conf = 0.99` landmarks.

---

## D. Temporal Modelling, Timestamps, Receptive Field & Causality

### D.1 Actual Timestamps & Variable Frame Rate (VFR) Invariance Audit

Timestamps propagate in `float64` through [`validate_times`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L24-L28), clip alignment, and origin shift (`rel_times = (video_times - video_times[0]).astype(np.float32)`, `data.py:L342`). In [`convert_video_pose_npz.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/convert_video_pose_npz.py#L50-L75), OpenCV falls back to constant `25.0` fps if `CAP_PROP_FPS <= 0` or PTS is non-monotonic. [`ReliabilityAwareFusion`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L165) falls back to `25.0` fps if `timestamps=None`.

**Is the model VFR-invariant? No — only half of the temporal operations use elapsed seconds:**

| Component | Uses Real $\Delta t$ / `timestamps`? | VFR-Invariant? | Exact Defect Under VFR Changes (`15` vs `25` vs `60` fps) |
|---|---|---|---|
| `align_pose_to_video` (`data.py`) | Yes (`gap <= max_gap_seconds`) | **Yes** | Linear interp over physical gap $\le 0.20\text{s}$. |
| `PoseExpert._kinematic_features` | Yes (`/ dt`) | Partial | Velocity magnitudes VFR-scaled, **except `tanh(v/5), tanh(a/25)`** still respond to frame-step noise jitter scaled by $1/\Delta t^2$ at 60 fps (`x 5.76` variance amplification over 25 fps!). |
| `MotionExpert` (`motion.py`) | **No** ($\lvert I_t - I_{t-1} \rvert$, `Conv3d(k=3)`, `Conv1d(k=7)`) | **No** | Frame difference $\lvert I_t - I_{t-1} \rvert$ halves when FPS doubles (`60` vs `30`); receptive field shrinks from `0.48s` (`25 fps`) to `0.20s` (`60 fps`). |
| `FingerspellingExpert` (`fingerspelling.py`) | **No** (`BiGRU`, `spot_conv(k=5)`) | **No** | RNN gates and `k=5` spotting kernel advance per frame, not per second. |
| `ContinuousTimeLagBias` (`fusion.py`) | Yes (`dt` + Fourier basis), clipped by `r=4` frames | **No (Truncated at High FPS!)** | See Eq. 8 below: `offsets = [-4..+4]` truncates $W=0.20\text{s}$ down to $\pm 0.067\text{s}$ at `60 fps`! |
| `MultiScaleTemporalStem` (`temporal.py`) | **No** (`Conv1d` `k=(3,5,9,15)`, `d=(1,1,2,4)`) | **No** | Effective window `57 frames` spans **`3.80s` at `15 fps` vs `0.95s` at `60 fps`**! |
| `_adaptive_pool` (`temporal.py`) | Partial (`dt` in `salience_in` & `slot_std`) | **No** | Slot budget `m_valid = ceil(L / 2)` is hardcoded to **frame count** (`L/2` slots regardless of clip FPS or duration); frame `0` hardcodes `step_dt[:, 0] = 0.04` (`25 fps`); `delta = fine[t] - fine[t-1]` is not divided by `dt`. |
| `RelativeTimeBias` (`temporal.py`) | Yes (`coarse_times[i] - coarse_times[j]`) | **Severely Quantized** | See Eq. 9 below: ~47%–55% of log-buckets sit below physical frame spacing (`< 0.028s`). |
| `GlobalTemporalBlock` | Self-attn: Yes (`time_bias`); Conv: **No** | Partial | Attention edges use seconds; `depthwise(k=15)` operates over coarse slot indices (`+-7` slots). |
| `LatentPlanGrounder` | Yes (normalized $\tau \in [0,1]$ in `order_loss`) | Partial | Dynamic slot budget `ceil(valid_len / target_stride)` uses **frame count**, not clip duration (`target_stride = 4` frames = `0.067s` at 60 fps vs `0.27s` at 15 fps). |

---

### D.2 Continuous Lag Bias & High-FPS Truncation

#### Equation 8 — `ContinuousTimeLagBias` ([`fusion.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L20-L54, L157-L173))

With neighborhood index window $k \in \{-r, \dots, +r\}$ (`r = lag_radius_frames = 4`, $K = 2r + 1 = 9$), $W = \text{lag\_window\_seconds} = 0.20\text{s}$, $\Delta t_{b,t,k} = t_{b, \text{clip}(t+k)} - t_{b,t}$, normalized lag $\tilde{\tau} = \operatorname{clamp}(\Delta t / W,\, -2,\, 2)$:

$$\phi(\tilde{\tau}) = \big[\,\tilde{\tau},\; \tilde{\tau}^2,\; |\tilde{\tau}|,\; \sin(\pi\tilde{\tau}),\; \cos(\pi\tilde{\tau}),\; \sin(2\pi\tilde{\tau}),\; \cos(2\pi\tilde{\tau})\,\big] \in \mathbb{R}^7$$

$$\text{bias}_{b,t,h,k,s} = \operatorname{MLP}_{\phi}\!\big(\phi(\tilde{\tau}_{b,t,k})\big)_h + \alpha_{h,s}\,\tilde{\tau}_{b,t,k} + \beta_{h,s}$$

masked whenever $(t+k \notin [0, T)) \lor (|\Delta t_{b,t,k}| > W \land k \neq 0) \lor (\neg m_{b,t+k,s})$.

- **Mathematical Check & Critical Frame-Rate Limitation:** When $|\Delta t| \le W$, $\tilde{\tau} \in [-1, 1]$, on which $\{1, \cos(\pi \tilde{\tau}), \sin(\pi \tilde{\tau}), \cos(2\pi \tilde{\tau}), \sin(2\pi \tilde{\tau})\}$ forms a clean Fourier harmonic basis supplemented by $\tilde{\tau}, \tilde{\tau}^2, |\tilde{\tau}|$, and per-stream signed anticipation/delay slopes $\alpha_{h,s}$ (initialized `0`, differentiable directly wrt sign/lag asymmetry). However:
  - At **`r = 4` frames** (`configs/research_v3_compact.yaml:L56` & `research.yaml:L43`):
    - At **`25 fps`** ($\delta = 0.04\text{s}$): $r \cdot \delta = 0.16\text{s} < 0.20\text{s}$ (only $80\%$ of configured `lag_window_seconds`).
    - At **`30 fps`** ($\delta = 0.0333\text{s}$): $r \cdot \delta = 0.133\text{s}$ ($66.7\%$ coverage).
    - At **`50 fps`** ($\delta = 0.02\text{s}$): $r \cdot \delta = 0.08\text{s}$ ($40\%$ coverage).
    - At **`60 fps`** ($\delta = 0.0167\text{s}$): $r \cdot \delta = \mathbf{0.067\text{s}}$ (**`66.7%` of the intended $0.20\text{s}$ window is physically unreachable!**).
  - Thus `lag_radius_frames=4` silently overrides `lag_window_seconds=0.20` on any video faster than 20 fps!

---

### D.3 Relative Time Position Encoding & Dead Sub-Frame Log-Buckets

#### Equation 9 — `RelativeTimeBias` ([`temporal.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L48-L61))

Given coarse event timestamps $t^{\text{coarse}} \in \mathbb{R}^{B \times M}$, $\Delta_{ij} = t^{\text{coarse}}_i - t^{\text{coarse}}_j$, `min_dt` $= 10^{-5}\text{s}$, $T_{\max} = \text{max\_elapsed\_seconds} = 30.0\text{s}$, and $B_{\text{half}} = \text{buckets}/2$:

$$\text{scaled}_{ij} = \frac{\log\!\big(\operatorname{clamp}(|\Delta_{ij}|,\, 10^{-5},\, T_{\max}) / 10^{-5}\big)}{\log(T_{\max} / 10^{-5})} \in [0, 1]$$

$$b_{\text{mag}}(i,j) = \operatorname{clamp}\!\Big(\big\lfloor \text{scaled}_{ij} \cdot (B_{\text{half}} - 1) \big\rfloor,\; 0,\; B_{\text{half}} - 1\Big), \qquad b(i,j) = b_{\text{mag}}(i,j) + \mathbf{1}_{[\Delta_{ij} \ge 0]} B_{\text{half}}$$

**Exact Quantized Interval Audit (Verified via Python against `temporal.py:L57-L60`):**

Because `min_dt` is set to **$10^{-5}\text{s}$ (`0.01 ms`, 3,300x smaller than one 30 fps frame)** while uniform logarithmic spacing spends **half the dynamic range (`10 us .. 28 ms`) below a single frame**:

| Profile (`buckets`, `B_half`) | Exact Sub-Frame Bucket Intervals (`local 0 .. k`) | Dead Buckets at `25 fps` (`coarse dt ~ 0.08s`) | Human Sign Scales Squeezed Into Remaining Active Positive Buckets |
|---|---|---|---|
| **`v3` compact** (`32`, `16`) | `0`: `[10us, 27us)`, `1`: `[27us, 73us)`, `2`: `[73us, 197us)`, `3`: `[197us, 534us)`, `4`: `[0.53ms, 1.44ms)`, `5`: `[1.44ms, 3.90ms)`, `6`: `[3.90ms, 10.5ms)`, `7`: `[10.5ms, 28.5ms)` | **`17 / 32` (`53.1%`) dead**! (`neg 0..8` + `pos 1..7` + diagonal goes to `16`; `8`: `[28.5ms, 77.0ms)` barely touched at `dt=0.08s`!) | **All non-diagonal lags `[0.077s .. inf)` share just `7` positive (`25..31`) & `7` negative (`9..15`) buckets!** E.g. `0.08s, 0.10s, 0.20s` all collide in `local=9`; `5.0s` & `10.0s` collide in `local=13`. |
| **`v2` / `v1` full** (`64`, `32`) | `0`: `[10us, 16us)`, ..., `15`: `[13.6ms, 22.0ms)`, `16`: `[22.0ms, 35.6ms)`, `17`: `[35.6ms, 57.7ms)` | **`35 / 64` (`54.7%`) dead** at `25 fps` coarse `dt=0.08s` (`neg 0..17` + `pos 1..17` unused; `48.4%` still dead at `30 fps` raw!) | Almost **55% of `RelativeTimeBias.table` receives zero gradient updates** during standard 25–30 fps training. |

*(Additional detail: Diagonal $i=j$ (`dt = 0`) hits `local=0, positive` (`table[B_half]`), whereas `table[0]` (`dt < 0, |dt| < 27us` or `16us`) is mathematically unreachable for any valid strictly-increasing clip sequence with `dt_min > 0.03 ms`.)*

---

### D.4 Differentiable Adaptive Consolidation & Unpooling Equations

#### Equation 10 — Event Salience Pooling, Retrieval & Reverse Unpooling ([`temporal.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L155-L195, L235-L291, L316-L329))

Let $X_{\text{fine}} \in \mathbb{R}^{B \times T \times D}$ be the `MultiScaleTemporalStem` output, $m_{\text{fine}} \in \{0,1\}^{B \times T}$ the `lengths` mask, $M = \lceil T/2 \rceil$, $M_b = \lceil L_b / 2 \rceil$, $\Delta X_t = (X_t - X_{t-1}) \mathbf{1}_{m_t \land m_{t-1}}$, $\Delta \tau_t = \max(\tau_t - \tau_{t-1}, 10^{-4})$ ($\Delta\tau_0 := 0.04$):

$$s_t = m_{\text{fine},t} \cdot \Big(\operatorname{softplus}\!\big(\operatorname{MLP}_{\text{sal}}([X_t,\, \Delta X_t,\, \tfrac{\sqrt{\|\Delta X_t\|_2^2 + 10^{-8}}}{\sqrt{D}},\, \log(1 + \Delta \tau_t)])\big) + 0.1\Big) \ge 0.1$$

$$\text{slot\_pos}_t = \frac{\sum_{u \le t} s_u - 0.5\, s_t}{\max\!\big(\sum_{u=0}^{L_b - 1} s_u,\; 10^{-6}\big)} \cdot M_b \in [0, M_b], \qquad c_j = j + 0.5, \qquad \sigma_s = \operatorname{softplus}(\log\sigma_0) + 0.30$$

$$L_{b,j,t} = -\frac{(\text{slot\_pos}_{b,t} - c_j)^2}{2 \sigma_s^2}, \qquad A_{b,j,t} = \operatorname{softmax}_{t:\, m_{\text{coarse},j} \land m_{\text{fine},t}}\!(L_{b,j,t}) \in \mathbb{R}^{B \times M \times T}$$

$$t^{\text{coarse}}_{b,j} = \sum_t A_{b,j,t}\,\tau_{b,t}, \qquad \text{std}_{b,j} = \sqrt{\sum_t A_{b,j,t} (\tau_{b,t} - t^{\text{coarse}}_{b,j})^2 + 10^{-6}} - 10^{-3}$$

$$B_{\text{rev}}[b, j, t] = \frac{A_{b,j,t}}{\max\!\big(\sum_{k=0}^{M_b-1} A_{b,k,t},\; 10^{-6}\big)}, \qquad X^{\text{up}}_{b,t} = \sum_{j=0}^{M_b-1} B_{\text{rev}}[b, j, t] \cdot X^{\text{retrieved}}_{b,j}$$

**Mathematical Verification & Subtle Properties:**
1. **Verified Exact Batch-Padding Invariance & Singularity Safety:** Because `total_s` sums only over masked $m_{\text{fine}}$, $L_{b,j,t}$ only depends on valid frames $0..L_b-1$, and `slot_std` uses smooth offset $\sqrt{\text{var}+10^{-6}} - 10^{-3}$ instead of `clamp_min` (`L284`), `out[:, :L_b]` is mathematically invariant to batch padding length $T$ and finite/differentiable even at $L_b = 1$ or static inputs.
2. **Coarse Row Normalization + Reverse Column Normalization:** `A` is row-normalized over $t$ (weighted frame mean into coarse slots), and `B_rev` normalizes columns over $j$ so each frame $t$'s unpooled feature $X^{\text{up}}_{b,t}$ is a convex combination of event slots ($\sum_j B_{\text{rev}}[b,j,t] = 1$ wherever any slot assigns positive mass). *Edge nuance:* if an edge frame $t$ receives near-zero weight $\sum_k A_{b,k,t} \ll 10^{-6}$ across all slots (possible if $\sigma_s$ shrinks toward its $0.30$ floor and a small slot lies $> 5\sigma_s$ away from a sharp peak), `.clamp_min(1e-6)` attenuates $X^{\text{up}}_{b,t}$ toward zero instead of renormalizing.
3. **Strict Non-Causality & Non-Streaming:** Global clip total $\sum_{u=0}^{L_b-1} s_u$ appears in the denominator of every $\text{slot\_pos}_t$ (`L262`), preventing causal/streaming evaluation before the end of the utterance.
4. **Adaptive Slot Count Not Adaptive To Motion Duration:** Although *locations* of the $M_b$ slots warp around kinematic transitions ($s_t \ge 0.1$), total slot capacity $M_b = \lceil L_b/2 \rceil$ still grows linearly with frame count $L_b$ rather than total change $\int s(t)\,dt$ or clip duration.

---

### D.5 Temporal Receptive Field & Causality Analysis

| Stage | Module | Exact Temporal Receptive Field (Frames) | Wall-Clock RF at `15 / 25 / 30 / 60 fps` | Temporal Symmetry / Causality |
|---|---|---|---|---|
| **Pose Kinematics** | `PoseExpert._kinematic_features` | `[t-2, t]` (3 frames) | `0.13s / 0.08s / 0.07s / 0.03s` | **Backward causal** (`v`: 1st diff, `a`: 2nd diff) |
| **Motion 3D Stem + 1D** | `MotionExpert` (`v1`/`v2`) | `diff[t-1..t]` + `k_t=3` + `k_t=3` + `k=7` -> `[t-6, t+5]` (`12` frames; `10` if `k=5`) | `0.80s / 0.48s / 0.40s / 0.20s` (`k=7`) | **Mixed** (backward `diff`, symmetric non-causal convs) |
| **Fingerspelling** | `FingerspellingExpert` (`BiGRU`) | **Full clip `[0, L_b - 1]`** (`spot_conv` `k=5` = `[t-2, t+2]`) | Full utterance | **Strictly bidirectional / non-causal** |
| **Cross-Stream Lag** | `ReliabilityAwareFusion` | `[t - 4, t + 4]` (9 frames) clipped by $\lvert \Delta t \rvert \le 0.20\text{s}$ | `<= 0.40s` (`15-25fps`), `0.27s` (`30fps`), `0.13s` (`60fps`) | **Strictly bidirectional / symmetric window** |
| **Temporal Stem** | `MultiScaleTemporalStem` | `k=(3,5,9,15), d=(1,1,2,4)` -> `[t-28, t+28]` (**`57` frames**) | **`3.80s / 2.28s / 1.90s / 0.95s`** | **Strictly bidirectional / symmetric `padding=(k//2)*d`** |
| **Global Conformer** | `GlobalTemporalBlock x 6 / 12` | Attention: **Full `[0, L_b - 1]`**; Conv branch (`k=15` coarse): $\pm 7 L_{\text{global}}$ slots ($\approx \pm 84 / \pm 168$ fine frames) | Full utterance (attn); `+-3.36s` conv (`v3` at `25 fps`) | **Strictly bidirectional / non-causal** (zero attn mask in future direction) |
| **Detail Retrieval** | `AdaptiveDetailRetrieval` (`v2/v3`) / `FineDetailRetrieval` (`v1`) | `v2/v3`: **Full `[0, L_b - 1]`**; `v1`: local `window = 7` | Full utterance (`v2/v3`) / `~0.28s` at 25 fps (`v1`) | **Strictly bidirectional / non-causal** |
| **Latent Planner** | `LatentPlanGrounder` | **Full `[0, L_b - 1]`** (`K=16` or `32` global cross-attn slots) | Full utterance | **Strictly bidirectional** (with soft clip-relative monotonicity penalty if enabled) |
| **Language Model** | `T5VisualDecoder` (`FLAN-T5`) | Encoder: **Full `[0, L_b + K - 1]`**; Decoder: **Target-causal** | Full visual + plan context | **Encoder bidirectional; Target decoder autoregressive causal** |

**Asymmetric Kinematic Phase Lag:** Note that `PoseExpert` (`v_t, a_t`) and `MotionExpert` (`|I_t - I_{t-1}|`) use strictly backward finite differences (`t - 1` -> `t`), placing derivative peak responses $\approx 0.5\text{--}1.0$ frame *later* than centered zero-phase convolutions (`MultiScaleTemporalStem`, `Conv3d`), rather than central differences $\frac{p_{t+1}-p_{t-1}}{2\Delta t}$. Meanwhile, the rest of the backbone is completely non-causal, precluding real-time/online sign translation without redesigning every temporal block.

---

## Master Verification Table of Key Equations

| Eq. | Component | Output Shape | Consistent? | Core Mathematical Assumptions | Matches Code? | Primary Limitations |
|---|---|---|---|---|---|---|
| **1** | Letterbox affine (`data.py`) | `[T, 66, 2]` | Yes | Square padding centered; aspect ratio preserved for points | Exact (`L52-65`) | `_bbox` + `grid_sample` squash rectangular ROIs onto square ViT crops |
| **2** | `estimate_torso_scale` (`pose.py`) | `[B, T]` | Yes (clamped `[0.05, 2.0]`) | Constant shoulder visibility ratio `1 : 0.65_body : 1.25_face`, 2D fronto-parallel shoulders | Exact (`L61-86`) | **Discrete 4-tier cascade jumps on single-frame dropout**, amplifying velocity/accel spikes |
| **3** | Local anchors (`pose.py`) | `[B, T, N_g, 2]` | **Partial (Bug in Body)** | Wrist or both shoulders consistently visible; centroid representative | Match (`L362-389`), **contradicts `docs` on body `L378`** | Body `.any()` jumps `0.5` shoulder-width on single shoulder; wrist fallback unflagged |
| **4** | 11-d kinematics (`pose.py`) | `[B, T, 66, 11]` | Yes | Smooth coordinates, bounded physical $v < 5\text{s}^{-1}, a < 25\text{s}^{-2}$ | Exact (`L300-324`) | `[0:2]` translation-variant; global `v` differentiated across jumping $\sigma(t)$; $\tanh$ saturation; fake `conf=1.0` |
| **5** | Mirror canon. (`pose.py`) | `[B, T, 21, 11]` | **Inconsistent Cross-Modally** | Signer centered at `x=0.5`; L/R handshapes mirror symmetric | Exact (`L396-404`) | `1-x` shifts off-center signers; **pose mirrored while RGB ViT crops unmirrored**; L/R not swapped |
| **6** | `CrossArticulatorGraph` (`pose.py`) | `4 x [B, T, N_g, D]` | Yes (autograd-safe) | Key fingertips+wrist summarize intra-hand pose; 1 layer bipartite suffices | Exact (`L198-273`) | **Diameter 8 > layers 3** + zero intra-hand edges = no fingertip interaction; 15 nodes bypassed |
| **7** | `ReliabilityAwareFusion` (`fusion.py`) | `[B, T, D_f]` | Algebraic yes; **Statistically No** | Shared 5-d `q` semantics across all streams; no calibration needed | Exact (`L124-155`) | **Uncalibrated learned gate**, $\delta(q)$ non-identifiability, `71`-param semantic collision, no Dirichlet |
| **8** | `ContinuousTimeLagBias` (`fusion.py`) | `[B, T, H, K*S]` | Yes within `r` | Async lag bounded inside `[-r, +r]` frames AND $\lvert \Delta t \rvert \le W$ | Exact (`L20-53, 157-172`) | `r=4` clips $W=0.20\text{s}$ down to `0.067s` at `60 fps` |
| **9** | `RelativeTimeBias` (`temporal.py`) | `[B, H, M, M]` | Yes | Uniform log density across `[10us, 30s]` | Exact (`L48-61`) | **`~47%–55%` of buckets dead `< 28ms`**; `[0.08s, 30s]` crushed into 7 buckets in `v3` |
| **10** | `_adaptive_pool` + `B_rev^T` (`temporal.py`) | `[B, M, D] -> [B, T, D]` | Yes (padding-invariant) | Fixed $M = \lceil L/2 \rceil$ suffices per clip; soft Gaussian RBF events cover frames | Exact (`L235-291, 316-329`) | Global denominator non-streaming; slot count $M$ not VFR/duration invariant |

---

## E. The Five Most Serious Mathematical & Representation Problems

### 1. Intra-Hand & Fingertip Graph Reachability Blindness (`HAND_EDGES` Diameter = 8 vs. `layers = 3`)
- **Code Locations:** [`pose.py:L19-L26, L47-L58, L281`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L19-L58), [`configs/research_v3_compact.yaml:L31, L39`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/research_v3_compact.yaml#L31-L39).
- **Root Mathematical Failure:** `HAND_EDGES` forms a 5-finger kinematic tree of depth `4` (diameter `8`) with zero inter-knuckle edges, yet both `research_v3_compact.yaml` and `research.yaml` configure `pose.layers = 3`. Consequently, **not even wrist `0` reaches any fingertip (`4, 8, 12, 16, 20`)** in `RegionGraphEncoder`, and fingertips sit `8` hops apart. Furthermore, `CrossArticulatorGraph` sets intra-hand adjacency to $I_6$, so **two fingertips on the same hand (`thumb_tip 4` vs. `index_tip 8`) never communicate across any graph edge in the entire pose network**. Phonological handshape apertures (pinch, contact, finger splay) can only be inferred indirectly by the global permutation-invariant `NodePool` weighted average over single-joint features (with 15 non-key joints never even touched by cross-graph updates) rather than relational edge biases.

### 2. Uncalibrated, Gauge-Non-Identifiable & Cross-Modally Conflicted `reliability_net`
- **Code Locations:** [`fusion.py:L76-L85, L142-L155`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L76-L155), [`model.py:L107-L136`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/model.py#L107-L136), [`losses.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py), [`train.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/train.py).
- **Root Mathematical Failure:** Despite documentation framing it as calibrated structural uncertainty (`reliability_uncertainty_gate`), **zero Dirichlet evidence, zero calibration targets, and zero scoring rules exist** (`fusion_aux["reliability"]` has zero consumers). Worse: (a) summing $\text{MLP}_{\text{gate}}([z_s, q_s]) + \log r_s(q_s)$ creates an exact shift gauge symmetry $\pm\delta(q)$, making $r_s$ mathematically unidentifiable; and (b) one **shared `71`-parameter MLP without stream identity** consumes 10 incompatible $q_s$ definitions—equating RGB `area`/`sharpness` (`~0.02–0.05`) with `pose` availability (`1.0`), missing-confidence proxy (`1.0`), and fingerspelling's $\max_c \operatorname{softmax}(\text{ctc})$ (`~0.99` when predicting `<blank>` outside fingerspelling!).

### 3. Vision–Pose Chirality Contradiction, Off-Center Reflection Error & False Handedness Invariance
- **Code Locations:** [`pose.py:L393-L405`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L393-L405), [`vision.py:L190-L203`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L190-L203), [`losses.py:L161`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L161), [`fingerspelling.py:L90`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fingerspelling.py#L90).
- **Root Mathematical Failure:** Enabling `share_hand_encoder=True` flips `right_hand` pose (`1 - x, -rel_x, -vx, -ax`) inside `PoseExpert`, creating three simultaneous defects: (a) **`AppearanceExpert` does NOT flip right-hand RGB crops** before its shared `self.hand` ViT, yet InfoNCE (`losses.py:L161`) pulls unflipped `R_rgb` directly toward features of flipped `R_pose`; (b) $x \mapsto 1 - x$ reflects across image center $0.5$ rather than torso midline, corrupting global node channel `[0]` whenever the signer is laterally shifted; and (c) downstream `CrossArticulatorGraph`, `Fusion`, and `FingerspellingExpert` maintain separate un-swapped left/right channels—so mapping `R -> L` does **not** solve left- vs. right-handed signer dominance asymmetries, while suppressing directional sign-space symmetry inside intra-hand kinematics.

### 4. Severe `RelativeTimeBias` Dead-Bucket Waste (~53–55%) & VFR Architectural Coupling
- **Code Locations:** [`temporal.py:L48-L61, L245-L258`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L48-L61), [`fusion.py:L157-L168`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L157-L168), [`motion.py:L23-L33`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/motion.py#L23-L33).
- **Root Mathematical Failure:** Setting `min_dt = 1e-5` in `RelativeTimeBias` wastes **`17 / 32` (`53.1%`)** buckets in `v3_compact` and **`35 / 64` (`54.7%`)** buckets in `v2/v1` below physical coarse-token frame intervals at `25 fps` (`0.08s`), compressing the entire linguistically active `[0.08s .. 30s]` range into only `7` positive log-buckets in `v3` (`dt=0.08s, 0.10s, 0.20s` collide into a single bucket). Simultaneously, `lag_radius_frames=4` clips `ContinuousTimeLagBias`'s `0.20s` window down to `0.067s` at `60 fps`, while `MultiScaleTemporalStem` (`57 frames` = `3.8s` at 15fps vs `0.95s` at 60fps), `GlobalTemporalBlock.depthwise`, `MotionExpert`, `FingerspellingExpert`, `_adaptive_pool`'s slot count `m_valid = ceil(L/2)`, and `LatentPlanGrounder`'s active slot budget remain strictly frame-count-based rather than VFR-invariant.

### 5. Metric Discontinuities & Absent Adapter in the 66-Point Pose Pipeline
- **Code Locations:** [`pose.py:L61-L86, L300-L390`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L61-L86), [`data.py:L3-L5, L77-L121, L253-L286`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L253-L286), [`docs/data_contract.md:L63-L80`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/data_contract.md#L63-L80).
- **Root Mathematical Failure:** Multiple low-level geometric/contract breaks undermine clean kinematic learning: (a) **No 543->66 adapter exists**, while parent `CNN-BiLSTM` exports (`BODY` first, Face Mesh eye-first) and prior audit script `07` (`468:489` indexing) conflict silently with [`data.py:L278`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L278)'s shape-only validation; (b) `estimate_torso_scale` switches discretely across 4 incompatible fallback regimes (`shoulder_span -> 0.65*body -> 1.25*face -> 1.0`) and multiplies finite differences of **global** (un-anchored) `xy`, manufacturing velocity/acceleration spikes upon single-frame shoulder drops; (c) `body` uses `.any()` instead of `.all()` (`pose.py:L378`), snapping the body origin by `0.5` shoulder-width on single-shoulder occlusion, and wrist-to-centroid fallback emits no anchor-mode flag; (d) omitting `pose_confidence` injects synthetic `1.0` into node channel `[8]` & quality channel `[1]`; and (e) MediaPipe Pose wrists (`46, 47`) duplicate MediaPipe Hands wrists (`0, 21`) inside `CrossArticulatorGraph`.
