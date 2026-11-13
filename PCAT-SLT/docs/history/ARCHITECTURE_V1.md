# Progress-Coupled Area Transport (PCAT)

**A candidate architectural primitive for common-speed-invariant, order-sensitive sign modeling**  
Research specification, 13 November 2026. Implementation: `src/pcat_slt/operators.py`.

## Abstract

Sign language translation must combine articulators whose coordination is meaningful while accommodating nuisance variation in acquisition and signing rate. We isolate an architectural tension: quotienting out independent articulator clocks destroys some cross-articulator order distinctions, whereas using only frame-indexed representations supplies no exact common-clock invariance. We propose Progress-Coupled Area Transport (PCAT), a differentiable operator that constructs a shared learned path-length coordinate, integrates polygonal signed areas over progress cells, and converts those areas into norm-preserving feature transport. The encoder consumes actual 3D landmarks and preserves their signed position relative to an anatomical reference frame. We prove invariance under constant proper 3D similarities and common orientation-preserving reparameterization under specified assumptions, demonstrate a separation unavailable to separate first-order path descriptors, and bound the transport's conditioning and sensitivity. The repository implements the complete landmark-to-text model, training, evaluation and controlled ablations. These are structural guarantees and executable research hypotheses, not evidence of superior natural-language translation. Novelty relative to all published and unpublished work, benchmark superiority and publication suitability remain unestablished.

## 1. Research finding and precise scope

Recent work pursues several complementary directions: pretrained language decoders and scaling [R1,R2], graph representations [R3], multilingual contrastive pretraining [R4], contextual cues [R5], multimodal LLM adaptation [R6], sign-specific visual self-supervision [R7], and specialized expert alignment or retrieval [R8,R9]. None should be characterized as mathematically incapable of learning articulator coordination. A Transformer can represent many of the functions studied here; the proposed difference is an explicit architectural guarantee and a particular computational bias.

The target failure mode is **choosing the wrong temporal equivalence relation**. Let a joint articulatory trajectory be

\[
z(t)=(z^1(t),\ldots,z^A(t))\in\mathbb R^{N},\qquad N=Ar.
\]

A common warp acts as \(z\mapsto z\circ\phi\), with the SAME increasing bijection \(\phi\) on every channel. An independent warp acts as \(z^a\mapsto z^a\circ\phi_a\). The latter can change coordination. PCAT is invariant to the former, and deliberately not generally invariant to the latter. It retains geometric evidence of relative order, not every aspect of physical timing.

**Important qualifications.** Physical duration and holds can be linguistic information. Even common-warp invariance is therefore a hypothesis about nuisance variation, not a universal property of meaning. The optional timing channel preserves some duration information and explicitly relinquishes the full model's temporal invariance. Landmark-only input also omits visual texture and may inadequately capture mouthings, eye gaze or subtle finger contacts. This project is an end-to-end *landmark-to-text* model, not a video-to-3D estimator or a trained universal interpreter.

## 2. Why 2D cannot be made universally 3D-correct

For orthographic projection \(\pi(x,y,z)=(x,y)\), points \((x,y,z_1)\) and \((x,y,z_2)\) have identical observations for arbitrary depths. Under perspective projection, all positive scalar multiples of a camera ray share its image point. Thus projection is noninjective. If two distinct 3D signing configurations with different labels have identical 2D projections, any deterministic function of those projections assigns the same output to both and must fail on at least one. Architectural sophistication cannot remove that identifiability obstruction without additional assumptions or information.

We therefore require a common Euclidean 3D coordinate system supplied by depth, calibrated multiview sensing, motion capture, or an external estimator whose errors are measured separately. Independently normalized hand/face landmark coordinates are not interchangeable with such coordinates. A monocular estimator's output can be used, but the theorem then applies to its numerical 3D output, not to otherwise unobserved physical motion or arbitrary camera views.

## 3. Anatomical coordinate map

The first frame supplies reliable left shoulder \(l\), right shoulder \(r\), and neck \(n\). Define

\[
o=(l+r)/2,\quad s=\|r-l\|_2,\quad e_1=(r-l)/s,
\]
\[
u=n-o-\langle n-o,e_1\rangle e_1,\quad
e_2=u/\|u\|_2,\quad e_3=e_1\times e_2,\quad B=[e_1,e_2,e_3].
\]

For every joint and frame,

\[
\widetilde p_j(t)=B^\top(p_j(t)-o)/s. \tag{1}
\]

The code uses row-vector coordinates and right multiplication by \(B\), the same map. The reference is fixed throughout the clip; subsequent torso motion is preserved. Do not dynamically rotate away head or torso movement. Do not sort hands by motion energy: anatomical left/right labels remain explicit.

**Theorem 1 (proper-similarity invariance).** For a constant transform \(p'_j(t)=aRp_j(t)+b\), with \(a>0\), \(R\in SO(3)\), and unchanged joint identities/confidences, (1) is unchanged, provided \(s>0\) and \(u\ne0\).

**Proof.** The transformed reference satisfies \(o'=aRo+b\), \(s'=as\), and \(e'_1=Re_1\). Orthogonal projection commutes with \(R\), so \(e'_2=Re_2\). Since \(\det R=1\), \((Re_1)\times(Re_2)=R(e_1\times e_2)\), hence \(B'=RB\). Substitution gives \(B'^\top(p'-o')/s'=B^\top R^\top aR(p-o)/(as)=B^\top(p-o)/s\). ∎

This excludes reflections, perspective distortion, changing camera transforms and anatomically nonrigid rescaling. Reflections intentionally change the signed depth coordinate, rather than silently merging chirality. The implementation rejects degenerate reference frames. Its numerical degeneracy thresholds limit literal invariance for extremely small or nearly collinear anchors.

## 4. The atomic PCAT operator

### 4.1 Affine articulator embedding

Each region \(a\) uses an affine map of its canonical coordinates and confidence values:

\[
z^a_i=W_a\operatorname{vec}([\widetilde P^a_i,c^a_i])+b_a\in\mathbb R^r.
\tag{2}
\]

Concatenate the regions into \(z_i\in\mathbb R^N\) and linearly interpolate consecutive vertices. Keeping (2) affine makes insertion of interpolated vertices commute with the embedding. A nonlinear pointwise network would preserve a continuous reparameterization theorem for its continuous output path but would generally break exact finite polygonal subdivision invariance when only its sampled endpoints are interpolated. Projection to rank \(r\) may lose information; no injectivity is claimed.

### 4.2 A shared trainable progress coordinate

Let

\[
w_j=\frac{\operatorname{softplus}(\theta_j)+\mu}
 {N^{-1}\sum_q(\operatorname{softplus}(\theta_q)+\mu)},\quad
M=\operatorname{diag}(w),\quad\mu>0.
\tag{3}
\]

For a bounded-variation path, define metric arclength

\[
\ell(t)=\int_0^t\|dz\|_M,\qquad L=\ell(T).
\tag{4}
\]

For a polygonal path, \(d_i=z_{i+1}-z_i\), \(\ell_i=\sqrt{d_i^\top M d_i}\), and \(L=\sum_i\ell_i\). All articulators share this clock. Metric normalization fixes an otherwise redundant global scale. For finite parameters \(M\) is positive definite; hence \(\ell_i=0\) exactly when \(d_i=0\) in embedded coordinates. The embedding itself can still have a nullspace.

Split \([0,L]\) into \(K\) cells \(I_k=[kL/K,(k+1)L/K]\). If \(s_i=\sum_{j<i}\ell_j\), the segment fraction in cell \(k\) is

\[
f_{ki}=\begin{cases}
\frac{\max(0,\min(s_i+\ell_i,(k+1)L/K)-\max(s_i,kL/K))}{\ell_i}, &\ell_i>0,\\
0,&\ell_i=0.
\end{cases}\tag{5}
\]

Set \(v_{ki}=f_{ki}d_i\). This clips a segment exactly at cell boundaries rather than assigning an entire frame interval to one cell. No additive epsilon is inserted into nonzero arclengths, since that would make the result depend on how many vertices were inserted. Safe denominators are used only for zero-length segments. Min/max/clamp operations are differentiable almost everywhere; cell-crossing boundaries can be nondifferentiable.

### 4.3 Ordered area as a transport generator

Within a cell, define the local prefix \(p_{ki}=\sum_{j<i}v_{kj}\), displacement \(D_k=\sum_i v_{ki}\), and

\[
H_k=\frac12\sum_i(p_{ki}v_{ki}^{\top}-v_{ki}p_{ki}^{\top})
   =\frac12\sum_{j<i}(v_{kj}v_{ki}^{\top}-v_{ki}v_{kj}^{\top}).
\tag{6}
\]

Thus \(H_k^\top=-H_k\). This is the antisymmetric level-two path signature, a classical object, NOT a newly discovered integral [R10–R14]. With articulator blocks, \(H_k^{ab}\) records cross-channel ordered areas and \(H_k^{ba}=-(H_k^{ab})^\top\). It is not calibrated elapsed lag, causal influence, or a universally interpretable lead/lag score. Different trajectories can have the same area.

The proposed operation makes this chronological statistic act directly on the joint feature state:

\[
A_k=\tanh(g)\frac{H_k}{1+\|H_k\|_F},\qquad
Q_k=(I-A_k)^{-1}(I+A_k),\qquad
\widehat m_k=Q_km_k.\tag{7}
\]

Here \(m_k=(z_{\text{start},k}+z_{\text{end},k})/2\) is the endpoint midpoint, not the temporal average. The full statistic supplied to a learned readout is

\[
u_k=[m_k,D_k,\widehat m_k,
 \operatorname{upper}(H_k/(1+\|H_k\|_F)),\log(1+L)].\tag{8}
\]

The atomic design is the coupling of (3)–(7): a common learned progress measure, exact progress-local chronological area, and a Cayley action on anatomically signed features. This is a candidate operator-level contribution assembled from established mathematics. The narrower empirical question is whether *using order as stable feature transport*, beyond simply exposing signature features, improves SLT with matched resources. A name and a failed exact-phrase search do not establish priority.

### 4.4 Optional duration channel

With `timing: true`, append \([\log(1+\Delta t_k),\Delta t_k/T_{\rm elapsed}]\). A moving segment's time is distributed in proportion to its clipped fraction under the linear within-segment interpolation assumption. A stationary segment is assigned to the cell immediately following its progress position, clipped to the last cell at \(L\). For a fully stationary clip all elapsed time is assigned to the first cell. These fractions sum to one. Holds and total duration can then affect the result. Timing-enabled full-model predictions are intentionally NOT guaranteed common-warp invariant.

## 5. Structural results and their limits

### Theorem 2: common-clock invariance

Assume a fixed metric \(M\), a continuous bounded-variation embedded path, \(L>0\), and a common continuous increasing bijection \(\phi\) of the time domains. Metric arclength, progress-cell start/end states, \(D_k\), \(H_k\), and (7)–(8) are unchanged by \(z\mapsto z\circ\phi\).

**Proof.** Variation along the same oriented curve is independent of the parameter: \(\ell_{z\circ\phi}(t)=\ell_z(\phi(t))\). Thus corresponding progress cells contain the same ordered curve pieces with the same boundary states. Stieltjes integration of \((z-z_{\text{start}})\otimes dz\) over corresponding pieces is unchanged by an orientation-preserving change of variable. Its antisymmetric part is (6); displacement and endpoints are also unchanged. Equations (7)–(8) depend only on these quantities. ∎

**Exact discrete corollary.** Inserting vertices on existing straight segments, with compatible interpolation and unchanged reference frame, does not change (3)–(8) in exact arithmetic. For a split \(d=\alpha d+(1-\alpha)d\), metric lengths add by positive homogeneity, and the introduced same-segment wedge is zero. This remains true when a segment crosses multiple progress cells because (5) clips each subsegment exactly. Duplicate vertices contribute zero motion. Fully stationary paths have zero displacement/area and retain their static position through \(m_k\).

The guarantee is NOT exact for arbitrary resampling that omits corners, nonlinear interpolation, noisy reconstruction, changing confidence preprocessing, or a nonlinear camera projection. A continuous warp and a polygonal path through finitely sampled warped observations are different mathematical objects. The full invariant encoder inherits the discrete corollary because the embedding is affine, cell count is fixed, and source positional embeddings index progress cells rather than original frames. Evaluation mode is required for deterministic equality with dropout. Decoder predictions inherit matching memories, except floating-point ties can make argmax discontinuous.

### Theorem 3: independent-clock invariance cannot preserve all coordination

Consider two scalar articulators and trajectories \(x(t)=(t,t^2)\) and \(y(t)=(t^2,t)\), \(t\in[0,1]\). Each scalar coordinate of \(x\) can be independently reparameterized into its corresponding coordinate of \(y\): use \(\phi_1(t)=t^2\) and \(\phi_2(t)=\sqrt t\). Both are increasing bijections. Consequently any representation invariant under all independent channel warps must identify these trajectories.

However,

\[
H_{12}(x)=\tfrac12\int_0^1(t\,d(t^2)-t^2\,dt)=1/6,
\quad H_{12}(y)=-1/6.\tag{9}
\]

The joint area separates them. Separate scalar signatures or endpoint displacements both give displacement one in each channel and cannot. This proves a strict representational advantage over that specified class of descriptors, not over arbitrary attention networks or all existing SLT systems. It also demonstrates why independent timing invariance would be the wrong universal design target. A downstream learned readout can still discard the difference.

### Theorem 4: orthogonality and solve conditioning

For any real skew-symmetric \(A\), (7) is well defined and \(Q^\top Q=I\). Therefore \(\|Qm\|_2=\|m\|_2\). For a fixed \(A\), the Jacobian with respect to the transported value \(m\) has singular values exactly one.

**Proof.** \((I-A)^\top(I-A)=I-A^2=I+A^\top A\succeq I\); hence \(I-A\) is invertible and its inverse has spectral norm at most one. Since all polynomials in \(A\) commute, substitution of \(A^\top=-A\) gives \(Q^\top Q=I\). Also \(\|A\|_2\le\|A\|_F<1\) in this implementation, so

\[
\kappa_2(I-A)\le\sqrt{1+\|A\|_2^2}<\sqrt2.\tag{10}
\]

This is a conditioning guarantee for the small solve. It does not bound all network gradients because \(A\) itself depends on the path and cell boundaries, and the readout and decoder are ordinary learned networks. Norm preservation avoids amplification by this transport; it is not evidence that training always converges.

### Proposition 5: sensitivity to area perturbations

For two skew matrices, define \(Q(A)=2(I-A)^{-1}-I\). The resolvent identity and \(\|(I-A)^{-1}\|_2\le1\) imply

\[
\|Q(A)-Q(B)\|_F\le2\|A-B\|_F.\tag{11}
\]

The radial normalization \(f(H)=H/(1+\|H\|_F)\) is at most 2-Lipschitz: add and subtract \(G/(1+\|H\|_F)\), apply the triangle inequality, and use the reverse triangle inequality for norms. Therefore, with fixed gain,

\[
\|Q(H)-Q(G)\|_F\le4|\tanh(g)|\|H-G\|_F.\tag{12}
\]

For fixed corresponding polygonal cell segments with increments \(d_i,e_i\), let \(V_d=\sum_i\|d_i\|\), \(V_e=\sum_i\|e_i\|\), and \(E=\sum_i\|d_i-e_i\|\). Expanding each pairwise wedge in (6) and using \(\|uv^\top\|_F=\|u\|\|v\|\) yields the conservative bound \(\|H(d)-H(e)\|_F\le E(V_d+V_e)\). Combined with (12), transport values satisfy

\[
\|Q(H)m-Q(G)n\|\le\|m-n\|+4|\tanh(g)|\|H-G\|_F\|n\|.
\tag{13}
\]

These bounds assume corresponding fixed clipped cells. They do not prove global stability of learned progress-cell boundaries near a zero-motion path. That is an explicit limitation.

### Proposition 6: concatenation is ordered, not commutative

For adjacent pieces with increments \(u,v\) and areas \(H_x,H_y\),

\[
H_{x*y}=H_x+H_y+\tfrac12(uv^\top-vu^\top).\tag{14}
\]

This follows by splitting the double sum in (6) into within-piece and cross-piece terms; it is the level-two Chen identity. Reversing the traversal negates the local area and inverts the corresponding Cayley transport. However, products of Cayley transforms of separate areas do NOT in general equal the Cayley transform of the concatenated area. We do not claim that PCAT is an exact path-development or holonomy representation.

## 6. Complete model and objective

```mermaid
flowchart LR
    P["3D landmarks + confidence"] --> B["Fixed anatomical frame"]
    B --> E["Affine articulator maps"]
    E --> C["Common learned progress clock"]
    C --> H["Exact clipped signed areas"]
    H --> Q["Cayley feature transport"]
    Q --> R["K progress tokens"]
    R --> T["Small context encoder"]
    T --> D["Causal byte decoder"]
    D --> Y["Written-language text"]
```

The readout is Linear–GELU–LayerNorm. Learned progress-index positions and a pre-norm Transformer context encoder process the \(K\) tokens. A conventional causal Transformer decoder predicts a fixed vocabulary of 259 tokens: PAD, BOS, EOS and all 256 UTF-8 bytes. This supports arbitrary written languages without downloading or implicitly changing pretrained language models. Byte sequences can be long and generated bytes can be invalid UTF-8; decoding uses the Unicode replacement character. The baseline favors controlled research over the language priors and speed of a pretrained subword decoder.

The objective is token cross-entropy, optionally label smoothed:

\[
\mathcal L=-\frac1{\sum_b U_b}\sum_{b,u}\log p_\Theta(y_{bu}\mid y_{b,<u},P_b,c_b).
\tag{15}
\]

No gloss, pseudo-gloss or CTC target is assumed. Gradient accumulation sums token losses across microbatches and divides gradients by the actual count of non-padding target tokens at each update, including partial accumulation groups. Validation uses unsmoothed token NLL. Best checkpoints are selected by validation NLL; all reported text metrics must be computed from autoregressive generation, not teacher-forced argmax outputs.

For defaults: \(A=4\), \(r=4\), \(N=16\), \(K=32\), model width 256, two context encoder layers and four decoder layers. Exact dimensions follow the YAML and are not silently inferred from another model's layout. CPU and CUDA are implemented. The solve is FP32 even under CUDA AMP; FP64 is supported for mathematical audits. CUDA execution is not certified by a CPU-only verification run.

## 7. Computational cost

Affine pose embedding scales linearly in frame count and input coordinates. The reference PCAT implementation iterates over \(K\) cells, computes prefix sums and tensor contractions, and costs \(O(BKTN^2+BKN^3)\), with the cubic term from small dense solves. Forward working memory is approximately \(O(BTN+BKN^2)\), excluding the embedding and readout. Autograd retains intermediates from all cells and may require \(O(BKTN+BKN^2)\) storage; the complete decoder has its own attention memory. This is not an optimized fused kernel or a claimed linear-time replacement for all attention. With \(N=16\) the solve is small; high ranks should be profiled. The context attention is \(O(BK^2d)\) instead of attention on every original frame, at the cost of lossy compression.

## 8. Falsifiable comparisons

The implemented controls share the same decoder, affine maps, token count, readout dimensions and training schedule:

| Config change | Question tested |
|---|---|
| `operator: first_order` | Does area add useful information beyond endpoint/midpoint/displacement? |
| `operator: signature` | Does the proposal beat unbounded classical area features without transport? |
| `operator: no_transport` | Does the Cayley action help when the same bounded area is exposed to the readout? |
| `operator: temporal` | Do shared progress cells help compared with elapsed-time cells, with the same transport? |
| `clock: euclidean` | Does learning the diagonal metric help? |
| `timing: true` | Does preserving duration outweigh imposing common-speed invariance? |

`first_order` zeroes the area readout slot and uses identity transport; duplicated midpoint slots maintain the same readout shape. `signature` uses raw antisymmetric level-two statistics and identity transport. It is a specific controlled signature baseline, not a reproduction of the full 2019 or 2024 gesture architectures. Frozen ablation parameters mean nominal and trainable parameter counts can differ slightly; report both.

The minimum credible publication program is specified in `docs/EXPERIMENTS.md`: multiple real sign languages, matched preprocessing and decoders, signer/source separation, prior signature/path-development baselines, confidence-aware pose error analyses, timing tests and multiple seeds. PCAT should be rejected or revised if its gains disappear against `no_transport` or matched signature baselines. A positive result solely on our constructed event task is insufficient.

## 9. What is proved, tested and still unknown

| Claim | Status |
|---|---|
| Invariance of body coordinates to constant proper similarities | Proved under nondegenerate anchors; numerical tests |
| Common-warp invariance of continuous embedded PCAT statistics | Proved under stated path assumptions |
| Exact polygonal refinement invariance | Proved; tested for cell crossings and full invariant encoder |
| Distinguishing some order-swapped articulator paths | Analytic counterexample and numerical mechanism audit |
| Orthogonal transport and well-conditioned solves | Proved; tested numerically |
| End-to-end gradients, masking, causal decoding and epoch resume | Executable tests |
| Superior SLT quality, robustness or sample efficiency | Unknown; no real-corpus experiment included |
| ResNet/Transformer-level novelty or top-tier acceptance | Not established and cannot be guaranteed |

Finite-order areas are noninjective; motion that leaves a cell and backtracks may collide with other paths. Cells lying inside straight segments have zero area, and cross-cell interactions are then handled by the context encoder, not a missing local area term. Rank reduction can discard discriminative directions. Metric learning can emphasize detector noise. The first reference frame may be noisy or tilted, and landmark occlusion/imputation can create artificial motion. Camera invariance in reconstructed coordinates does not prove invariance of a camera-dependent reconstruction procedure. A fixed number of source tokens may omit rapid fingerspelling or long utterance structure. These are concrete failure modes to measure.

## References

Primary links, dates, publication/preprint distinctions and novelty conflicts are recorded in `docs/LITERATURE.md` under identifiers R1–R16. The theory above uses classical arclength, path signatures and Cayley transforms; the proofs concern this specified construction and do not claim those mathematical ideas as new.
