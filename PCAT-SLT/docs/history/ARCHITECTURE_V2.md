# PCAT v0.2: multiscale progress-coupled area transport

**A strengthened, tested research architecture for continuous 3D-landmark-to-text translation.**

This version addresses specific defects of PCAT v0.1: repeated integration work, order lost across cell boundaries, certain retracing collisions, unbounded clock anisotropy, compulsory rank/token compression, early-frame reference failure and malformed byte generation. It does not claim to solve every limitation of sign-language translation. Real-corpus accuracy and novelty remain experimental questions. The original derivations are preserved in [docs/history/ARCHITECTURE_V1.md](ARCHITECTURE_V1.md); the current implementation is specified here.

## 1. Two explicit operating regimes

| Configuration | Information preserved | Temporal invariance claim |
|---|---|---|
| `configs/invariant.yaml` | Multiscale progress statistics, signed anatomical position, progress-augmented areas | Exact polygonal subdivision invariance under the assumptions below; no physical duration |
| `configs/base.yaml` | The same coarse statistics **plus every valid frame's canonical coordinates, confidence and relative timestamps** | Deliberately sensitive to physical timing and frame multiplicity |

The enhanced default retains observed detail instead of forcing all information through a fixed number of low-rank tokens. The invariant regime provides the cleaner scientific test of the PCAT primitive. Neither regime reconstructs missing appearance or unobserved motion.

## 2. Anatomical coordinates and reference selection

Given reliable left shoulder l, right shoulder r and neck n from one reference frame, set

\[
o=(l+r)/2,\quad s=\|r-l\|,\quad e_1=(r-l)/s,
\]
\[
u=n-o-\langle n-o,e_1\rangle e_1,\quad e_2=u/\|u\|,\quad
B=[e_1,e_2,e_1\times e_2],\quad \widetilde p=B^\top(p-o)/s. \tag{1}
\]

The reference is fixed throughout a clip; later torso/head motion is retained. Anatomical handedness is not sorted by energy or erased by reflection invariance.

`anchor_policy: first_reliable` searches valid frames for the first reliable, finite, nondegenerate anchor triplet. This fixes rejection of a usable clip merely because its first frame is occluded. No padded frame is eligible. If no such frame exists, the sample is rejected with an explicit error. The data loader uses the same selection rule when imputing entirely absent joints.

`anchor_policy: first` admits only the first frame. The invariant configuration uses this stricter policy: subdivision must not change the chosen anatomical reference. Under the fallback policy, inserting an earlier newly reliable reference can change canonical coordinates; exact full-model subdivision equality is not promised in that case. Threshold selection is discrete and can be discontinuous near an eligibility boundary.

**Proper-similarity theorem.** With unchanged confidences and eligible reference identity, a constant transform p'=aRp+b, a>0 and R in SO(3), leaves (1) unchanged. Indeed o'=aRo+b, s'=as and B'=RB, so B'^T(p'-o')/s'=B^T(p-o)/s. For the implemented fallback selector, reliability and relative noncollinearity are invariant under that transform away from numerical thresholds. This excludes perspective projection, reflections, moving cameras and nonrigid scale. Tiny absolute degeneracy thresholds restrict literal numerical invariance at extreme scales.

## 3. A bounded learned common clock

Region a maps its canonical joint coordinates and confidences affinely into r channels:

\[
z_i^a=W_a\operatorname{vec}([\widetilde P_i^a,c_i^a])+b_a,
\qquad z_i=\operatorname{concat}_a z_i^a. \tag{2}
\]

Affine maps commute with interpolation and therefore preserve the exact polygonal refinement property. They can lose information; the enhanced memory path in Section 7 bypasses them.

For a user-specified condition limit kappa>=1, define

\[
\bar w_j=\exp(\tfrac12\log(\kappa)\tanh\theta_j)+\mu,\quad
w_j=\bar w_j/\operatorname{mean}(\bar w),\quad M=\operatorname{diag}(w),\quad \mu>0. \tag{3}
\]

The clock uses d_i=z_{i+1}-z_i and

\[
\ell_i=\sqrt{d_i^\top M d_i},\quad L=\sum_i\ell_i,\quad
s_i=\sum_{j<i}\ell_j. \tag{4}
\]

All articulators share this clock. Unlike a positive floor followed by unconstrained normalization, (3) supplies a global condition bound.

**Clock-conditioning theorem.** For all finite theta, max(w)/min(w)<=kappa and 1/kappa<=w_j<=kappa. Before adding mu, the ratio is bounded by exp(log(kappa)); adding the same positive mu cannot increase that ratio. Mean normalization preserves the ratio and puts the mean at one, yielding the componentwise bounds. Consequently the clock remains coercive in embedded coordinates. This prevents an arbitrarily ill-conditioned metric; it does not prevent a poor embedding, noisy coordinates or semantically inappropriate speed invariance.

## 4. Progress augmentation and multiple scales

Define normalized common progress rho_i=sum(j<i) ell_j/L when L>0; use zero when L=0. With `progress_augmentation: true`, integrate the path

\[
x_i=[z_i,\rho_i],\qquad \Delta x_i=[d_i,\ell_i/L]. \tag{5}
\]

The metric clock is computed from the original z coordinates, not recursively from the augmented coordinate. Progress increases monotonically along every nonconstant embedded segment and is independent of physical traversal speed.

The default coarse scales are K in {1,4,16,32}, producing 53 source tokens. Each scale partitions [0,L] into equal progress cells. The whole-clip scale preserves cross-cell order that would be missing if every fine cell lay inside a straight segment. Other scales give finer temporal localization. Scales use the same clock, metric, transport gain and readout; positional embeddings identify the resulting token positions.

For the clipped increments v_i within a cell, define

\[
D=\sum_i v_i,\quad H=\tfrac12\sum_{j<i}(v_jv_i^\top-v_iv_j^\top),\quad H^\top=-H. \tag{6}
\]

**A resolved collision.** In one channel, paths 0→1→0 and 0→-1→0 have equal start/end, equal total length and zero ordinary antisymmetric area. The v0.1 one-cell features identify them. After (5), their increments are respectively (1,1/2),(-1,1/2) and (-1,1/2),(1,1/2), giving H_12=+1/2 and -1/2. Progress augmentation therefore separates this particular retracing collision. It does not make a finite collection of low-order moments injective over all paths.

These ingredients have substantial prior art: signatures, path augmentation, learned signature layers, multiscale trajectory descriptors and Lie-group path development are established concepts. The narrower PCAT hypothesis is the value of common-progress signed-area **transport**, assessed against matched identity-transport/signature controls. See [docs/LITERATURE.md](../LITERATURE.md), including Deep Signature Transforms and Path Development Networks.

## 5. Exact integration without a per-cell rescan

For a prefix ending at path position q=x(t)-x(0), store its displacement q and signed area G(t). For a fraction alpha of one straight segment d after prefix p,

\[
q=p+\alpha d,\qquad G=G_{\rm before}+\tfrac\alpha2(pd^\top-dp^\top). \tag{7}
\]

For cell boundaries a,b, its local displacement is D=q_b-q_a and its local area is

\[
H_{[a,b]}=G(b)-G(a)-\tfrac12(q_aD^\top-Dq_a^\top). \tag{8}
\]

**Exactness proof.** Split the prefix at a. Ordered concatenation gives G(b)=G(a)+H_[a,b]+(q_a D^T-D q_a^T)/2: the cross term contains every prefix increment paired with every interval increment. Rearranging proves (8). Formula (7) is the same identity for a partial straight segment, whose intrinsic skew area is zero. Thus only boundary positions must be searched after one prefix pass. Zero-length edges and padding contribute no movement.

`integration: prefix` implements (7)–(8). Its integration cost is O(B*T*N^2+B*sum(K)*N^2), plus boundary searches, rather than O(B*T*sum(K)*N^2). Prefix area storage is O(B*T*N^2); autograd and the decoder add further memory. Transport solves cost O(B*sum(K)*N^3), where N includes the optional progress coordinate. This is not a universal memory reduction or an end-to-end speedup claim.

`integration: clipped` retains the independent explicit clipping implementation, including progress augmentation and multiple scales. Its values and gradients are tested against the prefix backend. This is also a useful alternative when subtracting large prefixes is numerically problematic.

**Floating-point caveat and resolution policy.** Algebraic exactness does not remove cancellation. The supplied long-path benchmark exhibited a maximum float32 area disagreement around 1.1e-3, versus about 2.4e-12 in float64 on that particular case. Recommended base/invariant configs therefore use `integration_precision: float64`; outputs are cast to the learned readout's precision. This costs memory and can be slower on consumer GPUs. Float64 does not guarantee stability for all extreme inputs. Choose the clipping backend and assess precision on the actual corpus when necessary.

## 6. Orthogonal area transport and invariance

With cell midpoint m=(x_start+x_end)/2, set

\[
A=\tanh(g)H/(1+\|H\|_F),\quad Q=(I-A)^{-1}(I+A),\quad \widehat m=Qm. \tag{9}
\]

The readout receives [m,D,Qm,upper(H/(1+||H||_F)),log(1+L)] and optional elapsed-time features. It is Linear–GELU–LayerNorm, followed by a small Transformer context encoder over coarse tokens only.

**Transport theorem.** Since A is real skew-symmetric, (I-A)^T(I-A)=I+A^TA is positive definite, so the solve exists. Polynomial factors in A commute, giving Q^TQ=I and ||Qm||=||m||. Since ||A||_2<1, kappa_2(I-A)<sqrt(2). The resolvent identity yields ||Q(A)-Q(B)||_F<=2||A-B||_F; radial area normalization gives a conservative bound 4|tanh(g)| ||H-G||_F when comparing unnormalized areas. The complete network is not norm preserving, and gradients through clock/cell selection are not bounded by this theorem.

**Refinement theorem.** Inserting a vertex alpha d on a straight segment splits metric length additively, leaves every progress cell on the same oriented curve, and introduces no same-segment wedge because d wedge d=0. The extra progress coordinate also splits linearly, so (5)–(9) are unchanged in exact arithmetic at every scale. This applies with a fixed anatomical reference, interpolated confidence, unchanged embedding and evaluation-mode dropout. Continuous common reparameterization has the analogous bounded-variation/Stieltjes-integral proof. Timing-enabled readouts and detail-memory attention are excluded from full invariance claims.

Independent channel retiming can change H and is deliberately not quotiented out. For example x(t)=(t,t^2) and y(t)=(t^2,t) are related by independent increasing channel warps yet have H_12=+1/6 and -1/6. Their joint area distinguishes a coordination difference that independent scalar signatures cannot. This is a counterexample for a particular descriptor class, not a claim that Transformers cannot represent chronology.

## 7. A detail path around both compression bottlenecks

In the enhanced regime, for each valid original frame define

\[
f_i=\operatorname{vec}([\widetilde P_i,c_i]),\quad
r_i=[f_i,\operatorname{Fourier}(\rho_i),\log(1+t_i-t_0),(t_i-t_0)/(t_{last}-t_0),0,\ldots]. \tag{10}
\]

Require model width >=4J+detail_position_dim+2. Timing slots are zero when timing is disabled. These tokens are appended **after** the coarse context encoder and passed directly to decoder cross-attention with a source padding mask. There is no learned projection, normalization or pooling over the f_i subspace. The coarse branch still uses the PCAT primitive; detail tokens avoid O(T^2) source self-attention, although decoder cross-attention grows with T.

**Observed-detail preservation proposition.** At memory precision, selecting the first 4J coordinates of valid r_i exactly recovers f_i. Therefore neither the learned rank-r map nor the fixed coarse-cell count forces loss of observed canonical joint/confidence values before decoder attention. With timing enabled, the timing coordinates also retain relative sample times. This does not prove injective decoder outputs, recovery of physical motion between samples, accurate input pose, or preservation of raw RGB information. AMP may reduce memory precision; this proposition concerns the actual stored numeric representation.

`encode_with_mask` is the correct memory interface for padded batches. `encode_invariant` exposes the geometry-only coarse branch when timing is disabled, even if the model is configured with detail memory. `encode` returns only the memory tensor and is not a substitute for the mask-aware interface on variable-length detail batches.

Physical durations/holds remain available through both coarse elapsed-time features and frame-level timing. The full enhanced model intentionally relinquishes common-speed invariance and duplication invariance. Timing values are in seconds; inventing a frame-rate convention would undermine the intended semantics.

## 8. Strict UTF-8 autoregressive generation

The decoder predicts a fixed 259-token vocabulary: PAD/BOS/EOS and all bytes. Its finite-state generation constraint follows the legal byte-sequence ranges in [RFC 3629](https://www.rfc-editor.org/rfc/rfc3629). It excludes overlong forms, surrogate encodings, out-of-range code points, invalid continuation bytes and EOS inside a character. A prospective leading byte is masked if the remaining generation budget cannot complete its code point.

**Validity proof.** Start in the accepting boundary state. Each allowed byte either stays at that boundary for ASCII, enters an admissible continuation state, or reduces the number of required continuation bytes within the range mandated by its leading byte. EOS is available only at a boundary. The budget mask permits a transition only if its destination needs fewer bytes than the number of slots available before that transition; by induction the final slot must leave zero continuation bytes. Thus every generated byte prefix at termination is valid UTF-8, even if EOS is not produced. This proves encoding validity, not grammaticality, factuality, adequacy or a chosen language.

Training still uses ordinary teacher-forced token cross-entropy, with token-correct accumulation and validation-NLL model selection. The constraint changes inference and does not mask the training softmax. Invalid-byte rate is reported explicitly during evaluation.

## 9. Implementation and evidence

New configuration controls include progress scales/augmentation, bounded metric condition limit, detail memory width/positions, anchor policy, integration backend and precision. Nominal and trainable parameter counts must be reported for each configuration. V0.2 checkpoints use format version 2; v0.1 checkpoints are rejected rather than silently reinterpreted under a changed metric parameterization. Original source and results are retained in the local v0.1 release/history.

Tests cover exact/oracle integration and gradient agreement, multiscale augmentation/refinement, retracing and boundary-order counterexamples, extreme metric parameters, masked detail preservation/backward, reference fallback, UTF-8 validity/budget cases, provenance leakage and exact coarse/enhanced CPU checkpoint resume. Actual run and benchmark evidence is in [reports/VERIFICATION.md](../../reports/v2/VERIFICATION.md).

A change in the richer model's synthetic learning curve is not evidence that area transport improved SLT: source memory, model width and training schedules differ. The matched ablations and real-corpus tests in [docs/EXPERIMENTS.md](../EXPERIMENTS.md) remain required. Every original limitation is classified in [docs/LIMITATIONS.md](LIMITATIONS_V2.md); none is hidden by a claim of perfection.
