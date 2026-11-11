# ISL-MEGA architecture specification

## 1. Purpose and boundary

ISL-MEGA is a proposed multimodal, gloss-free continuous ISL-to-English research architecture. It consumes the full RGB frame sequence, a timestamp-aligned 66-point articulated pose sequence and optional detector ROIs. It does not require gloss labels. It can use clip-level fingerspelling strings where those have been annotated.

The model integrates several established research directions. The synthesis itself is a hypothesis, not a proven novel method. A paper contribution must identify and test a precise mechanism beyond stacking existing encoders.

## 2. Canonical inputs

For a batch of clips padded to the maximum clip length `T`:

- RGB frames: `frames [B,T,3,H,W]`, RGB floating-point values in `[0,1]`.
- Timestamps: `timestamps [B,T]`, seconds relative to the start of each clip.
- Pose: `pose_xy [B,T,66,2]`, `pose_confidence [B,T,66]`, `pose_valid [B,T,66]`.
- Confidence provenance: `confidence_known [B,T,66]`.
- Region boxes: `boxes [B,T,4,4]` and `box_valid [B,T,4]`, ordered face/left hand/right hand/body.
- Clip lengths: `lengths [B]`.

The loader preserves the original number of decoded video frames. Pose can originate at a different sampling rate and is linearly interpolated onto video timestamps, but only when the bracketing observation gap does not exceed the configured maximum. Long-gap landmarks are invalidated rather than extrapolated. Frames are spatially letterboxed; no time dimension is cropped or resampled.

## 3. Specialist encoders

### 3.1 RGB appearance expert

Four regional views are extracted per frame:

1. Face.
2. Left hand.
3. Right hand.
4. Upper body / global signer view.

Left and right hands share the same ViT encoder; face and body have independent encoders/adapters when `separate_hand_face_body=true`. The high-capacity configuration initializes from `timm` DINOv2-L/14. It then supports an additional global-teacher/local-region-student self-distillation stage. The pretraining stage is an ISL-MEGA implementation variant inspired by SignDINO; it does not reproduce the paper's exact recipe.

### 3.2 Articulatory pose expert

The 66 nodes are partitioned into four graph pathways:

| Region | Nodes | Output width (research config) |
|---|---:|---:|
| Left hand | 21 | 512 |
| Right hand | 21 | 512 |
| Upper body | 6 | 512 |
| Face | 18 | 512 |

Each node has 11 channels: absolute `x,y`, anchor-relative `x,y`, bounded velocity `vx,vy`, bounded acceleration `ax,ay`, confidence, validity, and confidence-known indicator. Hand graphs use the MediaPipe 21-point finger chains and wrist-relative coordinates. The upper-body graph uses shoulder/elbow/wrist links and the shoulder midpoint as its preferred coordinate anchor (visible-body centroid fallback if both shoulders are absent). The sparse 18-point face graph is fully connected and face-centroid-relative. Confidence thresholding is applied only when detector confidence is actually known.

When v2 mechanisms are enabled (`configs/research.yaml` / `configs/smoke_v2.yaml`), three articulatory upgrades activate cleanly via YAML ablation flags:

- `normalize_by_shoulder_span`: divides anchor-relative offsets, intra-graph edge displacements, and kinematics by estimated frame torso/shoulder scale (`estimate_torso_scale` with bounded geometric fallback).
- `share_hand_encoder`: ties left/right 21-point hand graph weights with mirror-canonical right-hand coordinate projection and a learned 2-way `hand_side_emb`.
- `cross_articulator_graph`: runs `CrossArticulatorGraph` message passing over the 36 phonological landmarks (6 key points per hand: wrist + 5 fingertips, 6 upper-body joints, 18 face points) prior to node pooling, biased by torso-normalized pairwise offsets and 4 multi-scale spatial proximity Gaussian RBF kernels (`sigma in {0.10, 0.25, 0.60, 1.50}`).

### 3.3 Temporal motion expert

The body/global crop is concatenated with its absolute frame difference and passed through two 3-D convolutional stages. Temporal stride is one; spatial dimensions are reduced. A depthwise temporal refinement follows. Masks are reapplied after temporal convolutions so batch padding cannot affect valid frame features.

### 3.4 Fingerspelling expert

The two hand appearance streams and two hand pose streams enter a bidirectional GRU and character CTC head. Character posteriors are softly gated back into the frame representation. The initial feedback gate is biased mostly closed. CTC loss is trained only for records with an explicit `fingerspelling_text` label and enough valid frames.

When `span_routing=true`, a local temporal spotting conv + head predicts calibrated per-frame fingerspelling activity logits/probabilities (`spot_logits`, `spot_prob`), gating posterior character feedback (`gate * spot_prob`) so non-fingerspelled lexical frames suppress character spillover without altering raw CTC logits. Optional frame-level span annotations (`fingerspelling_spans` `[Tv]` in NPZ or `[start_sec, end_sec]` intervals in JSONL) can supervise `compute_fingerspelling_spotting_loss` directly, falling back to weak clip-level MIL (noisy-OR + sparsity on positive clips, zero-target suppression on negative clips) when only clip strings exist.

## 4. Reliability-aware fusion

Ten streams are projected to the shared `fusion.dim`:

- Left hand RGB, right hand RGB, face RGB, body RGB.
- Left hand pose, right hand pose, body pose, face pose.
- Motion.
- Fingerspelling.

Each stream provides a five-value quality descriptor: presence, ROI/pose coverage or area proxy, sharpness/motion or gate/spotting proxy, detector-confidence/availability proxy, and frame validity.

- **v1 baseline (`continuous_time_lag=false`):** Discrete cross-stream lag attention over `-2..+2` frame indices.
- **v2 continuous-time + reliability fusion (`continuous_time_lag=true`, `reliability_uncertainty_gate=true`):**
  1. Structural reliability calibrator `r_s(t) in (0,1)` adds `log(r)` to content utility logits (log-odds sensor gating), plus optional `compute_fusion_entropy_loss` to prevent single-stream collapse.
  2. Continuous elapsed-time cross-stream lag attention (`ContinuousTimeLagBias`) encodes continuous `dt = t_neighbour - t_query` in seconds using normalized Fourier/polynomial features `[tau, tau^2, |tau|, sin(pi tau), cos(pi tau), sin(2 pi tau), cos(2 pi tau)]` plus signed per-stream slope priors, masking out neighbors beyond `lag_window_seconds` (invariant across variable frame rates).

All-missing observations are represented by a learned null vector and remain finite. Missing streams receive zero final mixture weight.

## 5. Hierarchical temporal encoder

### 5.1 Fine-rate front end

A parallel multi-scale temporal convolution stem keeps the original temporal length `T`. The high-capacity configuration uses four branches:

| Branch | Kernel | Dilation | Channels |
|---|---:|---:|---:|
| 1 | 3 | 1 | 256 |
| 2 | 5 | 1 | 256 |
| 3 | 9 | 2 | 256 |
| 4 | 15 | 4 | 256 |

The branch outputs are concatenated and projected to 1,024 dimensions. This operation retains all fine temporal positions.

### 5.2 Coarse global reasoning (uniform v1 vs kinematic event-boundary v2)

- **v1 baseline (`adaptive_consolidation=false`):** Consecutive fine features are grouped uniformly with `coarse_stride=2`.
- **v2 adaptive event consolidation (`adaptive_consolidation=true`):** A boundary salience network scores per-frame visual-kinematic transition energy `[fine, delta_fine, ||delta_fine||, log1p(dt)]`. Valid cumulative salience partitions clip length `L_i` into `m_i = ceil(L_i / coarse_stride)` non-uniform soft event slots via differentiable Gaussian-RBF slot assignment `A [B, M, T]`. Static holds collapse into fewer slots while rapid signing strokes and fingerspelling retain higher slot density, with exact batch-padding invariance (`out[:, :L_i]` independent of padded batch length `T`).

Twelve pre-normalized global Transformer blocks use 16 attention heads, 1,024 model dimensions, a 4,096-wide feed-forward layer and relative timestamp bias.

### 5.3 Fine detail retrieval & reversible event unpooling

In v1 (`adaptive_consolidation=false`), `FineDetailRetrieval` queries a local 7-frame window and broadcasts via `repeat_interleave(stride)`. In v2 (`adaptive_consolidation=true`), `AdaptiveDetailRetrieval` refines each event token across valid fine frames biased by its soft event assignment prior `assign_logits` and unpools back onto the full `T`-frame timeline using column-normalized transpose weights `B_rev^T` (`global_up = B_rev^T @ retrieved`).

## 6. Latent plan / evidence aggregation

A learned bank of up to 32 latent queries (research config) cross-attends to the fine-rate temporal memory. When `dynamic_plan_routing=true`, active plan slots scale cleanly with clip valid length (`active = clamp(ceil(valid_len / target_stride), min=min_plan_queries, max=num_queries)`), masking padded queries before self-attention and inside T5 (`plan_mask`). When `diverse_ordered_planning=true`, slot positional priors combine with normalized attention/embedding Gram orthogonality (`plan_diversity_weight`) and intrinsic clip-relative center-of-mass ordering (`plan_order_weight`). Grounding maps remain inspectable attention diagnostics, not verified human gloss labels.

## 7. Text decoder & counterfactual visual grounding

A learned linear/LayerNorm bridge maps frame and active plan tokens to the configured T5 hidden dimension. When `plan_semantic_ot=true`, unconstrained entropic OT computes a weighted combination of frame-level OT (`visual_semantic_ot_loss`) and active-plan OT (`plan_semantic_ot_loss`).

When `use_counterfactual_grounding=true` and `counterfactual_weight > 0`, a detached counterfactual decoder forward pass swaps visual+plan encoder embeddings across the batch (`B >= 2`) or supplies a single blank zero token (`B == 1`), computing a token-level NLL margin hinge (`relu(margin - (nll_cf - nll_real))`) and symmetric distribution shift floor (`relu(kl_margin - KL(p_real || p_cf))`) without backpropagating gradients through synthetic mismatched inputs. This explicitly penalizes language-prior bypass where FLAN-T5 ignores visual conditioning.

## 8. High-capacity v2 configuration & v1 baseline ablation

`configs/research.yaml` enables all six v2 upgrades (`normalize_by_shoulder_span`, `share_hand_encoder`, `cross_articulator_graph`, `continuous_time_lag`, `reliability_uncertainty_gate`, `span_routing`, `adaptive_consolidation`, `diverse_ordered_planning`, `dynamic_plan_routing`, `plan_semantic_ot`, `use_counterfactual_grounding`); `configs/ablation_v1_baseline.yaml` disables all six at matched parameter scale for controlled ablations. All settings remain research proposals pending empirical runs on real data.

## 9. Implemented vs not implemented

Implemented: all v1 and v2 modules/ablation toggles above, optional `fingerspelling_spans` dataset/collate support plus clip-level MIL fallback, counterfactual visual sensitivity loss, split disjointness checks, SignDINO-inspired self-distillation variant, synthetic unit tests covering both v1 (`tests/test_architecture.py`) and v2 (`tests/test_novel_v2.py`).

Not implemented: undocumented raw `.pose` binary parser; landmark detector/tracker; actual ISL videos/annotations; trained model weights; distributed multi-GPU trainer; real-data empirical results or human fluent-signer evaluation.
