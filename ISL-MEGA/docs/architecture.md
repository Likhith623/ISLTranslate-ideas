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

### 3.3 Temporal motion expert

The body/global crop is concatenated with its absolute frame difference and passed through two 3-D convolutional stages. Temporal stride is one; spatial dimensions are reduced. A depthwise temporal refinement follows. Masks are reapplied after temporal convolutions so batch padding cannot affect valid frame features.

### 3.4 Fingerspelling expert

The two hand appearance streams and two hand pose streams enter a bidirectional GRU and character CTC head. Character posteriors are softly gated back into the frame representation. The initial feedback gate is biased mostly closed. CTC loss is trained only for records with an explicit `fingerspelling_text` label and enough valid frames.

**Current scope limitation:** one character transcript per clip is supported. Timestamped sub-clip spans, character timestamps and segmentation-level routing are not implemented. If only a segment is fingerspelled, prepare a corresponding segment-level clip/record with the appropriate label rather than attaching a segment transcript to a whole-sentence clip.

## 4. Reliability-aware fusion

Ten streams are projected to the shared `fusion.dim`:

- Left hand RGB, right hand RGB, face RGB, body RGB.
- Left hand pose, right hand pose, body pose, face pose.
- Motion.
- Fingerspelling.

Each stream provides a five-value quality descriptor: presence, ROI/pose coverage or area proxy, sharpness/motion or gate proxy, detector-confidence/availability proxy, and frame validity. The gating network learns per-frame stream weights. A local cross-stream attention stage searches residual offsets of -2 through +2 frame **indices** around each reference frame.

These offsets are frame-index lags, not milliseconds. The base pose pipeline aligns landmarks using timestamps; the residual lag module handles small remaining sampling/processing offsets. For variable-frame-rate recordings, evaluate whether a timestamp-based lag parametrization is preferable.

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

### 5.2 Coarse global reasoning

Consecutive fine features are grouped with `coarse_stride=2`. Each group is summarized using a masked mean and a first-to-last valid difference. The final partial group uses its last valid member, not a padded zero. The coarse sequence length is `ceil(T/2)` for the default configuration.

Twelve pre-normalized global Transformer blocks use 16 attention heads, 1,024 model dimensions, a 4,096-wide feed-forward layer and relative timestamp bias. A depthwise temporal convolution with kernel 15 provides local continuity. Self-attention provides global interaction across the full valid coarse sequence.

### 5.3 Fine detail retrieval

Each coarse token queries a configured local window (default seven fine positions) around its temporal centre. This is sparse local retrieval, not dense all-to-all fine/coarse attention. The retrieved context is repeated across the associated fine group and concatenated with the original fine features. The final temporal representation has the original frame length `T`.

This means the network does not discard the original fine-resolution pathway; it does **not** imply lossless preservation of every pixel-level detail inside learned features.

## 6. Latent plan / evidence aggregation

A learned bank of 32 latent queries (research config) cross-attends to the full fine-rate temporal memory. The resulting plan tokens and their query-to-frame attention maps are exposed. The frame attention is inspectable and can be visualized for qualitative analysis.

This is not a human-supervised grounding loss. The package does not include timestamped ground-truth evidence regions, human frame rationales or a direct attention-steering reinforcement-learning objective. Those are potential research extensions, not current implemented features.

## 7. Text decoder

A learned linear/LayerNorm bridge maps frame and plan tokens to the configured T5 hidden dimension. They are concatenated as encoder `inputs_embeds` and supplied to the full T5 encoder-decoder. The high-capacity config uses `google/flan-t5-xl`; the smoke config uses `google/flan-t5-small`.

The decoder supports teacher-forced translation cross-entropy, optional label smoothing, generation, and an optional unconstrained entropic semantic OT regularizer between visual memory and target text encoder states. OT is not a monotonic alignment and should be evaluated as an ablation, not assumed beneficial.

## 8. Proposed high-capacity configuration

The committed `configs/research.yaml` uses:

| Component | Setting |
|---|---|
| Appearance | DINOv2 ViT-L/14 via timm, 1,024 dimensions |
| Pose | 66 nodes, 512 dimensions, 3 graph layers, 8 heads |
| Motion | 512 dimensions, two temporal 3-D conv stages |
| Fingerspelling | 512-dimensional, 3-layer bidirectional GRU + CTC |
| Fusion | 1,024 dimensions, 3 interaction layers, 16 heads |
| Temporal | 1,024 dimensions, 12 global layers, 16 heads, FFN 4,096 |
| Detail path | Coarse stride 2, local detail window 7 |
| Planner | 32 latent queries, 3 cross-attention layers |
| Decoder | FLAN-T5-XL |
| Main objective | Translation CE + source-aware RGB↔pose InfoNCE + configurable CTC + configurable semantic OT |

These are high-capacity starting settings, not validated optima. The T5-XL and three independent 518-pixel appearance backbones require substantial memory. Batch size must be established empirically on the actual hardware; GPU capacity alone does not establish training stability or data quality.

## 9. Implemented vs not implemented

Implemented: the pathways and interfaces described above, source/signer split checks, self-distillation variant, translation training loop, checkpointing, metric script, and automated synthetic regression tests.

Not implemented: a raw undocumented `.pose` parser; landmark detector/tracker; actual ISL annotations; exact reproduction of SignDINO, SignMusketeers, Uni-Sign, SignBind-LLM or AttnSign; gloss/phrase segmentation; localized fingerspelling spans; human frame-grounding supervision; reinforcement-learning cadence steering; distributed multi-GPU trainer; human evaluation workflow; model weights.
