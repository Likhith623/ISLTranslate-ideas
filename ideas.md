# Proposed Architecture for Continuous Indian Sign Language Translation

**Status:** Research architecture specification — to be implemented and validated  
**Working name:** Reliability-Aware Multi-Rate Articulatory Graph-Conformer (RA-MAGC)  
**Task:** Gloss-free, sentence-level ISL-to-English translation from complete pose/keypoint sequences  
**Primary text model:** The repository's existing `google/flan-t5-small` sequence-to-sequence model

> This is the recommended primary architecture to investigate. It is not an empirically proven optimum, a guaranteed accuracy improvement, or a guarantee of Q1-journal acceptance. The final architecture may be frozen only after the prescribed ablations and real-data validation. Do not describe an untested configuration as state of the art.

---

## 1. Design objective

The model must translate the complete signing sequence while retaining the short, fine-grained movements of the fingers and face and learning longer relationships across the sentence. It must not silently truncate a long sample, confuse a missing landmark with a padded time step, or treat automatically generated English captions as verified gold translations.

The recommended design has five components:

1. **Anatomical pose streams:** separate left hand, right hand, upper body and (when reliably available) face representations.
2. **Topology-aware spatial encoding:** model the connections between joints within each anatomical region.
3. **Reliability-aware cross-region fusion:** combine region representations according to the quality and availability of their observations.
4. **Dual-rate temporal encoding:** process every supplied frame at full temporal resolution, aggregate adjacent frames into a shorter global sequence, and retain the original high-resolution features for local detail retrieval.
5. **Global temporal modelling and text generation:** use a Conformer encoder followed by the repository's FLAN-T5-small encoder-decoder.

The full model is:

```text
Complete pose sequence with original timestamps
                  │
                  ▼
     Anatomical feature construction
                  │
       ┌──────────┼───────────┐
       ▼          ▼           ▼             ▼
   Left hand  Right hand  Upper body       Face*
   21 joints   21 joints   6 joints       18 points
       │          │           │             │
       ▼          ▼           ▼             ▼
   Shared hand graph encoder       Body graph encoder
   + left/right identity           Face landmark set encoder
       └──────────┬───────────┬─────────────┘
                  ▼
       Per-frame region interaction
                  │
                  ▼
       Reliability-aware gated fusion
                  │
                  ▼
       Full-resolution temporal features
                  │
                  ▼
       Two multi-scale temporal blocks
                  │
           ┌──────┴─────────┐
           │                │
           ▼                ▼
    Fine memory Z       Pairwise aggregation
       length T          length ceil(T/2)
           │                │
           │                ▼
           │        4-block Conformer
           │                │
           └──── local detail retrieval ────┐
                                            ▼
                                  Fused visual sequence
                                            │
                                            ▼
                              LayerNorm + visual projection
                                            │
                                            ▼
                         FLAN-T5-small encoder-decoder
                                            │
                                            ▼
                                  English translation

* The face stream is enabled only when the selected source data provides
  sufficiently reliable face landmarks. It is never fabricated or silently
  replaced by zeros interpreted as valid observations.
```

The architecture is designed for **offline translation of complete sentence/phrase clips**, which is the task supported by the current project. It is not a streaming sentence-boundary detector.

## 2. Input and feature contract

### 2.1 Landmark groups

Use the repository's current named-landmark selection as the starting point:

| Stream | Landmarks | Current project definition |
|---|---:|---|
| Left hand | 21 | Wrist and four joints for each finger |
| Right hand | 21 | Wrist and four joints for each finger |
| Upper body | 6 | Left/right shoulders, elbows and wrists |
| Face (optional) | 18 | Current sparse eye, eyebrow and mouth selection |
| Total without face | 48 | Current baseline selection |
| Total with face | 66 | Current selection when `features.face: true` |

Do not describe the current configuration as 48 landmarks plus 24 body/face landmarks. It is 42 hand landmarks plus 6 upper-body landmarks; enabling the currently implemented sparse face selection adds 18, giving 66.

The project reads actual component and landmark names from the pose header. Preserve that behavior. Never assume that a pose file's array order is identical across sources.

### 2.2 Per-landmark features

For each valid landmark, retain:

- Shoulder-centred, shoulder-width-normalized position `(x, y)`.
- Velocity `(vx, vy)` computed using the actual timestamp difference in seconds.
- Observation, imputation, position-validity and velocity-validity indicators.
- For hand landmarks, wrist-relative position `(x_local, y_local)` and a separate local-coordinate-validity indicator.
- For face landmarks, optional face-relative position normalized by the distance between the selected eye anchors, with its own validity indicator. If the anchors are unavailable, retain the global normalized position and mark face-relative coordinates invalid.

The proposed structured node tensors are:

| Stream | Tensor per batch | Channels per node |
|---|---|---:|
| Left hand | `[B, T, 21, 11]` | Global XY (2), velocity (2), wrist-relative XY (2), observed/imputed/position-valid/velocity-valid/local-valid (5) |
| Right hand | `[B, T, 21, 11]` | Same as left hand |
| Upper body | `[B, T, 6, 8]` | Global XY (2), velocity (2), observed/imputed/position-valid/velocity-valid (4) |
| Face (optional) | `[B, T, 18, 11]` | Global XY (2), velocity (2), face-relative XY (2), observed/imputed/position-valid/velocity-valid/local-valid (5) |

`B` is batch size and `T` is the original, per-example number of pose frames before batching padding. These are the **new structured inputs**, not the existing flattened baseline tensor. With the current default settings (48 landmarks, velocity enabled, no local-hand or bone blocks), the legacy flat input has 384 channels. Enabling the current 18-point face selection makes the corresponding base flat input 528 channels; optional local-hand and bone blocks add further channels. The feature-preparation and collation contracts must be extended explicitly.

### 2.3 Bone and topology features

Supply edge attributes to the graph encoder rather than flattening all bone vectors into an undocumented extra channel block. For an edge from joint `i` to joint `j`, use:

`e_ij = [x_j - x_i, y_j - y_i, ||p_j - p_i||, edge_valid]`.

Use the existing 20-edge hand topology (`HAND_EDGES`) as the starting hand adjacency. Use the existing five-edge upper-body skeleton as the starting body adjacency. A face stream should use a documented sparse landmark set-attention design or an explicitly documented facial graph; do not invent dense facial connections between unrelated landmarks.

Global coordinates must be retained. Wrist-relative hand coordinates are an additional representation, not a replacement: hand location relative to the torso and face can carry meaning.

### 2.4 Missingness, confidence and time

Keep these concepts separate:

- **Time padding:** positions introduced only to make examples in a batch the same length.
- **Missing landmark:** a landmark not observed in a real video frame.
- **Imputed landmark:** a value estimated across a short, explicitly permitted gap.
- **Invalid motion:** velocity unavailable because a landmark is absent or a continuity boundary was crossed.
- **Detector confidence:** retain only if the source pose format actually supplies a calibrated confidence value. The current repository mainly exposes thresholded validity masks; do not claim that continuous detector confidence is used when it is not available.

Keep original timestamps and segment/continuity identifiers. Calculate motion in units per second. Do not interpolate across a segment boundary or a gap longer than the configured limit. Do not replace unknown coordinates with plausible-looking coordinates.

### 2.5 Normalization

Retain the existing shoulder-anchor normalization as the body-relative reference frame. For hands, add wrist-relative geometry while preserving global/body-relative coordinates. For the optional face-relative representation, normalize by valid inter-eye distance. Any population standardization parameters must be fitted on the training partition only and then frozen for validation, test and inference.

Do not claim calibrated 3D geometry from uncalibrated depth values. If a source has only 2D coordinates, this remains a 2D model.

## 3. Spatial encoder: anatomical graph streams

### 3.1 Hand encoders

Use one **shared-weight hand graph encoder** for the left and right hand. Add a learned side-identity embedding (`left` or `right`) before region interaction. Shared parameters exploit the common hand anatomy and reduce unnecessary parameter duplication; the side embedding allows the network to retain laterality.

For each hand and each frame:

1. Project the 11-D node feature to 128 dimensions.
2. Add a learned joint-identity embedding.
3. Apply two residual edge-aware graph-attention layers, each with 4 attention heads and 128 output dimensions.
4. Use the bone-edge attributes and validity mask when computing messages.
5. Apply masked learned attention pooling across the 21 joints to obtain one 128-D hand token.

The left and right streams produce `[B, T, 128]` each, plus a region-availability mask and per-frame quality statistics.

### 3.2 Upper-body encoder

For each frame:

1. Project the 8-D node feature to 128 dimensions.
2. Add joint-identity embeddings.
3. Apply two residual edge-aware graph-attention layers over the six-joint upper-body graph.
4. Apply masked learned attention pooling to obtain `[B, T, 128]`.

The six selected body landmarks are sparse. Do not describe this branch as a full-body skeleton encoder.

### 3.3 Face encoder

Enable this stream only when face landmarks are available with adequate coverage and useful temporal resolution. The current implementation has 18 sparse face points, not a dense 468-point face mesh.

Project face landmark features to 128 dimensions, add landmark-identity embeddings, apply two small set-attention layers (4 heads), then masked attention pooling. A sparse set encoder is preferable to pretending the selected face points form a complete dense facial mesh.

If face landmarks are absent for an example, the stream is unavailable and its fusion gate is masked. The system must still function with the left-hand, right-hand and upper-body streams.

### 3.4 Per-frame region interaction

For each frame, collect the valid region tokens:

`R_t = [h_left, h_right, h_body, h_face]`, with shape `[B, T, K, 128]`, where `K=4` when face is enabled and `K=3` otherwise.

Add region-identity embeddings and apply a two-layer region-interaction Transformer with 4 heads and width 128. This allows the model to represent relationships such as two-hand coordination, hand-to-body location and hand-to-face interactions rather than merely concatenating region vectors.

### 3.5 Reliability-aware fusion

For each region and frame, compute quality descriptors from values that are genuinely available:

- Fraction of selected landmarks observed.
- Fraction imputed.
- Fraction with valid position and valid motion.
- Optional mean detector confidence, only when the input source exposes it.

A small gating MLP predicts one scalar reliability logit per region. Set the logit to negative infinity for a region with no usable landmarks, then normalize over the remaining regions:

`alpha_r,t = softmax_r(g([h_r,t, q_r,t]))`.

Fuse the gate-weighted region summary with the masked mean of the region-interaction outputs, then project to 256 dimensions:

`f_t = Linear_256(concat(sum_r(alpha_r,t * h_r,t), masked_mean_r(h'_r,t)))`.

If every region is unavailable at a real frame, use an explicit learned missing-observation representation and a missingness indicator. Do not confuse this case with batch padding.

**Output:** `F ∈ R^[B,T,256]` plus the frame mask, region masks, timestamps and segment identifiers.

## 4. Full-resolution multi-scale temporal detail encoder

The purpose of this stage is to learn short and medium motion patterns **before** temporal aggregation.

Use two stacked residual multi-scale temporal blocks. Each block operates at stride 1 and width 256. Each block contains four parallel depthwise-separable Conv1D branches:

| Branch | Kernel | Dilation | Local receptive field |
|---|---:|---:|---:|
| Fine | 3 | 1 | 3 frames |
| Short | 5 | 1 | 5 frames |
| Medium | 5 | 2 | 9 frames |
| Broad-local | 9 | 2 | 17 frames |

Each branch uses a depthwise temporal convolution over 256 channels followed by a pointwise projection to 64 channels, normalization and GELU. Concatenate the four 64-D outputs to 256 dimensions, apply a pointwise projection and dropout, then add the residual connection and apply LayerNorm.

All branches use odd kernels and same-length padding. Reapply the true-time mask after every convolution, normalization, activation and residual update. Local temporal operations must not mix frames across a declared discontinuity boundary.

For two such blocks in series, the largest convolutional receptive field along the broad-local path is:

`RF = 1 + 2 * (17 - 1) = 33 original frames`.

This 33-frame value describes the stacked local convolutional path only. It is **not** the receptive field of the complete model; the later Conformer provides global sequence attention.

## 5. Dual-rate temporal aggregation with retained fine-detail memory

### 5.1 Why this stage exists

Running full quadratic self-attention over every raw frame is unnecessary for every clip. However, selecting only every second or fourth frame can miss short hand/facial events. The model therefore computes fine temporal features at the original resolution, creates a compact sequence for global modelling, and keeps the fine features available for local retrieval.

### 5.2 Pairwise aggregation

Let the full-resolution temporal features be `Z ∈ R^[B,T,256]`. Within each continuous segment, group adjacent valid frames into pairs. For pair `i`, let the fine features be `a_i` and `b_i`:

`m_i = (a_i + b_i) / 2`

`d_i = b_i - a_i`

Concatenate `[m_i, d_i]` and project `512 -> 256`. Add a learned embedding of the pair's centre timestamp, pair duration and observation-validity pattern.

For an odd-length segment, the final group contains one real frame; its mean is that frame and its difference is zero. Do not create a synthetic second frame that is treated as observed.

For one uninterrupted segment:

`T_coarse = ceil(T / 2)`.

For multiple continuous segments, pair within each segment separately and concatenate the resulting tokens in time order:

`T_coarse = sum_s ceil(T_s / 2)`.

Never form a pair across a declared discontinuity boundary.

### 5.3 Retain the fine path

Keep the full-resolution tensor `Z` as a fine-detail memory. Do not discard it immediately after pair aggregation.

After global temporal modelling, each coarse token queries a small local window of the fine memory (the two frames in its pair and, when valid and in the same segment, the immediately adjacent fine frames). Use a small multi-head local cross-attention module and a gated residual update to refine the coarse token.

This gives the model:

- Fine temporal processing at all original input positions.
- A compact global sequence of approximately half as many tokens.
- A path to retrieve fine-grained details after sentence-level context has been learned.

**Important wording:** this is not mathematically lossless compression. The learned projection reduces dimensionality. The defensible claim is that every original frame is processed and contributes to the pair representation, while the full-resolution feature memory remains available for local detail retrieval. Do not claim that stride-2 aggregation is identical to retaining every frame as an independent global-attention token.

## 6. Global temporal backbone: Conformer

Use four Conformer blocks over the coarse sequence.

| Parameter | Primary specification |
|---|---:|
| Model dimension | 256 |
| Number of blocks | 4 |
| Attention heads | 8 |
| Head dimension | 32 |
| Feed-forward dimension | 1,024 |
| Depthwise convolution kernel | 15 coarse tokens |
| Convolution stride | 1 |
| Dropout | 0.1 |
| Normalization | Pre-LayerNorm / residual-stable Conformer layout |
| Attention | Bidirectional full self-attention over valid coarse tokens |

Each block follows the Macaron Conformer pattern:

1. Half-step feed-forward module.
2. Multi-head self-attention with relative-time attention bias.
3. Depthwise temporal convolution module (pointwise expansion, GLU, depthwise Conv1D, normalization, SiLU/GELU and pointwise projection).
4. Second half-step feed-forward module.
5. Final normalization and masked residual output.

Use a learned **relative elapsed-time bias** based on actual timestamp differences, not only raw frame indices. Bucket signed `delta_t` values in seconds using a documented log-spaced bucket scheme. This allows the attention mechanism to distinguish nearby and distant events when frame intervals are not perfectly uniform. Relative-time encoding can help represent timing differences; it does not guarantee speed invariance.

The self-attention layer gives every valid coarse token access to the full valid coarse sequence. Therefore, the global temporal context is the entire supplied clip, irrespective of the finite local convolutional receptive field.

For a 450-frame, uninterrupted sample, the nominal sequence sizes are:

- Input and fine temporal memory: 450 positions.
- Pair-aggregated sequence: 225 positions.
- Conformer output: 225 globally contextualized positions.

These are illustrative dimensions, not a claim that 450 is the dataset maximum. The actual maximum must be measured from the real pose manifest.

Use PyTorch scaled-dot-product attention / Flash Attention when the installed backend supports it. This reduces attention-memory overhead but does not make dense attention linear in sequence length. Keep frame-budget batching and dynamic micro-batches. There is no fixed `max_frames` truncation in the primary architecture.

## 7. Visual-to-text bridge and language decoder

The Conformer and local-detail fusion output has shape `[B, T_coarse, 256]`.

Use:

`LayerNorm(256) -> Linear(256, t5.config.d_model)`.

For `google/flan-t5-small`, obtain `d_model` from the loaded model configuration rather than hardcoding it. Pass the projected visual embeddings and the correct coarse attention mask to the existing `T5ForConditionalGeneration` through `inputs_embeds`, preserving the repository's current teacher-forced training and free-generation evaluation interface.

This is the full FLAN-T5 encoder-decoder path used by the current repository: the visual embeddings enter the T5 encoder and the T5 decoder generates English. Do not call it a decoder-only design.

Keep the same tokenizer, EOS handling, target padding convention (`-100` for ignored labels), reference-free prediction interface and generation-selection procedure as the existing model unless an independently controlled experiment justifies changing them.

## 8. Training objective

### 8.1 Primary objective

The mandatory objective is autoregressive translation cross-entropy over non-padding target tokens:

`L_translation = token_mean(CrossEntropy(logits, target_ids))`.

Preserve the repository's accumulation-window token normalization so the result is independent of how valid target tokens are distributed among micro-batches.

### 8.2 Weak lexical objective

The existing weak lexical-presence BCE head may be retained as an optional auxiliary objective:

`L = L_translation + lambda_lex(epoch) * L_lexical_BCE`.

The lexical targets are weak English word-presence labels. They are not gloss annotations, frame-level labels or sign-word alignments. Keep the auxiliary head optional and compare it against translation-only training.

### 8.3 Temporal/semantic alignment: controlled research experiment, not an assumed requirement

The 2026 TransAlign-SLT work combines decayed TranslationCTC with Masked Entropic Optimal Transport (MOT) between visual temporal features and contextualized target-word representations. This is a relevant experiment, not a component that should be copied into the core model without testing.

Evaluate, separately:

1. Translation CE only.
2. Translation CE + existing weak lexical BCE.
3. Translation CE + training-only MOT semantic alignment.
4. Translation CE + decayed TranslationCTC.
5. Translation CE + MOT + decayed TranslationCTC.

TranslationCTC is not a gloss CTC loss. It imposes a monotonic alignment pressure on a spoken-language token sequence, which may conflict with non-identical ISL and English word order. If tested, it must be explicitly described as weak auxiliary regularization, use a decaying weight, and be removed if it hurts validated translation quality. MOT requires a frozen contextual text encoder and must be masked for padding. No auxiliary target may be constructed from validation/test references.

Prefer human-validated or carefully quality-filtered targets for semantic alignment. Automatically harvested captions can be noisy and should not be treated as universally correct semantic supervision.

## 9. Training and regularization policy

- Start from the same source-held-out training partition and same text decoder as the baseline.
- First train the new visual encoder and visual-to-text bridge with the T5 weights frozen. Then unfreeze the T5 model using a lower learning rate for pretrained parameters than for new visual parameters. Choose the schedule on validation data; do not hardcode a universally optimal number of warm-up epochs.
- Use AdamW, warmup followed by cosine decay, gradient clipping and early stopping on a predeclared validation-selection policy.
- Use multiple fixed random seeds. Store the exact feature schema, data/split hashes, model configuration, tokenizer/model revision and numerical environment in every checkpoint.
- Fit all feature statistics and text-derived resources on the training partition only.
- Use pose-coordinate jitter and bounded, physically plausible temporal warping only after testing label preservation. Recompute velocities after a temporal transformation.
- Use landmark/region masking as a controlled robustness augmentation, not as a substitute for real missingness handling.
- Do not horizontally mirror examples by default: left/right laterality and spatial signing can be linguistically meaningful. Mirroring requires evidence that it preserves the target translation and a documented transformation policy.
- Keep RGB features outside the primary pose-only architecture initially. Once the pose model is stable, test synchronized RGB crops as a separate modality ablation using actual timestamps and explicit crop-validity masks.

## 10. Complete-frame policy and length handling

The primary policy is:

1. Read every valid frame in the source pose sequence.
2. Keep `resample_fps: null` by default; no fixed-rate resampling or hard temporal truncation.
3. Retain original timestamps and continuity boundaries.
4. Run spatial encoders and the fine temporal path at original sequence length `T`.
5. Use pairwise learned aggregation to form the global sequence at approximately `ceil(T/2)` tokens, retaining the full-resolution feature memory for local detail retrieval.
6. Apply global attention over every valid coarse token.
7. Batch by frame/token budget and reduce batch size for long examples rather than clipping the sequence.

The exact sequence-length distribution must be computed from the real training/validation/test manifests before the implementation is frozen. Report minimum, median, mean, 90th, 95th, 99th percentile and maximum frame counts, plus duration and effective FPS. Do not use the illustrative 450-frame example as a verified dataset maximum.

If an input file contains a corrupted or invalid timeline, reject or explicitly repair it under a documented data-quality rule. Never silently delete frames to satisfy a GPU budget.

## 11. Padding and discontinuity invariants

These requirements are mandatory implementation tests:

- Distinguish `frame_padding_mask` from per-joint and per-region validity masks.
- Apply the frame mask after every convolution, normalization, activation, projection and residual addition. LayerNorm on a zero vector can produce its learned bias; padded activations must be masked again.
- Use masked graph attention and masked landmark pooling. Missing nodes must not contribute as real zero-valued landmarks.
- Use correct key-padding masks in region attention, Conformer attention and T5 attention.
- Never allow a local temporal convolution, pair aggregation or local-detail retrieval window to cross a segment discontinuity.
- Keep actual elapsed time through pair aggregation and global attention.
- For odd lengths, verify the single-frame final pair and its mask.
- When every anatomical region is missing at a real time step, use the explicit missing-observation path, not the padding path.
- Outputs for a sample must be invariant (within numerical tolerance) to the amount of unrelated padding added by other examples in its batch.

## 12. Required architecture comparisons and ablations

Use matched data splits, target preprocessing, decoder, decoding policy, seeds and training budget wherever possible.

### 12.1 Baseline comparisons

| ID | Architecture | Purpose |
|---|---|---|
| A0 | Current two-layer temporal CNN + two-layer BiLSTM + FLAN-T5 | Reproducible project baseline |
| A1 | Current CNN + four-block Transformer/Conformer + FLAN-T5 | Isolate recurrent-to-attention change |
| A2 | Current flat input + proposed dual-rate Conformer | Isolate temporal design |
| A3 | Anatomical streams without reliability gate | Isolate region decomposition |
| A4 | Anatomical streams + reliability gate, without cross-region attention | Isolate gated fusion |
| A5 | Full RA-MAGC | Evaluate complete proposed design |
| A6 | Full RA-MAGC without fine-detail memory | Test whether retained high-resolution details matter |

### 12.2 Input and temporal ablations

- 48-landmark hands + upper-body input versus 66-landmark input with the sparse face branch, where face data are available.
- Global hand coordinates only versus global plus wrist-relative coordinates.
- With and without explicit bone-edge attributes.
- Shared versus separate left/right hand encoder weights.
- Reliability gate versus uniform region averaging versus concatenation.
- One temporal scale versus the full multi-scale block.
- No learned pair aggregation versus pair aggregation using mean only versus mean + signed difference.
- Full-resolution fine-memory retrieval on/off.
- Pairwise reduction factor 2 versus factor 4 as an ablation only; the primary design uses factor 2.
- Relative elapsed-time bias versus relative frame-index bias.
- Four versus six Conformer blocks; keep four as the initial default to control capacity on a low-resource dataset.

### 12.3 Robustness and dependency diagnostics

Evaluate valid inputs alongside:

- All-zero pose values with correct missingness masks.
- Temporally shuffled pose features, with timestamps and masks transformed consistently.
- Donor poses paired with an unrelated target, where this diagnostic can be constructed without leaking target identity.
- Left-hand, right-hand, body and face ablations.
- Controlled landmark occlusion and short missing intervals.
- Long examples and examples near segment discontinuities.

A model that produces nearly unchanged fluent English for blank or unrelated pose inputs may be exploiting the decoder's language prior rather than the visual sequence. Report this explicitly.

## 13. Evaluation protocol for credible research claims

### 13.1 Data partitions and contamination control

The current repository uses source-held-out splitting. Preserve it, but report the grouping key and the exact partition membership hashes. Where reliable signer identities are available, additionally create signer-independent evaluation. Also report sentence/translation overlap across partitions and remove duplicated or near-duplicated source material where possible.

Recent SLT evaluation research shows that signer-dependent splits and repeated target sentences can strongly inflate performance. Report both the split protocol and the train/dev/test overlap audit. If the dataset lacks verified signer identity, state that limitation rather than claiming signer-independent evaluation.

If combining ISLTranslate and iSign, resolve the parent source, original video identity, clip timestamps and duplicate/near-duplicate examples before training. Do not assume these datasets are independent. The iSign benchmark describes more than 118K video-sentence/phrase pairs and a human-validated sample of 593 pairs; that sample is not the same as a fully human-verified label set for the entire corpus.

### 13.2 Metrics

Report at minimum:

- Corpus BLEU-1, BLEU-2, BLEU-3 and BLEU-4, with tokenizer and smoothing stated.
- chrF++ with exact configuration.
- TER.
- ROUGE-L.
- METEOR, with resource/configuration stated.
- WER as a diagnostic, not the sole translation-quality measure.
- Semantic similarity metric, with model/version and limitations reported.

Do not compare published scores as if they used identical splits, data, preprocessing, reference quality, tokenizer or decoding. Report the complete evaluation signature with every result.

### 13.3 Human evaluation

On an independently selected sample, obtain translations or judgments from qualified ISL signers/interpreters. Assess meaning preservation, omission, addition/hallucination, grammatical adequacy and acceptability. Use multiple references where practical, report annotator qualifications, adjudication and agreement. Separate results on human-validated samples from results on automatically harvested captions.

### 13.4 Statistical reliability

- Train at least three independent random seeds; five is preferable when compute permits.
- Report mean and standard deviation across seeds.
- Use paired bootstrap confidence intervals over the same held-out examples for system comparisons.
- Use length-stratified reporting (short, medium, long) with thresholds derived from the observed dataset distribution, not arbitrary unsupported sign-duration claims.
- Report parameter counts, trainable parameter counts, training time, GPU/accelerator, peak memory, throughput and generation latency.

## 14. Implementation changes required in this repository

The current model accepts a flat `[B,T,F]` tensor. The proposed architecture requires structured anatomical streams and continuity metadata. It cannot be implemented correctly by merely increasing `cnn_channels` or replacing the LSTM class.

| File/module | Required change |
|---|---|
| `src/isl_slt/features/pose.py` | Build structured hand/body/optional-face features; expose local coordinates, edge attributes, per-node validity, original timestamps and segment IDs. Keep the old flat feature path for baseline reproducibility. |
| `src/isl_slt/data/dataset.py` | Return/pad each anatomical stream independently; preserve frame lengths, node masks, region masks, timestamps and segment IDs. Do not truncate. |
| `src/isl_slt/models/encoders.py` | Add hand graph encoder, body graph encoder, optional face set encoder, region interaction, reliability gate, two full-resolution multi-scale temporal blocks, pair aggregator, Conformer stack and local-detail retrieval. Keep the old `VisualEncoder` intact. |
| `src/isl_slt/models/factory.py` | Register a new architecture identifier (for example `ra_magc`) and keep `baseline_cnn_lstm` selectable. |
| `src/isl_slt/models/t5.py` | Keep the existing T5 input-embedding/attention-mask contract; accept the new visual encoder output dimension through the existing bridge mechanism. |
| `src/isl_slt/config.py` and `configs/` | Add validated architecture configuration for stream selection, graph widths, temporal branches, pair reduction, Conformer parameters and face availability. |
| Feature caches/checkpoints | Bump feature-schema and architecture versions. Include stream order, landmark names, normalizers, adjacency, temporal policy and all configuration in fingerprints. |
| `tests/` | Add shape, mask, boundary, padding-invariance, odd-length, missing-stream, full-frame-contribution, generation and checkpoint-compatibility tests. |

Do not overwrite the existing baseline architecture. A fair paper requires the baseline and proposed architecture to coexist and run on identical splits.

## 15. Minimum acceptance tests before any training run

1. **Shape test:** 48-landmark input produces left/right hand and body streams; 66-landmark input additionally produces the optional face stream.
2. **Header test:** shuffled source landmark order produces the same structured tensor after name-based selection.
3. **Full-frame test:** an input with `T` frames has `T` positions in the fine path before pair aggregation.
4. **Pair test:** an uninterrupted sequence produces `ceil(T/2)` tokens; every valid full-resolution temporal position is assigned to one pair, including the final odd position. Because the preceding temporal convolutions have overlapping receptive fields, an original input frame may influence more than one neighbouring pair token.
5. **Fine-memory test:** local-detail retrieval can access the original fine features and does not index padding or cross segment boundaries.
6. **Long-sequence test:** test at the observed maximum and above it without hard-coded truncation; batch size may reduce, sequence length must not.
7. **Missingness test:** absent face, absent hand joints, imputed joints and entirely missing real frames do not become valid observations.
8. **Padding-invariance test:** a sample evaluated alone and in a padded mixed-length batch gives matching valid outputs within a predefined numerical tolerance.
9. **Continuity test:** no convolution, pair or local retrieval crosses a declared discontinuity.
10. **Time test:** irregular timestamps and speed changes preserve finite velocity values and correctly update relative-time features.
11. **Gradient test:** all intended branches, gates, pair aggregation, Conformer and visual bridge receive finite gradients.
12. **Dependency test:** blank, shuffled and unrelated-pose diagnostics expose excessive decoder-prior dependence.
13. **Reproducibility test:** checkpoint reload restores exactly the same feature schema, masks, model architecture, tokenizer, T5 revision and decoding configuration.

## 16. Research positioning and novelty discipline

The following components are already present in relevant literature and must not be claimed as individually novel:

- Temporal convolution + Transformer/Conformer + pretrained T5 for ISL translation.
- Dedicated left-hand, right-hand, body and face pose streams.
- Region-wise graph encoders and multi-scale temporal convolution for sign recognition.
- Reliability/gating-based fusion of anatomical streams.
- TranslationCTC and semantic optimal-transport alignment as auxiliary supervision.

The research hypothesis to test is the **joint use of reliability-aware anatomical interaction with a dual-rate temporal design that retains the original fine-resolution feature memory and retrieves local detail after global sequence modelling**, evaluated specifically for gloss-free ISL translation under source-held-out and (where possible) signer-independent protocols.

This is a hypothesis, not a verified novelty claim. Before submission, conduct a systematic prior-art search for event-aware temporal pooling, multi-rate pose encoders, cross-scale memory retrieval and quality-aware articulated SLT. If an equivalent method exists, refine the contribution or reposition the work around rigorous ISL evaluation and generalization rather than asserting a new architecture.

## 17. Research references to include in the related-work comparison

1. Joshi, A., Agrawal, S. and Modi, A. (2023). **ISLTranslate: Dataset for Translating Indian Sign Language.** Findings of ACL 2023. DOI: [10.18653/v1/2023.findings-acl.665](https://doi.org/10.18653/v1/2023.findings-acl.665). Establishes the 31K-pair ISLTranslate resource.
2. Joshi, A. et al. (2024). **iSign: A Benchmark for Indian Sign Language Processing.** Findings of ACL 2024. DOI: [10.18653/v1/2024.findings-acl.643](https://doi.org/10.18653/v1/2024.findings-acl.643). Establishes the larger iSign benchmark and describes its 593-pair human-validation sample.
3. Ahmed, S. and Mishra, D. (2026). **ISLiM: A Compact End-to-End Sign Language Translation Framework.** Proceedings of WSLP 2026, pp. 18–31. DOI: [10.18653/2026.wslp-1.3](https://doi.org/10.18653/2026.wslp-1.3). Directly relevant keypoint-based ISL translation using temporal convolution, Transformer encoding and T5-small.
4. Krishna, P. M. et al. (2026). **TransAlign-SLT: Complementary Temporal and Semantic Supervision for Gloss-Free Sign Language Translation.** Proceedings of WSLP 2026, pp. 213–223. DOI: [10.18653/2026.wslp-1.21](https://doi.org/10.18653/2026.wslp-1.21). Relevant comparison for TranslationCTC and masked entropic optimal-transport supervision.
5. Al-Azani, S. et al. (2026). **TWASEL at SignEval 2026: Adaptive Multi-Stream Pose Fusion for Continuous Sign Language Recognition.** CVPR Workshops 2026, pp. 10473–10480. Relevant multi-region pose fusion and Conformer prior art; note that this is recognition, not sentence translation.
6. Tran, S. et al. (2025). **Generalizable Sign Language Recognition via Local Temporal Convolutions and Region-Aware Pose Encoding.** ICCV Workshops 2025. Relevant region-wise graph pose encoding and local temporal modelling; note that this is gloss-level recognition, not gloss-free translation.
7. Artiaga, K. et al. (2025). **Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation.** Findings of EMNLP 2025. DOI: [10.18653/v1/2025.findings-emnlp.997](https://doi.org/10.18653/v1/2025.findings-emnlp.997). Motivates signer-independent and sentence-overlap evaluation.

---

## Final architecture decision

**Implement RA-MAGC as the primary proposed model, with four anatomical streams when face data are available and three streams otherwise; two full-resolution multi-scale temporal blocks; learned pairwise stride-2 aggregation; a retained fine-detail memory with local cross-attention retrieval; four 256-dimensional Conformer blocks; and the existing FLAN-T5-small sequence-to-sequence text model.** Keep the current CNN-BiLSTM as the reproducible baseline. Use full original input sequences with no fixed frame cap. Treat CTC/MOT, RGB, six-block Conformer and stride-4 reduction as controlled experiments rather than mandatory assumptions.

Freeze this architecture only after the dataset frame audit, implementation acceptance tests, matched baseline comparison, multi-seed evaluation and human-validated translation assessment. A Q1 publication remains a research outcome to earn through evidence, not a property of the architecture specification.
