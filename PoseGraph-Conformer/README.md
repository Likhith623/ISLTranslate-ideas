# RA-MAGC: Reliability-Aware Multi-Rate Articulatory Graph-Conformer

**Task:** continuous, gloss-free Indian Sign Language (ISL) → English translation  
**Status:** research prototype with synthetic regression tests; not a trained or benchmark-validated model  
**Python package:** `ramagc`

RA-MAGC is a structured-pose visual encoder prototype. It separates hands, upper body and
optional facial landmarks, tracks explicit observation/confidence provenance, keeps a
stride-one temporal memory, constructs a shorter global sequence, applies a Conformer, and
retrieves local detail from the retained fine memory. `RAMAGCTranslator` can connect this visual
encoder to an already-instantiated compatible T5 encoder-decoder model.

The package does **not** establish that RA-MAGC improves translation quality, is state of the
art, or is novel relative to all prior work. Those statements require controlled experiments.

## 1. Architecture

```text
Structured pose [B,T,...] + timestamps + observation/adjacency masks
        │
        ├── Left-hand edge-aware graph encoder (21 nodes)
        ├── Right-hand edge-aware graph encoder (21 nodes; shared hand weights)
        ├── Upper-body graph encoder (6 nodes)
        └── Optional face set encoder (18 sparse points)
                        │
            Per-frame region interaction
                        │
             Reliability-aware soft gating
                        │ [B,T,256]
          Two stride-one multi-scale temporal blocks
                        │
             Fine memory [B,T,256]
                        │
       Mean/difference pairwise aggregation
                        │ ceil(T/2)
       Four-block temporal Conformer [256, 8 heads]
                        │
       Local gather-based cross-attention into fine memory
                        │
          Visual-to-text projection [B,ceil(T/2),512]
                        │
      Compatible pretrained T5 encoder-decoder (external)
                        │
                English sentence
```

All valid supplied frames pass through the stride-one front end. Adjacent features are combined
to build a shorter global sequence. The original-resolution fine memory is retained and each
global token can retrieve a local window from it. This is **not lossless compression**: the
Conformer operates on the aggregated sequence, while a separate local retrieval path preserves
access to fine features.

There is no hard-coded universal maximum frame count. However, the Conformer self-attention
still has quadratic cost in the coarse sequence length. The actual maximum practical duration
must be measured on the target GPU and real sequence-length distribution.

## 2. Default architecture dimensions

| Component | Default |
|---|---:|
| Left/right hand landmarks | 21 each |
| Upper-body landmarks | 6 |
| Optional face landmarks | 18 |
| Hand node features | 13 |
| Body node features | 10 |
| Face node features | 13 |
| Edge features | 4 |
| Hand/body graph hidden size | 128 |
| Graph attention layers | 2 per region |
| Region interaction layers | 2; 4 heads |
| Fused feature size | 256 |
| Multi-scale temporal blocks | 2; stride 1 |
| Temporal branch kernels | (3, 5, 5, 9) |
| Temporal branch dilations | (1, 1, 2, 2) |
| Pairwise global aggregation | factor 2; output length ceil(T/2) |
| Conformer blocks | 4 |
| Conformer dimension / heads | 256 / 8 |
| Conformer FFN dimension | 1,024 |
| Conformer depthwise kernel | 15 |
| Visual bridge dimension | 512 by default |
| Text decoder | externally instantiated compatible T5 model |

The face stream is disabled by default. Enable it only after confirming that the 18 selected
facial points exist, are consistently ordered, are in the same coordinate system as the body
and hands, and have sufficient real-data coverage. A face-enabled model keeps a fixed four-stream
layout even when an entire batch has no valid face observations.

## 3. Feature data contract

`extract_structured_sample()` accepts already-decoded NumPy arrays; it does not decode `.pose`
files. All region coordinates must share the same image/camera coordinate system and units.

### Input shapes

- `times`: finite, one-dimensional timestamps in seconds, strictly increasing.
- `segments`: optional `[T]` segment IDs. A sample containing more than one ID is rejected.
- `raw_coords['body']`: `[T,6,2+]`.
- `raw_coords['left_hand']`, `raw_coords['right_hand']`: `[T,21,2+]`.
- Optional `raw_coords['face']`: `[T,18,2+]`.
- `raw_valid[region]`: `[T,N]`, boolean or finite numeric 0/1.
- Optional `raw_conf[region]`: `[T,N]`, finite values in `[0,1]` where present; NaN means unavailable.

At least one valid left/right shoulder pair is required somewhere in each clip to define the
torso reference. The torso centre is interpolated over time between valid shoulder-pair anchors;
the scale is the median valid shoulder distance for the clip. Absolute source timestamps are
converted to clip-relative elapsed seconds before storage to avoid loss of precision when
absolute timestamps are later represented in float32 tensors.

### Node channels

Hand features (`[T,21,13]`) in order:

1. `x`, `y`: torso-normalized global position;
2. `vx`, `vy`: finite-difference velocity in normalized units/second;
3. `x_local`, `y_local`: wrist-relative position;
4. detector confidence;
5. confidence-availability flag;
6. observed flag;
7. imputed flag (zero; this package does not impute);
8. position-valid flag;
9. velocity-valid flag;
10. wrist-relative-position-valid flag.

Body features (`[T,6,10]`) are `x,y,vx,vy,confidence,confidence_available,observed,imputed,
position_valid,velocity_valid`.

Face features (`[T,18,13]`) use the hand schema, with eye-relative coordinates in place of
wrist-relative coordinates. In the selected point list, indices 0 and 4 are the eye anchors
(landmark IDs 33 and 362).

Confidence availability is separate from detector confidence. The code does not infer a
confidence score from visibility or observation masks. An unavailable confidence is represented
by value 0 and availability flag 0.

Edges carry `[dx,dy,length,edge_valid]`. The hand graph uses 20 finger-bone edges plus
valid-node self-loops. The upper-body graph connects the shoulder bridge and the two arm chains.

### Temporal integrity

Timestamps must be in seconds and strictly increasing. Velocity uses each actual positive time
difference and is valid only when the same joint is valid in both adjacent frames. The first
frame has no finite-difference velocity. Edited/concatenated clips must be split before feature
extraction; the extractor rejects multiple segment IDs but cannot detect every edit without
reliable segment metadata.

## 4. Batching, padding and decoder targets

`RAMAGCCollator` pads variable-length samples and returns `lengths`, `timestamps`, all anatomical
features and masks, IDs and target strings. With a tokenizer, it also returns:

- `labels`, with target padding replaced by `-100` for the T5 loss;
- `decoder_attention_mask`, so target padding is not attended to by the decoder.

The visual encoder masks padded frames before temporal processing, and the Conformer re-masks
normalization/GLU outputs before depthwise convolution. Synthetic tests verify that valid
outputs do not change when the same example is right-padded in a batch.

A real frame with every articulator missing receives a learned missing-observation embedding;
a batch-padded frame is always reset to zero. These two situations are intentionally distinct.

## 5. Reliability descriptors

The fusion gate receives six per-region descriptors:

1. mean detector confidence over points where confidence is available;
2. fraction of points with confidence available;
3. observed-point ratio;
4. imputed-point ratio;
5. position-valid ratio;
6. velocity-valid ratio.

No imputation is currently performed; the imputed flag remains zero. If another imputation stage
is added, retain observed/imputed provenance and test for leakage from future frames or targets.

## 6. Model interface

Example feature extraction:

```python
from ramagc import extract_structured_sample, RAMAGCCollator, ModelConfig, RAMAGCVisualEncoder

sample = extract_structured_sample(
    uid="example-001",
    text="a sample translation",
    times=times_seconds,             # [T], seconds
    segments=segment_ids_or_none,    # [T] or None
    raw_coords=region_coordinates,
    raw_valid=region_validity,
    raw_conf=region_confidence_or_none,
    include_face=False,
)

batch = RAMAGCCollator()([sample])
encoder = RAMAGCVisualEncoder(ModelConfig())
visual_embeddings, visual_mask = encoder(
    **{key: value for key, value in batch.items()
       if key in {
           "left_hand_nodes", "left_hand_edges", "left_hand_adj", "left_hand_valid",
           "right_hand_nodes", "right_hand_edges", "right_hand_adj", "right_hand_valid",
           "body_nodes", "body_edges", "body_adj", "body_valid",
           "timestamps", "lengths", "face_nodes", "face_valid",
       }}
)
```

To use `RAMAGCTranslator`, provide an already-loaded T5 encoder-decoder object whose
`config.d_model` equals `ModelConfig.t5_d_model`. For FLAN-T5-small the expected width is 512,
but the checkpoint, tokenizer and Transformers version must be pinned by the integration
project. This package does not download model weights automatically.

## 7. Scope and limitations

This package does **not**:

- download ISLTranslate or ISH-NEWS;
- parse `.pose` files or extract pose keypoints from raw video;
- construct or validate source-, signer- or sentence-disjoint data splits;
- verify manifests, file hashes, licenses or source mappings;
- train, checkpoint, resume or run an evaluation loop;
- instantiate/download FLAN-T5 weights or tokenizer automatically;
- establish a measured BLEU/chrF/ROUGE score or generalization claim.

The `DataConfig` path fields are integration conventions, not evidence that these paths exist
or are read by the package. `RAMAGCDataset` is an in-memory wrapper over pre-extracted
`StructuredPoseSample` objects.

For scientific evaluation, build group-aware splits from trustworthy metadata. A parent-source
ID is not automatically a signer ID. Report source-disjoint and signer-disjoint results
separately where possible, plus exact train/test sentence overlap. Artiaga et al. report severe
performance reductions for several SLT systems under signer-independent evaluation and identify
sentence overlap as another source of optimistic results:
https://aclanthology.org/2025.findings-emnlp.997/

## 8. Research positioning and related work

The major building blocks are established precedents, not novelty claims:

- **ISLiM (WSLP 2026):** keypoint-based gloss-free ISL translation with temporal convolution,
  Transformer encoding and T5-small. The paper reports BLEU-4 5.85 on its iSign evaluation;
  do not compare this number directly with results from different splits or preprocessing.
  https://aclanthology.org/2026.wslp-1.3/
- **TWASEL (CVPR Workshops 2026):** left/right hand, body and face streams with learned fusion and
  a Conformer for continuous sign-language recognition. Recognition is related to, but distinct
  from, free-form sentence translation.
  https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Al-Azani_TWASEL_at_SignEval_2026_Adaptive_Multi-Stream_Pose_Fusion_for_Continuous_CVPRW_2026_paper.html
- **Camgoz et al. (ECCV Workshops 2020):** multi-channel Transformers for multi-articulatory sign
  language translation; multi-articulator modelling is not new by itself.
  https://doi.org/10.1007/978-3-030-66823-5_18
- **TransAlign-SLT (WSLP 2026):** TranslationCTC and masked entropic optimal transport for
  complementary temporal/semantic supervision. Any auxiliary alignment loss in future RA-MAGC
  experiments should be compared and ablated against this direction.
  https://aclanthology.org/2026.wslp-1.21/
- **ISH-NEWS (Scientific Reports, 2026):** reports 4,222 sentence videos derived from 162 broadcast
  videos and four professional interpreters. The paper notes broadcast-caption/signing lag and
  limited signer/style diversity; its videos are not a neutral substitute for diverse
  conversational ISL evaluation. Check data access, copyright, and split details before use.
  https://www.nature.com/articles/s41598-026-60893-0

A testable RA-MAGC hypothesis is that explicit graph edge modelling, shared torso-relative
coordinates, quality provenance, retained fine memory and local retrieval improve robustness
to missing/noisy articulators without losing long-range context. This is an experimental
hypothesis; neither novelty nor improved accuracy is established by this code.

## 9. Required experiments before performance claims

1. Reproduce the existing CNN-BiLSTM + FLAN-T5 baseline with matching data, tokenizer,
   optimization budget and split protocol.
2. Ablate edge-aware graphs vs. coordinate MLPs; hand/body/face streams; region interaction;
   reliability gate; multi-scale branches; pairwise aggregation; Conformer; fine retrieval.
3. Test torso-global vs. local coordinates, confidence present/absent, face present/absent, and
   controlled missing/noisy landmarks.
4. Compare no truncation against controlled truncation and temporal aggregation choices. Report
   GPU peak memory, throughput and latency alongside translation metrics.
5. Report BLEU-1..4, chrF++, ROUGE-L, TER and a semantic metric, including exact metric/tokenizer
   versions. Obtain qualified human ISL review where feasible.
6. Report signer-disjoint, source-disjoint and sentence-overlap analyses only where reliable
   identities are available. Do not infer signer independence from source IDs.
7. Use multiple seeds and paired/grouped bootstrap confidence intervals on the same fixed test
   examples. Report every run, not only the best seed.

## 10. Installation and tests

Requires Python >=3.10, NumPy >=1.24 and PyTorch >=2.1.

```bash
python -m pip install -e ".[test]"
pytest -q
```

The optional Hugging Face wrapper requires:

```bash
python -m pip install -e ".[t5]"
```

The regression suite uses synthetic data only. It is not a substitute for testing actual pose
files, training or benchmark evaluation.

## 11. Package modules

- `ramagc/config.py`: typed configurations and cross-component dimension checks.
- `ramagc/dataset.py`: structured feature construction, validation, normalization and collation.
- `ramagc/graph_layers.py`: edge-aware graph attention and face set encoder.
- `ramagc/fusion.py`: region interaction and reliability-aware gating.
- `ramagc/temporal.py`: stride-one temporal branches, pair aggregation, relative-time bias,
  Conformer and local fine-detail retrieval.
- `ramagc/model.py`: visual encoder and optional T5 wrapper.
- `tests/test_ramagc.py`: synthetic regression tests.
