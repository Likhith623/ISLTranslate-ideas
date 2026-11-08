# RA-MAGC: Reliability-Aware Multi-Rate Articulatory Graph-Conformer

**Task:** continuous, gloss-free Indian Sign Language (ISL) → English translation  
**Status:** research prototype / architecture implementation; not a validated benchmark result  
**Primary interface:** `RAMAGCVisualEncoder` and `RAMAGCTranslator`

RA-MAGC is a proposed structured-pose architecture. It is not claimed to be state of the art,
and this package does not establish that its architecture improves translation quality. Such
claims require controlled experiments on documented data splits.

## 1. Architecture

```text
Structured pose sequence [B,T,...] + timestamps + validity masks
        │
        ├── Left hand graph encoder (21 nodes, shared with right hand)
        ├── Right hand graph encoder (21 nodes, left/right side embedding)
        ├── Upper-body graph encoder (6 nodes)
        └── Optional sparse face-set encoder (18 points)
                        │
           Per-frame region interaction
                        │
             Reliability-aware gated fusion
                        │  [B,T,256]
          Two stride-1 multi-scale temporal blocks
                        │
              Fine-resolution memory [B,T,256]
                        │
       Mean/difference pair aggregation (stride 2)
                        │
        Coarse sequence [B,ceil(T/2),256]
                        │
       Four-block Conformer with signed relative
             elapsed-time attention bias
                        │
       Local cross-attention into fine memory
                        │
             Visual-to-text projection
                        │
            FLAN-T5-small encoder-decoder
                        │
                 English text
```

All valid frames pass through the stride-1 front-end. The pair aggregator creates a shorter
global sequence, but the full-resolution feature memory is retained for local detail retrieval.
This is **not lossless compression**: the fine memory remains accessible, but the global
Conformer sees the aggregated sequence. No universal maximum frame count is hard-coded.

### Default model dimensions

| Part | Default |
|---|---:|
| Hand nodes per side | 21 |
| Upper-body nodes | 6 |
| Optional face points | 18 |
| Hand node features | 13 |
| Body node features | 10 |
| Face node features | 13 |
| Edge features | 4 |
| Region token width | 128 |
| Fused feature width | 256 |
| Multi-scale branches | kernels (3,5,5,9), dilations (1,1,2,2) |
| Multi-scale blocks | 2 |
| Pairwise aggregation | factor 2 |
| Conformer blocks | 4 |
| Conformer width / heads | 256 / 8 |
| Conformer FFN width | 1024 |
| Conformer depthwise kernel | 15 |
| Text model width | 512 (FLAN-T5-small default) |

The face stream is **disabled by default** because its use depends on the availability, quality,
coordinate consistency, and permissions of facial keypoints in the selected data. Enable it
only after inspecting actual face coverage and running face/no-face ablations.

## 2. Feature data contract

All region coordinates must be in the same camera/image coordinate system.

Hand node features (`[T,21,13]`) are ordered as:

1. `x`, `y`: shoulder-reference-normalized global position
2. `vx`, `vy`: finite-difference velocity in normalized units/second
3. `x_local`, `y_local`: wrist-relative hand position
4. `detector_confidence`
5. `confidence_valid`
6. `observed`
7. `imputed` (currently zero; this extractor does not impute)
8. `position_valid`
9. `velocity_valid`
10. `local_valid`

Body node features (`[T,6,10]`) are:

1. `x`, `y`, `vx`, `vy`
2. `detector_confidence`
3. `confidence_valid`
4. `observed`
5. `imputed` (currently zero)
6. `position_valid`
7. `velocity_valid`

Face node features (`[T,18,13]`) use the hand feature schema, with eye-relative rather than
wrist-relative local coordinates. Face points 0 and 4 (33 and 362 in `FACE_POINTS`) are the
eye anchors.

If detector confidence is unavailable, the confidence value and its validity flag are both
zero. The system does not infer detector confidence from a boolean visibility mask.

Edges contain `[dx, dy, length, edge_valid]`. The hand graph contains 20 finger-bone edges
plus valid-node self-loops. The upper-body graph contains the shoulder bridge and two arm chains.

### Coordinate normalization

The body, hands and face use one common torso reference per frame:
- center: left/right shoulder midpoint when observed; otherwise the median valid midpoint
  within the same continuous clip;
- scale: median valid shoulder distance within the clip.

This avoids normalizing hand coordinates around the wrist/thumb base or facial coordinates
around arbitrary first-listed points. Wrist-relative and eye-relative coordinates are additional
features; they do not replace global signing-space position.

### Temporal integrity

`times` must be finite, strictly increasing seconds. A sample must be one continuous clip.
If source timestamps contain edits, concatenations or discontinuities, split those into separate
samples before feature extraction. This avoids velocity, convolution and attention crossing a
video cut. The extractor rejects multiple segment IDs in one sample.

`compute_velocity` does not estimate velocity across the first frame or missing joints.
It uses the true time difference rather than assuming a fixed FPS.

## 3. Batching and padding

`RAMAGCCollator` pads variable-length samples and returns `lengths`, `timestamps`, anatomical
features, masks, IDs and texts. If a tokenizer is supplied, it additionally returns `labels`
with target padding replaced by `-100`.

Padding invariants:
- `lengths` identifies all valid frame positions.
- all padded feature positions are zero;
- the temporal CNN explicitly zeros padded positions;
- the Conformer masks padded attention keys and zeros the normalized input to its temporal
  depthwise convolution, preventing LayerNorm bias from leaking into valid boundary frames;
- T5 receives the coarse visual attention mask.

The package tests compare a sample run alone and as the shorter item of a padded batch.

## 4. Reliability features

The gate receives six per-region statistics:
1. mean detector confidence over points with an available confidence value;
2. fraction of points with a confidence value;
3. observed-point ratio;
4. imputed-point ratio;
5. position-valid ratio;
6. velocity-valid ratio.

The learned missing-observation token is used only for a real frame with no valid region.
Batch padding is reset to zero before temporal processing.

The current feature extractor does not impute joints. If a separate imputation stage is added,
it must preserve observed/imputed provenance and be tested for information leakage.

## 5. Data and split limitations

`RAMAGCDataset` is an in-memory dataset over pre-extracted `StructuredPoseSample` objects.
This package does **not**:
- download ISLTranslate or ISH-NEWS;
- parse `.pose` files;
- construct source/signer splits;
- verify manifest hashes or source mappings;
- train, checkpoint, resume, or evaluate a model;
- load FLAN-T5 weights automatically.

The path fields in `DataConfig` are conventions for integration with the larger repository,
not proof that the paths exist or are read by these modules.

For research evaluation, use group-aware partitions. Do not use ordinary random clip-level
splitting when clips share signers, parent videos, sessions or near-duplicate content.
When reliable metadata are available, report signer-disjoint and parent-source-disjoint results
separately, and report train/validation/test sentence overlap. A grouped or stratified-group
cross-validation design can be valid; the relevant requirement is preventing leakage, not
banning the word “stratified”.

A source ID is not automatically a signer ID. The split generator must define and verify each
group key from trustworthy metadata. See Artiaga et al. (Findings of EMNLP 2025), which
documents substantial score drops under signer-independent evaluation and recommends
signer-independent and sentence-disjoint reporting:
https://aclanthology.org/2025.findings-emnlp.997/

## 6. Current research positioning

The broad combination of temporal convolution, Transformer/Conformer sequence modelling
and a pretrained T5 text model is not novel by itself. ISLiM (Ahmed & Mishra, WSLP 2026)
already presents a keypoint-based gloss-free ISL translation system with temporal
convolutions, a Transformer encoder and T5-small:
https://aclanthology.org/2026.wslp-1.3/

TWASEL (CVPR Workshops 2026) uses left-hand, right-hand, body and face streams with learned
gating and a Conformer for continuous sign-language recognition. It is related work, but
recognition and free-form translation are different tasks:
https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Al-Azani_TWASEL_at_SignEval_2026_Adaptive_Multi-Stream_Pose_Fusion_for_Continuous_CVPRW_2026_paper.html

A 2026 Scientific Reports paper introduces the ISH-NEWS sentence-level ISL resource with
4,222 sentence videos. Check its access, annotation, licensing and split suitability before
treating it as an additional evaluation source:
https://www.nature.com/articles/s41598-026-60893-0

Therefore, RA-MAGC must establish its contribution empirically. A plausible hypothesis is
that a shared anatomical coordinate frame, explicit keypoint-quality provenance, and
full-resolution memory retrieval improve robustness to missing or noisy articulators while
the coarse sequence handles long-range context. This is a hypothesis, not a result.

## 7. Required experiments before making performance claims

1. **Baseline parity:** reproduce the existing CNN-BiLSTM + FLAN-T5 configuration under the
   same data, tokenizer, optimization budget and split protocol.
2. **Architecture ablations:** graph vs. coordinate MLP; body/hand/face streams; reliability
   gating; multi-scale branches; pairwise aggregation; Conformer; fine-detail retrieval.
3. **Data ablations:** torso-global vs. wrist/eye-relative features; confidence available vs.
   unavailable; face enabled vs. disabled; noisy/missing keypoints.
4. **Temporal ablations:** no truncation vs. controlled truncation; stride 1 vs. aggregation
   factor 2; fine-detail retrieval on/off. Report memory, latency and throughput as well as
   translation metrics.
5. **Evaluation:** BLEU-1..4, chrF++, ROUGE-L, TER and at least one semantic metric; add human
   review by qualified ISL signers where feasible. Report the exact metric implementation and
   tokenization.
6. **Generalization:** signer-disjoint, source-disjoint and sentence-overlap analyses where
   metadata permit. Do not claim unseen-signer generalization if only parent-video IDs are known.
7. **Statistics:** multiple seeds and paired/grouped bootstrap confidence intervals on the
   same fixed test set. Report all runs, not only the best seed.

No target BLEU score or improvement is guaranteed by this architecture.

## 8. Installation and smoke tests

Requires Python 3.10 or newer and PyTorch 2.1 or newer. This audit was run on Python 3.13
with PyTorch 2.10.0 on CPU; other Python/PyTorch combinations have not been tested here.
Transformers is needed only for the pretrained text-model wrapper.

```bash
python -m pip install -e ".[test]"
pytest -q
```

To use the FLAN-T5 wrapper, install the `transformers` optional dependency and provide an
instantiated compatible encoder-decoder model whose `d_model` matches `t5_d_model`.

## 9. Project modules

- `config.py`: typed architecture and path configuration.
- `dataset.py`: structured in-memory pose feature construction, normalization, confidence
  bookkeeping and variable-length collation.
- `graph_layers.py`: edge-aware hand/body graph layers and optional face set encoder.
- `fusion.py`: per-frame region interaction and reliability-aware fusion.
- `temporal.py`: multi-scale temporal CNN, pair aggregator, elapsed-time bias, Conformer and
  fine-detail retrieval.
- `model.py`: visual encoder and optional Hugging Face encoder-decoder wrapper.
- `tests/`: synthetic shape, masking, normalization, temporal and model integration tests.
