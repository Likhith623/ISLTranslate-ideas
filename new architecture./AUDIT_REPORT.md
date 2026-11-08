# RA-MAGC archive audit and correction record

**Source archive:** `new architecture..zip`  
**Audit date:** 2026-11-08  
**Scope:** every human-readable source/document file in the supplied ZIP, runtime smoke tests,
tensor/data-contract review, and targeted research verification.

## 1. Executive result

The supplied archive was **not ready to use unchanged**. It contained seven Python modules and
one README, plus macOS metadata and CPython 3.14 bytecode caches. The Python sources compiled
in isolation, but targeted execution exposed multiple defects, including a face-path crash and
a batch-padding-dependent Conformer output.

The corrected package in this directory addresses the reproducible defects below and adds a
test suite. It is a more reliable **architecture prototype**, not a scientifically validated
ISL translation system. No real pose corpus, FLAN-T5 checkpoint, training log, validation/test
result, or GPU target was included in the source archive. Those empirical claims cannot be
verified here.

## 2. Original archive inventory

| Original file | Purpose | Audit disposition |
|---|---|---|
| `README.md` | Architecture overview, dataset claims, evaluation guidance | Rewritten to remove unsupported enforcement/novelty claims and state actual package scope |
| `__init__.py` | Package exports and package description | Updated; removed unsupported “state-of-the-art” label |
| `config.py` | Dataclass configuration and tensor dimensions | Updated to match the corrected feature schema and validate core dimensions |
| `dataset.py` | Pose normalization, velocity, edges, sample extraction and collation | Reworked; fixes torso-reference normalization, face broadcasting, timestamps and confidence provenance |
| `graph_layers.py` | Hand/body graph attention and face set encoder | Tightened adjacency with node-validity masks; fixed all-invalid face-frame broadcast |
| `fusion.py` | Per-frame region interaction and reliability gate | Retained the basic design; corrected TransformerEncoder construction warning |
| `model.py` | Anatomical streams, fusion, dual-rate temporal path and T5 wrapper | Fixed padded-frame handling, feature-quality indices and input-contract checks |
| `temporal.py` | Multi-scale CNN, pair aggregation, time bias, Conformer, fine retrieval | Reworked masking, timestamp handling, configurable branches and padding-safe Conformer convolution |

The ZIP also contained `__pycache__/*.pyc` files and `__MACOSX` entries. These are generated artifacts,
not source-of-truth files, and are excluded from the corrected package.

## 3. Defects reproduced in the original code

### 3.1 Optional face feature extraction crashed

A synthetic 100-frame sample with valid face landmarks raised a NumPy broadcasting `ValueError`.
The original code mixed `[T,1]` eye-validity masks with `[T,1,1]` eye distances and squeezed a
temporal axis before combining it with `[T,18]` landmark masks.

**Correction:** eye validity is represented consistently as `[T]` or `[T,1]`; the local-face
calculation explicitly broadcasts it across landmark and coordinate axes. The 100-frame
face extraction path now passes.

### 3.2 Face set encoder crashed when a frame had no valid face points

The original `torch.where` condition had shape `[B*T,1]`, while the result had shape
`[B*T,18,D]`; the missing trailing dimension caused a runtime size mismatch.

**Correction:** expand the invalid-frame condition to `[B*T,1,1]`. Face-enabled inference now
passes when one sample has face points and another has none.

### 3.3 Hand and face coordinates were not normalized by the reference claimed in comments

The original generic `normalize_positions()` used the first two points of each region as
anchors. For hands this meant wrist and thumb-CMC; for face it meant the first two listed face
points. It did not use the shoulder midpoint/width described in the comments and intended data
contract.

**Correction:** establish one per-frame torso reference from the left/right shoulder pair and
robust per-clip shoulder width, then normalize body, both hands and face in that same coordinate
system. Wrist-relative hand and eye-relative face coordinates remain additional channels.

A clip without any valid shoulder pair is rejected with an explicit error rather than silently
producing an invalid coordinate frame.

### 3.4 Timestamp validation was missing

The original velocity code divided by timestamp differences without checking for duplicate,
decreasing or non-finite timestamps. This could produce infinities or NaNs.

**Correction:** timestamps must be finite and strictly increasing; each sample must represent one
continuous segment. Clips containing edits/discontinuities must be split before extraction.
Velocity is computed only for adjacent valid points.

### 3.5 Confidence was claimed but was not represented

The original feature extractor used an always-zero imputation placeholder and had no separate
detector-confidence input. Its reliability gate therefore could not distinguish an unavailable
confidence value from a low confidence value.

**Correction:** node schemas now contain detector confidence and a separate confidence-available
flag. When confidence is missing, both channels are zero; no confidence is fabricated from a
boolean observation mask. The extractor still does not impute joints, and its imputation flag
remains zero until a separately documented imputation stage exists.

### 3.6 Conformer output depended on batch padding

A direct test compared four valid tokens alone with the same tokens followed by two padding
tokens. After setting the Conformer input LayerNorm bias to a non-zero value, the original valid
outputs differed by a maximum absolute value of approximately `0.1839`.

**Cause:** `LayerNorm(0)` at padded positions produced its learned bias, which then entered the
depthwise temporal convolution and affected adjacent valid positions.

**Correction:** reapply the temporal mask immediately after the input LayerNorm and after GLU,
before the depthwise convolution. In the corrected code, the same test returns a maximum
absolute difference of `0.0`. The complete visual encoder also passes a right-padding invariance
test within numerical tolerance.

### 3.7 Padded frames could be represented as real missing-observation frames

The reliability gate deliberately emits a learned missing-observation embedding when a real
frame has no valid articulator. The original top-level model did not reset this embedding to
zero at batch-padded positions before temporal convolutions.

**Correction:** use `frame_mask` to preserve the missing-observation embedding on real frames
but force all batch-padding positions to zero before temporal processing.

### 3.8 Dataset and README claims exceeded the actual code

The original README said the package enforced cryptographic source manifests and source-held-out
splits. No split generator, source-mapping reader, manifest-hash checker, or signer identity
checker exists in the supplied Python modules.

**Correction:** the README now states that the package accepts pre-extracted in-memory samples
and does not generate or verify splits. It distinguishes parent-source identity from signer
identity and recommends reporting the appropriate group-disjoint evaluations when reliable
metadata are available.

## 4. Research verification and implications

The following sources materially affect how the architecture should be described:

1. **ISLiM (Ahmed & Mishra, WSLP 2026)** already combines temporal convolution, a Transformer
   encoder and pretrained T5-small for gloss-free keypoint-based ISL translation. Thus, that
   broad stack must not be claimed as a novel contribution:
   https://aclanthology.org/2026.wslp-1.3/

2. **TWASEL (CVPR Workshops 2026)** uses separate left-hand, right-hand, body and face streams,
   learned gated fusion, and a Conformer for continuous sign-language recognition. It is a
   close architectural precedent; recognition is not the same task as free-form translation:
   https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Al-Azani_TWASEL_at_SignEval_2026_Adaptive_Multi-Stream_Pose_Fusion_for_Continuous_CVPRW_2026_paper.html

3. **Multi-channel Transformers for Multi-articulatory Sign Language Translation (Camgoz et al.,
   ECCV Workshops 2020)** established multi-articulator modelling of hand, face and body channels
   in sign-language translation. Multi-stream articulation modelling is therefore not new by
   itself:
   https://doi.org/10.1007/978-3-030-66823-5_18

4. **TransAlign-SLT (Krishna et al., WSLP 2026)** uses auxiliary TranslationCTC and masked
   entropic optimal transport for temporal and semantic supervision in gloss-free SLT. Any
   CTC/semantic-alignment proposal must be compared against this work and independently ablated:
   https://aclanthology.org/2026.wslp-1.21/

5. **Artiaga et al. (Findings of EMNLP 2025)** report substantial performance drops under
   signer-independent evaluation and recommend signer-independent and sentence-disjoint
   reporting. Source-disjoint splits alone do not prove unseen-signer generalization unless
   source and signer identities are linked by verified metadata:
   https://aclanthology.org/2025.findings-emnlp.997/

6. **ISH-NEWS (Scientific Reports, 16 July 2026)** describes a sentence-level ISL resource with
   4,222 sentence videos. It is a potentially relevant additional data source, but accessibility,
   licensing, annotation quality, overlap and split compatibility need separate checks:
   https://www.nature.com/articles/s41598-026-60893-0

These sources support a cautious research positioning. They do not establish RA-MAGC novelty,
superior BLEU, or journal acceptance. The central RA-MAGC hypothesis—shared torso-relative
anatomical features, explicit confidence provenance, dual-rate temporal processing and local
fine-memory retrieval—must be tested against strong, properly matched baselines.

## 5. Corrected package test evidence

The corrected package contains ten tests covering:

- hand/body/face tensor dimensions;
- shared shoulder-reference normalization;
- confidence-availability provenance;
- face feature extraction on a 100-frame sequence;
- rejection of duplicate timestamps and multi-segment clips;
- odd/even pairwise aggregation in a mixed-length batch;
- Conformer padding invariance;
- end-to-end visual-encoder output and batch-padding invariance;
- face-enabled mixed-availability forward pass;
- real frames where all articulators are missing (learned fallback path);
- finite gradients from a backward pass.

**Observed result:** `10 passed` on Python 3.13 with PyTorch 2.10.0 CPU. All corrected Python
modules also compile. Editable installation succeeded with local build isolation disabled.
A normal isolated editable install could not retrieve the build dependency because this
execution environment has no external package-index access.

These are synthetic software tests. They do not validate real pose files, data provenance,
GPU memory requirements, convergence, translation quality, or generalization.

## 6. Remaining work before research use

1. Integrate the actual `.pose` parser and source metadata pipeline from the main ISLTranslate
   repository; this archive contains no parser for those files.
2. Confirm the raw coordinate systems, landmark order, units, confidence semantics and validity
   conventions against real pose files.
3. Build and audit source-, signer- and sentence-aware splits from verified metadata.
4. Connect a pinned FLAN-T5-small checkpoint/tokenizer and record their exact versions/hashes.
5. Implement the training, checkpointing, decoding and evaluation pipeline.
6. Run baseline parity, architecture ablations, data-quality ablations and multi-seed evaluation.
7. Profile long samples on the actual target GPU; full-resolution memory and attention cost must
   be measured on the real length distribution.

## 7. Final judgement

The original ZIP was **not perfect and should not be used unchanged**. The corrected files remove
the reproducible runtime and data-contract defects found in this audit, add regression tests, and
state the research limitations honestly.

No software package can be certified “perfect” by a finite synthetic test suite. The corrected
package is suitable as a starting implementation for integration and controlled experiments;
it is not yet a completed, empirically validated SLT research result.
