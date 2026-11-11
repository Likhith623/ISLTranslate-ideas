# RA-MAGC source archive: rigorous audit and corrected release

**Input archive:** `new architecture.(1).zip`  
**Audit date:** 8 November 2026  
**Corrected release:** `RA-MAGC-final.zip`  
**Package name:** `ramagc`

## 1. Executive result

The submitted archive was **not ready for use unchanged**. I found a critical mathematical error in
its edge-aware graph-attention score calculation, a broken test/import layout, missing package
metadata despite documented editable-install instructions, configuration values that were not
actually used, unnecessary dense computation in the supposedly local retrieval module, and a
missing decoder target-attention mask.

The corrected release addresses these issues, adds explicit input/configuration validation and
expands the regression suite. The corrected release passes **22 synthetic tests** on the tested
CPU environment. This is a software verification result only. It is not evidence of real-data
translation accuracy, target-GPU suitability, publication novelty, or Q1-journal readiness.

## 2. Files inspected in the uploaded archive

Source and documentation files:

- `__init__.py`
- `config.py`
- `dataset.py`
- `fusion.py`
- `graph_layers.py`
- `model.py`
- `temporal.py`
- `tests/__init__.py`
- `tests/test_ramagc.py`
- `README.md`
- `AUDIT_REPORT.md`

The ZIP also contained `__MACOSX` metadata and Python bytecode caches. These are not source of
truth and have been excluded from the corrected release. The original directory name included
spaces and a trailing period; the corrected project uses a standard importable `ramagc/` package.

## 3. Critical findings in the submitted archive

### 3.1 Edge-aware graph-attention equation was indexed incorrectly — critical

The original `EdgeAwareGraphAttention.forward()` used:

```python
scores = torch.einsum("bhnd,bhmnd->bhnm", q, k.unsqueeze(2) + edge_bias)
```

The projected tensors were laid out as query nodes `[i]`, key nodes `[j]`, and pair-specific
edges `[i,j]`, but the einsum labels aligned the query with the second pair axis and the key
projection was not paired with the key-node axis as intended. It therefore did not implement

\[
s_{ij}=\frac{q_i^\top(k_j+W_e e_{ij})}{\sqrt{d_h}}.
\]

This is a semantic/model-correctness defect, not a cosmetic style issue. Shape-only and
finite-gradient tests can pass while this equation is wrong.

**Correction:** form pair keys as `k.unsqueeze(2) + edge_bias`, form pair queries as
`q.unsqueeze(3)`, and reduce over the head dimension. A dedicated numerical test now compares
the method against an explicit nested-loop reference for a hand-constructed example.

### 3.2 The original test suite could not be collected as distributed

The source modules used relative imports such as `from .config import ...`, but the test inserted
the module directory into `sys.path` and imported `from dataset import ...` as a top-level module.
That raises `ImportError: attempted relative import with no known parent package` on a clean
checkout. I reproduced that failure using `pytest -q` in the extracted archive.

**Correction:** use a conventional `ramagc/` package directory, import through `ramagc.*`, and
add `pyproject.toml` package metadata.

### 3.3 Installation instructions referenced a missing build file

The original README instructed users to run `python -m pip install -e ".[test]"`, but the ZIP
contained neither `pyproject.toml` nor `setup.py`/`setup.cfg`.

**Correction:** add `pyproject.toml` with setuptools build metadata, runtime dependencies, and
optional `test` and `t5` extras. Editable installation and the documented test command were
executed successfully in this environment.

### 3.4 Multi-scale configuration was partly decorative

`MultiScaleTemporalConfig` exposed `kernel_sizes` and `dilations`, but the original
`MultiScaleTemporalBlock` hard-coded branch kernels/dilations and `RAMAGCVisualEncoder` did not
pass those configuration values to the block. Users could change configuration without changing
the architecture.

**Correction:** construct branches from the configured tuples, pass them through the model, and
validate their lengths, parity, and dilation values.

### 3.5 Fine-detail retrieval materialized dense attention before applying its local mask

The original `FineDetailRetrieval` computed scores of shape `[B,H,T_coarse,T_fine]`, then masked
all but a local band. The attention was local in semantics but still allocated/computed a dense
coarse-by-fine matrix.

**Correction:** gather each coarse token's bounded fine-memory window before computing attention.
The implementation now scales with the selected local window width rather than materializing the
full `[T_coarse,T_fine]` score tensor.

### 3.6 Pairwise aggregation encoded absolute recording-clock time

The original pair aggregator projected `(pair_duration, centre_timestamp)` directly. Absolute
recording-clock offsets are not sign-language timing features and can introduce spurious
variation between otherwise identical clips.

**Correction:** represent centre time as elapsed seconds relative to the clip's first frame.
Feature extraction also converts timestamps to clip-relative seconds before they are cast to
float32, avoiding precision loss for epoch-like source timestamps.

### 3.7 Decoder target padding was not passed as an attention mask

The original collator replaced target padding IDs with `-100` in `labels`, which correctly
excluded those positions from the loss, but did not return a decoder attention mask. The T5
wrapper did not pass one. Depending on model defaults, decoder self-attention could therefore
include target padding positions as keys.

**Correction:** the collator returns `decoder_attention_mask`; the translator passes it to T5
and derives it from `labels != -100` if not explicitly supplied.

### 3.8 Shoulder fallback could create artificial coordinate jumps

The original torso centre used the current frame's shoulder midpoint when available, but switched
to a clip-median midpoint when the shoulder pair was missing. Switching between these two
references can cause artificial coordinate and velocity jumps.

**Correction:** interpolate the torso-centre trajectory in timestamp space between valid
shoulder-pair anchors, extend the nearest valid anchor at clip boundaries, and keep the robust
clip-level median shoulder distance as scale. A clip with no valid shoulder pair remains an
explicit error.

### 3.9 Face stream could disappear for an all-face-missing batch

When face encoding was enabled but the collated batch had no face arrays at all, the original
model silently ran with three regions instead of four. The active region count could therefore
vary by batch.

**Correction:** when the model is configured with face enabled, missing face inputs are replaced
with zero features and all-false validity masks, preserving a fixed four-region interface.

### 3.10 Test and audit documentation were inconsistent

The original test module contained eleven test functions, while its audit report said ten passed.
The original backward test docstring also said every parameter had a finite, non-zero gradient,
although the code only checked finite values for parameters whose gradient was not `None`.

**Correction:** update the test descriptions and audit evidence. The current suite has 22
individually named synthetic tests; it checks finiteness of gradients that are present, not that
every conditional parameter must receive a non-zero gradient in every test.

### 3.11 Input/configuration validation was incomplete

The original feature extractor did not fully validate timestamp rank, region shapes, validity
mask values, confidence ranges or confidence tensor dimensions. Several cross-module dimensions
were not checked when constructing the top-level model configuration.

**Correction:** explicit validation is added for timestamp shape/order, segment-array length,
region shapes, binary validity masks, confidence tensor shape/range, model dimensions, head
divisibility, temporal kernel parity, dilation values, and residual channel compatibility.

## 4. Corrected release contents

The corrected ZIP contains:

```text
RA-MAGC-final/
├── pyproject.toml
├── README.md
├── AUDIT_REPORT.md
├── ramagc/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── fusion.py
│   ├── graph_layers.py
│   ├── model.py
│   └── temporal.py
└── tests/
    └── test_ramagc.py
```

## 5. Validation performed

**Environment:** Python 3.13.5, PyTorch 2.10.0 CPU, NumPy 2.3.5.

Commands executed from the corrected project root:

```bash
python -m compileall -q ramagc tests
python -m pip install -e ".[test]" --no-build-isolation --no-deps
pytest -q
```

Observed result:

```text
22 passed
```

The tests cover:

1. Hand/body/face tensor shapes and clip-relative timestamps.
2. Interpolation of missing shoulder-centre frames in timestamp space.
3. Rejection when no valid shoulder pair exists.
4. Velocity based on actual elapsed seconds.
5. Confidence-availability provenance.
6. Rejection of out-of-range detector confidence.
7. Timestamp rank/order and multi-segment validation.
8. Face feature extraction over 100 frames.
9. Odd/even pair aggregation and invariance to absolute timestamp offset.
10. Rejection of non-monotonic valid timestamps by the pair aggregator.
11. Actual use of configured multi-scale kernels and dilations.
12. Correct query/key/edge graph-attention score equation.
13. Finite graph-attention forward pass.
14. Conformer padding invariance.
15. Fine-detail local retrieval and output-mask behaviour.
16. Learned missing-observation output from the reliability gate.
17. Visual-encoder output shapes and batch-padding invariance.
18. Fixed four-stream face-enabled behaviour when face observations are absent.
19. Decoder attention-mask creation by the collator.
20. Passing the decoder attention mask into the T5 wrapper.
21. Finite gradients for parameters that receive gradients.
22. Rejection of incompatible architecture dimensions.

All tests are synthetic. The test suite does not load actual ISLTranslate `.pose` files, instantiate
or download FLAN-T5 weights, train a model, calculate BLEU/chrF/ROUGE on real translations, or
profile an intended GPU.

## 6. Research verification and positioning

The cited related-work claims were checked against the publisher/ACL pages available on
8 November 2026:

- **ISLiM**, Ahmed and Mishra, Proceedings of the Second Workshop on Sign Language Processing
  (October 2026), describes a keypoint-based gloss-free ISL translation framework using temporal
  convolutions, a Transformer encoder and T5-small. It reports BLEU-4 5.85 on its iSign evaluation.
  This confirms that the broad convolution + Transformer + T5 stack is not sufficient novelty by
  itself. https://aclanthology.org/2026.wslp-1.3/
- **TWASEL**, CVPR Workshops 2026, uses separate hand/body/face streams, learned gated fusion and
  a Conformer for continuous sign-language recognition. It is a close architectural precedent,
  but is a recognition task rather than free-form translation.
  https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Al-Azani_TWASEL_at_SignEval_2026_Adaptive_Multi-Stream_Pose_Fusion_for_Continuous_CVPRW_2026_paper.html
- **Camgoz et al.**, ECCV Workshops 2020, explicitly studied multi-channel multi-articulatory sign
  language translation. Multiple articulator streams are not novel in themselves.
  https://doi.org/10.1007/978-3-030-66823-5_18
- **TransAlign-SLT**, WSLP 2026, uses TranslationCTC and masked entropic optimal transport for
  complementary temporal and semantic supervision in gloss-free SLT. It is a relevant comparator
  for any future auxiliary alignment objective.
  https://aclanthology.org/2026.wslp-1.21/
- **Artiaga et al.**, Findings of EMNLP 2025, show that signer-overlapping evaluation can
  substantially overstate generalization and recommend signer-independent evaluation and
  sentence-overlap reporting.
  https://aclanthology.org/2025.findings-emnlp.997/
- **ISH-NEWS**, Scientific Reports (published 16 July 2026), reports 4,222 sentence videos derived
  from 162 broadcast videos and four professional interpreters. The paper describes subtitle/sign
  timing lag and limited signer/style diversity. This source may help broaden data studies but is
  not, by itself, an independent signer-diverse benchmark.
  https://www.nature.com/articles/s41598-026-60893-0

These papers establish architectural precedents and evaluation risks. They do not establish
RA-MAGC novelty or improved performance.

## 7. Still required before research claims

1. Integrate the real ISLTranslate `.pose` parser and verify landmark order, coordinate units,
   confidence semantics and validity conventions on actual examples.
2. Confirm whether 18 sparse face points are available and reliable; compare face-enabled and
   face-disabled runs.
3. Verify source, parent-video, signer and sentence identities from trustworthy metadata. A source
   ID must not be treated as a signer ID without evidence.
4. Construct leakage-audited train/validation/test splits and report their exact membership rules.
5. Pin FLAN-T5-small checkpoint/tokenizer revisions and Transformers version; measure its actual
   `d_model` before setting `t5_d_model`.
6. Implement and validate the training, checkpointing, resumption, decoding and metric pipeline.
7. Reproduce the CNN-BiLSTM baseline under matched conditions; run architecture, feature, temporal
   and loss ablations with multiple seeds.
8. Report BLEU, chrF++, ROUGE-L, TER and semantic/human evaluation with exact metric versions,
   tokenization and confidence intervals.
9. Profile peak memory, throughput and latency on the intended GPU over the real sequence-length
   distribution. The Conformer remains quadratic in the coarse token count.

## 8. Final judgement

The uploaded archive had a critical graph-attention equation defect and packaging/test issues; it
should not be treated as perfect. The corrected release fixes the reproduced implementation
problems and passes its 22 synthetic tests. It is a more reliable starting point for integration
and controlled experimentation, not a proven high-accuracy ISL translation system.
