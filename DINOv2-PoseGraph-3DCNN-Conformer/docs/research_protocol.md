# Experimental and publication protocol

This document is a recommended protocol, not a claim that experiments have already been completed.

## 1. Freeze the corpus before training

1. Record dataset release/version, download date, licence/consent basis and source URLs.
2. Preserve original video IDs, parent source IDs, signer pseudonyms and sentence IDs in a provenance table.
3. Resolve duplicate source videos, re-encoded copies, overlapping clips and repeated sentence translations before making splits.
4. Build source-disjoint train/validation/test manifests. For a signer-independent claim, require complete signer metadata and use signer-disjoint splits as a separate, stricter evaluation.
5. Keep the test split sealed until model selection and hyperparameter tuning are finished.
6. Audit the overlap between datasets before combining corpora. Do not assume two benchmark names imply independent recordings.
7. Maintain a human-reviewed validation subset from fluent ISL signers. Dataset transcripts and automatically assembled captions must not be described as uniformly signer-verified unless the provenance says so.

## 2. Reproduce baselines first

At minimum, implement and evaluate under the same data, text normalization and split rules:

- Existing project CNN-BiLSTM + FLAN-T5 baseline.
- A pose-only temporal CNN/Transformer + T5 baseline (ISLiM-style direction).
- RA-MAGC pose pathway without RGB.
- RGB appearance-only model.
- RGB + pose model without the motion and fingerspelling specialists.
- Full ISL-MEGA.

Published BLEU scores across different corpora, splits, references or text normalization pipelines are not directly comparable. Report exact benchmark protocol beside every number.

## 3. Required architecture ablations

| Experiment | Change | Question |
|---|---|---|
| A0 | Pose-only baseline | What does the existing pose signal achieve? |
| A1 | RGB appearance only | Does raw visual appearance add information? |
| A2 | RGB + pose | Does complementary geometry help? |
| A3 | Add 3-D motion | Does explicit temporal appearance improve transitions? |
| A4 | Add fingerspelling expert | Does character supervision help names/technical terms? |
| A5 | Replace learned reliability gate with uniform fusion | Does reliability-aware gating help? |
| A6 | Remove residual asynchronous lag attention | Does temporal offset modelling help? |
| A7 | Remove coarse global transformer | Does global sentence reasoning help? |
| A8 | Remove fine detail retrieval | Does retained high-resolution memory help? |
| A9 | Remove latent plan tokens | Does latent evidence aggregation help? |
| A10 | Remove self-supervised visual pretraining | Does sign-specific SSL help over DINOv2 initialization? |
| A11 | Remove semantic OT | Does the unconstrained OT objective help? |
| A12 | Remove CTC | Does clip-level fingerspelling CTC help? |
| A13 | Remove source-aware RGB↔pose InfoNCE | Does cross-modal same-frame alignment improve translation? |
| A14 | Disable cross-articulator graph, shared hand encoder & torso normalization (`cross_articulator_graph=false`, `share_hand_encoder=false`, `normalize_by_shoulder_span=false`) | Does explicit hand-face-torso relational geometry improve unseen-signer and scale robustness? |
| A15 | Disable continuous-time lag & uncertainty-gated fusion (`continuous_time_lag=false`, `reliability_uncertainty_gate=false`) | Does Gaussian-lag alignment and heteroscedastic uncertainty gating outperform static stream gating under blur/occlusions? |
| A16 | Disable fingerspelling spotting & span-gated routing (`span_routing=false`) | Does span-gated character feedback reduce lexical-sign interference vs unconstrained CTC feedback? |
| A17 | Disable salience-adaptive temporal consolidation (`adaptive_consolidation=false`) | Does motion-salience boundary pooling beat uniform stride consolidation across variable signing speeds? |
| A18 | Disable diverse ordered planning & dynamic plan routing (`diverse_ordered_planning=false`, `dynamic_plan_routing=false`) | Do orthogonality, temporal ordering, and adaptive query halting improve multi-clause translation fidelity? |
| A19 | Disable plan semantic OT & counterfactual visual sensitivity (`plan_semantic_ot=false`, `use_counterfactual_grounding=false`) | Do hierarchical plan OT and counterfactual grounding curb language-prior hallucinations? |

Do not change multiple unlinked components outside matched config pairs (`configs/research.yaml` vs `configs/ablation_v1_baseline.yaml` for the full v1-at-matched-capacity baseline) in a purported single-factor ablation.

## 4. Required robustness and grounding checks

Evaluate the trained model on:

- Blank RGB and blank pose inputs.
- Temporally shuffled frames and pose sequences.
- Correct video with donor pose from another signer/clip.
- Correct pose with unrelated RGB frames.
- Missing one hand, face, or body ROI.
- Low-confidence and missing landmarks.
- Cropped/occluded hands and motion blur.
- Different signing speeds and variable frame rates.
- Long-sequence duration buckets, with original frames retained.

Measure whether generated translations change appropriately. A fluent decoder output on blank or mismatched inputs is a warning of language-prior hallucination.

Inspect planner attention and modality weights, but do not call attention maps proof of causal grounding. Stronger evidence needs human-labelled temporal/rationale annotations or controlled input interventions.

## 5. Metrics and human evaluation

Automated metrics currently implemented: corpus BLEU (SacreBLEU), chrF++, TER, optional ROUGE-L.

Report metric signatures, tokenization, casing, reference count, SacreBLEU version and exact split. Add other semantic metrics only after selecting and validating an appropriate model for English/Indian-English translation; do not assume generic embedding metrics are reliable for ISL meaning preservation.

Human evaluation should be performed by qualified/fluent ISL signers or interpreters, blinded to the system identity. Evaluate at least:

- Semantic adequacy / preservation of intended meaning.
- Omission and insertion of information.
- Handshape/fingerspelling-sensitive names and entities.
- Non-manual markers and negation/question distinctions.
- Grammatical acceptability in English.
- Hallucination or unsupported additions.

Report the annotation protocol, number and qualifications of raters, agreement and adjudication.

## 6. Statistics and reproducibility

- Use at least three, preferably five, independent training seeds for the final comparison.
- Report mean and standard deviation across seeds.
- Use paired bootstrap confidence intervals over source-video groups for metric differences.
- Preserve the best validation checkpoint selection rule before viewing test results.
- Report parameter count, total training compute, GPU model, precision, batch size, optimiser, scheduler, tokenizer/checkpoint revisions and code commit.
- Log all resolved configuration values and package versions.
- Archive predictions and evaluation scripts for every reported test checkpoint.

## 7. Novelty positioning

The literature already contains multi-region hand/face/body encoders, RGB-pose fusion, sign-specific self-supervised learning, fingerspelling/lipreading experts, attention steering and latent planning approaches. ISL-MEGA is therefore a synthesis proposal. A credible contribution must be narrowed to a testable new mechanism (for example, timestamp-aware reliability fusion across asynchronous anatomical experts with full-rate evidence retrieval) and shown to improve over controlled, matched baselines.

A Q1 journal quartile is a journal-level classification, not an architecture property. Acceptance depends on novelty, technical correctness, significance, writing, fit and peer review. No implementation or benchmark score guarantees it.

## 8. Proposed primary contribution: quality-calibrated asynchronous fusion

The active research configuration adds `fusion_calibration_weight` to the existing
timestamp-aware reliability fusion. For each frame and anatomical stream, the
model receives five contract-level quality signals. Their masked mean is
normalized across available streams and used as a training target for the
predicted fusion distribution through a KL divergence. Missing streams are
excluded from both target construction and loss reduction.

This is a mechanism hypothesis, not a novelty claim by itself. The paper must
compare it against the same model with the calibration weight set to zero,
uniform fusion, and the original learned gate. Required stress tests should
inject controlled hand, face, body, pose, blur, and timestamp corruption while
reporting translation quality, stream-weight retention, calibration error, and
latency error. The contribution is publishable only if the matched ablations
show consistent gains across independent seeds and held-out signers.
