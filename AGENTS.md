# MASTER RESEARCH AND IMPLEMENTATION PROMPT
## RA-MAGC Benchmark → Full ISL-MEGA Multimodal Indian Sign Language Translation System

### ROLE

Act as a senior research scientist, deep-learning architect, computer-vision engineer, dataset engineer and experimental auditor specializing in:

- Continuous Sign Language Translation (CSLT).
- Indian Sign Language (ISL).
- Multimodal deep learning.
- Spatiotemporal representation learning.
- Graph neural networks.
- Vision Transformers and video Transformers.
- Conformer architectures.
- Cross-modal attention and alignment.
- Gloss-free translation.
- Research reproducibility and statistical evaluation.

Your task is to develop a scientifically defensible, complete and reproducible Indian Sign Language-to-English translation research system, with the ultimate objective of producing research suitable for submission to a reputable Q1-ranked journal.

Do not promise publication or manufacture research novelty. The objective is to develop a genuinely novel contribution and prove its effectiveness experimentally.

I have already prepared the following resources:

1. Original ISLTranslate project, including the existing CNN-BiLSTM implementation and its data pipeline.
2. RA-MAGC corrected implementation ZIP.
3. Full ISL-MEGA research implementation ZIP.

Locate these resources in the current workspace or attached files. Identify them by inspecting their contents. Do not assume filenames or directory structures.

Your work MUST proceed sequentially through two major research stages.

---

# PART I — NON-NEGOTIABLE RESEARCH RULES

1. Never fabricate training results, validation results, dataset statistics, GPU performance, metrics, annotations or human-validation results.

2. Never describe synthetic testing as real-dataset validation.

3. Never assume that two datasets are independent because they have different names.

4. Never combine training, validation and test examples without checking parent-video overlap.

5. Never train on the human-validated evaluation set and subsequently claim independent evaluation.

6. Preserve complete original video duration. Do not impose arbitrary 128-, 200-, 256- or 450-frame truncation merely for convenience.

7. Preserve original timestamps, frame ordering, variable-length sequences and observation masks.

8. Use the existing CNN-BiLSTM implementation as the baseline. Do not silently replace or modify it.

9. Never change the original baseline split when reporting a direct reproduction comparison. If it has leakage problems, additionally produce a properly grouped evaluation protocol.

10. Use the actual pretrained checkpoints and verify their architecture, dimensions, tokenizer, configuration and licensing.

11. Make all experiments reproducible through saved configurations, source manifests, seeds, model checkpoints, dependency versions and evaluation signatures.

12. Do not claim that RA-MAGC or ISL-MEGA is state of the art without comparable, independently reproducible experimental evidence.

13. Do not delete or overwrite the original project, either ZIP, existing results or checkpoints. Work in a separate version-controlled research directory.

14. You have access to substantial GPU resources. Do not artificially simplify the proposed architecture solely because of assumed computational limitations. Nevertheless, investigate parameter efficiency and overfitting rather than equating parameter count with research quality.

---

# PART II — PHASE 0: COMPLETE AUDIT OF EXISTING PROJECTS

Before implementing anything, thoroughly inspect:

A. Original ISLTranslate project.

B. RA-MAGC corrected ZIP.

C. ISL-MEGA research implementation ZIP.

Read every Python module, YAML configuration, Markdown document, test, packaging file and training/evaluation script.

Produce:

`reports/INITIAL_CODEBASE_AUDIT.md`

Include:

- Exact project structure.
- Existing CNN-BiLSTM architecture.
- Original input dimensions.
- Original preprocessing.
- Dataset identifiers.
- Existing split policy.
- Sequence-length handling.
- Pose extraction and coordinate conventions.
- Decoder architecture.
- Tokenizer.
- Optimizer and scheduler.
- Loss functions.
- Existing test coverage.
- Missing modules.
- Integration incompatibilities.
- Actual external dependency requirements.
- Research limitations.

Do not infer that the two ZIPs are perfectly compatible.

Identify the differences between their APIs, configurations and tensor contracts.

Then build an integration plan without modifying the original source files.

---

# PART III — PHASE 1: REPRODUCE THE ORIGINAL CNN-BiLSTM BASELINE

This is the first mandatory experimental milestone.

Use EXACTLY the same ISLTranslate training corpus, video identifiers, translation annotations, preprocessing, data split and evaluation protocol used by the original project.

Primary dataset:

https://github.com/exploration-lab/isltranslate

Original dataset paper:

https://aclanthology.org/2023.findings-acl.665/

The original repository contains:

- ISLTranslate.csv.
- ISL sentence/phrase videos.
- Thirteen MediaPipe Holistic pose archives.
- Human-validation information for 291 sign-text pairs.

Do not add iSign, GloSL, ISL-FS or ISH-NEWS to this experiment.

### Phase 1A: Reproduce the original implementation

First establish whether the original CNN-BiLSTM code runs correctly.

Recover its actual architecture and training configuration.

Record:

- Train/validation/test sample IDs.
- Number of unique parent videos.
- Number of signers where identifiable.
- Maximum, minimum, mean and median sequence lengths.
- Tokenization configuration.
- Training steps and epochs.
- Optimizer.
- Scheduler.
- Loss.
- Validation selection criterion.
- Checkpoint selection policy.

Run the original baseline.

Save:

`results/baseline/`

Include the trained checkpoint, predictions, references, logs, metrics, configurations and environment details.

If the published benchmark results cannot be reproduced, diagnose the reason instead of pretending that they have been reproduced.

### Phase 1B: Integrate RA-MAGC

Use the provided RA-MAGC implementation.

Do not redesign it arbitrarily.

Integrate its corrected modules into the original ISLTranslate training and evaluation pipeline.

The experiment must use exactly the same source video IDs and train/validation/test membership as the reproduced CNN-BiLSTM baseline.

Model-specific representations may differ, but they must originate from the same original video frames, timestamps and underlying pose observations.

Create a verified conversion pipeline for the required anatomical features.

The RA-MAGC representation must preserve:

- Left hand.
- Right hand.
- Upper body.
- Face where valid observations exist.
- Relative coordinates.
- Global spatial coordinates.
- Motion and velocity.
- Confidence.
- Missing-observation masks.
- Actual temporal information.

If the original data provides 543 MediaPipe landmarks, create a documented named-index mapping to the intended RA-MAGC representation.

Never assume landmark ordering.

Verify the mapping against the source files, MediaPipe Holistic metadata and rendered landmark overlays.

Do not silently substitute missing facial landmarks with fabricated points.

Run visual inspection on randomly selected real examples.

### Phase 1C: Fair baseline comparison

Train RA-MAGC using the same data and split as the original CNN-BiLSTM.

Keep the decoder and text-processing configuration fixed for the principal architecture comparison wherever technically compatible.

Also report a separate comparison if decoder configurations differ.

Use at least three independent training seeds where feasible.

Metrics:

- SacreBLEU BLEU-1, BLEU-2, BLEU-3 and BLEU-4.
- chrF++.
- ROUGE-L.
- TER.
- WER.
- METEOR.
- Semantic evaluation where technically justified.

Record SacreBLEU version and complete metric signature.

Use paired bootstrap confidence intervals grouped by parent source video.

Do not choose models using the test set.

### Phase 1D: Mandatory diagnostic tests

Evaluate:

1. Normal inputs.
2. Completely blank pose inputs.
3. Completely blank visual inputs where applicable.
4. Temporally shuffled frames.
5. Incorrect donor poses.
6. Incorrect donor RGB videos.
7. Mismatched RGB-pose pairs.
8. Hand-only inputs.
9. Face-only inputs.
10. Missing-region inputs.

The model must demonstrate meaningful dependence on the visual signing evidence.

### Phase 1 deliverables

Create:

`reports/RA_MAGC_BENCHMARK_REPORT.md`

Include:

- Original baseline results.
- RA-MAGC results.
- Statistical comparisons.
- Error analysis.
- Dataset split verification.
- Performance by sequence length.
- Performance on human-validated translations.
- Failure examples.
- Reproducibility instructions.

Do not proceed to a publication claim simply because the architecture passes software tests.

Once this stage is complete, proceed to the multimodal system.

---

# PART IV — PHASE 2: DATASET ACQUISITION AND PROVENANCE AUDIT

The final ISL-MEGA architecture requires RGB videos, pose representations and supplementary specialist supervision.

Investigate the following official datasets.

## Dataset A: ISLTranslate

Official repository:

https://github.com/exploration-lab/isltranslate

Use this to preserve continuity with the original CNN-BiLSTM and RA-MAGC experiments.

It provides sentence-level RGB videos, pose features and English translations.

## Dataset B: iSign v1.1

Official dataset:

https://huggingface.co/datasets/Exploration-Lab/iSign

Official website:

https://exploration-lab.github.io/iSign/

Paper:

https://arxiv.org/abs/2407.05404

This is the primary larger-scale multimodal ISL resource.

The release contains:

- Video archives.
- Pose archives.
- English translation CSV.
- Word-presence resources.
- Semantic-description resources.

Important technical instructions from its official documentation:

The video archives are divided into parts:

`iSign-videos_v1.1_part_aa`

`iSign-videos_v1.1_part_ab`

The pose archives are divided into parts:

`iSign-poses_v1.1_part_aa`

`iSign-poses_v1.1_part_ab`

`iSign-poses_v1.1_part_ac`

`iSign-poses_v1.1_part_ad`

The official instructions use binary concatenation to reconstruct these archives.

Use the original CSV UID structure:

`video_id-sequence_number`

Group all examples from the same `video_id` into one split.

Accept the required Hugging Face dataset conditions and verify the non-commercial CC-BY-NC-SA-4.0 restrictions before use.

Do not count ISLTranslate examples inside iSign as new independent examples.

## Dataset C: GloSL-ISL

Dataset:

https://huggingface.co/datasets/pedroodb/glosl-isl

Documentation:

https://pypi.org/project/glosl/

Reported dataset statistics:

- 126,846 samples.
- 269.03 hours.
- 101,481 training examples.
- 12,683 development examples.
- 12,682 test examples.

It uses 543-point MediaPipe Holistic representations and a source-video-disjoint, topic-stratified split.

Inspect the actual downloaded `annotations.csv`, `card.json`, diagnostics and pose files.

The dataset viewer currently exposes fewer rows than the dataset card claims. Verify the complete underlying release before treating the reported count as established.

The hosted GloSL-ISL release lists annotations, metadata and pose files. Do not assume all original RGB videos are hosted there. Verify `video_file` and `video_ref` resolution.

Use its existing split where appropriate for a separate harmonized pose benchmark.

Do not merge it blindly with iSign or ISLTranslate.

## Dataset D: ISL-FS-Dense

Dataset:

https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling

Official project:

https://kirandevraj.github.io/ISL-Fingerspelling/

Official code:

https://github.com/Kirandevraj/ISL-Fingerspelling-Dense

Paper:

https://aclanthology.org/2025.wslp-main.6/

Use this dataset for the dedicated fingerspelling expert.

The resource includes:

- 1,308 manually annotated video segments.
- Word-level transcriptions.
- Character-level frame annotations.
- Letter timestamps.
- A signer-independent split.
- More than 11,700 automatically discovered supplementary segments.

Use the manual character-level annotations as gold supervision.

Keep automatic discoveries separate as weak supervision.

The automatic segments refer to underlying iSign videos and are not all independently supplied video clips. Resolve their source IDs against the iSign video corpus.

Do not leak the fingerspelling evaluation source videos into the primary translation test set.

Use `signer_split.csv` for a separate signer-independent fingerspelling evaluation.

## Dataset E: ISH-NEWS (2026)

Research paper:

https://www.nature.com/articles/s41598-026-60893-0

Published annotations and preprocessing resources:

https://drive.google.com/drive/folders/1cr2CETamR8qrkJQ-eHDreAsTavBXMOj6?usp=drive_link

This resource reports 4,222 sentence-level ISL videos.

Important:

The authors do not redistribute the original raw video files because of copyright restrictions. Only derived annotations, timestamps, splits and preprocessing resources are publicly available.

Determine whether the original video material can lawfully be accessed for research.

Do not assume that annotations alone are equivalent to access to complete RGB videos.

Check for source overlap with ISL-FS and other ISH News material.

Use it as an additional evaluation or training resource only when the underlying video data is legitimately available.

## Optional auxiliary resources

CISLR:

https://huggingface.co/datasets/Exploration-Lab/CISLR

Use for isolated sign recognition or auxiliary visual pretraining.

It is not a continuous sentence-to-English translation dataset.

How2Sign:

https://how2sign.github.io/

Use only for carefully controlled cross-lingual visual representation pretraining, if useful.

It is American Sign Language, not Indian Sign Language.

Never report its ASL test results as ISL performance.

### Dataset integration rules

Build one canonical dataset registry.

For each example, preserve:

- UID.
- Parent source-video ID.
- Dataset origin.
- Signer ID when reliably available.
- Original video file.
- Pose file.
- Timestamp range.
- Native FPS.
- Original frame count.
- English reference.
- Human-validation status.
- Fingerspelling annotation status.
- Licence and usage restrictions.
- Original train/validation/test membership.
- Exact media hashes.

Detect:

1. Exact duplicates.
2. Same parent video under different UIDs.
3. Overlapping video segments.
4. Near-duplicate videos.
5. Cross-dataset source reuse.
6. Signer overlap.
7. Repeated English translations that could create retrieval leakage.

Produce:

`data/manifests/master_manifest.parquet`

`data/manifests/source_provenance.csv`

`data/manifests/deduplication_report.csv`

`data/manifests/split_assignments.csv`

`reports/DATASET_AUDIT.md`

Never silently remove examples. Every excluded or merged example must have an explicit reason.

---

# PART V — PHASE 3: IMPLEMENT THE FULL ISL-MEGA ARCHITECTURE

The main research candidate is:

ISL-MEGA: Multi-Expert, Evidence-Grounded Architecture for Continuous Indian Sign Language Translation.

Use the provided ISL-MEGA ZIP as the implementation starting point.

Preserve useful, verified code. Correct defects where necessary.

Do not rewrite everything unnecessarily.

## Expert 1: Appearance encoder

Use pretrained DINOv2 visual representations.

Create specialist representations for:

- Left hand.
- Right hand.
- Face.
- Upper body.

Inspect the existing backbone-sharing arrangement.

Compare shared-weight and separate-weight alternatives.

Verify pretrained checkpoint availability, exact architecture, image preprocessing, feature dimensions and checkpoint licensing.

Use high-resolution ROI crops without losing anatomical context.

Derive boxes from verified pose geometry where external boxes are unavailable.

When landmark observations are missing, use a robust fallback such as a tracked person crop or global visual context. Do not treat an invalid ROI as a valid hand observation.

Use native timestamps and preserve synchronization.

Do not resize RGB frames independently of the coordinate system used for pose landmarks.

## Expert 2: Articulatory geometry encoder

Integrate the corrected RA-MAGC graph encoder.

Use:

- Left-hand graph.
- Right-hand graph.
- Upper-body graph.
- Facial landmark graph.
- Joint relationships.
- Global and relative coordinates.
- Velocity and acceleration.
- Confidence and missingness indicators.

Preserve the original full temporal sequence.

This pathway is mandatory even when RGB is available because explicit geometric structure may complement appearance representations.

## Expert 3: Temporal motion expert

Use a genuine spatiotemporal video encoder.

Retain the existing ISL-MEGA 3D temporal convolution pathway as a candidate.

Investigate:

- Temporal convolution.
- Motion representations.
- Frame differences.
- Long-range temporal attention.
- Cross-articulator movement.
- Temporal event representations.

Do not assume that a simple frame-difference CNN is equivalent to a modern video Transformer.

Compare both.

## Expert 4: Fingerspelling expert

Initialize and evaluate this pathway using ISL-FS-Dense.

Use:

- RGB hand appearance.
- Hand pose information where available.
- Character-level temporal supervision.
- CTC-based character decoding.
- Confidence estimates.
- Temporal segment detection.

Use a learned routing mechanism so this expert contributes when fingerspelling is present.

Do not force it to hallucinate character sequences on ordinary signs.

## Stage 2: Reliability-aware multimodal fusion

Combine:

- Left-hand appearance.
- Right-hand appearance.
- Face appearance.
- Body appearance.
- Hand pose.
- Body/face pose.
- Temporal motion.
- Fingerspelling representations.

Use timestamp-aware cross-attention, reliability gating, modality-validity masks and learned interactions.

All streams must retain their original temporal provenance.

Do not assume all streams have identical FPS, frame counts or timestamps.

## Stage 3: Hierarchical temporal reasoning

Use a hierarchical Transformer with local and global attention.

Retain fine-resolution temporal features for retrieval.

Use a global sequence representation for sentence-level grammar and context.

Use relative temporal position and real elapsed-time embeddings.

Process complete sequences through dynamic length batching.

## Stage 4: Visual grounding and latent evidence planning

Retain the existing latent planning mechanism as a candidate.

Make its attention inspectable.

However, do not claim that attention weights prove evidence grounding.

Where reliable annotations exist, explicitly supervise important spatial regions and temporal events.

Otherwise describe these outputs as model attention diagnostics rather than verified explanations.

## Stage 5: English translation decoder

Retain the FLAN-T5 interface.

Experiment with:

- FLAN-T5-large.
- FLAN-T5-XL.

Compare frozen, parameter-efficient and full fine-tuning strategies.

Verify actual checkpoint sizes and tokenizer compatibility.

Use the same decoder configuration in controlled architecture comparisons whenever possible.

Do not assume that a larger decoder automatically produces better visual translation.

---

# PART VI — PHASE 4: TRAINING AND OPTIMIZATION

Use staged training.

### Stage A: Specialist pretraining

Train or fine-tune:

1. Appearance representations using unlabeled signing videos.
2. Pose representations using available ISL pose resources.
3. Fingerspelling recognition using ISL-FS-Dense.
4. Motion representations using complete signing video sequences.

Use self-supervised learning where labels are unavailable.

Do not introduce English targets as fake annotation for visual pretraining.

### Stage B: Cross-modal alignment

Align RGB and pose representations.

Use a source-aware contrastive objective.

Ensure that negatives come from genuinely different source examples.

Evaluate representation alignment under missing or noisy modalities.

### Stage C: Translation training

Use:

- Translation cross-entropy.
- Optional translation CTC as weak auxiliary supervision.
- Semantic entropic optimal transport.
- RGB-pose contrastive alignment.
- Fingerspelling character supervision.
- Spatial/temporal attention supervision where annotations exist.

Investigate each auxiliary loss separately.

Do not enable all losses at full weight without experimentation.

### Stage D: Multimodal refinement

Compare:

- Pose-only.
- RGB-only.
- RGB + pose.
- RGB + pose + motion.
- RGB + pose + motion + fingerspelling.
- Complete ISL-MEGA.

Use modality dropout and controlled missing-stream experiments.

Investigate decoder hallucination using blank, shuffled and mismatched visual inputs.

---

# PART VII — PHASE 5: EXPERIMENTAL VALIDATION

Create the following comparisons:

| Model | Input |
|---|---|
| Original CNN-BiLSTM | Original pose |
| RA-MAGC | Pose |
| RA-MAGC + RGB | Pose + RGB |
| ISL-MEGA appearance model | RGB |
| ISL-MEGA without motion | RGB + pose |
| ISL-MEGA without fingerspelling | RGB + pose + motion |
| Full ISL-MEGA | All valid streams |

Run controlled ablations with identical source-level test membership.

At minimum compare three independent random seeds.

Prefer five seeds for the central final comparisons where feasible.

Use confidence intervals and significance testing.

### Primary translation metrics

- Corpus BLEU-1 through BLEU-4.
- chrF++.
- ROUGE-L.
- TER.
- WER.
- METEOR.

Use consistent tokenization and metric configurations.

### Secondary evaluation

- Performance on the human-validated ISL examples.
- Source-video-held-out performance.
- Signer-held-out performance when signer metadata is reliable.
- Long-sequence performance.
- Fingerspelling character error rate.
- Missing-landmark robustness.
- RGB corruption robustness.
- Pose corruption robustness.
- Cross-source generalization.

Never mix ASL evaluation with ISL evaluation.

### Data leakage diagnostics

Perform:

- Source-video overlap checks.
- Signer overlap checks.
- Text duplication checks.
- Exact media hash checks.
- Perceptual duplicate checks.
- Train/test nearest-neighbour retrieval checks.

The final test set must remain untouched during training, hyperparameter tuning and model selection.

---

# PART VIII — SOFTWARE ENGINEERING REQUIREMENTS

The final implementation should be a complete installable Python research project.

Use a clear structure such as:

```text
ISLTranslate-Research/
│
├── src/
│   └── isl_mega/
│       ├── data/
│       │   ├── isltranslate.py
│       │   ├── isign.py
│       │   ├── glosl.py
│       │   ├── fingerspelling.py
│       │   ├── pose_format.py
│       │   ├── video_loader.py
│       │   ├── mapping.py
│       │   └── manifest.py
│       │
│       ├── models/
│       │   ├── vision.py
│       │   ├── pose.py
│       │   ├── graph.py
│       │   ├── motion.py
│       │   ├── fingerspelling.py
│       │   ├── fusion.py
│       │   ├── temporal.py
│       │   ├── planner.py
│       │   ├── decoder.py
│       │   └── model.py
│       │
│       ├── training/
│       │   ├── pretraining.py
│       │   ├── alignment.py
│       │   ├── translation.py
│       │   ├── losses.py
│       │   └── checkpointing.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── diagnostics.py
│       │   ├── leakage.py
│       │   └── human_eval.py
│       │
│       └── utils/
│
├── configs/
│   ├── baseline.yaml
│   ├── ramagc.yaml
│   ├── multimodal.yaml
│   ├── pretraining.yaml
│   └── ablations/
│
├── data/
│   ├── manifests/
│   ├── splits/
│   └── dataset_cards/
│
├── experiments/
│   ├── baseline/
│   ├── ramagc/
│   └── isl_mega/
│
├── checkpoints/
├── logs/
├── results/
├── reports/
├── tests/
├── scripts/
├── pyproject.toml
├── README.md
└── AUDIT_REPORT.md
```

Use lazy video decoding and efficient feature caching rather than requiring every original RGB frame to be stored permanently as uncompressed NumPy data.

Implement:

- Training resume.
- Checkpoint validation.
- Mixed precision where appropriate.
- Gradient accumulation where useful.
- Multiple GPUs where available.
- Experiment tracking.
- Configurable seed management.
- Automated dataset integrity checks.
- Automated split leakage checks.
- Reproducible decoding.
- Complete prediction export.

Do not redistribute raw videos or restricted pretrained weights inside the research ZIP.

Provide download scripts, access instructions, metadata and licence information instead.

---

# PART IX — FINAL QUALITY CONTROL

Before declaring completion:

1. Audit every Python file.
2. Check every configuration file against the actual model interfaces.
3. Validate every tensor dimension.
4. Run every unit test.
5. Run integration tests.
6. Test the actual `.pose` reader on real samples.
7. Test real RGB video decoding.
8. Test RGB-pose temporal synchronization.
9. Test missing-modality behaviour.
10. Test model checkpoint save and restore.
11. Test actual external pretrained checkpoint loading.
12. Test one complete real-data training batch.
13. Run a genuine short end-to-end experiment.
14. Verify the full training and evaluation commands.
15. Reopen and validate the final ZIP.
16. Run the complete test suite against the extracted ZIP.

A passing synthetic test suite is not proof of real-data correctness.

Never claim full implementation verification when external weights or real data were not available.

### Required final deliverables

Produce:

1. `RA_MAGC_COMPLETE_IMPLEMENTATION.zip`
2. `ISL_MEGA_FINAL_IMPLEMENTATION.zip`
3. `INITIAL_CODEBASE_AUDIT.md`
4. `DATASET_AUDIT.md`
5. `RA_MAGC_BENCHMARK_REPORT.md`
6. `ISL_MEGA_ARCHITECTURE_REPORT.md`
7. `ABLATION_RESULTS.md`
8. `EXPERIMENTAL_RESULTS.md`
9. `RESEARCH_NOVELTY_ASSESSMENT.md`
10. `FINAL_AUDIT_REPORT.md`
11. Complete executable Python source.
12. All YAML configurations.
13. Complete test suite.
14. Installation and execution instructions.
15. Reproducibility documentation.
16. Dataset citations and licensing documentation.
17. A complete experiment manifest with seeds, versions, configurations and checkpoint hashes.

Create individual corrected replacement files whenever any file is defective. Do not provide partial snippets as a substitute for complete corrected modules.

### FINAL EXECUTION INSTRUCTION

Start with Phase 0 immediately.

Then reproduce the original CNN-BiLSTM baseline.

Then implement and train RA-MAGC on the exact same original ISLTranslate source examples and splits.

Do not skip or merge these two experiments.

After producing the RA-MAGC comparison, construct the unified, provenance-aware dataset registry and proceed to the complete multimodal ISL-MEGA implementation.

Do not stop at writing an architecture proposal or generating synthetic mocks. Execute as much of the real implementation and validation as the available environment permits.

When credentials, access approval or unavailable raw videos genuinely prevent a task, identify the exact blocker and continue with every independent task that remains possible.

Keep a continuous record of completed work and unresolved issues.

The final objective is a reproducible, rigorously evaluated, original ISL-to-English translation research system whose scientific contributions can be defended before peer reviewers.

**Do not optimize for impressive parameter counts. Optimize for genuine translation accuracy, visual grounding, unseen-signer generalization, reproducibility and scientific validity.**