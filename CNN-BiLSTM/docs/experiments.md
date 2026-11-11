# Experiment configurations and comparison protocol

The CE-only baseline is the reference experiment. Every optional method has an explicit configuration; none is a proven improvement. Keep input revisions, immutable memberships and reporting conventions fixed when comparing methods. No experimental results have been produced.

## Configuration behavior

`configs/baseline.yaml` contains the complete shared settings. Other configurations inherit it; `rgb_images.yaml` inherits `rgb_cached.yaml`. Inheritance paths are relative to the child YAML, dictionaries merge recursively, and lists replace their parent values. Unknown fields and incompatible feature/stage combinations fail validation.

Use the same experiment's configuration for feature preparation, training and evaluation. Changes to feature/context settings select different caches; changed model/RGB settings must match the checkpoint at evaluation. Numeric variants and combinations can use explicit overrides with unique experiment names/run directories instead of a preset for every combination.

## Retained experiments

| Config | Change from baseline | Additional inputs or constraints |
| --- | --- | --- |
| `baseline.yaml` | Pose CNN → two-layer BiLSTM → full FLAN-T5-small; translation CE only | Audited target poses, source maps, interpreter metadata and generic local T5 assets |
| `local_hand.yaml` | Wrist-local hand coordinates while retaining body-relative wrists | Same underlying audited poses |
| `bone.yaml` | Anatomical bone vectors and validity masks | Required landmark names in pose headers |
| `face.yaml` | Sparse brow/eye/mouth landmarks | Actual stored face component/points |
| `lexical.yaml` | Weak word-presence BCE, weight 0.1, three-epoch ramp, gradient diagnostics | Train-only lexical vocabulary; regex mode needs no external NLP model |
| `transfer.yaml` | Additional filtered iSign stages before target fine-tuning | Extra annotations/poses, exact revision and verified cross-dataset parent identities |
| `context.yaml` | 0.5 seconds before and after the center clip | Verified timing, source continuity and coverage in `data/metadata/target_timing.csv` |
| `rgb_cached.yaml` | Frozen synchronized ResNet18 vectors, fused before recurrence | Videos, explicit geometry in `data/metadata/target_rgb.csv`, initializer and prepared RGB cache |
| `rgb_images.yaml` | Actual crops with last-block tuning in the final stage | Same verified videos/geometry and image access throughout training |
| `attention_lstm.yaml` | Attention-LSTM decoder with training-only BPE instead of T5 | Training-only vocabulary fitting; no pretrained T5 initializer |
| `diagnostics.yaml` | Paired input diagnostics with BLEU/chrF++/TER/WER | Baseline checkpoint; these metrics avoid optional language-resource requirements |

## AMD precision experiments

`amd_fp32.yaml` uses eight workers, strict determinism and FP32 recurrence with a
required GPU. `amd_bf16.yaml` changes autocast precision only; recurrence stays FP32
and training/resume requires a successful environment preflight. The matched
`amd_fp32_pilot.yaml` and `amd_bf16_pilot.yaml` run two epochs each with fixed greedy
validation. Follow the [pilot comparison](runbook.md#fp32-versus-bf16-validation-pilot)
before using BF16 for a full experiment. Their distinct run directories prevent
cross-precision resume. The shared baseline stays unchanged.

## Baseline and anatomy variants

The baseline trains one T5-frozen adapter epoch and up to 19 joint epochs. Keep this schedule unchanged for direct anatomy comparisons. Example commands for the wrist-local variant, after baseline data/split/model preparation:

```bash
uv run --no-sync python -m isl_slt prepare-features --config configs/local_hand.yaml
uv run --no-sync python -m isl_slt train --config configs/local_hand.yaml
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/local_hand.yaml --checkpoint runs/local-hand/checkpoints/best-bleu.pt --partition validation --output runs/local-hand/evaluation/validation
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/local_hand.yaml --checkpoint runs/local-hand/checkpoints/best-bleu.pt --partition test --decoding runs/local-hand/evaluation/validation/decoding.json --output runs/local-hand/evaluation/test
```

The equivalent config and run directories for bones and face are declared in their YAML files. For a combined hand/bone experiment, create an override with both flags, a new name and a distinct run directory. Reuse the same frozen memberships; fit no text resource on held-out rows.

## Weak lexical presence

Lexical labels describe an unordered set of English words in the target sentence. They are weak auxiliary supervision, not glosses or frame-aligned signs. The head pools masked recurrent visual states and uses BCE with its own valid-label denominator. This experiment adds no CTC loss.

Vocabulary fitting uses training records only, with configured frequency and size limits. Regex extraction is the provided default. `method: precomputed` needs declared JSONL annotations with `uid,words`; `method: spacy` needs a prepared local pipeline and the `lexical` dependency extra. Record the method/resource revision and inspect gradient diagnostics when enabled.

```bash
uv run --no-sync python -m isl_slt train --config configs/lexical.yaml
```

It reuses baseline pose caches because its visual preprocessing is unchanged. Run evaluation with `configs/lexical.yaml` and `runs/weak-lexical/checkpoints/best-bleu.pt`.

## Filtered additional-iSign transfer

Fill in `data.pretrain.dataset_revision`, exact export columns, paths and authoritative mappings. Target and pretrain IDs must refer to the same parent-identity namespace. Auditing/splitting produce a contamination report; do not assume any fixed eligible example count.

```bash
uv run --no-sync python -m isl_slt audit --config configs/transfer.yaml
uv run --no-sync python -m isl_slt split --config configs/transfer.yaml
uv run --no-sync python -m isl_slt check-leakage --config configs/transfer.yaml
uv run --no-sync python -m isl_slt prepare-features --config configs/transfer.yaml
uv run --no-sync python -m isl_slt train --config configs/transfer.yaml
```

The configured chronology is generic T5 initialization → one frozen-adapter pretrain epoch → four joint pretrain epochs → up to 20 target epochs. This continues the model within one staged run rather than assuming a task-specific external checkpoint is uncontaminated.

The transfer split directory is separate, but its target memberships must match the baseline comparison. Inspect hashes and the filter report. Every target parent (including training), target UID, unresolved identity, known content duplicate and normalized-text duplicate is excluded from the extra pool. Acquisition selects eligible whole source groups with an explicit compressed-byte budget and seed. Missing duplicate-detection coverage remains a documented limitation.

## Verified context

The provided config requests half a second on both sides. Set actual timing metadata before freezing the relevant manifests. Run audit, split/leakage checks and feature preparation with the context configuration, then train using the same config.

Context is assembled within each partition using verified physical time and coverage, not UID adjacency. Preserve segment boundaries and center/context masks. Change `previous_seconds` or `next_seconds` for a named duration experiment. The following-context window adds explicit lookahead, and the BiLSTM remains noncausal even with zero extra context.

## RGB cached and image modes

Install the selected backend with `--extra rgb` during later setup. Include actual video paths and verified pose-to-video transforms in the frozen manifests. Prepare the generic T5 and declared ResNet18 initializer explicitly:

```bash
uv run --no-sync --env-file .env python -m isl_slt prepare-model --config configs/rgb_cached.yaml --allow-network
uv run --no-sync python -m isl_slt prepare-features --config configs/rgb_cached.yaml
uv run --no-sync python -m isl_slt prepare-rgb --config configs/rgb_cached.yaml
uv run --no-sync python -m isl_slt train --config configs/rgb_cached.yaml
uv run --no-sync python -m isl_slt train --config configs/rgb_images.yaml
```

These commands follow the required RGB-aware data audit/split checks. `prepare-rgb` only supports cached mode. The default RGB view is upper body; `hand_crops: true` adds two hand views with validity masks and an upper-body fallback.

Cached features cannot tune a CNN or support pixel augmentation. Image mode loads actual synchronized crops and configures pixel augmentation. It has one adapter epoch, 17 joint epochs, then two final-block tuning epochs; only the declared final stage unfreezes the last residual block. Its smaller batch/frame limits reflect the extra image path, without claiming measured memory use or runtime.

## Attention-LSTM comparison

`attention_lstm.yaml` uses the same visual encoder family and a training-only BPE tokenizer with configured target vocabulary size 4000. Its attention decoder trains for up to 20 target epochs. Use the same source memberships and pose features. The T5 preparation step is unnecessary for this decoder.

```bash
uv run --no-sync python -m isl_slt train --config configs/attention_lstm.yaml
```

Report that this comparison changes both decoder architecture and text initialization. Tokenizer fitting uses eligible training records only.

## Evaluation and reporting

Select checkpoints and decoding on validation free generation. Test and gold references never choose hyperparameters, beam count, checkpoint or averaged parents. Keep the primary source-held-out test separate from gold transcript and interpreter-reference results.

Report cumulative corpus BLEU-1/2/3/4 with 13a/signatures, chrF++, TER, ROUGE-1/2/L, METEOR and corpus WER where their resources exist. Normalization, metric versions and scales belong in the saved reports; TER/WER can exceed 100. Missing metrics remain explicitly unavailable. Optional BERTScore requires its dependency plus an explicit local model directory and revision.

Use seeds 13, 42 and 2026 with fixed memberships, report individual scores and variation, and compare saved aligned predictions using paired source-group bootstrap where appropriate. The [runbook](runbook.md) gives exact seed, averaging and comparison commands.

Input diagnostics compare real, blank, incorrect and shuffled visual inputs using fixed pairing/seeds. For a nonbaseline checkpoint, inherit that experiment's config in a dedicated diagnostic override. Report donor/source flags and input dependence without claiming that diagnostics prove semantic faithfulness.

No target BLEU value is an acceptance threshold or guaranteed result. Outcomes, throughput, latency, memory usage and numerical stability remain unmeasured.

## Explicit scaling and motion ablations

`standardized.yaml` enables valid continuous-channel standardization: prepare its features, run `prepare-stats`, then train using the same config. Statistics use immutable target-training records only and travel with the checkpoint. Binary masks and invalid/padded zeros retain their meaning.

The baseline already enables per-second velocity, so the redundant velocity-enabled preset has been removed. `no_velocity.yaml` removes motion values plus their validity channels, reducing 384 channels to 240. The original baseline is not silently replaced by an undocumented v7 design.

`overfit.yaml` is the separate unregularized diagnostic preset; it is never a held-out-quality result. Worker count/mixed precision/standardization need actual measurements before any efficiency or quality claim.

The `evaluation.repeat_seeds` and `bootstrap_samples` fields are declarative experiment notes. They do not run seeds or override the independent `bootstrap --samples` CLI argument. Use the runbook's explicit seed loop and comparison commands.

Cased metrics remain primary; _lc variants have their own signatures. Paper score equivalence also requires matching memberships, references and scoring policies, which are not established here. Strict seed summaries reject different effective settings or selected beams; use a predeclared common decoding protocol when necessary.
