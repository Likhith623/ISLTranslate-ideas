# Architecture and engineering decisions

The default experiment translates complete pose sequences into English using translation cross-entropy. It uses a temporal CNN, a two-layer bidirectional LSTM and the full generic FLAN-T5-small encoder-decoder. Optional components require explicit configuration. Synthetic CPU tests cover the implemented contracts; real-pose quality and GPU validation remain separate acceptance checks.

## Data path

1. Explicit CSV adapters produce canonical records with dataset revisions, annotation provenance and parent-source identity.
2. Auditing checks input availability and pose structure, records local asset checksums and exports immutable manifests.
3. Splitting reserves interpreter-related parent sources, creates source-disjoint partitions and records membership hashes and leakage reports.
4. Header-driven pose reading preserves coordinate masks and confidence policy. Feature preparation creates revision-aware caches.
5. Length buckets form complete-sequence batches under example and frame budgets. Collation keeps sequence lengths separate from missing-landmark masks.
6. Training or free-generation evaluation loads the same feature contracts. References enter training losses and scoring, while prediction accepts visual inputs only.

See [data contracts](data_contracts.md) for the input and artifact schemas.

## Pose representation

The baseline selects 48 landmarks: left/right shoulders, elbows and wrists, plus 21 points in each hand. It reads component and landmark names from actual `pose-format` headers rather than assuming an array order. Multiple stored people require an explicit selection.

XY coordinates are centered using valid shoulder anchors and scaled using median shoulder width within a continuous segment. Short gaps may be interpolated within the configured time limit. Motion uses actual elapsed seconds, with validity masks that prevent differences across discontinuities. Missing values, imputation, position validity, velocity validity and sequence padding have distinct meanings.

Feature blocks include normalized positions, velocity and validity information. The resulting input dimension comes from the prepared schema; model construction does not assume an unverified fixed width. Invalid numerical values are masked before entering the model.

Optional wrist-local hand coordinates retain body-relative wrists. Anatomical bone vectors and a sparse face selection add separate channels. Explicit per-component depth scales can expose local depth channels; these do not establish a calibrated shared 3D skeleton. Full-duration temporal resampling is an explicit configuration change.

## Baseline model

| Stage | Default behavior |
| --- | --- |
| Temporal CNN | Two kernel-5 convolutions, widths 256 then 512, GELU, per-frame LayerNorm and dropout |
| Temporal reduction | Strides 1 then 2; exact ceiling division for odd sequence lengths |
| Recurrent encoder | Two bidirectional LSTM layers, hidden size 256 per direction, packed sequences |
| Visual bridge | LayerNorm and a linear map from recurrent states to the loaded T5 `d_model` |
| Text model | Generic `google/flan-t5-small`, including its encoder and decoder |
| Objective | Token cross-entropy; auxiliary weight is zero in the baseline |

The CNN masks padding before convolution and after each block. Packed recurrence excludes padded time steps from the BiLSTM. Bridged visual states enter T5 through `inputs_embeds` with an encoder attention mask. Target tokenization retains EOS; label padding is `-100`. Training uses teacher forcing, and validation/test predictions use autoregressive generation.

The first baseline epoch freezes T5 parameters, followed by up to 19 joint-training epochs. Frozen T5 operations remain in the autograd graph so gradients can reach the bridge and visual encoder. Prediction runs without gradients. The BiLSTM consumes the complete sequence and is noncausal.

## Loss and optimization

Translation loss is summed over valid target tokens and divided by the token count of the entire accumulation window. The final partial window uses its actual count. Optional lexical BCE has its own window-wide label denominator and configured ramp. There is no English-content-word CTC objective.

AdamW groups distinguish new layers, T5 and optional RGB parameters. Frozen parameters stay registered with the optimizer, preserving parameter-group identity across stage changes. Warmup and cosine scheduling operate within explicit stages. Gradient clipping follows scaling/unscaling; nonfinite or skipped updates do not advance the schedule.

Defaults use FP32 with recurrent operations explicitly in FP32. BF16 and FP16 require a later backend probe. Length bucketing and frame budgets bound batch size without silently truncating or discarding long examples. Out-of-memory errors require an explicit recorded configuration change.

## Optional model paths

- Lexical presence uses masked pooling of visual states and a multilabel head. Its labels describe weak English word presence, without assuming sign alignment or a gloss sequence.
- Transfer stages train on the filtered additional iSign pool before target fine-tuning. Held-out target information is excluded from fitting text resources.
- Context concatenates verified neighboring source time windows within the same partition and carries center/context regions and continuity boundaries.
- RGB projects synchronized ResNet18 vectors and crop-validity masks into features fused before the BiLSTM. Cached mode keeps the CNN fixed. Image mode can tune the final residual block in its declared last stage.
- The attention-LSTM comparison fits a training-only BPE vocabulary and uses an attention decoder in place of T5.

Configuration details and prerequisites are in [experiments](experiments.md).

## Reproducibility and checkpointing

Epoch retention defaults to zero; epoch files are written only when explicitly requested for averaging. Same-epoch best snapshots share one serialized inode through atomic hard links, with a logged copy fallback on unsupported filesystems. `training.inference_snapshot_dtype: bf16` is an optional lossy storage conversion of inference weights only; it clears decoding and requires fresh validation selection. Native storage remains default and `last.pt` always keeps full training precision/state. Exclusive run ownership protects checkpoint cleanup and removes stale `.last.pt.*` files before continuation.

Manifests, caches and checkpoints use content fingerprints and atomic writes. Split membership is immutable. Checkpoint compatibility includes architecture, tokenizer, feature schema, cache artifacts, split hashes, lexicon, RGB initializer and numerical policy. Prepared text assets record a resolved model revision and file checksums.

Continuation checkpoints (`last.pt`) contain model, optimizer, scheduler, scaler, RNG states, stage/epoch/batch position, tokenizer bytes, resolved config, initialization provenance, package versions and backend information. Wall-time exits save at accumulation boundaries. Exact resume requires matching data, configuration, numerical policy and backend. Load checkpoint files only from trusted runs; full optimizer/RNG restoration uses local Python serialization.

`last.pt` supports continuation; `best-bleu.pt` and `best-chrf.pt` serve distinct validation objectives. `best-guarded.pt` requires a new historical BLEU-4 maximum and chrF++ no more than `training.guarded_chrf_tolerance` (default 0.5 percentage points) below the historical chrF++ maximum, evaluated before updating either maximum. Checkpoint averaging accepts one compatible trajectory, checks nonfloating buffers and produces inference-only weights. Averaged weights require fresh validation decoding selection.

## Package ownership

Each run writes `parameter_counts.json` and embeds the inventory in checkpoints/evaluation reports. It records unique total, visual, T5 and other parameters plus trainable counts for every configured stage (including RGB final-block tuning). Tied T5 embeddings are counted once and buffers are excluded. Checkpoints additionally record the trainable count at save time; inventory calculation does not change model modes or gradient flags.

| Package | Responsibility |
| --- | --- |
| `data` | Input adapters, canonical records, pose loading, source splits, context, batches and text resources |
| `features` | Pose transforms, feature cache contracts and RGB extraction |
| `models` | Visual encoder, T5 integration, attention-LSTM and model construction |
| `losses` | Window-normalized translation and optional lexical objectives |
| `training` | Stages, optimization, checkpoints and resume |
| `evaluation` | Free generation, metrics, paired diagnostics, summaries and timing |
| `inference` | Checkpoint restoration and reference-free prediction reused by evaluation |
| `utils` | Atomic I/O, hashes, versions, device/precision policy and RNG handling |

`cli.py` dispatches explicit commands, `config.py` validates configuration, and `workflows.py` connects preparation and evaluation. Keeping prediction separate lets evaluation restore the exact trained model while keeping references outside its prediction interface.

See [implementation status](implementation_status.md) for the execution boundary.

## Review-era implementation decisions

Baseline velocity stays in units per actual second. Its 48 joints produce 384 value/mask channels; the explicit no-velocity ablation removes 96 motion values plus 48 validity channels, leaving 240. Optional train-only standardization is applied at dataset/raw-inference boundaries, with a complete continuous-validity map and checkpointed statistics. It does not transform binary masks.

Per-epoch validation uses `generation.train_beams`, default [1]. Final validation selection uses `generation.beams`, default [1,4], and writes a checkpoint/membership/generation-bound selection. The shared loader is used by evaluate, predict, benchmark and diagnostics. Test/gold must use the final selection when policies differ.

Only `last.pt` is resumable. Best/epoch snapshots retain model/T5 configuration, input dimensions, tokenizer, lexicon, feature statistics, RGB initializer, decoding state, compatibility and provenance, while dropping optimizer/scheduler/scaler/RNG. Averaging preserves these reconstruction fields and clears stale decoding.

Resume gates declared runtime dependency versions, Python minor and torch/CUDA/HIP builds. Full installed-package/platform/lock inventories remain provenance, so unrelated notebook packages do not block continuation. `--allow-environment-change` logs a full diff and marks continuation non-exact; it is separate from `--allow-code-change`. Neither overrides data/model/cache/numerical contracts or incompatible RNG topology. Old environment identities require an explicit transition. Strict determinism remains default.

Dataset construction verifies cache arrays once. Per-item fetches check immutable file identities and manifest-bound metadata. Indexed source timelines and command-scoped signatures avoid global context scans and repeated raw hashing. Worker count remains zero until profiled; persistent workers are enabled only with workers > 0 and custom batches support GPU pinning.

RGB hand-slot masking is deliberate: a failed hand crop receives upper-body fallback pixels for tensor construction, but its hand mask stays false. Fusion zeros that hand vector and includes the false mask; the valid upper-body slot supplies the visual fallback. Missing decoded frames have false masks. BatchNorm running statistics remain fixed even during final-block tuning.

Pose noise applies only to position-valid normalized/standardized feature channels, including permitted imputation. It does not alter raw coordinates, binary masks, RGB vectors, or recompute derived velocity/bones. All optional changes remain explicit experiments.
