# PCAT v0.3 executed verification

## Dataset catalog and Kaggle notebook addendum

The expanded `docs/DATASET_CATALOG.md`, `docs/KAGGLE_RUNBOOK.md` and `notebooks/Kaggle_PCAT_SLT.ipynb` are linked from the main training guide. `kaggle_notebook_checks.json` records notebook syntax/local links, source ZIP bootstrap/hash verification, a CPU diagnostic train/dev-generation run, a second fresh Python process with workspace recovery to epoch two, run-bundle hashes, and prepared-manifest pose/visual setup checks using synthetic observations. Dependencies were reused locally; notebook installation, Kaggle service/CUDA execution and real pretrained/corpus runs were not exercised. CPU diagnostic timings must not be extrapolated to GPU/corpus performance.

**14 November 2026**. Python 3.12.14, PyTorch 2.15.0, Transformers 4.57.6, macOS ARM64, CPU. Optional multimodal dependencies were installed; exact versions are in `requirements-verified.txt`. Historical v0.1/v0.2 evidence is in separate subdirectories.

## Test and implementation checks

`python -m pytest -q`: **111 passed, 1 skipped**. The skipped test requires an available MPS device; this execution context does not expose one. CUDA execution was not tested. Ruff and source/script compilation passed. New console entry points were installed and exercised.

The suite covers the previous geometry, integration, invariance, shape, gradient, masking, causality, byte validity and exact resume tests, plus:

- Solve-free rotation orthogonality, reversal, one-dimensional behavior and finite-difference gradients.
- Optimized exact Cayley vector action against dense-matrix values **and gradients**.
- Sketched areas against the projected dense-area identity using the same full-state clock.
- Per-example adaptive cell counts/masks, refinement and batch independence, including local clock streams.
- Time-aware local body tracking, preserved global detail and conditional-prefix causality.
- Depth-ignore behavior under arbitrary changes to input depth.
- Actual RGB perturbation sensitivity, masked/padded-pixel isolation and nonzero visual gradients.
- Correct temporal binding of visual observations when geometric progress is zero.
- Common-origin RGB alignment, constant offsets and within-clip timestamp corruption.
- Named-frame crop preprocessing with actual images, aspect-preserving output and preserved timestamps.
- Explicit unanchored RGB fallback, rejection of no-observation samples, and masking absent motion.
- Large epoch-valued timestamps rebased before float32 conversion; explicit time requirements.
- Local T5/DINO weight loading, training gradients, generation and checkpoint reload **after original assets are moved away** with Hub offline mode set.
- Pretrained-decoder interrupted/offline resume matching uninterrupted CPU training exactly for every model tensor and validation NLL.
- Annotation/target/split contracts for three translation corpora, rejection of CISLR-as-SLT and object-bearing pickle metadata.
- Validity of all **18** generated multimodal ablation configurations without model downloads.

Tests use small local random-weight T5/DINO fixtures to exercise real library loading/forward/generation/checkpoint paths. They do not imply that large pretrained weights were fine-tuned or evaluated on an SLT corpus.

## Failed and final pixel diagnostics

The generated diagnostic has 24 train / 6 validation / 6 test examples, static pose, and artificial RGB bar-pattern labels `a`/`b`. It is not a sign-language dataset or a mouthing task.

| Development run | Updates | Validation NLL (first → last) | Held-out exact match |
|---|---:|---|---:|
| Initial global-pool/unnormalized visual setup, 16 epochs | 48 | 1.15370 → 0.296451 | 50% (3/6) |
| Final spatial-grid/normalized/time-bound setup, 40 epochs | 120 | 1.01667 → 0.000337682 | 100% (6/6) |

The initial failure is preserved as `visual_initial_failed_*`; final history, predictions and metrics are `visual_final_*`. Architecture, normalization and horizon changed, so the table is **not** a controlled ablation establishing the effect of one component.

A within-checkpoint observation intervention on the final model gave **100% exact match with observed RGB and 50% with all RGB explicitly masked**, while keeping the anchored pose input. Raw results and reproducing script are included. This supports functional use of pixels on this toy task, not linguistic grounding or natural-language translation quality. EOS fraction was 1.0 and invalid-byte fraction 0.0. BLEU-2..4 are zero for the one-token references with effective order disabled; these are not meaningful SLT scores.

All generated sample arrays, geometry, timestamps, pixel masks, exact duplicates and supplied signer/source separation were validated. The generator's synthetic identifiers are not real people or recordings.

## Exact versus solve-free CPU actions

`benchmark_transports.py`: batch 2, 32 cells, one CPU thread, medians of 10 repetitions. Cayley uses the **optimized single-right-hand-side exact action**, not a full transport-matrix construction.

| Dimension | Exact Cayley action | Palindromic rotations |
|---|---:|---:|
| 16 | 0.0775 ms | 0.1121 ms |
| 64 | 0.7255 ms | 0.2367 ms |
| 128 | 3.0133 ms | 0.6187 ms |

Cayley was faster in the small case; rotations were faster in the larger cases. The maps are mathematically different. Timing excludes the encoder, area construction, backward, decoder and I/O. These CPU measurements do not establish a GPU/MPS training speedup or translation advantage. FP32 norm residuals were below 2e-6 in these samples.

The supplied review's third-order Neumann shortcut has a **0.12109375** orthogonality defect on the documented 2D counterexample despite generator Frobenius norm below one. The script records this directly. Orthogonality of an exact infinite series does not transfer to that finite truncation.

## Portability and model scope

Format-3 checkpoints include active pretrained weights, architecture snapshots and tokenizer assets. Offline fixture reload and exact pretrained-decoder resume were exercised. The existing format-2 enhanced byte checkpoint was also loaded and retained its 6/6 synthetic output in a compatibility check. This is checkpoint-interface compatibility evidence, not a blanket promise of bit-identical output across library/hardware versions.

Self-contained multimodal smoke configuration: **1,592,707 parameters**, global maximum 18 tokens plus local/detail/motion/visual memories. Base/invariant pose references remain 9,596,417 / 6,235,265 parameters. Full DINOv2-small/FLAN-T5-base initialization and target-device memory/throughput were not profiled here; the real example configuration is not a trained translation checkpoint.

The archive is checked by fresh extraction, per-file hashes, tests/lint and an installable-wheel build. `release_check.json` records that final packaging check. Generated data, run weights, caches, environments and recursive release archives are excluded.

## Remaining scientific requirements

No licensed real-corpus training/evaluation, measured mouthing/gaze accuracy, real-view robustness gain, verified CUDA/MPS deployment, SOTA improvement, universal novelty priority or publication outcome is established. See the complete audit decisions and limitation ledger. Provider-controlled data was not acquired by sending agreements/messages. Importer tests verify declared schemas, not every restricted release variant.
