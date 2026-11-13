# PCAT v0.2 executed verification

Environment: Python 3.12.14, PyTorch 2.15.0, macOS ARM64, **CPU**, 13 November 2026. Package versions are recorded in `requirements-verified.txt`. Current machine-readable summary: `verification.json`. Historical v0.1 evidence is retained under `reports/v1/` and must not be read as a current-version benchmark.

## Automated checks

`python -m pytest -q`: **86 passed**. `ruff check .`: passed. Source/script compilation also passed.

The suite includes the original geometry, area, Cayley, shape, gradient, masking, causality, data and checkpoint tests, plus:

- Prefix integration versus independent explicit clipping on variable-length paths with holds, across progress/temporal cells and multiple cell counts.
- Gradient agreement between the two integration algorithms, rather than only comparing their outputs.
- Multiscale progress-augmented refinement invariance and augmented-prefix/clipping agreement.
- Explicit retracing and cross-cell-order counterexamples addressed by the new representation.
- Bounded metric condition under extreme raw parameters.
- Missing-initial-anchor recovery, strict-first-reference policy and exclusion of padded reference frames.
- Exact recovery of the complete canonical coordinate/confidence vector from detail memory, source padding masks and finite enhanced backward passes.
- Legal Unicode, prohibited byte sequences, special UTF-8 leading-byte ranges and randomized byte-budget completion tests.
- Source-recording overlap rejection and invalid enhanced config rejection.
- Float64 integration in a float32 learned network, including backward propagation.
- **Exact CPU epoch resume for both coarse and enhanced regimes**, including dropout and a partial accumulation group. The enhanced resume test integrates in float64.
- Forward and backward passes with finite gradients for the actual recommended `base.yaml` and `invariant.yaml` configurations.

Tests support the stated mathematical/implementation properties, not a claim of real-world translation quality.

## Numerical mechanism audit

`python scripts/mechanism_audit.py --output reports/mechanism_audit.json`, seed 17, float64:

| Quantity | Observed |
|---|---:|
| Max invariant-token subdivision residual | 1.7764e-15 |
| Max transport orthogonality residual | 4.4409e-16 |
| Max solve condition in this sample | 1.03122 |
| Max Cayley reversal residual | 1.1102e-16 |
| Deterministic area-sign classification on 200 artificial order paths | 100% |

The classifier is an analytic mechanism check, not a trained sign-language translator.

## Integrator benchmark and discovered precision tradeoff

`python scripts/benchmark_operator.py`, seed 17, one CPU thread, batch 2, 512 frames, 16 dimensions, scales `[1,4,16,32]`, median over 10 repetitions:

| Backend | FP32 median | FP64 median |
|---|---:|---:|
| Shared-prefix integration | 1.577 ms | 1.796 ms |
| Explicit clipping | 5.556 ms | 5.790 ms |

This case showed approximately **3.52× FP32 / 3.22× FP64** integration speedup. It excludes backward, Cayley solves, the decoder, data I/O and hardware transfers; it is not an end-to-end training or inference speedup claim.

The maximum area disagreement between algorithms was **1.1024e-3 in FP32** and **2.3768e-12 in FP64** on the same paths. Prefix subtraction can lose precision. The recommended base, invariant and enhanced-smoke configurations therefore use float64 integration; the clipping backend remains selectable. Raw measurements are in `operator_benchmark.json`. Neither arithmetic choice is claimed universally stable or optimal on every device/input range.

## Complete enhanced training and generation run

Executed `configs/enhanced_smoke.yaml` with **float64 integration**, the shared-clock pyramid `[1,2,4]`, progress augmentation, frame-level detail memory, physical timing, a scratch byte decoder and seed 17:

- Generated artificial samples: 24 train / 6 validation / 6 test.
- Optimization: 12 epochs / 36 updates.
- Validation NLL: **2.17908 → 0.233095**.
- Autoregressive held-out code exact match: **100% (6/6)**.
- EOS fraction: **1.0**; invalid-UTF-8 fraction: **0.0**.
- Single-file prediction on `test_00000.npz`: `hf`, matching the synthetic reference.

Actual history, predictions and corpus metrics are included as `enhanced_history.jsonl`, `enhanced_predictions.jsonl` and `enhanced_metrics.json`. Full sample validation checked the generated arrays, geometry, exact decoded duplicates and split/signer separation. Synthetic examples have no real source recordings; known source-video count is zero. Source-group rejection is tested separately using explicit overlapping metadata.

These codes are not signs or natural-language sentences. They contain too few word n-grams for BLEU-2..4 with effective order disabled, so those metrics remain zero even for exact output. The enhanced model differs in width and memory capacity from v0.1/coarse smoke models; its synthetic result is not evidence that PCAT transport improved SLT or improved sample efficiency in a controlled comparison.

## Model sizes and release boundaries

| Configuration | Parameters | Coarse source cells | Detail memory |
|---|---:|---:|---|
| Base enhanced | 9,596,417 | 53 | All admitted original frames |
| Invariant | 6,235,265 | 53 | Disabled |
| Enhanced smoke | 1,511,969 | 7 | All admitted original frames |
| Minimal coarse smoke | 41,057 | 4 | Disabled |

The ten-variant ablation generator was exercised in configuration-writing mode. Real-corpus ablations have not been run. CUDA/AMP execution, distributed/streaming deployment and large-scale memory/throughput remain unverified.

The deliverable contains all source, tests, configuration, mathematical derivations, literature/limitation/experiment documentation and actual verification artifacts. Generated arrays, run weights, environments, caches and recursive release archives are excluded from the source ZIP. Hashes and a clean extraction/build check verify release integrity; see `release_check.json` for the recorded release check.

**No real SLT benchmark, real-world robustness gain, universal novelty priority or publication outcome is established.** The limitation audit separates resolved defects from mitigations and required observations/experiments in `docs/LIMITATIONS.md`.
