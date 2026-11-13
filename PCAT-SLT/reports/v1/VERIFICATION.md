# Executed verification

Date: 13 November 2026. Environment: Python 3.12.14, PyTorch 2.15.0, macOS ARM64, **CPU**. Exact package versions are recorded in `requirements-verified.txt`; machine-readable summary is `verification.json`. CUDA/AMP runtime execution and real-corpus SLT performance were not tested.

## Automated checks

`python -m pytest -q`: **43 passed**. Tests cover:

- Closed-form signed area, skew symmetry, ordered concatenation and reversal.
- Exact progress-boundary clipping under polygonal subdivision at multiple cell counts.
- Duplicate-vertex and timestamp-warp invariance of the invariant branch.
- Full encoder refinement invariance, proper 3D similarity invariance and reflection sensitivity.
- Orthogonal transport, value-norm preservation, solve sensitivity and stationary paths.
- Finite-difference autograd checks and nonzero gradients to clock/gain parameters.
- Timing-channel behavior, elapsed-time conservation and mixed timestamp precision.
- Shapes and finite backward passes for all five operator modes.
- Causal target masking, padded-input isolation, batch independence and EOS generation.
- UTF-8 tokenization, missing-joint imputation, bad-data/config rejection and leakage validation.
- Token-count loss aggregation, exact metric cases, RNG roundtrip and source-file fingerprints.
- **Exact epoch-boundary resume:** two uninterrupted CPU epochs match a one-epoch checkpoint followed by resume, for every model tensor and validation NLL. The test uses dropout and a partial gradient-accumulation group.

`ruff check .`: passed. `python -m compileall -q src scripts`: passed.

## Independent numerical mechanism audit

Executed `python scripts/mechanism_audit.py --output reports/mechanism_audit.json` with seed 17 in float64:

| Measure | Observed |
|---|---:|
| Max token difference after exact polygonal subdivision | 2.6090e-15 |
| Max orthogonality residual | 4.4409e-16 |
| Max condition number of the tested solve matrices | 1.03122 |
| Max reversal/Cayley-transpose residual | 1.1102e-16 |
| Analytic area-sign classifier on 200 artificial order examples | 100% |
| First-order feature distance on a paired order counterexample | 0 |

The area-sign classifier is a deterministic test of the mathematical construction, not a trained translator or evidence of linguistic accuracy. Exact values are in the JSON file.

## Complete training and generation runs

Generated 24 training, 6 validation and 6 test **nonlinguistic geometric examples**, with proper rotations, scales, translations, variable frame counts and durations. Validation checked every sample, exact decoded-array duplicates, ID/path overlap and synthetic signer separation. The validation report is included as `data_validation.json`.

| Run | Epochs / optimizer updates | First → last validation NLL | Held-out code exact match |
|---|---|---|---:|
| `configs/smoke.yaml` | 8 / 24 | 4.01230 → 2.13330 | 50% (3/6) |
| `configs/synthetic_order.yaml` | 80 / 240 | 4.00649 → 0.00732333 | 100% (6/6) |

These configurations have different cosine schedules because their total optimization horizons differ. They are not a controlled comparison of architectural variants. Both use seed 17, a 41,057-parameter model, a scratch byte decoder and the same generated data. The base configuration has 6,111,297 parameters. Model counts include all affine maps, transport/readout, encoder and decoder parameters.

Actual training histories, generated predictions and corpus metrics are copied into this directory. The artificial references are one-token codes, so BLEU-2..4 with effective order disabled are zero even for exact generations. Their reported BLEU-4 is not a natural-language benchmark result. EOS was generated for all six examples in both runs. Single-file prediction on `test_00000.npz` returned `hf` after the longer run, matching its synthetic reference.

The seven-variant ablation generator was executed in configuration-writing mode. The paired-bootstrap script was exercised on identical predictions for 20 replicates and returned zero differences/intervals, as expected. This is an implementation check; no statistically meaningful architecture comparison was performed.

## Release contents and reproducibility

The deliverable includes the implementation, source tests, all YAML configurations, the whitepaper, primary-source literature audit, data/experiment documentation and these actual result files. It excludes generated NPZ data, trained weights, temporary environments and caches. The data generator and configurations reproduce the sanity workflow without licensed datasets or pretrained downloads. A source ZIP includes per-file SHA-256 hashes and has a companion whole-archive checksum.

The source archive was extracted into a fresh temporary directory. Every per-file hash and ZIP integrity check passed; importing resolved to the extracted source, all **43 tests passed again**, and building an installable wheel from that extracted project succeeded. Model and training source remain unchanged since that test; release packaging now writes archives and checksums into the project's own `releases/` directory.

Tests and algebraic checks support correctness within the stated assumptions. There is **no trained real-corpus SLT model, measured real-world robustness, demonstrated SOTA improvement or established publication-level novelty** in this deliverable. Those require the real-data and prior-art comparisons specified in `docs/EXPERIMENTS.md`.
