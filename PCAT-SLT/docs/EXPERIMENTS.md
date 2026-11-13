# Experiments required to evaluate PCAT

The hypothesis is: **shared progress-local signed-area transport improves compositional articulator modeling beyond matched signature features and identity transport, especially under nuisance sampling-rate changes**. A decrease in training loss or success on an artificial order task is not evidence that this hypothesis holds for SLT.

## 1. Data and comparison regime

Use licensed continuous translation corpora with fixed official train/dev/test partitions. PHOENIX-2014T and CSL-Daily can support comparison with earlier SLT work, while an open-domain corpus such as How2Sign or BOBSL tests broader language/context behavior. Their standard releases do not automatically satisfy this repository's 3D landmark contract. Establish a reproducible extraction/mapping process and report whether geometry is sensor-derived, calibrated multiview, or estimated from monocular video. Do not call an uncalibrated mixture of local depth coordinates metric 3D.

For Indian Sign Language, use an appropriate consented, sentence-level ISL translation corpus with documented splits. Results in another sign language cannot establish ISL performance. Isolated sign datasets may support auxiliary mechanism experiments but cannot substitute for continuous translation evaluation.

Keep preprocessing, landmark estimator, confidence treatment, decoder, source token budget, target tokenizer, optimization, external training data and validation budget fixed across variants. Use the same missing-data policy for all variants. Maintain signer/source-recording separation; report overlaps when official splits do not provide signer independence. Freeze the test set until model selection is complete.

## 2. Executable internal controls

```bash
python scripts/run_ablations.py --config configs/invariant.yaml --seeds 17 23 41
```

This writes ten fully resolved configurations per seed, including scale, progress-coordinate and detail-memory controls. Add `--execute` to train them sequentially. It does not download data or claim to reproduce external papers. Train and evaluate each checkpoint on the same manifests. Compare:

1. PCAT versus `no_transport`: the primary test of the proposed action, with the same bounded area statistics.
2. PCAT versus `signature`: a basic learned-embedding level-two signature baseline.
3. PCAT versus `first_order`: whether cross-channel chronological features matter.
4. PCAT versus `temporal`: progress windows versus elapsed-time windows with transport retained.
5. Learned versus Euclidean metric: whether learned common progress helps.
6. Invariant versus timing-enabled: whether duration information is needed.

Also reproduce appropriate external signature and path-development models from the primary references, as well as a strong pose Transformer/graph model with the same decoder. Those external baselines are an outstanding research experiment, not silently claimed as implemented by a similarly named internal control. For comparisons to large RGB/LLM systems, disclose modality and pretraining differences rather than asserting encoder superiority from unmatched scores.

## 3. Stress tests

Distinguish mathematically exact tests from physical generalization tests:

| Test | Expected structural outcome | Empirical question |
|---|---|---|
| Constant proper 3D similarity of input coordinates | Invariant canonical representation, up to numeric error | Does a real detector preserve its reconstructed geometry across cameras? |
| Insert vertices on every existing straight segment | Same invariant PCAT tokens | Does robustness persist under realistic rate changes and interpolation error? |
| Drop vertices at corners | No exact equality claim | How much meaning or translation quality is lost? |
| Common smooth retiming | Continuous-path invariance; finite sampled reconstructions may differ | Which speed variations are truly nuisance in this corpus? |
| Retiming face independently from hands | May change joint area and output | Can the model detect linguistically meaningful coordination changes? |
| Holds/duration minimal pairs | Invariant branch can collapse duration distinctions | Does the timing side channel recover those distinctions? |
| Noise, occlusion and confidence shifts | No exact invariance guarantee | Is the learned clock dominated by detector jitter? |
| Mirroring | No blanket invariance guarantee | Are anatomical handedness and signed-space distinctions preserved? |

Retiming or mirroring can change a valid sign's meaning; do not assume labels remain correct for every augmentation. Have qualified signers assess any purported semantic minimal pairs. The geometric order counterexample in the unit tests is not itself a linguistic minimal pair.

## 4. Measures and reporting

Use autoregressive BLEU-1..4 with the saved SacreBLEU signatures, chrF++, and precisely defined ROUGE-L. The included ROUGE metric is macro whitespace-token LCS F1 with beta=1, case sensitivity and no stemming; it must not be compared to another ROUGE variant without recalculation. UTF-8 byte accuracy is not a substitute for translation evaluation. Report EOS rate and invalid-byte effects. For short artificial codes, BLEU-4 with effective order disabled is inherently zero and uninformative.

Select checkpoints by development NLL as implemented, or preregister a different development-only criterion for all baselines. Report all seeds, mean and standard deviation, paired corpus-bootstrap confidence intervals, parameter counts (both nominal and trainable), wall time and peak training/inference memory. A small six-example synthetic test set has no scientific inferential value for SLT.

```bash
pcat-evaluate --checkpoint runs/pcat_v2/best.pt --manifest data/test.jsonl --output evaluation/pcat
python scripts/paired_bootstrap.py \
  --a evaluation/pcat/predictions.jsonl \
  --b evaluation/no_transport/predictions.jsonl \
  --samples 1000 --output evaluation/paired.json
```

The bootstrap script requires identical IDs/references and resamples examples in matched pairs. Its percentile intervals describe test-corpus uncertainty; they do not replace variation across independent training seeds. If examples share a recording or signer, use an appropriate grouped resampling design in the actual study rather than treating dependent clips as independent.

## 5. Decision criteria and known alternative explanations

The operator contribution is unsupported if PCAT fails to improve on the matched bounded-area/no-transport control, if any improvement disappears under equal validation search, or if gains occur only through extra timing/input information. A gain over endpoint pooling alone is insufficient because signature methods already encode chronology. Reject universal invariance claims if timing-sensitive cases degrade. Reject reconstruction-level camera claims if only synthetic coordinate transforms were evaluated.

Analyze errors by fingerspelling, manual/nonmanual coordination, signer, utterance length, confidence and camera condition. Sweep rank r and progress-cell count K: compression capacity can explain gains otherwise attributed to geometry. Inspect learned metric weights for concentration on unstable detector outputs. Human evaluation by fluent signers is needed to interpret linguistically meaningful changes and adequacy, beyond n-gram scores.

A publication-grade result requires real experiments, defensible novelty relative to R10–R16, reproducible data preparation and substantive analysis. The supplied mathematical guarantees and tested code are a starting point for that study, not its conclusion.

## V0.2-specific controls

Use the invariant regime to isolate operator effects, and the enhanced regime to test recovery of timing/detail. Add single-scale, removed-progress-coordinate and coarse-only controls. Keep model width fixed when testing detail memory and report its increased source-memory and decoder cross-attention cost. A richer model learning the synthetic task sooner is confounded by width and memory changes and is not an area-transport ablation. Measure anchor fallback sensitivity, FP32/FP64 disagreement and alternative clipping on actual pose distributions. Multiscale and progress augmentation have established trajectory-method precedents and do not independently establish novelty.

## V0.3-specific comparisons and unsupported audit claims

The generator now writes additional transport, sketch, adaptive-budget, local-clock, body-frame, pixel and text-backend controls. Use identical written targets, observations, training schedules and selection budgets for primitive comparisons. Changing pixels, pretrained text, width or source-memory count is a resource/modality diagnostic and must not be presented as an isolated operator gain. Report all admitted lengths, chosen adaptive counts/cap rates, visual observation rates, unanchored RGB rates, frame/pose alignment provenance, peak memory and per-device latency.

Compare Cayley with Givens as different orthogonal actions, not as numerically equivalent implementations. Measure hardware latency instead of extrapolating CPU timings. Sweep sketch dimension/head count against full area statistics with the same full-state clock. Retain joint-only and joint-plus-local streams to check whether asynchronous descriptors actually help. Compare fixed and dual body coordinates while retaining the same global detail path. Verify that visual improvements persist under controls for background/source leakage.

The supplied audit's +15–20 BLEU and 3× GPU claims are hypotheses without supporting experiments here. A pretrained language model can increase fluent hallucination; evaluate adequacy and visual grounding, not grammar alone. Validate learned mouth/eye/finger behavior with actual linguistic examples and qualified signers. A pixel bar-pattern diagnostic is not evidence about any sign-language minimal pair.

Before publication, use appropriate licensed corpora and reproduce relevant modern visual/pretrained SLT baselines with disclosed budgets. The existing primary-source novelty audit still applies: none of the new backbones, EMA, area sketches or plane rotations is individually claimed as a fundamental new mathematical invention.
