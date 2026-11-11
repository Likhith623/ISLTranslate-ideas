# Continuous ISL → English

The baseline is pose → temporal CNN → two-layer BiLSTM → FLAN-T5-small, trained with translation cross-entropy. Optional experiments use explicit configurations.

The offline CPU suite verifies synthetic pipeline behavior. Actual ROCm compatibility, real-pose quality and the FP32/BF16 comparison require execution on the target GPU; see [implementation status](implementation_status.md) for measured validation.

| Guide | Contents |
| --- | --- |
| [Runbook](runbook.md) | Required inputs, uv/AMD/managed setup, notebook launch, preparation, training, resume and evaluation commands |
| [Data contracts](data_contracts.md) | Local storage layout, provenance, immutable manifests/splits, cache/model assets and output files |
| [Architecture](architecture.md) | Model, losses, masks, checkpoint formats and reproducibility decisions |
| [Experiments](experiments.md) | Explicit optional configurations and comparison protocols |
| [Implementation status](implementation_status.md) | Current verification and external inputs |

```text
configs/                     baseline and explicit experiment settings
docs/                        operating guides and the only README
notebooks/run_pipeline.ipynb  one CLI runner with preview defaults
scripts/                     acquisition, pose inspection and run summaries
src/isl_slt/                  data, features, models, training, evaluation and inference
tests/                       offline synthetic CPU checks
```

Local inputs and generated outputs use ignored `data/`, `splits/`, `artifacts/` and `runs/` paths. Their [layout](data_contracts.md#local-storage-layout) is documented centrally; preparation creates output directories when needed. No empty placeholder folders are tracked.

Start with the [runbook](runbook.md). Use the [runner notebook](../notebooks/run_pipeline.ipynb) for an interactive interface to the same CLI; all stages require explicit execution opt-in.
