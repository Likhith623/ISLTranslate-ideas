# Preparation, training and evaluation runbook

Commands assume the repository root as the working directory. Use the managed setup below on an MI300X image with working ROCm PyTorch. Common settings live in `configs/baseline.yaml`; validation status is recorded in [implementation status](implementation_status.md).

## 1. Supply inputs and configure paths

Read the [data contracts and local storage layout](data_contracts.md#local-storage-layout). Supply annotations, matching pose files, verified parent-source mappings, the interpreter table and exact dataset revisions. Verify actual CSV headers instead of assuming the defaults match an export.

Set paths, revisions, column mappings, coordinate conventions and any metadata in the baseline before auditing. Distinct experiments inherit these settings. Local machine overrides can use an ignored `configs/*.local.yaml` file with `extends: baseline.yaml`; preserve the resolved config for every actual run.

When context or RGB will be evaluated, include its required timing/geometry metadata before freezing manifests. A changed immutable input needs new manifest/split paths and a recorded revision.

## 2. Prepare an environment

The repository includes a resolved `uv.lock`. For local CPU verification:

```bash
uv sync --locked --extra cpu --extra dev
uv run --no-sync pytest
```

Use the `cpu` extra for a project-owned CPU environment. Optional extras are `rgb`, `lexical`, `semantic`,
`notebooks`, and `dev`; include the desired extras together in the sync command.
Never use `--all-extras`. Later commands use `--no-sync` to avoid implicit changes.

### AMD MI300X: preserve the preinstalled ROCm environment

Use the Python interpreter that already imports the droplet's working ROCm torch.
The managed path does not use a backend extra or project `uv sync`. Record the
original environment and constrain the existing torch packages before installing:

```bash
export ISL_PYTHON="$(python -c 'import sys; print(sys.executable)')"
mkdir -p artifacts/environment
uv pip freeze --python "$ISL_PYTHON" > artifacts/environment/managed-before.txt
"$ISL_PYTHON" - <<'HOST'
import importlib.metadata as metadata
import json
from pathlib import Path
import torch
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("Select the interpreter with working ROCm torch first")
packages = {}
for name in ("torch", "torchvision"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
root = Path("artifacts/environment")
(root / "managed-torch.txt").write_text("".join(f"{name}=={version}\n" for name, version in packages.items()))
(root / "managed-backend.json").write_text(json.dumps({"packages": packages,
    "hip": torch.version.hip, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}, indent=2))
HOST
uv pip install --python "$ISL_PYTHON" -c artifacts/environment/managed-torch.txt -e ".[dev]"
uv pip freeze --python "$ISL_PYTHON" > artifacts/environment/managed-after.txt
```

If resolution cannot retain the constrained versions, resolve that incompatibility
before continuing. Do not replace the host torch build to make installation pass.
The package supports Python 3.11 and 3.12. Check the installed requirements:

```bash
"$ISL_PYTHON" -c 'import json; from isl_slt.utils.environment import check_managed_environment; print(json.dumps(check_managed_environment(), indent=2))'
```

Create `.env` from `.env.example` if absent, populate `HF_TOKEN` privately, and set
`chmod 600 .env`. Existing credentials must stay out of logs, YAML and commits.
The shared template contains no `CUBLAS_WORKSPACE_CONFIG`; remove it from older
local copies on AMD. Runtime only applies that setting on non-HIP builds.

Use this terminal helper for the commands below on the managed droplet. It loads
`.env`, explicitly selects the existing interpreter, and disables synchronization:

```bash
isl() {
  uv run --no-project --no-sync --no-python-downloads --python "$ISL_PYTHON" \
    --env-file .env "$ISL_PYTHON" -m isl_slt "$@"
}
isl env-check --config configs/amd_fp32.yaml
isl plan --config configs/amd_fp32.yaml
```

For every later `uv run --no-sync python -m isl_slt ...` example, use `isl ...`
on this managed host. Acquisition similarly uses the existing interpreter:

```bash
uv run --no-project --no-sync --no-python-downloads --python "$ISL_PYTHON" \
  --env-file .env "$ISL_PYTHON" scripts/fetch_isign.py --help
```

ROCm uses PyTorch's `cuda` device/API name. AMD configurations require the GPU
instead of silently falling back to CPU. FP32 recurrence stays outside autocast.
`env-check` exercises a packed bidirectional LSTM and random local tiny T5 in
FP32/BF16/FP16, without downloading weights. A failed configured precision exits
with an error. Strict determinism remains enabled; do not switch to warn mode to
silently bypass a missing deterministic kernel.

### GPU installation boundary

The committed lock verifies the CPU environment and ordinary dependencies. The
previous unverified `rocm`/`cuda` installer extras have been removed. GPU runs use
managed mode and the host's working PyTorch pair. The ROCm wheel index can require
multi-gigabyte downloads just to resolve metadata; it is not part of this lock.
If provisioning a fresh GPU image, install a driver-compatible pair using the
vendor's instructions first, then follow the constrained managed setup above.

### FP32 versus BF16 validation pilot

First supply the real inputs, revisions, frozen source-held-out splits, prepared
pose features and local T5 assets described below. Both pilots share them. Each
runs one frozen-adapter epoch and one joint epoch, seed 42, batch size 32,
accumulation 2, eight workers, and fixed greedy decoding. The LSTM remains FP32.
Run FP32 first; only continue to BF16 after both environment checks succeed:

```bash
isl env-check --config configs/amd_fp32_pilot.yaml > artifacts/environment/fp32-probe.json &&
isl train --config configs/amd_fp32_pilot.yaml &&
isl env-check --config configs/amd_bf16_pilot.yaml > artifacts/environment/bf16-probe.json &&
isl train --config configs/amd_bf16_pilot.yaml
```

BF16 training/resume also reruns preflight automatically and records
`env-check.json` in its run directory. A failed FP16 probe does not prevent a
successful FP32/BF16 configuration. Use fresh run directories; never resume a
trajectory under a different precision.

Evaluate both final checkpoints on the same validation partition:

```bash
for pilot in amd_fp32_pilot amd_bf16_pilot; do
  run="runs/${pilot//_/-}"
  isl evaluate --config "configs/$pilot.yaml" --checkpoint "$run/checkpoints/last.pt" \
    --partition validation --output "$run/comparison-validation"
  uv run --no-project --no-sync --no-python-downloads --python "$ISL_PYTHON" \
    --env-file .env "$ISL_PYTHON" scripts/summarize_run.py "$run" --output "$run/throughput.json"
done
```

Compare validation BLEU, chrF++ and WER, per-stage completed-epoch throughput
(including input loading), validation time and peak GPU memory. Update throughput
alone excludes input loading. BF16 remains an opt-in experiment; two epochs are
a compatibility/performance comparison, not evidence of final translation quality.
Keep the full FP32 baseline until you have reviewed these measurements. Test/gold
partitions are not used to select precision.

### Notebook workflow and managed Colab/Kaggle setup

The [pipeline notebook](../notebooks/run_pipeline.ipynb) calls the same CLI and
ships with preview defaults and empty outputs. Set `RUNNER="managed"` and select
the host interpreter on a managed ROCm, Colab or Kaggle image. Install the
`notebooks` extra together with `dev` in the constrained managed install above.
Launch Jupyter through the same environment loader so the managed kernel inherits
`.env` without displaying it:

```bash
uv run --no-project --no-sync --no-python-downloads --python "$ISL_PYTHON" \
  --env-file .env "$ISL_PYTHON" -m jupyter lab notebooks/run_pipeline.ipynb
```

For a project-owned kernel, use `uv sync --locked --extra cpu --extra notebooks`,
then launch with `uv run --no-sync --env-file .env jupyter lab notebooks/run_pipeline.ipynb`
and set `RUNNER="uv"`.
The notebook resolves the checkout from its root or `notebooks/`. Change
`CONFIG_PATH`, rerun helper cells, inspect printed commands, and enable only the
stages you intend to execute. Downloads require explicit network opt-in. Managed
mode inherits its kernel environment and does not load `.env` itself. Preserve
persistent data/run storage and immutable memberships on hosted notebook systems.

## 3. Probe and inspect the resolved plan

```bash
uv run --no-sync python -m isl_slt env-check --config configs/baseline.yaml
uv run --no-sync python -m isl_slt plan --config configs/baseline.yaml
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --dry-run
```

`env-check` executes synthetic recurrent forward/backward probes. `plan` and `--dry-run` print configuration/planning information without loading datasets or models. A printed plan does not validate files, cache contents, losses or convergence.

The environment check also constructs a tiny random T5 locally and probes forward/backward plus repeated greedy generation for each precision. No weights are downloaded. Reports separate recurrent and T5 outcomes; the CLI fails when the configured precision fails either. Probe success is preliminary host evidence, not a convergence or reproducibility guarantee for the full model. The committed CPU lock is verified separately from the managed GPU environment; successful GPU execution must be recorded on the target host.

## 4. Audit and freeze membership

First generate candidate source mappings, then review their real naming/cross-dataset evidence and supply a separate authoritative map:

```bash
uv run --no-sync python -m isl_slt make-source-map --config configs/baseline.yaml
uv run --no-sync python -m isl_slt make-source-map --config configs/transfer.yaml --dataset pretrain
```

Candidates remain unverified. Unresolved UIDs fail after report export. The command never overwrites an authoritative map. Exact duplicate rows are collapsed with kept/dropped row numbers and hashes; conflicting same-UID rows fail and raw annotations remain unchanged.

After pinning the dataset revision and reviewing the [publisher's UID convention](https://huggingface.co/datasets/Exploration-Lab/iSign/blob/main/README.md), promote with recorded evidence. DEF identities require the separate approval shown here; omit it for sentence-only input. No existing authoritative mapping is overwritten:

```bash
uv run --no-sync python -m isl_slt make-source-map --config configs/baseline.yaml --promote --evidence "REPLACE with reviewed revision-specific naming and DEF evidence" --approve-def
```

```bash
uv run --no-sync python -m isl_slt audit --config configs/baseline.yaml
uv run --no-sync python -m isl_slt split --config configs/baseline.yaml
uv run --no-sync python -m isl_slt check-leakage --config configs/baseline.yaml
```

Inspect `data/manifests/target.audit.json` before proceeding, including missing/invalid poses, labels, source verification, interpreter disagreements and any requested context/RGB errors. Check actual source/group counts and split memberships in `splits/source-v1/`. A report containing issues is not a successful scientific validation.

Membership is immutable and source-disjoint; different training seeds reuse it. Do not delete or overwrite a frozen split to make leakage checks pass. Revisions require distinct artifact paths.

## 5. Prepare features and generic model assets

```bash
uv run --no-sync python -m isl_slt prepare-features --config configs/baseline.yaml
uv run --no-sync python -m isl_slt verify-cache --config configs/baseline.yaml
uv run --no-sync --env-file .env python -m isl_slt prepare-model --config configs/baseline.yaml --allow-network
```

The feature command writes fingerprinted arrays and metadata. Model preparation is an explicit future download of the configured generic initializer. It resolves the requested T5 revision to a commit and records checksums in `artifacts/models/flan-t5-small/asset.json`. Ordinary training/evaluation loads local assets.

Review feature dimensions, lengths, masks, preparation settings and the initializer before training. Ordinary pose/cached-RGB training validates arrays and manifest dependency signatures without reopening raw assets. Arrays are hashed once at construction and file identities are checked during item reads. `verify-cache` defaults to full raw verification; `--no-verify-assets` checks caches alone. Train/evaluate/diagnostics accept `--verify-assets` explicitly.

Pose cache v2 adds continuous validity; old caches need regeneration. For optional standardization, prepare features with its config first:

```bash
uv run --no-sync python -m isl_slt prepare-features --config configs/standardized.yaml
uv run --no-sync python -m isl_slt prepare-stats --config configs/standardized.yaml
uv run --no-sync python -m isl_slt train --config configs/standardized.yaml
```

Statistics fit immutable target-training data only, preserve masks/invalid zeros, and are embedded in checkpoints. Image-mode RGB and raw prediction still require real assets.

## 6. Inspect a small training subset

Choose 32–64 unique manually inspected training UIDs and write one per line to `data/overfit_uids.txt`. The dedicated preset disables dropout, weight decay, noise and lexical loss, with T5 unfrozen:

```bash
uv run --no-sync python -m isl_slt plan --config configs/overfit.yaml
uv run --no-sync python -m isl_slt overfit --config configs/overfit.yaml --uids data/overfit_uids.txt
```

The protocol fits and selects on those training examples, records a diagnostic partition and excludes extra pretraining. Inspect loss/gradients/generated text; there is no BLEU threshold or held-out quality claim.

The preset is `training.debug_only: true`: ordinary training without an explicit UID file fails before split loading. In the notebook select `configs/overfit.yaml`, set `OVERFIT_UIDS_PATH` and use the dedicated overfit cell. Debug resume uses the same `overfit --uids` invocation plus `--resume`.

## 7. Train and resume the baseline

```bash
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml
```

Default training freezes T5 for one adapter epoch, then trains jointly for up to 19 more epochs. Per-epoch validation uses generation.train_beams (default greedy); the final validation sweep uses generation.beams (default 1 and 4). BLEU-4 controls early stopping in the final stage. Monitor `events.jsonl` and saved validation predictions rather than teacher-forced losses alone.

Continue a matching saved trajectory with:

```bash
uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --resume runs/baseline/checkpoints/last.pt
```

Resume checks data/cache/tokenizer/experiment/numerical fingerprints, source/Git identity and declared dependency/Python-minor/torch/CUDA/HIP versions. Full package and lock inventories are provenance. `--allow-code-change` approves only code changes; `--allow-environment-change` separately approves environment differences, logs the full diff and labels continuation non-exact. Neither bypasses data/model/numerical or RNG-topology requirements. A new run cannot reuse an existing run directory without explicit resume. If a wall-time limit returns `wall_time_limit`, retain `last.pt` and resume with the same compatible setup.

## 8. Prepare scoring resources and evaluate

The baseline requests METEOR, which needs local NLTK WordNet. Prepare it explicitly later:

```bash
uv run --no-sync --env-file .env python -m nltk.downloader -d artifacts/nltk_data wordnet
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --partition validation --output runs/baseline/evaluation/validation
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --partition test --output runs/baseline/evaluation/test
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --partition gold --output runs/baseline/evaluation/gold
uv run --no-sync python -m isl_slt diagnostics --config configs/diagnostics.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --partition test --output runs/baseline/diagnostics/test
```

The training checkpoint retains per-epoch selection; run the final validation beam sweep and pass decoding.json to every final prediction entry point. Test references are used only for scoring. Metrics and predictions are saved under the output path. Missing requested metric resources are recorded in the report; `evaluate` then exits with an error rather than reporting a substituted zero.

Gold reports retain all-parent-holdout transcript scores plus transcript/interpreter scores on exactly the same annotated UID subset, with shared counts/hashes. Diagnostics save real, blank, incorrect-input and temporal-shuffle predictions; they test input dependence rather than proving semantic faithfulness. For a nonbaseline experiment, use a diagnostic override inheriting that experiment's config so its model/features/RGB settings match the checkpoint.

## 9. Run seeds and summarize

Use seeds 13, 42 and 2026 with the same frozen memberships:

```bash
for training_seed in 13 42 2026; do
  uv run --no-sync python -m isl_slt train --config configs/baseline.yaml --seed "$training_seed"
  uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint "runs/baseline/seed-$training_seed/checkpoints/best-bleu.pt" --partition validation --output "runs/baseline/seed-$training_seed/evaluation/validation"
  uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint "runs/baseline/seed-$training_seed/checkpoints/best-bleu.pt" --partition test --decoding "runs/baseline/seed-$training_seed/evaluation/validation/decoding.json" --output "runs/baseline/seed-$training_seed/evaluation/test"
done
uv run --no-sync python -m isl_slt summarize --reports runs/baseline/seed-13/evaluation/test/metrics.json runs/baseline/seed-42/evaluation/test/metrics.json runs/baseline/seed-2026/evaluation/test/metrics.json --output runs/baseline/seed-summary.json
```

The seed override places outputs under `seed-<seed>`. Summaries require distinct seeds and identical effective decoding, chosen beams, numerical policy and metric signatures. If validation selects different beams, use a predeclared fixed-beam evaluation protocol for a strict comparable summary. Report individual runs and variation; do not concatenate seed predictions into a single score.

## 10. Average and reselect decoding

Only average at least two retained compatible checkpoints from the same trajectory. The glob below uses actual retained epoch files, not assumed epoch numbers:

Epoch retention defaults to zero. Before a trajectory intended for averaging, explicitly set `training.retain_epoch_checkpoints` to at least 2 in its config; this cannot recover discarded epochs afterward.

```bash
uv run --no-sync python -m isl_slt average --checkpoints runs/baseline/checkpoints/epoch-*.pt --output runs/baseline/checkpoints/averaged.pt
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/averaged.pt --partition validation --output runs/baseline/averaged/validation
uv run --no-sync --env-file .env python -m isl_slt evaluate --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/averaged.pt --partition test --decoding runs/baseline/averaged/validation/decoding.json --output runs/baseline/averaged/test
```

Averaged weights are inference-only and cannot resume optimizer training. Decoding selection is bound to the actual checkpoint checksum, generation configuration and validation membership.

For two compatible saved prediction files, source-group paired bootstrap is available:

```bash
uv run --no-sync python -m isl_slt bootstrap --first runs/baseline/evaluation/test/predictions.jsonl --second runs/local-hand/evaluation/test/predictions.jsonl --samples 1000 --seed 2026 --output runs/comparisons/local-hand-bootstrap.json
```

Run this only after both corresponding evaluations exist with aligned UIDs, references and source IDs.

## Prediction, measurement and recovery

`predict` and `benchmark` require reference-free visual JSONL records conforming to `VisualInput`. Required identities/paths and optional timing/geometry metadata must describe actual assets; text/reference fields are rejected.

```bash
uv run --no-sync python -m isl_slt predict --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --inputs data/prediction_inputs.jsonl --output runs/baseline/predictions.jsonl
uv run --no-sync python -m isl_slt benchmark --config configs/baseline.yaml --checkpoint runs/baseline/checkpoints/best-bleu.pt --decoding runs/baseline/evaluation/validation/decoding.json --inputs data/prediction_inputs.jsonl --output runs/baseline/benchmark.json
```

Benchmark reports distinguish model-only from end-to-end timing and record warmup/iteration settings. The BiLSTM is noncausal. The `estimate` command accepts actual measured throughput, validation time, example/epoch counts and an optional hourly price; no time, credit or quality guarantee is implied.

For missing assets, fix the configured paths or prepare the declared resources. For cache or membership mismatches, identify the changed inputs and use properly revised artifacts. For memory failures, make an explicit new batch/frame/resampling experiment. For resume incompatibility, restore the original environment or start a documented new trajectory. No recovery step silently drops examples or bypasses integrity checks.

See [experiments](experiments.md) for optional run sequences and [implementation status](implementation_status.md) for outstanding verification.

## Asset acquisition and layout

Plan target selection before extracting anything. This reads remote ZIP end/index ranges, prints exact selected compressed/uncompressed sizes, and downloads no member payloads. Obtain the annotation CSV separately and pin the configured dataset revision:

```bash
uv run --no-sync --env-file .env python scripts/fetch_isign.py --plan --uids-csv data/raw/ISLTranslate.csv --config configs/baseline.yaml --allow-network
uv run --no-sync --env-file .env python scripts/fetch_isign.py --fetch --uids-csv data/raw/ISLTranslate.csv --config configs/baseline.yaml --out data/raw/poses --allow-network
```

Remote parts resolve to one immutable Hub commit. Exact HTTP ranges are checked before any body is consumed, avoiding a silent whole-part download when ranges fail. Multipart input also accepts sized seekable streams, including HfFileSystem-compatible readers. No whole-archive concatenated copy is made. An explicit `--fetch --transport staged` fallback reads the last part first and evicts staging files before each new part, bounding staging to two parts including temporary bytes; its cost is printed before download. It is never automatically selected by `--plan`.

For optional transfer, promote both source maps first and pass `--dataset pretrain --uids-csv data/raw/iSign_v1.1.csv --config configs/transfer.yaml --budget-bytes YOUR_COMPRESSED_BYTE_BUDGET --selection-seed 2026`. Every target source, UID, known fingerprint and normalized-text duplicate is excluded. Eligible clips are grouped by source and selected in deterministic seeded order without exceeding the compressed-byte budget. No percentage or default budget is assumed. Point `data.pretrain.annotations` at the generated `selected-annotations.csv`, preserve the original revision and authoritative mapping, then audit the acquired poses; the same strict exclusions are rechecked. Changed transfer policy requires new split paths.

These commands are for later explicit network authorization. Listing does not download the pose parts; CSV and full-part flags are separate:

```bash
uv run --no-sync --env-file .env python scripts/fetch_isign.py --list --allow-network
uv run --no-sync --env-file .env python scripts/fetch_isign.py --csv --allow-network
```

The four pose files are ordered byte parts of one ZIP. Only a deliberately requested `--parts --allow-network` downloads them. Local extraction validates paths/UID uniqueness, builds the member index and decodes one selected sample before bulk extraction. For initial extraction, use the real annotation CSV as a UID request, avoiding immutable manifests with missing/fake asset checksums:

```bash
uv run --no-sync python scripts/fetch_isign.py --archives data/raw/isign-acquisition/iSign-poses_v1.1_part_aa data/raw/isign-acquisition/iSign-poses_v1.1_part_ab data/raw/isign-acquisition/iSign-poses_v1.1_part_ac data/raw/isign-acquisition/iSign-poses_v1.1_part_ad --uids-csv data/raw/ISLTranslate.csv --config configs/baseline.yaml --out data/raw/poses
uv run --no-sync python scripts/inspect_pose.py data/raw/poses/ACTUAL_UID.pose --config configs/baseline.yaml
```

Replace ACTUAL_UID with an extracted UID. `--manifest` can replace `--uids-csv` for a canonical selection. Extraction refuses overwrites, defaults to a 50 GiB selected-byte limit and records the selection/member index/output checksums. Change `--max-bytes` deliberately if justified. Configure `pose_pattern` and canonical-to-stored `features.component_names` from observed headers. The real archive layout and sample decoding remain unverified here.

## Measured training summaries and declared defaults

Interrupted acquisition resumes with the same arguments plus `--resume`. `extraction.journal.json` binds the resolved archives, dataset revisions, request hash and UID/member selection. Each completed pose is atomically published and journaled with size, CRC and SHA256. Resume verifies existing bytes and refuses corruption or a changed selection. `extraction.json` is finalized only after full coverage. Keep selected raw poses for feature ablations; clean staged/archive copies only after coverage, cache verification and a deliberate backup decision. The implementation performs no uploads or raw-pose deletion.

```bash
uv run --no-sync python scripts/summarize_run.py runs/baseline --output runs/baseline/training-summary.json
uv run --no-sync python -m isl_slt estimate --examples ACTUAL_TRAIN_COUNT --epochs 19 --run-summary runs/baseline/training-summary.json --stage joint
```

Logs contain actual applied group LRs, synchronized update seconds, examples/frames throughput, allocated GPU peaks and separate epoch train/validation time. Update throughput excludes input-window loading; epoch training time includes it. Setup/checkpoint I/O is excluded. Supply measured values to `estimate`; inference latency is not training throughput.

Completed-epoch example/frame counters and elapsed time persist across interruptions. Summary `stages` rates divide completed-epoch counts by their total training time, including loader setup/loading. Partial epochs and legacy epochs without counters cannot supply an estimate. `estimate --run-summary --stage` uses that rate and measured stage validation time; replace `ACTUAL_TRAIN_COUNT` with the audited count. Adapter/pretrain/joint stages are reported separately. `unique_checkpoint_bytes` counts hard-linked files once, while per-path sizes remain available.

Only `last.pt` retains continuation state. Best/epoch/averaged checkpoints keep complete inference reconstruction metadata without optimizer/RNG state. `best-guarded.pt` additionally uses the historical chrF tolerance policy.

`evaluation.repeat_seeds` and `evaluation.bootstrap_samples` are declarative notes, not automatic loops or CLI-default overrides. Operative controls are `train --seed`, explicit repetitions and `bootstrap --samples`. Workers remain zero until profiled; persistent workers/GPU pinning are supported. Pose noise changes only position-valid normalized/standardized feature channels; it does not edit raw poses or recompute velocity/bones.
