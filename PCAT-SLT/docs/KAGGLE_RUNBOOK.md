# Kaggle walkthrough: PCAT-SLT from setup to saved checkpoint

Checked against source on **14 November 2026**. Companion files: [datasets and training budget](../TRAINING_AND_DATASETS.md), [ready-to-import notebook](../notebooks/Kaggle_PCAT_SLT.ipynb). The notebook offers a generated-data diagnostic and separate prepared-real-data pose/visual modes. Local validation is not a claim that a Kaggle CUDA session was executed.

## 1. Exactly which accelerator to select

In the notebook editor's **Settings / Accelerator**, choose **GPU T4 ×2**. Its two cards each have 16 GB nominal memory; this single-process trainer uses **GPU 0 only**. Card 1 does not add memory to card 0. Do not use TPU, and do not launch this trainer twice through `torchrun`. [Kaggle staff announcement](https://www.kaggle.com/product-announcements/735239).

| Work | Choice | Starting profile |
|---|---|---|
| Inspect/download/convert metadata | CPU / Accelerator None when practical | No model training required |
| First setup/forward/backward/resume check | T4 ×2, one card | Notebook `MODE='diagnostic'` |
| First real pose-only run | T4 ×2, batch 1 | `configs/kaggle_pose.yaml` |
| First visual subset | T4 ×2, batch 1, frozen vision, small text model | `configs/kaggle_visual.yaml`; prepare four aligned 112px crops |
| Long-clip/full visual study | One full A100 40/80 GB or equivalent, if your allocation actually offers it; otherwise approved external compute | Profile before increasing batch/model size |
| TPU v5e/v6e or older TPU options | **Do not select for this code** | No TPU device/backend implementation |

P100 was retired on 15 September 2026. Current [official CLI docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) identify `NvidiaTeslaT4` as T4 ×2 and list some restricted accelerators; a listed A100/H100 identifier does not mean everyone can select it. If GPU options are unavailable, follow the account verification/quota messages actually displayed. Do not choose a TPU just because it is selectable.

The [notebook specifications](https://www.kaggle.com/docs/notebooks) describe a 12 h GPU/CPU session and 20 GB saved output; the [GPU guide](https://www.kaggle.com/docs/efficient-gpu-usage) describes a variable weekly quota around 30 h. Check your actual timer/quota/resources. Some documentation still mentions retired devices. Plan complete-epoch chunks with time left for output saving, rather than a single uninterrupted 40-epoch job.

## 2. Choose the route before uploading data

- **Route A — diagnostic:** no third-party corpus, videos, detector, Hugging Face token or pretrained download needed. The project generates toy geometry. Use this first to test setup, training and recovery. It does not evaluate SLT or the full pretrained visual architecture.
- **Route B — real pose:** obtain authorized prepared landmarks + written translations, for example by converting iSign's existing `.pose` files. Mode `pose` expects NPZ/JSONL, not original `.pose`, HDF5, pickle or MP4.
- **Route C — real visual:** Route B plus aligned hand/mouth/eye crop NPZs. Use mode `visual`. A raw video archive is insufficient input by itself.

For iSign, check its [access terms](https://huggingface.co/datasets/Exploration-Lab/iSign/blob/main/README.md) before any Kaggle data copy. The release asks users not to upload/share it elsewhere. Public or private hosting must not be assumed permitted. No gated agreement is accepted or data uploaded automatically by this project.

## 3. Files you need

Download this project's `releases/PCAT-SLT.zip` and companion `.sha256`. The source ZIP includes code/configs/docs/notebook, not third-party data or trained real-corpus weights. Keep one PCAT-SLT folder.

For Route B/C prepare a directory containing:

```text
prepared_release/
  train.jsonl
  val.jsonl
  test.jsonl
  poses/                 # NPZ files referenced by the manifests
  rgb/                   # Route C only
```

Example schema (IDs/paths here illustrate the required structure):

```json
{"id":"recording42-clip7","pose":"poses/recording42-clip7.npz","text":"The written reference.","source_video":"recording42","signer":"known-signer-id"}
```

Only include signer IDs if they are actually known. In visual mode add `"rgb":"rgb/recording42-clip7.npz"`. Use source-video metadata so leakage checks can enforce grouping. See [the exact arrays/layout](DATA.md). At least two frames are required; the provided profiles reject clips over 400 frames and excessive target lengths. Limits are not automatic truncation/sampling. Confirm the full distributions before training.

If official annotations have not been converted, use [the implemented importers](DATA.md). iSign needs actual landmark-layout conversion first. PHOENIX import accepts one official split file at a time into a fresh directory; point each YAML manifest path at the appropriate result or assemble stable generic manifests. Do not guess missing extraction steps.

## 4. Create the notebook and attach inputs

1. Sign in to Kaggle and create a **private Python notebook** (Create → New Notebook, or equivalent current UI).
2. Import/upload `notebooks/Kaggle_PCAT_SLT.ipynb` as the notebook source. This is distinct from adding a data input.
3. Add the source ZIP as a notebook input, using a source-code dataset or another input method available in your editor. Keep its filename `PCAT-SLT.zip`, or attach its extracted PCAT-SLT directory. Upload only the small release archive, not the entire local folder with runs/data/caches.
4. For real-data modes, separately attach the authorized prepared dataset. Inspect the Files/Input panel to confirm manifests and NPZs are accessible under `/kaggle/input`. Archives may be expanded by the platform; do not assume the exact mount path from a tutorial.
5. Select T4 ×2 and enable Internet for dependency installation and, in visual mode, initial model/tokenizer download. If a competition disables Internet, supply compatible dependency/model assets through an allowed offline route; the default notebook does not promise that route automatically.
6. Run the settings cell. Leave `MODE='diagnostic'` first. `PREPARED_ROOT=None` means find exactly one input directory with train/val/test manifests; if there are several, set it to the actual chosen path shown by the notebook. There are no real-data downloads in the notebook.

Input is read-only. The notebook copies/extracts project source to `/kaggle/working/PCAT-SLT`; generated configs/logs/weights remain there. Existing output is preserved. If automatic discovery finds ambiguous inputs it stops and lists candidates, rather than selecting a random dataset.

## 5. Run Route A: generated-data diagnostic

Run notebook cells in order. The sequence is:

1. Locate source; for ZIP input verify its embedded SHA-256 manifest before extraction. Copy an already extracted source tree if that is your input format.
2. Inspect CUDA/GPU; install the editable project and optional multimodal dependencies without forcing the local CPU lockfile over Kaggle's torch.
3. Run the unit suite; save package/GPU metadata.
4. Generate a tiny pose-order dataset under `data/kaggle_demo` and a runtime config under `configs/`. Default planned diagnostic schedule is eight epochs, but the first invocation stops after one.
5. Validate the dataset, train one epoch, record elapsed call time, peak CUDA allocation/reservation and checkpoint bytes.
6. Generate dev predictions from the best checkpoint, then create a run-artifact ZIP in `releases/`.

Expected artifacts include `runs/kaggle_demo/last.pt`, `best.pt`, `run.json`, `history.jsonl`, validation/timing/environment reports, dev predictions and an artifact archive. Exact accuracy/loss/time is not specified in advance. A successful synthetic run proves setup/plumbing, not natural-language translation quality.

If using commands manually after installation, the equivalent initial configuration can be produced from `configs/smoke.yaml`, changing train device to CUDA and giving it fresh data/output paths. The notebook writes and validates that concrete YAML for you. Do not run `configs/smoke.yaml` unchanged and assume it selects CUDA: its shipped device is CPU.

## 6. Save and recover a run before using a large corpus

The run bundle is not the source release ZIP: it deliberately includes checkpoints and diagnostic data or real manifests. Its name starts `PCAT-SLT-KAGGLE-`; it does **not** include referenced third-party raw/processed arrays. It contains a file checksum inventory. Keep these run outputs private as appropriate.

1. Download the run archive while the session is still available, or retain the notebook version's actual outputs. Inspect the saved output listing; Quick Save of notebook code is not evidence that checkpoint files were preserved.
2. For a fresh session, attach the same source release and same prepared data mounts, plus the **extracted saved run bundle** as an input. If you have only its ZIP, extract it locally into a dataset input before attaching, or extract it to a separate workspace folder and set the restore path accordingly.
3. Set `MODE` to the same mode, `RESUME=True` and `STOP_AFTER_EPOCH=2` after the first completed epoch. Keep every other training/data setting unchanged.
4. If checkpoint discovery is ambiguous, set `RESTORE_RUN_DIR` to the actual directory containing that run's `last.pt`, `best.pt`, `run.json`, `history.jsonl`. No arbitrary third-party checkpoint is downloaded.
5. Run setup/data/restore/validation/train cells. Diagnostic data are restored from the bundle when available. For real data, the same `/kaggle/input` observation paths must still resolve.
6. Confirm `history.jsonl` and the checkpoint's completed-epoch report show progress to epoch two. Save the new bundle again.

The trainer supports exact **epoch-boundary** continuation with identical config/input fingerprints. It rejects changed epochs, AMP, batch, data paths/content or decoder settings, except an output-directory change. An interrupted epoch is repeated. No mid-epoch resume, schedule extension or general weight-only dataset-transfer CLI is supplied. If real-data epoch time cannot fit the session, move that experiment to suitable persistent compute.

Kaggle's **Save Version / Save & Run All** may start a fresh execution. Ensure its settings point to the intended source, restored run and stop epoch; do not accidentally rerun a fresh diagnostic thinking it merely captures the currently running filesystem. Check the resulting output and logs.

## 7. Run Route B: real pose-only training

Once the prepared data are attached and permitted:

1. Set `MODE='pose'`, `RESUME=False`, `STOP_AFTER_EPOCH=1`. Select T4 ×2.
2. Select the actual `PREPARED_ROOT` or let unique manifest discovery locate it. The notebook rewrites observation references to stable absolute mounted paths in project-local manifests. It does not resplit or convert the observations.
3. Run full validation. Fix missing files, incorrect joint conventions, overlaps and target/length errors before training. Do not bypass validation to get a loss curve.
4. The starting model is `configs/kaggle_pose.yaml`: batch 1, accumulation 8, native byte decoder, float32 integration, CUDA AMP, intended 40-epoch schedule. The runtime YAML is saved separately. Small byte models are useful controls; they are not asserted to be the best final decoder.
5. Measure one train+validation+checkpoint call and dev generation. Check longest clips and resume peaks, not just average samples. Leave time for output saving.
6. Resume to later complete epochs using the saved bundle and unchanged data/config. Use dev metrics for choices. The notebook defaults generation to `val`; evaluate `test` only when the model/protocol is frozen.

The 40-epoch plan is an example, not a proven optimum. Do not change it to two epochs just to stop after two: use `STOP_AFTER_EPOCH=2` so the planned scheduler remains consistent. See [time formulas/scenarios](../TRAINING_AND_DATASETS.md) for budgeting based on actual measured calls.

## 8. Run Route C: real visual training

Set `MODE='visual'` for a **new run** with actual RGB paths on every manifest row. This selects frozen DINOv2-small + FLAN-T5-small, vision chunk 8, batch 1, accumulation 8 and one generation beam. DINO/text models initially download from Hugging Face. The default real models have not been benchmarked on this local host. Use [the model cards/precision discussion](../TRAINING_AND_DATASETS.md).

Prepare four actual 112×112 crops per frame; YAML does not resize arbitrary existing arrays. Masks and timestamps must match pose, including original clip origin. Creating all float crops can require enormous storage: measure representative compressed NPZ sizes before a full conversion. Decoder source memory includes per-frame details/pixels/motion, not only compressed PCAT cells.

If OOM occurs, distinguish initialization, forward/backward, optimizer, resume and generation peaks. First use batch 1 and smaller vision chunks; consider a smaller separately declared model or a larger GPU. Reducing `max_frames` alone does not resample clips. Turning AMP off may fix numerical issues but increases memory and requires a new training configuration/run. No DDP, TPU, feature-cache or gradient-checkpointing switch exists.

## 9. Common failures and exact responses

| Symptom | Likely cause / action |
|---|---|
| `CUDA requested but unavailable` | Confirm accelerator selection, restart/setup session if requested, inspect `nvidia-smi`/torch CUDA. Do not report a CPU run as a GPU run |
| Multiple source archives/manifests/checkpoints | Set `SOURCE_ZIP`, `PREPARED_ROOT` or `RESTORE_RUN_DIR` to the actual displayed choice |
| `No module named pcat_slt` | Run installation cell in this kernel; notebook also adds the copied `src` directory to its Python path |
| Missing NPZ / `.pose` unreadable | Raw dataset format has not been converted, or manifest path is wrong; consult the catalog and exact array contract |
| No reliable body frame | Verify shoulder/neck mapping, image-coordinate consistency, confidence and nondegenerate geometry; never mark fabricated anchors as confident |
| RGB time/shape/mask error | Fix conversion alignment; do not independently renumber pose and RGB timestamps |
| Target/frame limit exceeded | Audit lengths; revise a documented offline sampling/target policy or new config before a new run; no silent truncation |
| Duplicate/source/signer overlap | Audit group metadata and actual duplicates; preserve official versus new-experiment split distinctions |
| Existing `last.pt` | Resume deliberately or use a fresh run name; do not delete a useful run to hide the conflict |
| Resume config/fingerprint mismatch | Restore identical YAML/data/mount layout; changed training is a new experiment, not exact resume |
| HF access/network error | Internet/cache/credentials or repository access; use allowed model assets, not downloaded unofficial substitutes. Keep tokens out of outputs |
| Nonfinite AMP loss/gradients | Inspect input scales/confidence, reproduce a small batch, try a fresh float32 diagnostic/config and record the difference; CUDA numerical behavior is still unmeasured locally |
| GPU memory exhausted | Profile actual phase/lengths; accumulation does not reduce single-clip memory. Use a smaller configuration or larger device |
| Checkpoint absent next session | Live-session storage was not actually preserved; verify retained output/bundle before termination |
| Epoch exceeds session | Current epoch-only resume is unsuitable at that scale; use persistent compute or implement/test mid-epoch recovery |

## 10. What to retain for a defensible result

Keep the exact source ZIP/checksum, runtime YAML, immutable pretrained revisions for final research, corpus version/agreement, split assignments/source lineage, conversion mapping/version, data hashes/exclusions, hardware/software report, timing/memory logs, checkpoints and generated dev/final-test outputs. Report the scorer signature and language-appropriate normalization. Built-in checkpoint selection is validation NLL; it is not automatic BLEU-based early stopping.

The diagnostic, real pose and real visual modes test different things. A notebook that completes the first mode is a useful prerequisite; it does not certify the full architecture, every dataset or publication quality.
