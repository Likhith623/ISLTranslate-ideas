# ISL-MEGA

**Multi-Expert, Evidence-Grounded Architecture for Continuous Indian Sign Language Translation**

ISL-MEGA is a research implementation candidate for gloss-free, sentence-level ISL-to-English translation. It combines specialist RGB region encoders, an articulated 66-landmark graph pathway, a 3-D temporal motion pathway, a character-level fingerspelling expert, reliability-aware asynchronous fusion, hierarchical temporal reasoning, latent-plan evidence aggregation, and a pretrained T5/FLAN-T5 text decoder.

> **Research-status statement.** This repository is an implementation of a proposed architecture, not evidence that the method is state of the art, improves translation accuracy, or will be accepted by a Q1 journal. Those claims require experiments on the actual data, strict leakage-resistant splits, signer review, comparisons and ablations. The package does not contain the ISL dataset, trained model weights, Hugging Face checkpoints, SignDINO weights or benchmark results.

## What is implemented

| Pathway / system | Implementation |
|---|---|
| RGB appearance | Shared left/right hand ViT plus separate face and upper-body ViT backbones; optional `timm` DINOv2 initialization |
| Articulatory geometry | Shared bilateral hand graph (mirror-canonicalized) + shoulder-span scale normalization + 36-node cross-articulator phonological location graph with Gaussian proximity RBFs |
| Temporal motion | 3-D convolutional pathway over full-resolution body crops and frame differences; temporal stride is 1 |
| Fingerspelling | BiGRU + CTC specialist with calibrated frame-level span spotter (`spot_prob`) gating posterior character feedback (`span_routing`) |
| Fusion | Log-odds reliability calibration + continuous elapsed-time (`dt` seconds) Fourier/RBF asynchronous cross-stream alignment |
| Temporal reasoning | Multi-scale temporal stem, kinematic event-boundary adaptive slot consolidation, global Conformer, and reversible event-to-fine retrieval |
| Latent planning | Diversity- and ordering-regularized latent queries with dynamic clip-length budget routing (`plan_mask`) |
| Text generation | Optional Hugging Face T5/FLAN-T5 encoder-decoder conditioned on frame memory, active plan tokens, plan+frame semantic OT, and counterfactual visual grounding loss |
| Self-supervised pretraining | Global-teacher / local-region-student DINO-style objective; this is an ISL-MEGA variant inspired by SignDINO, not a reproduction of the paper |
| Data quality | Timestamp alignment, aspect-preserving letterboxing, coordinate/box transforms, optional `fingerspelling_spans`, NPZ validation and source/signer split checks |
| Training objectives | Translation CE, source-aware RGB↔pose InfoNCE, CTC + MIL/span spotting, plan diversity/order, fusion entropy, plan+visual semantic OT, counterfactual visual margin/KL |
| Evaluation | Corpus BLEU, chrF++, TER with SacreBLEU signatures/version, and optional ROUGE-L; human sign-language evaluation is external |

The RGB appearance and pose streams are separate specialists. The face appearance stream represents visual mouth/face information; the system does **not** claim to perform phoneme-level lipreading. The CTC fingerspelling expert only receives a reliable supervision signal when an explicit character transcript is provided.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full tensor flow, module dimensions, temporal policy and implemented-vs-planned boundaries. The principal data flow is:

```text
Complete RGB video + 66-point pose sequence + timestamps
       │
       ├── face / left hand / right hand / upper-body RGB appearance ViTs
       ├── left hand / right hand / upper-body / face graph encoders
       ├── 3-D temporal motion encoder
       └── fingerspelling BiGRU-CTC expert
                    │
          reliability-aware stream fusion
                    │
        multi-scale fine-rate temporal stem
                    │
       coarse global Transformer + local retrieval
                    │
             latent plan / evidence queries
                    │
         T5/FLAN-T5 visual-language bridge
                    │
              English translation
```

The original RGB frame count is retained by the dataset and by the fine-rate temporal and motion pathways. Pose samples are interpolated onto video timestamps only when the bracketing pose interval does not exceed the configured maximum gap; frames themselves are not temporally resampled or truncated. The global temporal branch groups fine features into `ceil(T / coarse_stride)` tokens, but the fine sequence remains available and is used for local detail retrieval. Spatial resizing uses letterboxing, not aspect-ratio distortion.

## Installation

Python 3.10 or newer is required. Install the full optional stack:

```bash
python -m pip install -e ".[all]"
```

Or install only the parts you need:

```bash
python -m pip install -e ".[test,video,vision,text,eval]"
```

`torch` should be installed using the command appropriate for your CUDA version before installing the package if you need a specific CUDA build. The high-capacity research configuration uses `timm` DINOv2-L/14 at 518×518 and Hugging Face FLAN-T5-XL; those weights are downloaded/cached separately and are not included in this ZIP. The cited `timm/vit_large_patch14_dinov2.lvd142m` model card declares CC-BY-NC-4.0; do not use those weights commercially without resolving the licence. Verify all model-weight licences and the terms of the video dataset before use.

## Data contract

The loader deliberately does not guess the binary format of an existing `.pose` file. Use the original ISLTranslate pose parser or your validated pose extractor to export the canonical NPZ schema described in [`docs/data_contract.md`](docs/data_contract.md).

Each NPZ contains:

- `frames`: `[Tv,H,W,3]` RGB uint8 or `[Tv,3,H,W]` RGB float in `[0,1]`.
- `video_timestamps`: `[Tv]` seconds, strictly increasing.
- `pose_xy`: `[Tp,66,2]` normalized original-frame coordinates in `[0,1]`.
- `pose_timestamps`: `[Tp]` seconds in the same clock/time origin as the video timestamps.
- Optional `pose_confidence`, `pose_valid`, `boxes`, `box_valid`, `box_scores`.

The JSONL manifest needs `uid`, `npz`, and `source_id`; supervised training additionally requires `translation`. `signer_id` is strongly recommended and required if signer-disjoint evaluation is enabled. `source_id` must identify the original source video, not a derived clip fragment. See the data contract for exact landmark and ROI ordering.

## Build and validate manifests

Example using a manifest of records that already point to exported NPZ files:

```bash
python scripts/make_splits.py data/all.jsonl --out data/manifests \
  --group-key source_id --ratios 0.8 0.1 0.1 --seed 2026
python scripts/validate_data.py data/manifests/train.jsonl \
  --image-size 518 --crop-size 518 --max-pose-gap 0.20 \
  --out reports/train_data_validation.json
```

For pretraining records without English translations:

```bash
python scripts/validate_data.py data/unlabeled.jsonl --unlabeled
```

Run the converter when you have an RGB video and a pose NPZ:

```bash
python scripts/convert_video_pose_npz.py \
  --video path/to/source_video.mp4 \
  --pose-npz path/to/canonical_pose.npz \
  --out data/npz/clip_0001.npz
```

The converter expects the pose NPZ to contain `pose_xy` and `pose_timestamps` already in the canonical 66-point order. Video presentation timestamps are preserved; pose timestamps must use the same time base. When pose timestamps are clip-relative but video PTS starts at a nonzero offset, pass `--pose-time-offset-seconds` with that offset. It does not extract poses from video or decode an undocumented `.pose` format.

## Self-supervised visual pretraining

The research config initializes specialist appearance backbones from pretrained DINOv2 through `timm`, then supports a SignDINO-inspired global-teacher / local-face-and-hand/body-student pretraining stage on records that do not need translation labels:

```bash
python scripts/pretrain_visual.py \
  --config configs/research.yaml \
  --manifest data/unlabeled.jsonl \
  --epochs 100 --batch-size 32 \
  --out artifacts/isl_mega_sign_ssl.pt
```

The implemented objective is a compact global-to-local self-distillation variant. It is **not** a faithful reproduction of SignDINO's complete augmentation, optimization, distributed training or inference recipe. For strictly inductive test evaluation, pretrain only on training videos or an explicitly approved external unlabeled corpus; do not include held-out test recordings. Reproduce the paper's exact method from its official source if you need a direct baseline.

## Fine-tuning and evaluation

After creating disjoint train/validation/test manifests and the visual pretraining checkpoint:

```bash
python scripts/train.py \
  --config configs/research.yaml \
  --ssl-checkpoint artifacts/isl_mega_sign_ssl.pt
```

Training requires `transformers` and access to the configured FLAN-T5 checkpoint. `--max-steps` is for debugging only and must not be used for reported results.

Evaluate a selected checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/research.yaml \
  --checkpoint runs/isl-mega-xl/last.pt \
  --out runs/evaluation/predictions.jsonl
```

Evaluation writes predictions and metric JSON. BLEU, chrF++ and TER require `sacrebleu`; ROUGE-L additionally requires `rouge-score`. Metric scores do not replace human evaluation by fluent ISL signers.

## Synthetic software smoke test

Synthetic data is for software testing only and must never be presented as research data:

```bash
python scripts/create_synthetic_dataset.py --out synthetic_data --examples 30
python scripts/make_splits.py synthetic_data/all.jsonl --out synthetic_data
isl-mega validate-config --config configs/smoke.yaml
```

The smoke configs point to `synthetic_data/{train,valid,test}.jsonl` (`configs/smoke.yaml` for the v1 baseline contracts; `configs/smoke_v2.yaml` for all 6 v2 mechanisms). Verify group disjointness with:

```bash
isl-mega check-splits --config configs/smoke.yaml
```

For controlled ablations on GPU, compare `configs/ablation_v1_baseline.yaml` against `configs/research.yaml` (full v2 architecture).

## Tests

```bash
pytest -q
```

The automated suites (`tests/test_architecture.py` for v1 contracts and `tests/test_novel_v2.py` for all v2 upgrades) cover tensor shapes, missing modalities, torso-scale pose normalization, cross-articulator graph interactions, continuous-time lag alignment, adaptive event consolidation padding invariance, fingerspelling span routing/MIL, dynamic plan diversity, plan-level OT, and counterfactual visual sensitivity. They do not download or execute actual DINOv2/FLAN-T5 weights.

## Research references

The architecture is a synthesis of public research directions, not a reimplementation of one paper:

- [SignMusketeers — Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1157/)
- [Uni-Sign — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html)
- [SignDINO — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html)
- [SignBind-LLM — ECCV 2026](https://cogvis-cvssp.github.io/papers/signbind/)
- [AttnSign — Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1435/)
- [Think in Latent Thoughts — ACL 2026](https://aclanthology.org/2026.acl-long.454/)
- [ISLTranslate — Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.665/)
- [Continuous Fingerspelling Dataset for ISL — WSLP 2025](https://aclanthology.org/2025.wslp-main.6/)
- [ISLiM — WSLP 2026](https://aclanthology.org/2026.wslp-1.3/)

These papers use different datasets and task settings. Do not compare their reported BLEU values as if they came from one common test set.

## Limitations and publication integrity

1. No actual ISL data, annotation files, pretrained weights or trained checkpoints ship in this repository.
2. The original `.pose` parser must export the documented canonical pose schema; no undocumented binary parser is included.
3. The fingerspelling CTC path currently supports one clip-level character transcript per record. Precise sub-clip fingerspelling spans and timestamped character alignments need an annotation/segmentation extension.
4. Face appearance features do not by themselves constitute validated ISL non-manual grammar recognition or lipreading.
5. The latent-plan attention is inspectable, but this repository does not include human-labelled frame-evidence annotations or a direct spatial grounding loss.
6. The implemented loss combines translation CE with source-aware same-frame RGB↔pose InfoNCE, optional clip-level CTC and optional entropic semantic OT. The OT term is unconstrained and is not a gold sign-to-English alignment. No human frame-grounding supervision or reinforcement-learning attention steering is implemented.
7. The default split utility enforces source disjointness. Signer disjointness must be explicitly requested and is only as reliable as the signer metadata. Deduplication of near-identical translations/video edits requires a separate corpus audit.
8. Human evaluation, inter-annotator agreement, confidence intervals, multi-seed experiments, baseline reproduction and ablations remain essential research work.

**A ZIP cannot guarantee Q1 publication.** A Q1 claim depends on a novel and well-positioned contribution, sound experimental evidence, appropriate comparison, reproducibility, dataset rights, ethical review and the target journal's review decision.
