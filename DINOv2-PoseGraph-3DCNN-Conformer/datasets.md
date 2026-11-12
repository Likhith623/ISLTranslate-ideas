# Dataset Research and Access Guide

This project requires continuous Indian Sign Language (ISL) video, English translation text, timestamps, and pose annotations that can be converted into the ISL-MEGA contract. The model does **not** download or consume these datasets directly. Every source must be legally acquired, provenance-audited, converted to RGB frames plus the 66-landmark schema, and split by source video and signer where metadata permits.

## Recommended dataset stack

| Dataset | Official access | What it provides | Fit for ISL-MEGA | Access and license notes |
|---|---|---|---|---|
| **ISLTranslate** | [Official GitHub](https://github.com/Exploration-Lab/ISLTranslate) · [ACL 2023 paper](https://aclanthology.org/2023.findings-acl.665/) | Continuous ISL-English sentence/phrase pairs; source videos and annotation resources are described by the project | **Primary translation corpus.** Requires video/pose acquisition and conversion to RGB frames, timestamps, and 66 points | Confirm the current repository/download instructions and the license for each redistributed artifact before use. The paper is citation-accessible, but paper access is not equivalent to video redistribution permission. |
| **iSign v1.1** | [Hugging Face dataset page](https://huggingface.co/datasets/Exploration-Lab/iSign) · [Project page](https://exploration-lab.github.io/iSign/) · [ACL/arXiv paper](https://arxiv.org/abs/2407.05404) | More than 118K ISL video/translation pairs and several ISL processing tasks, including SignVideo2Text and SignPose2Text | **Best scale-up corpus.** Requires gated access, video/pose download, provenance normalization, and conversion | Access may require Hugging Face account/gating approval. Record the exact release, acceptance of terms, and downloaded file hashes. Do not mix its source videos with ISLTranslate without duplicate/source overlap analysis. |
| **ISL Fingerspelling** | [Hugging Face dataset](https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling) · [Project page](https://kirandevraj.github.io/ISL-Fingerspelling/) · [GitHub tools](https://github.com/Kirandevraj/ISL-Fingerspelling-Dense) · [WSLP 2025 paper](https://aclanthology.org/2025.wslp-main.6/) | 1,308 manually annotated segments, character/frame boundaries, signer-independent split metadata, and weakly labelled iSign-derived segments | **Auxiliary fingerspelling corpus**, not a full sentence-translation corpus. Use for CTC/spotting pretraining or held-out fingerspelling evaluation | Dataset card reports CC-BY-NC-4.0. The automatic iSign-derived segments point back to iSign and do not redistribute the underlying videos. Respect the separate signer split and do not leak it into sentence-level test evaluation. |
| **GloSL-ISL** | [Hugging Face dataset](https://huggingface.co/datasets/pedroodb/glosl-isl) · [GloSL package](https://pypi.org/project/glosl/) | ISL-English annotations and 543-point MediaPipe pose files with source/timestamp metadata | **Pose-only auxiliary/resource corpus.** It is not directly compatible with the RGB-required dataset loader; it requires a verified 543-to-66 mapping and synchronized RGB access | Dataset card reports CC-BY-NC-4.0. The published pose files use MediaPipe's 543-point format and pixel/depth conventions, so normalization and landmark mapping must be tested before use. |
| **ISH-NEWS** | [Scientific Reports record](https://www.nature.com/articles/s41598-026-60893-0) | ISL news videos/annotations described by the publication | **Potential external evaluation or fingerspelling source**, subject to access | The Nature page may require institutional/cookie authorization. Raw broadcast video may be restricted and must not be redistributed without permission. Treat this as unavailable until the authors' official release and terms are verified. |

## Paper and benchmark references

These are useful for dataset definitions and baseline comparison, but they are not dataset download links:

- [ISLTranslate: Dataset for Translating Indian Sign Language](https://aclanthology.org/2023.findings-acl.665/)
- [iSign: A Benchmark for Indian Sign Language Processing](https://arxiv.org/abs/2407.05404)
- [Continuous Fingerspelling Dataset for Indian Sign Language](https://aclanthology.org/2025.wslp-main.6/)
- [Dense Frame Annotations for Low-Resource ISL Fingerspelling Recognition](https://github.com/Kirandevraj/ISL-Fingerspelling-Dense)

## Compatibility contract

Each training record must ultimately point to an NPZ containing:

- `frames`: `[T,H,W,3]` uint8 or `[T,3,H,W]` floating point in `[0,1]`;
- `video_timestamps`: finite, strictly increasing timestamps of length `T`;
- `pose_xy`: `[Tp,66,2]` normalized coordinates in `[0,1]`;
- `pose_timestamps`: finite, strictly increasing pose timestamps;
- optional `pose_valid`, `pose_confidence`, `boxes`, `box_valid`, `box_scores`, and frame-level fingerspelling spans.

Manifest records require unique `uid`, nonempty `source_id`, translation text for supervised training, and signer metadata for signer-independent evaluation. See [docs/data_contract.md](docs/data_contract.md).

## Required acquisition and conversion procedure

1. Record the official URL, release/version, access date, license, terms, and SHA-256 hash of every downloaded archive.
2. Preserve original dataset IDs, parent video IDs, signer IDs, sentence IDs, and absolute time intervals.
3. Detect duplicate or overlapping source videos before combining datasets. Dataset names do not establish independence.
4. Decode RGB frames with their native timestamps. Do not silently resample away timing information.
5. Convert source pose formats explicitly. MediaPipe 543-point data cannot be treated as the project's 66-point schema without a documented landmark map and validation report.
6. Normalize coordinates using the source frame dimensions and preserve validity/confidence masks. Reject malformed values rather than clipping them into apparently valid annotations.
7. Convert fingerspelling character intervals into relative seconds or frame-aligned boolean spans only after confirming the source FPS/timebase.
8. Generate source-disjoint train/validation/test manifests. Use signer-disjoint splits when complete signer metadata exists.
9. Store manifest hashes and conversion metadata beside every run. Keep the final test manifest sealed until model selection is complete.

## What is currently runnable

The repository currently contains a synthetic-data generator for software smoke tests. Synthetic samples are not research evidence. No external dataset becomes runnable merely because its link appears here; it must pass the conversion and validation procedure above and be represented by a project-compatible manifest.

## Reproducibility record

For every reported experiment, archive:

- dataset name, release, URLs, license/permission status, and file hashes;
- conversion commit and landmark mapping version;
- manifest hashes and split diagnostics;
- signer/source overlap audit;
- model/config/tokenizer/pretrained-weight identifiers;
- GPU, precision, seed, package lock, predictions, and evaluation metric signatures.
