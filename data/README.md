# Data Directory Architecture

This directory hierarchy implements the data management and provenance specification defined in **AGENTS.md** (Phase 2: Dataset Acquisition and Provenance Audit).

```text
data/
├── raw/
│   ├── isltranslate/
│   │   ├── videos/             # Raw MP4 video files ({uid}.mp4)
│   │   ├── poses/              # MediaPipe Holistic pose archives ({uid}.pose)
│   │   ├── ISLTranslate.csv    # Official sentence-level translation annotations
│   │   └── ISL-signer_validation.csv # 291 human-validated evaluation pairs
│   │
│   ├── isign/                  # iSign v1.1 multipart video and pose archives
│   ├── glosl-isl/              # GloSL-ISL 543-point holistic poses & annotations
│   ├── isl-fingerspelling/     # ISL-FS-Dense gold and discovered segments
│   └── ish-news/               # ISH-NEWS 2026 annotations and metadata
│
├── manifests/                  # Canonical manifests (Parquet / CSV / JSONL)
├── processed/                  # Preprocessed tensors, ROI crops, and features
└── cache/                      # Extraction caches and runtime buffers
```

All external dataset locations and paths are configurable via `configs/datasets.yaml` or environment variables (`ISLTRANSLATE_ROOT`, `ISIGN_ROOT`, etc.).

