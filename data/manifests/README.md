# Canonical Dataset Manifests

Manifests generated during Phase 2 Provenance Audit according to AGENTS.md Part IV:

- `master_manifest.parquet`: Full catalog of all registered samples across ISLTranslate, iSign, GloSL-ISL, ISL-FS-Dense, and ISH-NEWS with media hashes and parent source IDs.
- `source_provenance.csv`: Mapping of every UID to its original parent video, signer, and dataset origin.
- `deduplication_report.csv`: Exact duplicates, near-duplicates, and cross-dataset parent video overlap audit.
- `split_assignments.csv`: Source-disjoint and signer-disjoint split memberships.
- `target.jsonl` / `pretrain.jsonl`: Formatted records for training and evaluation.
