# Implementation status and pending verification

The package implements revisioned acquisition, source-held-out splits, pose/RGB
features, training/resume, reference-free prediction, evaluation and explicit
experiments. See the [runbook](runbook.md) for setup and execution.

## Local verification

CPU regression and integration results will be recorded after this change is tested.
The committed lock is for project-owned environments; managed installations retain
the host torch build and record their actual inventory separately.

## External inputs still required

- Actual ISLTranslate annotations, matching pose files, exact revisions and authoritative parent-source mappings.
- The interpreter-reference table with verified column mapping.
- Prepared generic T5 assets with a resolved commit and checksums.
- For transfer: additional-iSign annotations/poses, revision and verified cross-dataset identities.
- For context: verified timing, continuity, source duration where needed and actual requested coverage.
- For RGB: videos, pose-to-video transforms, synchronization metadata and the declared ResNet18 initializer.
- For optional linguistic metrics/labels: local WordNet, a declared local spaCy pipeline or a local BERTScore model/revision as configured.
- A compatible ROCm host and successful strict FP32/BF16 probes on that host.

## Target-host acceptance still pending

Run strict FP32 env-check, inspect a real pose, audit source provenance and frozen
memberships, prepare/verify caches and model assets, then run the paired two-epoch
FP32/BF16 validation pilot. Compare quality, loading-inclusive throughput and memory
before choosing BF16. The CPU suite does not establish AMD kernel compatibility,
real-data translation quality or GPU performance. No GPU results are claimed here.

Notebook launch/interruption, real optional RGB/context/transfer paths and optional
language resources need checks on the actual deployment environment. Keep revisions,
resolved configs, package/backend inventory, membership/checkpoint checksums,
decoding policy and raw reports with every experiment.
