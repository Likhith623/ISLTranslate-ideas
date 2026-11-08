# iSign v1.1 Dataset

- **Official Dataset**: https://huggingface.co/datasets/Exploration-Lab/iSign
- **Website**: https://exploration-lab.github.io/iSign/
- **Paper**: ACL Findings 2024 (https://arxiv.org/abs/2407.05404)
- **License**: CC-BY-NC-SA-4.0

## Archive Reconstruction

The dataset is partitioned into split byte parts:

- Video parts: `iSign-videos_v1.1_part_aa`, `iSign-videos_v1.1_part_ab`
  Reconstruct:
  ```bash
  cat iSign-videos_v1.1_part_* > iSign-videos_v1.1.zip
  ```

- Pose parts: `iSign-poses_v1.1_part_aa` to `part_ad`
  Reconstruct:
  ```bash
  cat iSign-poses_v1.1_part_* > iSign-poses_v1.1.zip
  ```

## UID Convention
All examples follow `{video_id}-{sequence_number}`. All splits must group by `video_id` to strictly prevent parent-video data leakage.
