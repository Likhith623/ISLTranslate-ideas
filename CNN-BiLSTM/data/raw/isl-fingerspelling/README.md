# ISL-FS-Dense Fingerspelling Dataset

- **Official Project**: https://kirandevraj.github.io/ISL-Fingerspelling/
- **Code Repository**: https://github.com/Kirandevraj/ISL-Fingerspelling-Dense
- **Hugging Face**: https://huggingface.co/datasets/kirandevraj/ISL-Fingerspelling
- **Paper**: ACL WSLP 2025 (https://aclanthology.org/2025.wslp-main.6/)

## Expected Files
- `manual_annotations.csv`: 1,308 manually annotated segments with letter-level timestamps (gold supervision).
- `signer_split.csv`: Official signer-independent train/dev/test split.
- `automatic_segments.csv`: 11,700+ discovered segments referencing iSign parent videos (weak supervision only).
- `segments/`: Extracted fingerspelling video clips and crops.

