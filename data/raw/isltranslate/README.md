# ISLTranslate Dataset

- **Official Repository**: https://github.com/exploration-lab/isltranslate
- **Paper**: ACL Findings 2023 (https://aclanthology.org/2023.findings-acl.665/)

## Expected Files

- `ISLTranslate.csv`: Sentence/phrase annotations with columns `uid`, `text`, etc.
- `ISL-signer_validation.csv`: 291 human-validated translation pairs.
- `videos/`: Original continuous RGB signing clips (`{uid}.mp4`).
- `poses/`: 13 MediaPipe Holistic pose archives or uncompressed `{uid}.pose` files.

## Instructions
Place the uncompressed pose archives or individual `.pose` files in `poses/` and RGB clips in `videos/`.
