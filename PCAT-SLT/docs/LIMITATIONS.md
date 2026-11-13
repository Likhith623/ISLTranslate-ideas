# Remaining limitations after the v0.3 audit

This is a scoped resolution record, not a claim that all SLT research is solved. The original v0.2 table is preserved in `history/LIMITATIONS_V2.md`; the supplied review and the complete response are in `history/EXTERNAL_AUDIT_V2.txt` and `AUDIT_RESPONSE.md`.

| Topic | Implemented resolution | What remains |
|---|---|---|
| Asynchronous articulators | Joint plus optional local progress streams; physical/frame binding remains available | Independent-warp invariance cannot preserve every coordination distinction. Local descriptors are not linguistic alignment ground truth. |
| Torso movement vs fixed reference | Dual local/global coordinates, time-aware EMA and explicit motion memory | Camera movement, detector error and meaningful body motion are not always identifiable from these observations. The full model is offline. |
| Quadratic area dimension | Multiple reproducible low-dimensional area sketches with the original clock | Sketches discard information; head count/dimension affect cost and expressivity. Raw detail does not guarantee the decoder uses it. |
| Matrix solves | Solve-free palindromic rotations with exact orthogonality/reversal in real arithmetic; Cayley baseline retained | Different transport family; sparse coupling may be weaker. CPU timings do not establish a GPU/MPS training speedup. |
| Truncated inverse error | Rejected the proposed short Neumann shortcut; numeric counterexample and tests included | Floating-point roundoff still exists in either valid orthogonal implementation. |
| Fixed progress-cell count | Per-example adaptive counts and proper padding masks, plus full admitted detail memory | Counts are capped and thresholded, not differentiable linguistic segmentation. Region-projection scale/noise affects L. Per-example dispatch can add overhead. |
| Early or complete reference dropout | Reliable-reference search; opt-in RGB fallback with absent geometry marked explicitly | No RGB/no reference is rejected. Weak confidence can select a poor reference; pixel fallback is not 3D recovery. Motion tokens are masked for unanchored clips. |
| No pixels | Actual crop arrays, masks, spatial CNN/DINOv2 and routed visual memories | Crop/tracking quality, resolution, occlusion and annotation matter. Visual access is not demonstrated understanding of mouthings, gaze or fingerspelling. |
| Appearance order at zero pose progress | Explicit time/frame positions for visual/motion tokens | Full multimodal output is intentionally not invariant to frame multiplicity/retiming. Frame positions are not word boundaries. |
| Byte-language prior | Optional pretrained T5/mT5 with correct masks, labels, generation, freezing and training | Only local fixture functionality was verified; large models were not fine-tuned on real SLT. Fluent hallucination, domain/language mismatch and adequacy remain empirical issues. |
| Decoder-family differences | Explicit T5/mT5 support; other families rejected | mBART and other token-shifting/language-code conventions require their own verified adapter, rather than an untested generic promise. |
| Monocular depth | Explicit depth-ignore image-plane mode; RGB can supply complementary observations | Hidden 3D configuration is nonidentifiable in general. No arbitrary viewpoint or reconstruction-quality guarantee follows. |
| Confidence/noisy imputation | Preserve confidence/observation masks, bound metric conditioning, mark absent inputs | Neither denoising nor uncertainty calibration nor true missing-motion reconstruction is proved. Confidence changes can influence clock length. |
| Prefix cancellation | Configurable float32/float64 and an independent explicit-clipping backend | Validate actual dynamic ranges. Float64 can be expensive; no finite-precision method is universally exact. |
| Apple/CUDA deployment | Explicit MPS/CPU-integration path, FP32 entry config, conditional hardware test and CUDA AMP implementation | This host exposes no MPS/CUDA for validation. MPS test is skipped; CPU transfers and unsupported non-operator kernels need actual-device testing. |
| RGB/pose alignment | Check common origin and every relative time; explicit frame names/natural sorting and crop contract | Incorrect but mutually consistent timestamps/crop IDs cannot be detected without source provenance or observation review. |
| Data leakage | ID/pose/RGB paths, decoded content, supplied source groups and resume fingerprints checked | Near duplicates, unknown provenance, external pretraining contamination and incomplete group IDs need curation. |
| Real corpus plumbing | Named annotation importers, explicit targets/splits, prepared observation checks | Licensed raw corpora/pose preparation were not supplied for full training. Primitive metadata support is not a claim that every restricted release variant was inspected. CISLR is not SLT. |
| Checkpoint portability | Store active model architecture/weights/tokenizer files; offline reload/resume tests | Initial real-model download, exact fresh-run revision pinning and hardware/library reproducibility remain deployment responsibilities. |
| Resource use | Remove unused T5 encoder blocks, chunk crop encoding, offer sketches/adaptive caps | Full visual/detail cross-attention and pretrained weights remain expensive. Peak loading/training memory and end-to-end accelerator throughput were not measured. |
| Publication and novelty | Primary-source review, claim corrections, matched-ablation plan and executable evidence | No real-corpus SOTA, universal priority or journal/conference acceptance is established. Extra backbones/EMA/rotations are not individually new inventions. |

## Evidence boundaries

The pixel diagnostic uses artificial bar patterns and code labels. Its successful final run demonstrates a functional learning path; it does not measure a sign-language phenomenon. The unsuccessful initial run is retained as well. The spatial representation, normalization and training horizon changed, so it is not a controlled causal ablation of any one fix.

No code change can distinguish two physical situations that provide identical observations but require different labels without additional information or assumptions. Likewise, an invariant representation cannot distinguish every semantic contrast carried by the exact transformation it removes. The architecture therefore offers explicit regimes and observation paths rather than promising mutually incompatible guarantees.

Actual-device tests, licensed real-corpus training, matched strong baselines, visual-grounding/adequacy analysis and qualified-signer review are the outstanding work needed for stronger claims. Those requirements cannot be replaced by a claim of “perfect” implementation.
