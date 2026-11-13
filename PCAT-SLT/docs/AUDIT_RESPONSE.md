# Response to the supplied external audit

Reviewed against the implementation on **14 November 2026**. The original review is preserved in `history/EXTERNAL_AUDIT_V2.txt`. Its identified gaps are useful, but its proposed numerical shortcut and promised outcomes are not all valid.

## Finding-by-finding decisions

| Review finding | Decision and implemented remedy | Limit of the remedy |
|---|---|---|
| A1: One common clock cannot represent asynchronous signing | A common parameterization does not require simultaneous articulator events. Added **local articulator progress streams alongside the joint stream** and explicit time-bound frame/visual memories. | Full invariance to every independent retiming remains incompatible with preserving all coordination. It is not claimed or desirable as a universal constraint. |
| A2: A fixed reference creates drift | Fixed coordinates preserve actual torso motion and are not inherently a numerical defect. Added **dual coordinates**: time-aware EMA local frames, explicit motion tokens and the original global detail path. No hard-coded MediaPipe shoulder indices. | EMA cannot distinguish every camera/detector artifact from meaningful body motion; the full model remains offline. |
| A3: N-by-N areas become expensive | Added fixed orthonormal **multi-head area sketches**, all evaluated with the original common clock. Dense statistics remain a baseline. | A sketch loses some area information. Raw canonical input is retained by the configured detail branch; no universal injectivity claim. |
| A4: Noisy monocular depth invalidates physical 3D claims | Added an explicit `depth_policy: ignore` for a planar image-coordinate mode, alongside the existing 3D mode. Recommended monocular configuration uses it. | This avoids relying on arbitrary depth; it does not recover hidden 3D motion or confer arbitrary camera-view invariance. |
| B1: Missing pixels/mouth/eye texture | Implemented **actual aligned RGB crop input**, masks, aspect-preserving preprocessing, spatial-grid CNN encoding and optional pretrained DINOv2 encoding. Crops route to articulators and decoder memories. | Access to pixels is not proof of recognizing every mouthing or gaze distinction. Crop quality, visual resolution and supervision matter. |
| B2: Byte decoder lacks language priors | Implemented a **pretrained T5/mT5 decoder adapter**, including proper source/target masks, target-token collation, optional freezing, generation, training/evaluation and portable tokenizer assets. Unused text-encoder transformer blocks are removed. | The adapter does not guarantee a grammar/adequacy improvement or a BLEU increase. Byte decoding remains a controlled offline baseline. Other model families are rejected rather than guessing their token-shifting rules. |
| B3: Fixed K overcompresses long clips | Added **per-example bounded adaptive cell counts**, padding-safe attention and full observed-frame detail memory. | Geometric progress is not word segmentation; the count is capped, thresholded and nondifferentiable. Finite source/target limits remain explicit. |
| C1: Exact Cayley solves may be expensive | Added a **solve-free palindromic Givens action**, which preserves orthogonality by construction. Exact Cayley remains selectable and its vector action now uses one right-hand side instead of constructing the full matrix. | It is a different orthogonal map, not an approximation guaranteed to reproduce Cayley or improve SLT. CPU timing is reported; GPU speed is not inferred. |
| C2: MPS numerical/kernel issues | Added an explicit MPS route with **differentiable CPU integration** and FP32 configuration, avoiding reliance on an unsupported device solve or device float64. | Hardware validation is unavailable here; the actual MPS test is skipped and no blanket compatibility/latency claim is made. |
| Upgrade 5: Real corpora rather than toy-only plumbing | Added corpus annotation importers for **PHOENIX-2014T, ISLTranslate and CSL-Daily**, preserving supplied targets/splits and requiring existing observation files. | Licensed raw data, landmark preparation and real-corpus training are not fabricated. CISLR is not a sentence-translation benchmark. |

## Why the suggested Neumann shortcut was not adopted

For A=[[0,1/2],[-1/2,0]], the Frobenius norm is less than one. The review's expression is

P=(I+A+A²+A³)(I+A)=(I-A⁴)Q=(15/16)Q,

where Q is exact Cayley. Thus PᵀP=(225/256)I: its diagonal orthogonality defect is **31/256 = 0.12109375** despite satisfying the convergence assumption. Convergence of an infinite series does not make a short truncation orthogonal or accurate enough. This expression is a truncated Neumann expansion, not a general guarantee from a Padé approximation.

The replacement rotates disjoint coordinate planes in a palindromic schedule. Each plane rotation is orthogonal; their product remains orthogonal and has the correct sign-reversal inverse. It avoids an LU solve entirely while retaining a mathematically defensible property. See `../ARCHITECTURE.md` and `../reports/transport_benchmark.json` for the precise definition and measurements.

## Claims in the review that need correction or evidence

- The ISLTranslate paper describes approximately **31k** sentence/phrase pairs, not approximately 3k. Its public CSV has `uid,text` columns. [Primary paper](https://aclanthology.org/2023.findings-acl.665/), [official metadata](https://raw.githubusercontent.com/Exploration-Lab/ISLTranslate/main/data/ISLTranslate.csv).
- CISLR is **word-level recognition**, according to its authors. Treating it as continuous sentence translation would confound evaluation. The importer explicitly rejects that substitution. [Authors' repository](https://github.com/Exploration-Lab/CISLR).
- The universal “over 30%” lexical/mouthing statement in the supplied review was not substantiated by a primary source. It is not used as a project claim. Visual nonmanual information is relevant, but its prevalence and linguistic role depend on the language and dataset.
- “+15–20 BLEU”, “3× faster GPU training”, “full MPS support”, “solves mouthings” and “turns it into a Q1 paper” are not consequences of these code changes. They require controlled experiments, hardware evidence and independent review.
- Physical 3D similarity invariance is conditional on the coordinate representation. It never implied recovery of ground-truth 3D from ambiguous 2D observations.

## Additional defects found in the follow-up audit

1. **Zero-progress appearance order loss.** Static pose and zero initial fusion gain can give every visual token the same progress position. Appearance could then become an unordered set. Added explicit frame/physical-time binding and a regression test showing reversed RGB order changes the model when pose progress is zero.
2. **Constant RGB timestamp offsets.** Comparing only rebased times accepted equal interval patterns with different clip origins. The loader now checks both the shared origin and all relative sample times. Both local perturbation and constant-offset cases are tested.
3. **Weak spatial/scale calibration.** An initial globally pooled CNN diagnostic stayed at 50% on pixel-only labels. Added configurable spatial-grid pooling and visual-memory normalization. The revised diagnostic learns the simple task, with the failed result retained. This is not a controlled architecture comparison: training horizon and representation changed.
4. **Complete pose-reference dropout.** Added opt-in unanchored RGB fallback. Canonical geometry is marked absent rather than fabricated, and a sample with neither usable pose nor observed pixels is still rejected.
5. **Checkpoint dependencies on local model files.** Text/visual architecture snapshots, complete weights and tokenizer files are now carried by format-3 checkpoints. Tests remove original model files and reload in offline mode, including a pretrained-decoder training/resume test.
6. **Source leakage through visual inputs.** Added duplicate RGB-path checks and cross-split decoded-RGB content checks, in addition to pose/ID/source-group checks.
7. **One-dimensional rotation edge case.** Fixed empty plane schedules while preserving a valid zero derivative; covered across several dimensions and by numerical gradient checks.
8. **Absolute timestamp precision.** Epoch-valued float64 times could collapse when cast to float32 before rebasing. Rebase first in both integrators and all time-bound memories; a full dual-frame/RGB regression test compares large-origin and relative times. Timing-enabled models now require explicit seconds even without detail memory.
9. **Unnecessary dense Cayley construction.** The exact baseline now solves only for the value vector; benchmark comparisons use that optimized baseline, not an artificially expensive full-matrix action.
10. **Fallback motion validity and control interactions.** Motion memories are masked when the pose reference is absent. Removing pixels in an ablation also disables RGB-only fallback; all 18 generated controls are tested for valid configurations.

## Primary implementation references

The text adapter follows the external encoder-memory interface documented for [T5](https://huggingface.co/docs/transformers/model_doc/t5); the visual adapter uses documented [DINOv2 hidden states](https://huggingface.co/docs/transformers/model_doc/dinov2). MPS float64 restrictions are explicit in the [PyTorch MPS implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/OperationUtils.mm). These references establish interfaces/constraints, not performance gains.

CSL-Daily's provider requires its data-access agreement; no email, agreement submission or unauthorized dataset acquisition was performed. [Official release instructions](https://ustc-slr.github.io/datasets/2021_csl_daily/).
