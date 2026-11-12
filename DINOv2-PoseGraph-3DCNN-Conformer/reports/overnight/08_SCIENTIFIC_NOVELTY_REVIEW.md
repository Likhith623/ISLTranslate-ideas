# Scientific Novelty and Prior-Art Investigation (2020–November 2026)

**Evaluation Scope:** Critical Q1 journal peer review (IEEE TPAMI / IJCV / ACL-grade scrutiny) of both:
1. **Implemented `ISL-MEGA` (v0.2.0)** in [`src/isl_mega/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/), [`configs/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/configs/), and [`scripts/`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/scripts/); and
2. **Proposed `CARP-S2T` redesign & rejected `EvoHARE-Net` concept** in [`reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md), [`reports/04_FINAL_ARCHITECTURE_SPECIFICATION.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/04_FINAL_ARCHITECTURE_SPECIFICATION.md), [`reports/05_IMPLEMENTATION_BLUEPRINT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/05_IMPLEMENTATION_BLUEPRINT.md), [`reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md), and [`reports/07_EXECUTIVE_DECISION_REPORT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/07_EXECUTIVE_DECISION_REPORT.md).

**Audit Baseline:** Independently cross-verified against primary literature (2020–November 2026), official ACL Anthology / CVF / ICLR / ECVA / IEEE / Nature / arXiv records, the repository's existing internal critique ([`02_CRITICAL_MATHEMATICAL_AND_ARCHITECTURAL_ISSUES.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/phase_1_verification/02_CRITICAL_MATHEMATICAL_AND_ARCHITECTURAL_ISSUES.md)), and prior overnight audits ([`01_COMPLETE_REPOSITORY_INTELLIGENCE.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/01_COMPLETE_REPOSITORY_INTELLIGENCE.md) through [`07_HARDWARE_AND_PERFORMANCE_REPORT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/07_HARDWARE_AND_PERFORMANCE_REPORT.md)).

---

## Executive Q1 Reviewer Verdict

> **Bottom-Line Peer-Review Judgment:** Neither **implemented `ISL-MEGA` (v0.2.0)** nor **proposed `CARP-S2T`** is defensible as a Q1 journal contribution in its current framing. `ISL-MEGA` is an un-benchmarked 10-stream architectural synthesis that combines independently published mechanisms (`SignDINO`-style region-to-global SSL, ST-GCN + cross-articulator pose attention, CTC fingerspelling, quality gating, Conformer, latent plan slots, entropic OT, and counterfactual visual margin regularization) across 9 loss terms, several of which either oppose one another (uniform-entropy regularization vs. stream specialization) or overlap directly with 2025–2026 published work (`TransAlign-SLT`, `SignBind-LLM`, `Think in Latent Thoughts`, `Uni-Sign`). Conversely, `CARP-S2T` exists as zero lines of code and rests on three unsound claims: (1) a mathematically invalid transplantation of `Trusted Multi-View Classification` (`TMC`) Dirichlet classification equations onto non-mutually-exclusive multimodal regression without ground-truth reliability targets; (2) naive horizontal hand mirroring that destroys dominant vs. base linguistic asymmetry in ISL; and (3) an unproven assertion that a pure-PyTorch or CUDA Bidirectional Mamba (`Bi-S6`) automatically outperforms a stride-2 Conformer on typical 90–180 frame sign sentences.

Four decisive facts govern scientific positioning:

1. **Zero Empirical Grounding in Repo:** Both [`README.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/README.md) and [`docs/research_protocol.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/docs/research_protocol.md) openly state that `ISL-MEGA` has zero real videos, zero real poses, zero trained checkpoints, and zero empirical runs; earlier novelty documents ([`reports/03`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md)–[`reports/07`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/07_EXECUTIVE_DECISION_REPORT.md)) additionally misattributed dates, tasks, and metrics across key 2024–2026 baselines (`SignBind-LLM`, `SignMamba`, `SignDAGC`, `TLAS`, `ISLTranslate`).
2. **Modular Composition vs. New Mechanism:** Swapping 3 ViT-Large backbones for LoRA-tuned DINOv2-Base, Conformer for `Bi-S6`, and sigmoid quality gating for `Dirichlet` evidence weights yields modular engineering rather than a new computational principle—especially when `Bi-S6`, `Region-LoRA`, and `DirichletEvidentialFusion` have not been benchmarked or even written.
3. **Genuine Unsolved Niche Exists—But Narrower:** Low-resource gloss-free Indian Sign Language translation (`iSign` / `ISLTranslate`) under real-world sensor degradation (tracking jitter, hand self-occlusion, variable frame rate, blurry low-bitrate YouTube frames) remains wide open (`ISLiM` reaches only 5.85 BLEU-4 on `iSign`; `iSign` baselines reach 0.56–1.47 BLEU-4).
4. **Constructive Path Forward:** Abandon `EvoHARE-Net`, drop the invalid simplex-classification Dirichlet equation from `CARP-S2T`, fix the source-code pose-slice/redundancy bugs audited in Reports 03–06, and test one sharp, falsifiable hypothesis uniting **provenance-aware observable quality calibration (`QMF`-grounded log-odds reliability)**, **36-landmark cross-articulator kinematic graphs**, and **stream-targeted counterfactual invariance** evaluated under corruption retention and meaning-preserving (`Beyond BLEU`) metrics.

---

## 1. Bibliographic & Task-Boundary Audit of Prior Repo Claims

Preliminary repository strategy documents ([`reports/03`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md), [`reports/06`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md), and [`reports/07`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/07_EXECUTIVE_DECISION_REPORT.md)) contain multiple factual inaccuracies, task conflations (Recognition vs. Translation vs. Generation vs. Gloss-to-Text Streaming), and unverified expectations that would be immediately challenged by Q1 referees. Table 1 audits every discrepancy against primary proceedings.

| System / Benchmark | Claim in Prior Repo Reports (`reports/03`, `06`, `07`) | Verified Primary Fact (2020–Nov 2026) | Critical Conflation / Methodological Risk |
| :--- | :--- | :--- | :--- |
| **`SignBind-LLM`** | Cited as `ECCV 2024` in [`reports/03` L14, L20](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md#L14-L20) and Tier-3 WMT-SLT baseline. | Thomas, Fish & Bowden (`arXiv:2509.00030`), accepted at **ECCV 2026**; evaluated on *How2Sign*, *BOBSL*, *ChicagoFSWild+*. | Wrong publication year (`2024` vs `2026`) and misattributed dataset (`WMT-SLT`). |
| **`SignDAGC`** | Cited as `(2024–2025)` Tier-2 graph baseline in [`reports/03` L22](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md#L22) and [`reports/06` L31](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md#L31). | Hu et al., *Pattern Recognition* Vol. 179 (**2026**), DOI `10.1016/j.patcog.2026.113947`; targets **CSLR and SLT** on *PHOENIX14/14T*, *CSL*, *CSL-Daily*. | Wrong publication year and requires clear distinction between CSLR vs. SLT tables. |
| **`SignMamba` / `Sign-Mamba`** | Described as `(2024–2025)` full-frame RGB **CSLT** baseline in [`reports/03` L19](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md#L19) and [`reports/06` L48](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md#L48). | Splits across three distinct papers: (a) **SignMamba** (Yang et al., *Neural Networks* **2026**, `10.1016/j.neunet.2026.109530`) is **CSLR** (gloss recognition); (b) **Sign-Mamba** (Feng et al., *ICASSP* **2025**, `10.1109/ICASSP49660.2025.10890373`) is **SLP** (sign generation); (c) **EvSLT** (Wang et al., `arXiv:2408.10488`, **2024**) uses Mamba for **event-camera stream** SLT. | Severe **task conflation** between CSLR, SLP, and Event-SLT vs. RGB+pose CSLT. |
| **`TLAS` vs. `TransAlign-SLT`** | Grouped together in [`reports/03` L25](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md#L25) as optimal-transport + visual-grounding models. | **TransAlign-SLT** (Krishna et al., *WSLP* **2026**, `2026.wslp-1.21`) uses TranslationCTC + Masked Entropic OT over BERT on *PHOENIX14T* & *CSL-Daily*. **TLAS** (Kermani et al., *ALVR @ ACL* **2026**, `2026.alvr-main.21`) is a **streaming gloss-to-text** scheduler (TPD + LRE + AFG) on *ASLG-PC12* discourse. | `TLAS` does **not** perform video-to-text visual grounding or optimal transport. |
| **`ISLTranslate` baseline** | Cited in [`reports/06` L44](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md#L44) as `CNN-BiLSTM (ACL 2023)`. | Joshi et al., *Findings ACL* **2023** (`2023.findings-acl.665`, 31,222 pairs) reports its **Pose-SLT** Transformer baseline in Table 4 (**BLEU-4 = 6.09**, ROUGE-L = 12.91); 48.93 BLEU-4 is human-reference agreement validation on a subset. | Must specify **Pose-SLT (6.09 BLEU-4)** vs. subset validation (48.93) and dataset overlap/leakage caveats. |
| **`GloSL-ISL`** | Treated in [`reports/06` L25](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md#L25) and [`reports/07` L68](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/07_EXECUTIVE_DECISION_REPORT.md#L68) as an independent `100k` ISL benchmark alongside `iSign` and `ISLTranslate`. | `glosl-isl` (curated on Hugging Face by Pedro Dal Bianco) is a **standardized preprocessing/packaging wrapper** built over underlying ISL corpora (*ISLTranslate* / *iSign*), not an independent raw dataset release. | Benchmarking both `GloSL-ISL` and `ISLTranslate`/`iSign` without deduplication guarantees train/test cross-release contamination. |
| **Predicted Ablation Gains (`A1`–`A12`) & Speedups** | Hard numeric assertions (`-2.4` BLEU, `2.8x` speedup, `65%` VRAM reduction, `>=75%` retention, `p < 0.01`) presented in [`reports/06` L74-L89](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md#L74-L89). | Purely **hypothetical projections**; zero real-data runs exist (`ISL-MEGA` is untested on real data, `CARP-S2T` has 0 lines of code). | Framing fabricated projections as expected proof triggers immediate desk rejection. |

> **Metric Non-Comparability Rule:** Published BLEU-4 figures across benchmarks are **strictly non-comparable** due to domain size, signer counts, annotation quality, capitalization/tokenization, and spoken-language priors: e.g., **SignDINO** reaches `25.46` BLEU-4 on *CSL-Daily* and `15.47` on *How2Sign*; **TransAlign-SLT** reaches `23.54` on *PHOENIX-14T*; **SignBind-LLM** reaches `23.1` on *How2Sign* and `7.0` on *BOBSL*; whereas **iSign** baselines reach `0.56–1.47` and **ISLiM** reaches `5.85` on *iSign*, while **ISLTranslate** Pose-SLT reaches `6.09` on *ISLTranslate*. Direct cross-benchmark BLEU tables are scientifically invalid.

---

## 2. Primary Literature Profiles (2020–November 2026)

Every key verified paper is profiled across the 12 mandatory Q1 dimensions. Systems verified strictly as Recognition (CSLR/ISLR), Generation (SLP), or Gloss-to-Text are tagged explicitly to forbid task conflation.

### 2.1 Directly Competing & Recent Sign-Language Systems (2024–2026)

#### P1. `SignDINO` (CVPR 2026 Highlight)
- **Exact Title:** Learning Effective Sign Features without Text for Gloss-free Sign Language Translation
- **Authors:** Shiwei Gan, Xiao Liu, Yafeng Yin, Nan Liu, Kuizhuang Liu, Desibieer Tuerdaken, Zhiwei Jiang, Lei Xie, Sanglu Lu, Hongkai Wen
- **Year:** 2026
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2026 Highlight)
- **DOI or Official URL:** [CVF Open Access Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html)
- **Architecture:** Student–teacher self-supervised ViT visual backbone trained from raw video frames prior to downstream gloss-free SLT encoder–decoder.
- **Main Mechanism:** Decoupled text-free pretraining: EMA teacher encodes full *global* sign frames while the student predicts teacher representations from *masked local views* cropped around hands and face; downstream inference uses only global frames.
- **Datasets:** *CSL-Daily*, *How2Sign* (gloss-free SLT; also standard benchmark suite comparisons).
- **Evaluation Protocol:** Gloss-free SLT evaluation reporting BLEU-1..4 and ROUGE-L (`CSL-Daily`: ~25.46 BLEU-4, 52.75 ROUGE-L; `How2Sign`: ~15.47 BLEU-4, 36.14 ROUGE-L).
- **Overlap with ISL-MEGA:** Directly inspired `SignAwareDistillation` in [`src/isl_mega/pretraining.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pretraining.py#L32-L108) (teacher on full body view, student on local hand/face crops) and motivates SSL in `vision.py`.
- **Differences:** SignDINO drops cropped local views at downstream inference (using global-only inputs) and uses no 66-point pose graph, no 10-stream reliability fusion, and no fingerspelling CTC head; `ISL-MEGA` retains all 4 RGB crops at inference (`separate_hand_face_body=True`).
- **Implications for Novelty:** **High overlap.** Global-to-local hand/face self-distillation for gloss-free SLT is established prior art; `ISL-MEGA`'s `pretraining.py` cannot be claimed as an algorithmic contribution unless improved fundamentally.

#### P2. `SignDino` (Temporal-Axis Self-Distillation Preprint, Sept 2026)
- **Exact Title:** SignDino: Self-Supervised Sign Language Representation Learning via Temporal-Axis Self-Distillation
- **Authors:** Junyi Hu, Zhewen He, Haomian Huang, Zhenhua Li, Zhifei Li, Yi Fang
- **Year:** 2026 (September 2026)
- **Venue:** arXiv preprint (cs.CV)
- **DOI or Official URL:** [arXiv:2609.06296](https://arxiv.org/abs/2609.06296)
- **Architecture:** YOLOv8n + ByteTrack cropper extracting left-hand, right-hand, and face tracks -> frozen **DINOv3 ViT-B/16** per-frame embeddings -> lightweight temporal Transformer student/EMA teacher.
- **Main Mechanism:** Shifts DINO/DINOv3 self-distillation from spatial patches to the *temporal axis* of localized articulator tracks, combining temporal self-distillation, frame-level masked token prediction (iBOT-style), KoLeo spreading, and frame-to-frame Gram anchoring over frozen ViT-B/16 features.
- **Datasets:** Sign-to-English translation, isolated sign recognition, and fingerspelling detection benchmarks.
- **Evaluation Protocol:** Frozen visual backbone evaluation across SSL downstream probing and end-to-end translation/fingerspelling metrics.
- **Overlap with ISL-MEGA:** Uses almost the identical design pattern recommended in `CARP-S2T` (`frozen DINOv2-Base` over hand/face crops + lightweight temporal model) and reinforces why freezing a DINO ViT-B over detector tracks works without training 3x ViT-Large backbones (`912.95M` params in `ISL-MEGA` `research.yaml`).
- **Differences:** Uses temporal-axis distillation and DINOv3 without 66-point ST-GCN or observable quality-aware fusion.
- **Implications for Novelty:** **Critical risk to visual novelty.** Demonstrates that frozen DINO-B over tracked hand/face crops + temporal adaptation is already explored directly in 2026 sign SSL literature.

#### P3. `SignBind-LLM` (ECCV 2026)
- **Exact Title:** SignBind-LLM: Multi-Stage Modality Fusion for Sign Language Translation
- **Authors:** Marshall Thomas, Edward Fish, Richard Bowden
- **Year:** 2026 (`arXiv` Sept 2025)
- **Venue:** European Conference on Computer Vision (ECCV 2026)
- **DOI or Official URL:** [arXiv:2509.00030](https://arxiv.org/abs/2509.00030) / [Project Page](https://cogvis-cvssp.github.io/papers/signbind/)
- **Architecture:** Three modular visual experts (continuous signing, fingerspelling, lipreading) + lightweight temporal alignment Transformer + pretrained LLM decoder.
- **Main Mechanism:** Multi-stage training where each specialist expert is independently pretrained with CTC over millions of automatically generated pseudo-gloss/mouthing/character sequences, before a learned temporal-alignment Transformer binds parallel token streams into the LLM.
- **Datasets:** *How2Sign*, *BOBSL*, *ChicagoFSWild+*.
- **Evaluation Protocol:** BLEU-4 for continuous SLT (*How2Sign* up to `23.1`, *BOBSL* up to `7.0`) and Letter Accuracy for fingerspelling (*ChicagoFSWild+* up to `73.6%`).
- **Overlap with ISL-MEGA:** Shares the core multi-expert decomposition (continuous body/sign stream + dedicated fingerspelling CTC stream + facial/mouth stream + temporal alignment + pretrained seq2seq/LLM decoder).
- **Differences:** SignBind-LLM operates on RGB appearance experts trained on large-scale pseudo-gloss CTC corpora (ASL/BSL), whereas `ISL-MEGA` relies on end-to-end synthetic-ready multi-stream training with 66-pt ST-GCN pose + 5-descriptor observable stream quality gating on ISL.
- **Implications for Novelty:** **Strong novelty erosion for multi-expert decomposition.** Simply combining continuous-signing + fingerspelling-CTC + face streams into a T5/LLM is already published at ECCV 2026; any claim of novelty must isolate what *anatomical graph + quality calibration* adds under degraded inputs where SignBind-LLM's uncalibrated experts have no pose fallback.

#### P4. `SignMusketeers` (Findings of ACL 2025)
- **Exact Title:** SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale
- **Authors:** Shester Gueuwou, Xiaodan Du, Greg Shakhnarovich, Karen Livescu
- **Year:** 2025
- **Venue:** Findings of the Association for Computational Linguistics: ACL 2025
- **DOI or Official URL:** [ACL Anthology 2025.findings-acl.1157](https://aclanthology.org/2025.findings-acl.1157/)
- **Architecture:** Efficient multi-stream frame-level SSL visual encoders over cropped face, hands, and body regions feeding an SLT sequence model.
- **Main Mechanism:** Avoids brittle off-the-shelf 3D/pose extractors and heavy full-video transformers by pretraining single-frame region encoders (hands, face, posture) via SSL on massive unlabeled sign corpora (*YouTube-ASL*), reducing pretraining compute to `<3%` of large video backbones.
- **Datasets:** *YouTube-ASL* (SSL pretraining), *How2Sign* (fine-tuning & test).
- **Evaluation Protocol:** Gloss-free SLT BLEU evaluation + FLOPs/GPU-hour compute efficiency comparison.
- **Overlap with ISL-MEGA:** Explicit 4-region crop decomposition (left hand, right hand, face, body) as in [`src/isl_mega/vision.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L20-L59) and [`crop_regions`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/vision.py#L20).
- **Differences:** SignMusketeers intentionally discards skeletal pose due to inconsistent MediaPipe hand/face detections, whereas `ISL-MEGA` explicitly retains both RGB crops and 66-pt pose precisely to handle cross-modal complementarity.
- **Implications for Novelty:** Eliminates any claim that multi-region visual cropping (hands + face + body) is novel; ironically underscores `ISL-MEGA`'s central research thesis—if off-the-shelf poses are noisy (SignMusketeers' rationale for dropping them), can *quality-aware reliability gating* extract pose value without brittle failure?

#### P5. `Uni-Sign` (ICLR 2025)
- **Exact Title:** Uni-Sign: Toward Unified Sign Language Understanding at Scale
- **Authors:** Ze-Cheng Li, Wen-Gang Zhou, Wei-Chao Zhao, Kepeng Wu, He-Zhen Hu, Hou-Qiang Li
- **Year:** 2025
- **Venue:** International Conference on Learning Representations (ICLR 2025)
- **DOI or Official URL:** [arXiv:2501.15187](https://arxiv.org/abs/2501.15187) / [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html)
- **Architecture:** Unified pose-and-RGB pretraining + generative SLT architecture with Prior-Guided Fusion (PGF) and score-aware keypoint sampling.
- **Main Mechanism:** Casts heterogeneous sign understanding tasks into a single generative SLT pretraining/fine-tuning framework over 1,985 hours (*CSL-News*), fusing RGB cues with skeleton representations via **Prior-Guided Fusion (PGF)** and **score-aware sampling** to mitigate noisy keypoints.
- **Datasets:** *CSL-News* (1,985h), *CSL-Daily*, *PHOENIX-2014T*, *WLASL*, *MSASL*.
- **Evaluation Protocol:** Multi-task CSLR, ISLR, and SLT (BLEU, ROUGE, WER).
- **Overlap with ISL-MEGA:** Directly overlaps with `ISL-MEGA`'s combination of skeleton pose + RGB regional cues + detector-confidence-aware weighting (`score-aware` mitigation of noisy keypoints).
- **Differences:** Requires 1,985-hour Chinese Sign Language pretraining (`CSL-News`) and lacks continuous elapsed-time lag attention, CTC fingerspelling feedback, or counterfactual stream ablation.
- **Implications for Novelty:** **Critical precursor.** Using detector quality/confidence priors to combine RGB and skeleton streams (`PGF` + score-aware sampling) already appeared at ICLR 2025; `ISL-MEGA`'s fusion must prove tangible differences over simple score-aware sampling/confidence gating.

#### P6. `AttnSign` (Findings of EMNLP 2026)
- **Exact Title:** Attention-Steered Vision-Language Models for Sign Language Translation
- **Authors:** Meibo Hu, Guohao Sun, Annemarie D. Ross, Sheng Li, Zhiqiang Tao
- **Year:** 2026
- **Venue:** Findings of the Association for Computational Linguistics: EMNLP 2026 (pp. 30596–30611)
- **DOI or Official URL:** [ACL Anthology 2026.findings-emnlp.1435](https://aclanthology.org/2026.findings-emnlp.1435/)
- **Architecture:** Pretrained Vision-Language Model (VLM) equipped with spatial attention steering and reinforcement-learning temporal keyframe policy.
- **Main Mechanism:** Diagnoses poor spatial-temporal visual grounding under cross-entropy-only training; introduces explicit spatial attention supervision onto hands/face regions plus RL motion-cadence keyframe steering.
- **Datasets:** *How2Sign*, *OpenASL*.
- **Evaluation Protocol:** Gloss-free SLT metrics (BLEU-1..4, ROUGE-L) and grounding inspection.
- **Overlap with ISL-MEGA:** Targets the exact failure mode motivated by `ISL-MEGA`'s [`adaptive_consolidation`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L197) (motion-cadence keyframe emphasis) and [`compute_counterfactual_visual_loss`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py#L135-L173) (preventing language-model visual ignore).
- **Differences:** Uses VLM spatial-attention supervision and RL keyframe steering rather than differentiable kinematic salience pooling or decoupled pose graphs.
- **Implications for Novelty:** Shows that "language models ignoring visual evidence" and "motion-cadence keyframe weighting" are already explicitly framed and solved in EMNLP 2026 (`AttnSign`).

#### P7. `Think in Latent Thoughts` (ACL 2026)
- **Exact Title:** Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation
- **Authors:** Yiyang Jiang, Li Zhang, Xiao-Yong Wei, Li Qing
- **Year:** 2026
- **Venue:** Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
- **DOI or Official URL:** [ACL Anthology 2026.acl-long.454](https://aclanthology.org/2026.acl-long.454/) / [arXiv:2604.15301](https://arxiv.org/abs/2604.15301)
- **Architecture:** Video encoder -> ordered **latent thoughts** bottleneck -> plan-then-ground text decoder.
- **Main Mechanism:** Treats gloss-free SLT as two-stage cross-modal reasoning: generates an ordered sequence of latent concept/plan tokens first, then grounds them back onto visual memory to bridge sign vs. spoken word-order divergence.
- **Datasets:** Author-curated large-scale gloss-free SLT dataset, *PHOENIX-2014T*, *CSL-Daily*.
- **Evaluation Protocol:** BLEU-1..4, ROUGE-L on gloss-free SLT benchmarks.
- **Overlap with ISL-MEGA:** Directly mirrored by `LatentPlanGrounder` (`K=32` ordered/diverse latent queries cross-attending to temporal memory) in [`src/isl_mega/planning.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/planning.py#L45-L197).
- **Differences:** In `Think in Latent Thoughts`, decoding strictly obeys a causal plan-then-ground formulation; in `ISL-MEGA`, [`T5VisualDecoder.encode_inputs`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py#L28-L43) simply concatenates `[visual; plan]` into T5's encoder so FLAN-T5 can bypass the plan altogether, and `budget_gate` does not constrain generation length.
- **Implications for Novelty:** **High overlap + weaker execution.** Intermediate ordered latent planning for gloss-free SLT is already published in ACL 2026 with a cleaner causal formulation than `ISL-MEGA`'s concatenated auxiliary queries.

#### P8. `ISLiM` (WSLP @ EMNLP 2026)
- **Exact Title:** ISLiM: A Compact End-to-End Sign Language Translation Framework
- **Authors:** Shoaib Ahmed, Dipti Mishra
- **Year:** 2026
- **Venue:** Proceedings of the Second Workshop on Sign Language Processing (WSLP 2026)
- **DOI or Official URL:** [ACL Anthology 2026.wslp-1.3](https://aclanthology.org/2026.wslp-1.3/)
- **Architecture:** Temporal 1D convolution stack -> lightweight Transformer encoder -> pretrained `T5-small` decoder operating purely on body-pose keypoints.
- **Main Mechanism:** Compact, gloss-free keypoint-only seq2seq translation optimized for edge inference (~384 ms Android CPU latency), eliminating RGB backbones while transferring cross-lingually via decoder swapping.
- **Datasets:** *iSign* (ISL), *PHOENIX-2014T* (cross-lingual DGS test).
- **Evaluation Protocol:** `iSign` SignPose2Text benchmark (**BLEU-4 = 5.85**), cross-lingual transfer, and mobile CPU latency profiling.
- **Overlap with ISL-MEGA:** Closest domain-specific ISL baseline; uses the exact target domain (`iSign`) and `T5` seq2seq family (`T5-small` / `FLAN-T5`).
- **Differences:** Strictly pose-only without visual appearance crops, fingerspelling CTC head, cross-articulator graph attention, or missing-modality stream reliability gating.
- **Implications for Novelty:** **Primary ISL comparison baseline.** Any claim made for `ISL-MEGA` or `CARP-S2T` on `iSign` MUST directly outperform `ISLiM` (`5.85` BLEU-4) under identical splits, especially on noisy/occluded clips where keypoint-only models degrade sharply.

#### P9. `TransAlign-SLT` (WSLP @ EMNLP 2026)
- **Exact Title:** TransAlign-SLT: Complementary Temporal and Semantic Supervision for Gloss-Free Sign Language Translation
- **Authors:** Pallapu Mohan Krishna, Uppala Sanjay Kumar, M. K. Bhuyan, Debanga Raj Neog
- **Year:** 2026 (October 2026)
- **Venue:** Proceedings of the Second Workshop on Sign Language Processing (WSLP 2026)
- **DOI or Official URL:** [ACL Anthology 2026.wslp-1.21](https://aclanthology.org/2026.wslp-1.21/)
- **Architecture:** Lightweight gloss-free encoder–decoder (~45.6M params) augmented at training time with auxiliary CTC and frozen corpus-adapted BERT teacher embeddings.
- **Main Mechanism:** Pairs **TranslationCTC** (coarse temporal regularization) with **Masked Entropic Optimal Transport (MOT)** aligning temporal visual representations to contextual word embeddings from frozen BERT; training-only alignment leaves inference lightweight.
- **Datasets:** *PHOENIX-2014T* (`23.54` BLEU-4, `48.46` ROUGE-L), *CSL-Daily*.
- **Evaluation Protocol:** Gloss-free SLT evaluation using BLEU-4, ROUGE-L, and parameter efficiency.
- **Overlap with ISL-MEGA:** Directly overlaps [`entropic_semantic_ot`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py#L200-L238) + auxiliary CTC regularization in `ISL-MEGA` (`decoder.py` aligns T5 encoder hidden states with detached T5 text embeddings via 30-step Sinkhorn entropic OT).
- **Differences:** `TransAlign-SLT` uses Masked OT against corpus-adapted BERT without dual T5-encoder visual pass or 10-stream multimodal fusion.
- **Implications for Novelty:** **Entropic Sinkhorn OT between visual frames and frozen text embeddings for gloss-free SLT is directly preceded/matched by WSLP 2026.**

#### P10. `TLAS` (ALVR @ ACL 2026)
- **Exact Title:** Temporal-Linguistic Adaptive Streaming for Continuous Sign Language Translation
- **Authors:** Arshia Kermani, Habib Irani, Deautaun Ross, Vangelis Metsis
- **Year:** 2026 (July 2026)
- **Venue:** Proceedings of the 4th Workshop on Advances in Language and Vision Research (ALVR @ ACL 2026)
- **DOI or Official URL:** [10.18653/v1/2026.alvr-main.21](https://doi.org/10.18653/v1/2026.alvr-main.21) / [ACL Anthology 2026.alvr-main.21](https://aclanthology.org/2026.alvr-main.21/)
- **Architecture:** Gloss-to-text streaming controller comprising Temporal Pause Detector (TPD), frozen-T5 Linguistic Readiness Estimator (LRE), and Adaptive Fusion Gate (AFG).
- **Main Mechanism:** Replaces fixed Wait-k streaming schedules by combining inter-gloss pause EMA statistics with T5 linguistic completeness scoring to trigger proactive sentence emission without oracle boundaries.
- **Datasets:** Synthetic ASL discourse benchmark (1,400 discourse groups built from *ASLG-PC12* glosses with LLM-generated timestamps).
- **Evaluation Protocol:** Streaming translation latency and quality on continuous gloss streams.
- **Overlap with ISL-MEGA:** Minimal (only shared T5 ecosystem and timing awareness); operates on discrete timestamped **gloss tokens**, NOT raw RGB/pose video.
- **Differences:** Completely distinct problem (online gloss-to-text segmentation vs. offline multimodal video-to-text SLT).
- **Implications for Novelty:** Corrects [`reports/03`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md#L25): `TLAS` is NOT a visual-grounding / optimal-transport competitor and should NOT be cited as one.

#### P11. `SignDAGC` (*Pattern Recognition*, 2026)
- **Exact Title:** SignDAGC: Dynamic axial graph structure for continuous sign language recognition and translation
- **Authors:** Hong-Xiang Hu, Xu-Hua Yang, Dong Wei, Gang-Feng Ma, Sheng Liu, Yuan Feng, Yong Wang
- **Year:** 2026
- **Venue:** *Pattern Recognition*, Vol. 179, Article 113947 (Elsevier)
- **DOI or Official URL:** [10.1016/j.patcog.2026.113947](https://doi.org/10.1016/j.patcog.2026.113947)
- **Architecture:** Hybrid CNN–GNN spatiotemporal backbone introducing a Dynamic Axial Graph Convolution module.
- **Main Mechanism:** Models intra-frame and cross-frame spatiotemporal correlations along structured graph axes, avoiding brute-force dense kNN graph construction across frames while decoupling articulatory dynamics.
- **Datasets:** *PHOENIX-2014*, *PHOENIX-2014-T*, *CSL*, *CSL-Daily*.
- **Evaluation Protocol:** Primarily Continuous Sign Language Recognition (WER) alongside Sign2Gloss2Text / SLT configurations (BLEU, ROUGE-L).
- **Overlap with ISL-MEGA:** Shares articulated spatial-temporal graph modeling across hands/face/body with [`PoseExpert`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L275-L434).
- **Differences:** SignDAGC runs a CNN–GNN dynamic axial topology trained on gloss-supervised CSLR/SLT benchmarks (*PHOENIX*, *CSL-Daily*) rather than gloss-free 10-stream reliability fusion on ISL.
- **Implications for Novelty:** Shows dynamic graph modeling of signing articulators is mature in top journals (2026); `EdgeGraphBlock` + `CrossArticulatorGraph` alone cannot claim SOTA architectural novelty without empirical proof.

#### P12. `ADAT` (arXiv Preprint, 2025)
- **Exact Title:** ADAT: Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation
- **Authors:** Nada Shahin, Leila Ismail
- **Year:** 2025 (April 2025)
- **Venue:** arXiv Preprint (`cs.CV` / `cs.CL`, **unreviewed preprint status** as of verification)
- **DOI or Official URL:** [arXiv:2504.11942](https://arxiv.org/abs/2504.11942)
- **Architecture:** Convolutional local temporal extractor + log-sparse self-attention + adaptive gating encoder–decoder.
- **Main Mechanism:** Combines 1D/temporal convolutions for short-range rapid hand motion with log-sparse attention for long-range context and adaptive gating to cut transformer quadratic cost.
- **Datasets:** *MedASL* (medical/clinical ASL dataset introduced in the paper) and standard SLT comparisons.
- **Evaluation Protocol:** Domain-specific SLT BLEU and training runtime reduction.
- **Overlap with ISL-MEGA:** Multi-scale conv stem + sparse/strided global Transformer + adaptive gating conceptually resembles [`HierarchicalTemporalEncoder`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L197-L375).
- **Differences:** No multimodal RGB+pose uncertainty fusion or cross-articulator anatomical skeleton graph; targets medical ASL.
- **Implications for Novelty:** Reinforces that "local conv + efficient Transformer + gating" is a standard engineering pattern; preprint status must be flagged explicitly.

#### P13. `SignMamba` (CSLR — Neural Networks 2026)
- **Exact Title:** SignMamba: Sparse Spatial–Temporal State-Space Modeling for Continuous Sign Language Recognition
- **Authors:** Xu-Hua Yang, Gang Wang, Hong-Xiang Hu, Dong Wei, Zhi-Wei Chen, Sheng Liu, Yuan Feng
- **Year:** 2026
- **Venue:** *Neural Networks* (Elsevier)
- **DOI or Official URL:** [10.1016/j.neunet.2026.109530](https://doi.org/10.1016/j.neunet.2026.109530)
- **Architecture:** Sparse spatial–temporal Selective State-Space (Mamba) encoder with CTC gloss prediction head.
- **Main Mechanism:** Uses sparse spatial–temporal state-space filtering to track signing articulators while suppressing background visual noise in linear time.
- **Datasets:** Continuous sign language recognition benchmarks (*PHOENIX-2014*, *PHOENIX-2014-T*, *CSL-Daily*).
- **Evaluation Protocol:** **Continuous Sign Language Recognition (CSLR)** Word Error Rate (WER), NOT end-to-end spoken text BLEU.
- **Overlap with ISL-MEGA:** Linear-time selective SSM sequence backbone targeted by `CARP-S2T` (`Bi-S6`).
- **Differences:** Solves **gloss recognition (CSLR)** rather than gloss-free spoken-language translation (SLT), and omits multi-descriptor stream quality fusion.
- **Implications for Novelty:** [`reports/03`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/03_SCIENTIFIC_NOVELTY_INVESTIGATION.md#L19) and [`reports/06`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md#L48) falsely conflate this CSLR paper with RGB continuous sign translation; cannot be cited as a CSLT SOTA proof without task separation.

#### P14. `Sign-Mamba` (SLP — ICASSP 2025)
- **Exact Title:** Sign-Mamba: Advanced Mamba-Based Sign Language Generation
- **Authors:** Guanwen Feng, Yilin Zhang, Yunan Li, An Liu, Qiguang Miao
- **Year:** 2025
- **Venue:** IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2025)
- **DOI or Official URL:** [10.1109/ICASSP49660.2025.10890373](https://doi.org/10.1109/ICASSP49660.2025.10890373)
- **Architecture:** Two-stage Mamba selective state-space latent sequence generator.
- **Main Mechanism:** Self-supervised latent trajectory learning + selective SSM decoding to synthesize continuous 3D skeletal sign motions from text prompts.
- **Datasets:** Continuous sign language generation benchmarks (*PHOENIX-2014T* pose trajectories).
- **Evaluation Protocol:** **Sign Language Production (SLP)** pose reconstruction error (MPJPE/DTW-MJE) and back-translation BLEU.
- **Overlap with ISL-MEGA:** Shares Mamba SSM temporal formulation proposed in `CARP-S2T`.
- **Differences:** Opposite problem direction (**Text-to-Sign Generation** rather than Video/Pose-to-Text Translation).
- **Implications for Novelty:** Further exposes task conflation in prior repo reports; Mamba for SLP provides zero empirical guarantee that Bi-Mamba improves CSLT over stride-2 Conformer.

#### P15. `EvSLT` (Event-Stream Mamba SLT — arXiv 2024)
- **Exact Title:** Event Stream-based Sign Language Translation: A High-Definition Benchmark Dataset and A Novel Baseline
- **Authors:** Shiao Wang, Xiao Wang, Duoqing Yang, Yao Rong, Fuling Wang, Jianing Li, Lin Zhu, Bo Jiang
- **Year:** 2024 (August 2024)
- **Venue:** arXiv Preprint (`cs.CV`, unreviewed preprint / *Event-CSL* release)
- **DOI or Official URL:** [arXiv:2408.10488](https://arxiv.org/abs/2408.10488)
- **Architecture:** Hybrid CNN visual stem + Mamba-based clip-level memory aggregation encoder + seq2seq translation decoder.
- **Main Mechanism:** Processes high-temporal-resolution neuromorphic event camera streams (*Event-CSL*) via CNN local extraction and Mamba state-space memory compression.
- **Datasets:** *Event-CSL* (high-definition neuromorphic event sign language corpus).
- **Evaluation Protocol:** End-to-end event-stream SLT evaluation using BLEU-1..4 and ROUGE-L.
- **Overlap with ISL-MEGA:** Demonstrates Mamba-style SSM temporal compression inside a sign translation pipeline (`CARP-S2T` direction).
- **Differences:** Built for dense asynchronous neuromorphic **event-camera streams** (not standard RGB+pose YouTube ISL video corpora like *iSign* or *ISLTranslate*).
- **Implications for Novelty:** Confirms Mamba already appeared in SLT in 2024, stripping pure "Mamba for SLT" novelty while leaving open whether `Bi-S6` outperforms stride-2 Conformer at `T_c <= 90` on RGB+pose.

#### P16. `Beyond BLEU` (Methodological Benchmark Critique — arXiv Sept 2026)
- **Exact Title:** Beyond BLEU: A Case for Redefining Sign Language Translation Benchmarks
- **Authors:** Oline Ranum, Edward Fish, Simon Hadfield, Richard Bowden
- **Year:** 2026 (September 2026)
- **Venue:** arXiv Preprint (`cs.CL`)
- **DOI or Official URL:** [arXiv:2609.03734](https://arxiv.org/abs/2609.03734)
- **Architecture:** Cross-architecture benchmark audit across 6 SLT models + open-weight LLM semantic QA evaluator.
- **Main Mechanism:** Proves empirically that BLEU-4 rewards spoken-language decoder priors and superficial n-gram matching; proposes a paraphrase-invariant LLM QA salient-content assessment aligned with human rankings.
- **Datasets:** *PHOENIX-2014T*, *CSL-Daily*.
- **Evaluation Protocol:** Rank correlation (`tau` / Spearman) between automated metrics (`BLEU-4`, `ROUGE-L`, LLM QA) and human comprehension judgments.
- **Overlap with ISL-MEGA:** Directly validates the anti-language-prior intuition behind [`compute_counterfactual_visual_loss`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py#L135-L173).
- **Differences:** An evaluation/benchmarking methodology rather than a multimodal fusion or translation architecture.
- **Implications for Novelty:** Makes BLEU-only ablation assertions in [`reports/06`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/06_EXPERIMENTAL_AND_PUBLICATION_PLAN.md) obsolete for 2026–2027 Q1 submission; requires meaning-based + visual-ablation controls.

---

### 2.2 Indian Sign Language (ISL) Datasets & Baselines (2023–2025)

#### P17. `ISLTranslate` (Findings of ACL 2023)
- **Exact Title:** ISLTranslate: Dataset for Translating Indian Sign Language
- **Authors:** Abhinav Joshi, Susmit Agrawal, Ashutosh Modi
- **Year:** 2023
- **Venue:** *Findings of the Association for Computational Linguistics: ACL 2023*
- **DOI or Official URL:** [ACL Anthology 2023.findings-acl.665](https://aclanthology.org/2023.findings-acl.665/)
- **Architecture:** Baseline **Pose-SLT** using extracted MediaPipe keypoints + Transformer encoder–decoder.
- **Main Mechanism:** Curates 31,222 continuous ISL–English video/sentence pairs from Indian educational and Deaf community YouTube channels (ISLRTC, Deaf Enabled Foundation, ISH News) for end-to-end gloss-free training.
- **Datasets:** *ISLTranslate* (31,222 sentence pairs; human signer validation subset check reached 48.93 BLEU-4 against certified translations).
- **Evaluation Protocol:** End-to-end gloss-free translation BLEU-1..4 & ROUGE-L; Table 4 `Pose-SLT` baseline scores **BLEU-4 = 6.09**, BLEU-1 = 13.18, ROUGE-L = 12.91.
- **Overlap with ISL-MEGA:** Primary intended corpus alongside `iSign`; motivates [`write_grouped_splits`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L432-L536) source/signer split checks.
- **Differences:** Baseline uses raw MediaPipe pose without multimodal RGB crops, 66-pt cross-articulator graphs, or quality-calibrated fusion.
- **Implications for Novelty:** Demonstrates how brittle uncalibrated pose-only baselines remain on continuous ISL while highlighting strict source-channel disjointness requirements documented in [`05_DATASET_AND_REAL_DATA_READINESS.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/05_DATASET_AND_REAL_DATA_READINESS.md).

#### P18. `iSign` (Findings of ACL 2024)
- **Exact Title:** iSign: A Benchmark for Indian Sign Language Processing
- **Authors:** Abhinav Joshi, Romit Mohanty, Mounika Kanakanti, Andesha Mangla, Sudeep Choudhary, Monali Barbate, Ashutosh Modi
- **Year:** 2024
- **Venue:** *Findings of the Association for Computational Linguistics: ACL 2024*
- **DOI or Official URL:** [ACL Anthology 2024.findings-acl.643](https://aclanthology.org/2024.findings-acl.643/)
- **Architecture:** Evaluates `Camgoz et al. (2020)` video/pose Transformers, `I3D + T5-small`, and `MediaPipe + T5-large` baselines.
- **Main Mechanism:** Scales ISL-English parallel supervision to >118,000 clips across 5 multi-task NLP/vision formulations (*SignVideo2Text*, *SignPose2Text*, *Text2Pose*, *Word Prediction*, *Sign Semantics*).
- **Datasets:** *iSign* (>118K ISL-English pairs). Note `GloSL-ISL` (`pedroodb/glosl-isl` on Hugging Face) is a community standardization wrapper over these ISL sources, NOT an independent dataset paper.
- **Evaluation Protocol:** Table 6 translation baselines score **0.56 BLEU-4** (`Camgoz` video), **0.24** (`I3D + T5-small`), **0.77** (`Camgoz` pose), and up to **1.47 BLEU-4** (`MediaPipe + T5-large`).
- **Overlap with ISL-MEGA:** Target large-scale benchmark matching `ISL-MEGA`'s gloss-free T5 seq2seq paradigm.
- **Differences:** Baseline pipelines do not combine regional DINOv2 appearance crops, 66-pt cross-articulator pose graphs, fingerspelling CTC, or quality-calibrated stream reliability gating.
- **Implications for Novelty:** Massive empirical headroom exists on *iSign* (`1.47` baseline -> `5.85` in `ISLiM` 2026), making *iSign* the highest-leverage empirical proving ground once leakage-free splits are verified.

#### P19. `Continuous Fingerspelling Dataset for ISL` (WSLP @ IJCNLP-AACL 2025)
- **Exact Title:** Continuous Fingerspelling Dataset for Indian Sign Language
- **Authors:** Kirandevraj R., Vinod K. Kurmi, Vinay P. Namboodiri, C. V. Jawahar
- **Year:** 2025 (December 2025)
- **Venue:** Proceedings of the Workshop on Sign Language Processing (WSLP @ IJCNLP–AACL 2025)
- **DOI or Official URL:** [ACL Anthology 2025.wslp-main.6](https://aclanthology.org/2025.wslp-main.6/)
- **Architecture:** `ByT5-small` byte/character-level seq2seq fingerspelling baseline (and follow-up dense frame-level CTC/recognition extensions).
- **Main Mechanism:** Annotates continuous ISL fingerspelling intervals and character transcripts extracted from 499 ISH News videos (1,308 segments, 14,814 chars, 70.85 mins, 3 signers).
- **Datasets:** Continuous ISL Fingerspelling Dataset (1,308 segments from *ISH News*; interpreter validation 90.67% match).
- **Evaluation Protocol:** Character Error Rate (CER): baseline `ByT5-small` reaches **82.91% CER** (dense frame-supervised follow-ups push CER to 4.87%).
- **Overlap with ISL-MEGA:** Directly motivates `fingerspelling_spans` frame spotting and CTC/MIL heads in [`src/isl_mega/fingerspelling.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fingerspelling.py) and [`losses.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L22-L102).
- **Differences:** Isolated fingerspelling-only recognition rather than full-sentence multimodal CSLT.
- **Implications for Novelty:** Critical split-hygiene hazard—*ISH News* overlaps *ISLTranslate*/*iSign*; joint training requires strict source-video deduplication.

---

### 2.3 Foundational SLT, Pose-Graph, Reliability, Causal, Continuous-Time & Plasticity Works (2020–2025)

#### P20. `Sign Language Transformers` (CVPR 2020)
- **Exact Title:** Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation
- **Authors:** Necati Cihan Camgoz, Oscar Koller, Simon Hadfield, Richard Bowden
- **Year:** 2020
- **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020)
- **DOI or Official URL:** [CVF Open Access Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.html)
- **Architecture:** Spatial 2D CNN visual embeddings + Transformer Encoder (CTC gloss recognition) + Transformer Decoder (spoken text translation).
- **Main Mechanism:** Jointly optimizes Sign2Gloss (CTC) and Sign2Text (cross-entropy) end-to-end inside a unified Transformer stack without manual frame alignment.
- **Datasets:** *RWTH-PHOENIX-Weather 2014T* (also used as `Sign2Text` baseline in *iSign* ACL 2024).
- **Evaluation Protocol:** Gloss WER (CSLR) and SacreBLEU / ROUGE (SLT).
- **Overlap with ISL-MEGA:** Establishes the standard Transformer + auxiliary CTC joint training blueprint.
- **Differences:** Originally relies heavily on gloss supervision for top performance and lacks multi-stream pose graphs or quality-aware missing-modality gating.
- **Implications for Novelty:** Baseline architectural ancestor; attention + CTC seq2seq alone offers zero novelty.

#### P21. `GFSLT-VLP` (ICCV 2023)
- **Exact Title:** Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining
- **Authors:** Benjia Zhou, Zhigang Chen, Albert Clapés, Jun Wan, Yanyan Liang, Sergio Escalera, Zhen Lei, Du Zhang
- **Year:** 2023
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2023)
- **DOI or Official URL:** [CVF Open Access Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Gloss-free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.html) / [arXiv:2307.14768](https://arxiv.org/abs/2307.14768)
- **Architecture:** Visual encoder + text encoder aligned via CLIP-style contrastive pretraining + seq2seq SLT decoder.
- **Main Mechanism:** Bridges raw sign videos and text directly without intermediate gloss labels using visual-language contrastive alignment plus masked self-supervised pretraining.
- **Datasets:** *RWTH-PHOENIX-Weather 2014T*, *CSL-Daily*.
- **Evaluation Protocol:** Gloss-free SLT BLEU-1..4 and ROUGE-L.
- **Overlap with ISL-MEGA:** Shares gloss-free cross-modal contrastive/semantic alignment principles (`losses.py`, `decoder.py`).
- **Differences:** RGB-only appearance modeling without anatomical pose graphs or reliability gating.
- **Implications for Novelty:** Defines the standard gloss-free VLP baseline; semantic contrastive loss without glosses is established prior art.

#### P22. `Sign2GPT` (ICLR 2024)
- **Exact Title:** Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation
- **Authors:** Ryan Wong, Necati Cihan Camgoz, Richard Bowden
- **Year:** 2024
- **Venue:** International Conference on Learning Representations (ICLR 2024)
- **DOI or Official URL:** [OpenReview Forum](https://openreview.net/forum?id=AXgA6qHwWw)
- **Architecture:** Sign visual encoder + lightweight adapter + LoRA-adapted pretrained causal LLM decoder.
- **Main Mechanism:** Generates pseudo-gloss semantic prototypes for pretraining and transfers visual sign representations to a parameter-efficient LoRA-adapted LLM for gloss-free translation.
- **Datasets:** *RWTH-PHOENIX-Weather 2014T*, *CSL-Daily*.
- **Evaluation Protocol:** Gloss-free BLEU-1..4 and ROUGE-L.
- **Overlap with ISL-MEGA:** Directly precedes `Region-LoRA` + pretrained seq2seq/LLM adaptation proposed in `CARP-S2T`.
- **Differences:** No 66-point ST-GCN, fingerspelling CTC head, or sensor-quality fusion.
- **Implications for Novelty:** Proves LoRA tuning of pretrained backbones/LLMs for gloss-free SLT is already ICLR 2024 prior art.

#### P23. `CoSign` (ICCV 2023)
- **Exact Title:** CoSign: Exploring Co-occurrence Signals in Skeleton-based Continuous Sign Language Recognition
- **Authors:** Peiqi Jiao, Yuecong Min, Yanan Li, Xiaotao Wang, Lei Lei, Xilin Chen
- **Year:** 2023
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2023)
- **DOI or Official URL:** [CVF Open Access Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_paper.html)
- **Architecture:** Group-specific spatial-temporal skeleton GCNs feeding temporal co-occurrence fusion + CTC gloss head.
- **Main Mechanism:** Partitions body skeletons into functional articulatory subgraphs (hands, face, upper body) and models complementary inter-group co-occurrence signals across continuous signing.
- **Datasets:** *PHOENIX-2014*, *PHOENIX-2014-T*, *CSL-Daily*.
- **Evaluation Protocol:** Continuous Sign Language Recognition (**CSLR** Word Error Rate / WER), NOT gloss-free spoken-text BLEU.
- **Overlap with ISL-MEGA:** Directly anticipates `RegionGraphEncoder` + `CrossArticulatorGraph` in [`src/isl_mega/pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L158-L274).
- **Differences:** Recognition-only skeleton model with zero RGB visual appearance stream, quality gating, or seq2seq text decoder.
- **Implications for Novelty:** Multi-region anatomical subgraphs + cross-part interaction cannot be framed as a novel graph topology without proving cross-modal occlusion resilience.

#### P24. `SignBERT` (ICCV 2021)
- **Exact Title:** SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition
- **Authors:** Hezhen Hu, Weichao Zhao, Wengang Zhou, Yuechen Wang, Houqiang Li
- **Year:** 2021
- **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2021)
- **DOI or Official URL:** [CVF Open Access Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html)
- **Architecture:** Hand-model-aware graph/Transformer backbone pretrained with masked pose-token reconstruction.
- **Main Mechanism:** Incorporates anatomical hand-prior hierarchies into self-supervised masked hand-state modeling before downstream sign recognition.
- **Datasets:** *MSASL*, *WLASL*, *SLR500*, *NMFs-CSL*.
- **Evaluation Protocol:** Isolated Sign Language Recognition (**ISLR** Top-1/Top-5 classification accuracy), strictly isolated recognition.
- **Overlap with ISL-MEGA:** Establishes hand-prior skeleton representation learning used inside [`PoseExpert`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L275-L434).
- **Differences:** Isolated sign classification only; no continuous translation, facial/body fusion, or stream uncertainty weighting.
- **Implications for Novelty:** Confirms hierarchical hand kinematic priors have been standard in sign representation since 2021.

#### P25. `Sem-SLT` (ACL 2024)
- **Exact Title:** Sign Language Translation with Sentence Embedding Supervision
- **Authors:** Yasser Hamidullah, Josef van Genabith, Cristina España-Bonet
- **Year:** 2024
- **Venue:** Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024, Short Papers)
- **DOI or Official URL:** [ACL Anthology 2024.acl-short.40](https://aclanthology.org/2024.acl-short.40/) / [arXiv:2510.19367](https://arxiv.org/abs/2510.19367)
- **Architecture:** Two-stage `sign2sem` continuous encoder + `sem2text` pretrained `mBART`/Transformer reconstructor.
- **Main Mechanism:** Replaces manual glosses with dense `sBERT` sentence embeddings of target translations as intermediate semantic supervision.
- **Datasets:** *PHOENIX-2014T*, *How2Sign*.
- **Evaluation Protocol:** Gloss-free BLEU-1..4 and ROUGE-L.
- **Overlap with ISL-MEGA:** Directly precedes frozen text-encoder semantic supervision (`entropic_semantic_ot` in [`decoder.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py#L111-L133) and semantic bottleneck plans).
- **Differences:** Sentence-level semantic matching rather than token-level Sinkhorn OT or multimodal quality fusion.
- **Implications for Novelty:** Reinforces that aligning visual states to frozen text embeddings without glosses is crowded 2024–2026 prior art.

#### P26. Online Continuous SLR/SLT (`Zuo et al.` EMNLP 2024)
- **Exact Title:** Towards Online Continuous Sign Language Recognition and Translation
- **Authors:** Ronglai Zuo, Fangyun Wei, Brian Mak
- **Year:** 2024
- **Venue:** Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)
- **DOI or Official URL:** [10.18653/v1/2024.emnlp-main.619](https://doi.org/10.18653/v1/2024.emnlp-main.619) / [ACL Anthology 2024.emnlp-main.619](https://aclanthology.org/2024.emnlp-main.619/)
- **Architecture:** Sliding-window online Transformer coupled with isolated sign dictionary alignment priors.
- **Main Mechanism:** Bounds memory and decoding latency for causal streaming sign recognition and translation instead of waiting for full-clip offline processing.
- **Datasets:** *PHOENIX-2014*, *PHOENIX-2014-T*, *CSL-Daily*.
- **Evaluation Protocol:** Streaming WER, BLEU-4, and end-to-end algorithmic latency.
- **Overlap with ISL-MEGA:** Benchmark point for causal vs. bidirectional sequence claims (`temporal.py` & `CARP-S2T` offline assumption).
- **Differences:** Explicitly causal online streaming, whereas `ISL-MEGA` and `CARP-S2T` operate purely offline on bidirectional full clips.
- **Implications for Novelty:** Prevents `ISL-MEGA` or `CARP-S2T` from claiming real-time streaming novelty without causal windowing.

#### P27. Continual Learning in Multilingual SLT (`Yazdani et al.` NAACL 2025)
- **Exact Title:** Continual Learning in Multilingual Sign Language Translation
- **Authors:** Shakib Yazdani, Josef van Genabith, Cristina España-Bonet
- **Year:** 2025
- **Venue:** Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the ACL (NAACL 2025)
- **DOI or Official URL:** [10.18653/v1/2025.naacl-long.546](https://doi.org/10.18653/v1/2025.naacl-long.546)
- **Architecture:** Seq2seq sign translation transformers evaluated across incremental continual-learning regularizers/replay vs. sequential fine-tuning.
- **Main Mechanism:** Tests 3 canonical continual-learning methods against 4 naive fine-tuning baselines across incremental sign language arrivals.
- **Datasets:** Cross-lingual ASL, BSL, CSL, and DGS translation suites.
- **Evaluation Protocol:** Forgetting, forward/backward transfer, and per-language SacreBLEU.
- **Overlap with ISL-MEGA:** Directly probes continual plasticity in SLT—the motivating rationale claimed by `EvoHARE-Net`.
- **Differences:** Evaluates multi-language continual shifts rather than prediction-error-triggered module insertion on a single offline ISL dataset.
- **Implications for Novelty:** Empirically shows specialized continual-learning mechanisms still trail standard fine-tuning in SLT, directly refuting `EvoHARE-Net`.

#### P28. `TMC` (Trusted Multi-View Classification — ICLR 2021 / IEEE TPAMI 2023)
- **Exact Title:** Trusted Multi-View Classification with Dynamic Evidential Fusion (ICLR 2021 original: *Trusted Multi-View Classification*)
- **Authors:** Zongbo Han, Changqing Zhang, Huazhu Fu, Joey Tianyi Zhou
- **Year:** 2021 (ICLR) / 2023 (IEEE TPAMI Vol. 45, No. 2)
- **Venue:** *ICLR 2021* / *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*
- **DOI or Official URL:** [OpenReview ICLR 2021](https://openreview.net/forum?id=Xu4bQvCBknW) / [10.1109/TPAMI.2022.3171983](https://doi.org/10.1109/TPAMI.2022.3171983)
- **Architecture:** Multi-view classification networks outputting per-view Dirichlet concentration vectors combined via reduced Dempster–Shafer belief mass combination.
- **Main Mechanism:** Applies Subjective Logic Dirichlet evidence (`alpha_c = e_c + 1`) over `K` mutually exclusive classes per view, combining views using uncertainty-aware Dempster–Shafer pooling.
- **Datasets:** Handwritten, Caltech101, Scene15, HMDB multi-view image/tabular datasets under noise and OOD corruption.
- **Evaluation Protocol:** Classification accuracy, ECE calibration error, and accuracy degradation under view noise.
- **Overlap with ISL-MEGA / CARP-S2T:** `CARP-S2T` (`reports/04`, `05`) copies TMC's exact Dirichlet formulas and loss structure (`alpha = e + 1`, `u = M / S`, MSE + variance + KL).
- **Differences:** TMC defines Dirichlet beliefs over *mutually exclusive target classes* (shared classification simplex `y_c`), NOT over *which sensory stream is reliable* (`y_m`).
- **Implications for Novelty:** Exposes `CARP-S2T`'s central mathematical flaw: transplanting TMC's equations onto modality weights violates subjective-logic semantics and leaves `y_m` undefined.

#### P29. `QMF` (Quality-Aware Multimodal Fusion — ICML 2023)
- **Exact Title:** Provable Dynamic Fusion for Low-Quality Multimodal Data
- **Authors:** Qingyang Zhang, Haitao Wu, Changqing Zhang, Qinghua Hu, Huazhu Fu, Joey Tianyi Zhou, Xi Peng
- **Year:** 2023
- **Venue:** International Conference on Machine Learning (ICML 2023, *PMLR* Vol. 202)
- **DOI or Official URL:** [arXiv:2306.02050](https://arxiv.org/abs/2306.02050)
- **Architecture:** Sample-wise dynamic multimodal fusion network driven by predictive modality confidence / energy uncertainty weighting.
- **Main Mechanism:** Replaces static fusion or ungrounded gating with confidence/energy-calibrated modality weighting derived directly from sample-level predictive uncertainty, proving generalization bounds under noisy multimodal data.
- **Datasets:** Multimodal image–text and multi-view benchmarks under synthetic noise, blur, and occlusion.
- **Evaluation Protocol:** Clean + noise-corrupted test classification accuracy and calibration metrics.
- **Overlap with ISL-MEGA:** Closest mathematically sound blueprint for what [`ReliabilityAwareFusion`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L56-L217) (`r_s(t)`) and `CARP-S2T` attempt to achieve.
- **Differences:** Studied on classification tasks rather than variable-frame-rate continuous seq2seq sign translation with 66-pt pose provenance (`confidence_known`).
- **Implications for Novelty:** **Critical pivot target.** Grounding `ISL-MEGA`'s observable quality gating in `QMF`-style sample-wise uncertainty calibration offers genuine mathematical rigor, unlike misusing `TMC`'s Dirichlet class simplex.

#### P30. `EDL` (`Sensoy` NeurIPS 2018) & Evidential Survey (`Ulmer` TMLR 2023)
- **Exact Titles:** (a) *Evidential Deep Learning to Quantify Classification Uncertainty* (`EDL`); (b) *Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods For Uncertainty Estimation*
- **Authors:** (a) Murat Sensoy, Lance Kaplan, Melih Kandemir; (b) Dennis Ulmer, Christian Hardmeier, Jes Frellsen
- **Year:** 2018 (`EDL`) / 2023 (`Survey`)
- **Venue:** *NeurIPS 2018* / *Transactions on Machine Learning Research (TMLR 2023)*
- **DOI or Official URL:** [arXiv:1806.01768](https://arxiv.org/abs/1806.01768) / [arXiv:2110.03051](https://arxiv.org/abs/2110.03051)
- **Architecture:** Deterministic classifier predicting Dirichlet `alpha_c` + KL uniform regularizer (`Sensoy`); comparative analysis of Prior/Posterior Dirichlet networks (`Ulmer`).
- **Main Mechanism:** Quantifies categorical uncertainty in one pass (`Sensoy`), while `Ulmer` formalizes failure modes including severe OOD overconfidence and KL hyperparameter sensitivity when likelihoods lack explicit density estimation.
- **Datasets:** MNIST, CIFAR-10/100, and synthetic OOD detection benchmarks.
- **Evaluation Protocol:** AUROC OOD detection, Expected Calibration Error (ECE), Brier score.
- **Overlap with ISL-MEGA / CARP-S2T:** Direct source of `CARP-S2T`'s MSE + variance + KL Dirichlet loss formula (`reports/05`).
- **Differences:** `CARP-S2T` provides zero calibration metrics and applies class-simplex math outside classification.
- **Implications for Novelty:** Directly explains why Q1 referees will reject uncalibrated "Dirichlet evidential fusion" without ground-truth reliability semantics.

#### P31. `CF-VQA` (CVPR 2021) & Missing-Modality Transformers (`Ma` CVPR 2022; `Lee` CVPR 2023)
- **Exact Titles:** (a) *Counterfactual VQA: A Cause-Effect Look at Language Bias* (`CF-VQA`); (b) *Are Multimodal Transformers Robust to Missing Modality?* (`Ma`); (c) *Multimodal Prompting with Missing Modalities for Visual Recognition* (`Lee`)
- **Authors:** (a) Yulei Niu et al.; (b) Mengmeng Ma, Jian Ren, Long Zhao, Davide Testuggine, Xi Peng; (c) Yi-Lun Lee, Yi-Hsuan Tsai, Wei-Chen Chiu, Chen-Yu Lee
- **Year:** 2021 (`CF-VQA`), 2022 (`Ma`), 2023 (`Lee`)
- **Venue:** IEEE/CVF CVPR 2021 / CVPR 2022 / CVPR 2023
- **DOI or Official URL:** [CF-VQA CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Niu_Counterfactual_VQA_A_Cause-Effect_Look_at_Language_Bias_CVPR_2021_paper.html) / [Ma CVPR 2022 CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html) / [Lee CVPR 2023 CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Multimodal_Prompting_with_Missing_Modalities_for_Visual_Recognition_CVPR_2023_paper.html)
- **Architecture:** Structural causal VQA fusion subtracting direct language branch (`CF-VQA`); optimal-fusion and missing-aware prompted multimodal Transformers (`Ma`, `Lee`).
- **Main Mechanism:** Computes total indirect vs. direct causal effect (`CF-VQA`) and prevents cross-modal attention collapse under dropped modalities (`Ma`, `Lee`).
- **Datasets:** *VQA v2*, *VQA-CP v2* (`CF-VQA`); *MM-IMDb*, *UPMC Food-101*, *Hateful Memes* (`Ma`, `Lee`).
- **Evaluation Protocol:** De-biased OOD accuracy and 10%–70% missing-modality drop curves.
- **Overlap with ISL-MEGA:** Directly parallels `ISL-MEGA`'s `roll(1)` language-prior hinge (`decoder.py#L135-L173`) and `available` masking + all-missing `null` token (`fusion.py#L135-L155`).
- **Differences:** Operates on causal VQA or static modal prompts rather than stream-specific reliability counterfactuals in continuous SLT.
- **Implications for Novelty:** Confirms `decoder.py`'s current post-fusion `roll(1)` hinge is less causally complete than `CF-VQA` and does not isolate per-stream reliability.

#### P32. `Neural CDE` (NeurIPS 2020), `Neural RDE` (ICML 2021), `STG-NCDE` (AAAI 2022)
- **Exact Titles:** (a) *Neural Controlled Differential Equations for Irregular Time Series* (`Neural CDE`); (b) *Neural Rough Differential Equations for Long Time Series* (`Neural RDE`); (c) *Graph Neural Controlled Differential Equations for Traffic Forecasting* (`STG-NCDE`)
- **Authors:** Patrick Kidger et al. (2020); James Morrill et al. (2021); Jeongwhan Choi et al. (2022)
- **Year:** 2020 / 2021 / 2022
- **Venue:** *NeurIPS 2020* / *ICML 2021* / *AAAI 2022*
- **DOI or Official URL:** [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html) / [arXiv:2009.08295](https://arxiv.org/abs/2009.08295) / [arXiv:2112.03558](https://arxiv.org/abs/2112.03558)
- **Architecture:** Continuous-time ODE solvers integrating hidden trajectories along cubic-spline / log-signature rough paths (plus spatial-temporal graph ODEs in `STG-NCDE`).
- **Main Mechanism:** Evolves latent state continuously `dZ(s) = f_theta(Z(s)) dX(s)` without fixed discrete clock ticks, naturally handling asynchronous irregular arrivals via adjoint backpropagation.
- **Datasets:** PhysioNet Sepsis, Character Trajectories, UEA irregular time series, PeMS traffic graphs.
- **Evaluation Protocol:** Accuracy/AUC/MAE across 30%–70% irregular missing timestamps.
- **Overlap with ISL-MEGA:** Conceptual inspiration behind `ContinuousTimeLagBias` (`fusion.py`) and `RelativeTimeBias` (`temporal.py`) for variable-FPS videos.
- **Differences:** `ISL-MEGA` runs discrete linear interpolation (`align_pose_to_video`) + discrete timestamp-biased attention, with zero differential-equation solver.
- **Implications for Novelty:** Prevents mislabeling `ISL-MEGA` as a Neural CDE while validating honest framing as timestamp-aware asynchronous attention.

#### P33. Dynamic Plasticity & Evolutionary Systems (`DEN` ICLR 2018; `NISPA` ICML 2022; `Dohare` *Nature* 2024)
- **Exact Titles:** (a) *Lifelong Learning with Dynamically Expandable Networks* (`DEN`); (b) *NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks* (`NISPA`); (c) *Loss of plasticity in deep continual learning*
- **Authors:** Jaehong Yoon et al. (2018); Mustafa Burak Gurbuz & Constantine Dovrolis (2022); Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton (2024)
- **Year:** 2018 / 2022 / 2024
- **Venue:** *ICLR 2018* / *ICML 2022* / *Nature* Vol. 632 (pp. 768–774)
- **DOI or Official URL:** [arXiv:1708.01547](https://arxiv.org/abs/1708.01547) / [arXiv:2206.01162](https://arxiv.org/abs/2206.01162) / [10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7)
- **Architecture:** Dynamic neuron expansion + split (`DEN`), fixed-density sparse rewiring (`NISPA`), and unit-utility selective reinitialization (*Continual Backpropagation*).
- **Main Mechanism:** Mitigates catastrophic forgetting or dormant-unit plasticity loss strictly over **non-stationary sequential task arrivals**.
- **Datasets:** Split-CIFAR/ImageNet, Permuted-MNIST, continual non-stationary RL/classification streams.
- **Evaluation Protocol:** Sequential task average accuracy, backward transfer, and active-unit rank recovery.
- **Overlap with ISL-MEGA / EvoHARE-Net:** Direct theoretical basis misappropriated by `EvoHARE-Net` (`reports/03`).
- **Differences:** Targets non-stationary continual-task streams, whereas *iSign*/*ISLTranslate* CSLT trains offline on a stationary corpus with a pretrained `FLAN-T5` decoder.
- **Implications for Novelty:** Confirms architectural growth / structural plasticity has zero empirical or theoretical justification in offline single-language CSLT (`EvoHARE-Net` stays rejected).

---

---

## 3. Synthesis Across All 17 Mandatory Research Areas (2020–Nov 2026)

| # | Research Topic | 2020–2026 Primary State of the Art | Status in Implemented `ISL-MEGA` | Status in Proposed `CARP-S2T` / `EvoHARE-Net` | Originality Assessment |
|---:| :--- | :--- | :--- | :--- | :--- |
| **1** | **Continuous Sign Language Translation (CSLT)** | Camgoz (2020) -> `Sign2GPT` (2024) -> `SignBind-LLM` / `Think in Latent Thoughts` / `AttnSign` (2026). | Implemented pipeline end-to-end (`ISLMEGA` + FLAN-T5), synthetic-only tested. | Proposed FLAN-T5-base replacement; 0 code. | Standard paradigm; no new CSLT framework claim without empirical runs. |
| **2** | **Gloss-Free Translation** | `GFSLT-VLP` (2023), Hamidullah (2024), `SignDINO` / `TransAlign-SLT` / `Think in Latent Thoughts` (2026). | Gloss-free by design + `entropic_semantic_ot` + latent plan slots + MIL CTC. | Drops OT & planner; retains CE + CTC + invalid Evid loss. | `entropic_semantic_ot` largely overlaps `TransAlign-SLT` (WSLP 2026). |
| **3** | **ISL-Specific Translation** | `ISLTranslate` (ACL 2023: 6.09 BLEU-4), `iSign` (ACL 2024: 1.47), `ISLiM` (WSLP 2026: 5.85). | Synthetic-only (`0` ISL videos/poses on disk; DSU split script in `data.py`). | Targets `iSign` + `ISLTranslate` + `GloSL-ISL` (wrapper). | **Strongest application gap:** huge empirical ceiling above `5.85` BLEU-4 on *iSign*. |
| **4** | **Sign Visual Representation Learning** | `SignMusketeers` (2025), `SignDINO` (CVPR 2026), `SignDino` (`2609.06296`), `Uni-Sign` (ICLR 2025). | 4 crops x DINOv2 (`912.95M` in research config) + SignDINO-inspired `pretraining.py`. | Frozen DINOv2-B + `Region-LoRA` (0 code). | Pretrained DINOv2 + LoRA/SSL crops is well-trodden territory (CVPR/arXiv 2026). |
| **5** | **Anatomical Graph Modelling** | `SignBERT` (2021), `CoSign` (ICCV 2023), `SignDAGC` (*PR* 2026). | 66-pt 4-region `EdgeGraphBlock` (`pose.py`) with slice-order discrepancy (`0:21` L, `21:42` R, `42:48` Body). | Naive `-x` mirror reflection breaks asymmetric signs. | Standard graph topology; torso normalization breaks (`=1.0`) when shoulders fail. |
| **6** | **Dynamic Graph Networks** | `SignDAGC` (dynamic axial graph, 2026), `ST-GCN` adaptive adjacency. | Fixed anatomical topology + feature-similarity distance bias inside `EdgeGraphBlock`. | Same static adjacency + dynamic gate. | Not genuinely dynamic topology; pairwise Euclidean distance RBF bias is modest. |
| **7** | **Cross-Articulator Interactions** | `CoSign` (co-occurrence 2023), `SignDAGC` (2026), `Uni-Sign` PGF (2025). | `CrossArticulatorGraph`: 36 phonological landmarks, 4 RBF scales (`sigma in {0.10, 0.25, 0.60, 1.50}`). | Claims `gate_init=0.1` (`pose.py` already has this). | Good domain engineering (36-pt phonological subset), incremental vs `CoSign`. |
| **8** | **State-Space Models (SSMs)** | S4 / Gu & Dao (2022–2024), continuous-time discretization in sequence modeling. | **Not implemented** (Conformer-12 / -6 active in `temporal.py`). | Proposes 6-layer `Bi-S6` without implementation. | Unimplemented; zero evidence over Conformer. |
| **9** | **Mamba (`Bi-S6`)** | `EvSLT` (Event SLT 2024), `Sign-Mamba` (SLP 2025), `SignMamba` (CSLR 2026). | **Not implemented.** | Core complexity claim (`O(T)` vs `O(T^2)`). | Negligible benefit for `T <= 180` (stride-2 -> `T_c <= 90`); slow on Apple M2 without CUDA kernels. |
| **10** | **Neural CDE & Continuous-Time Models** | Kidger (2020), `Neural RDE` (2021), `STG-NCDE` (AAAI 2022). | **Not Neural CDE.** Linear timestamp interp (`<=0.25s`) + `ContinuousTimeLagBias` / `RelativeTimeBias`. | Not included in `CARP-S2T`; claimed in `EvoHARE-Net`. | `ContinuousTimeLagBias` (Fourier `dt` + per-stream slope) is honest timestamp-aware attention, NOT a CDE. |
| **11** | **Multimodal Reliability Learning** | `QMF` (ICML 2023 provable bounds), `Uni-Sign` PGF + score-aware sampling (ICLR 2025). | 5 observable stream descriptors + `log(r_s(t))` gate in `fusion.py`; fought by `compute_fusion_entropy_loss`. | Claims Dirichlet ERF over `(vis, pose, fs)` quality vectors. | Observable `confidence_known`-aware calibration has merit once conflicting entropy loss is fixed. |
| **12** | **Evidential Deep Learning** | Sensoy (`NeurIPS` 2018), Han (`ICLR` 2021 / `TPAMI` 2023 `TMC`), Ulmer (`TMLR` 2023). | **Not implemented** (sigmoid reliability MLP in `fusion.py`). | **Invalid equations copied verbatim from classification EDL.** | Scientifically flawed in `CARP-S2T`: no ground-truth `y_m`, invalid simplex assumption. |
| **13** | **Counterfactual Multimodal Learning** | `CF-VQA` (Niu `CVPR` 2021), counterfactual causal debiasing & LVLM CFPO (2023–2025). | `roll(1)` detached post-fusion batch-swap hinge on NLL gap + KL (`decoder.py#L135-L173`). | Dropped in `CARP-S2T` (removed to reach 3 losses). | `decoder.py` acts on fused embeddings only (language-prior regularizer), NOT per-modality counterfactual reliability. |
| **14** | **Missing-Modality Robustness** | Ma (`CVPR` 2022), Lee (`CVPR` 2023 missing-aware prompts), `QMF` (2023). | Explicit `available` masks, null token for all-missing rows (`fusion.py#L135-L140`), `<=0.25s` gap invalidation. | Claims Dirichlet uncertainty `u -> 1.0`. | Clean engineering (`null` token + zero weight prevents NaN), conceptually similar to Ma/Lee masking. |
| **15** | **Continual Learning** | Yazdani (`NAACL` 2025 MSLT), Dohare (`Nature` 2024 plasticity loss). | None (offline i.i.d. training). | Only referenced inside rejected `EvoHARE-Net`. | Rightly excluded from static offline CSLT baseline. |
| **16** | **Evolutionary Neural Architectures** | `NEAT` (2002), `AmoebaNet` (2019), evolutionary NAS on fixed search budgets. | None. | Proposed & rejected (`EvoHARE-Net`). | Unjustified for offline gradient-based SLT with pretrained `FLAN-T5` decoders. |
| **17** | **Modular Structural Plasticity** | `DEN` (`ICLR` 2018), `NISPA` (2022 sparse rewiring), continual backprop (*Nature* 2024). | None. | Proposed & rejected (`EvoHARE-Net`). | Unjustified without non-stationary multi-task streams; worsens memorization of YouTube caption noise. |

---

## 4. Deep Critical Evaluation of Core Originality Questions

### 4.1 Does `CARP-S2T` Offer a Genuinely New Computational Mechanism vs. Known Combinations?

As defined across [`reports/04`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/04_FINAL_ARCHITECTURE_SPECIFICATION.md) and [`reports/05`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/05_IMPLEMENTATION_BLUEPRINT.md), **`CARP-S2T` is a modular composition of existing off-the-shelf mechanisms paired with a mathematically broken objective**. Inspecting its four advertised pillars exposes the gap:

1. **Frozen DINOv2-Base + `Region-LoRA` (`r=16`):** Standard rank-16 LoRA adaptation (Hu et al., ICLR 2022) applied to DINOv2-Base is routine engineering. Compared with `SignDINO` (CVPR 2026) and `SignDino` (`arXiv:2609.06296`), inserting unimplemented `LowRankAdapter` modules on `qkv`/`proj` adds zero new representation-learning principle.
2. **Bilateral Mirror-Canonicalized 4-Region ST-GCN (`BilateralMirrorCanonicalizer`):** Two flaws compromise originality here:
   - First, [`src/isl_mega/pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py#L275-L434) *already implements* `share_hand_encoder` (reflecting right-hand x-coordinates + adding a learned 2-way `hand_side_emb`) and `CrossArticulatorGraph` (`gate_init=0.1`). Claiming them as "new `CARP-S2T` upgrades" ignores what is already coded—and [`reports/05` L78-L88](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/05_IMPLEMENTATION_BLUEPRINT.md#L78-L88) drops `hand_side_emb` while assuming the broken `0..5` body slice order (`left_hand_slice=slice(6, 27)` vs. actual `0:21` in `pose.py`)!
   - Second, pure bilateral reflection without dominant vs. passive-base role encoding degrades Type-2 / Type-3 asymmetric ISL signs (already established in [`02_CRITICAL_MATHEMATICAL_AND_ARCHITECTURAL_ISSUES.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/phase_1_verification/02_CRITICAL_MATHEMATICAL_AND_ARCHITECTURAL_ISSUES.md#L57-L68) and [`04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md)).
3. **Dirichlet Evidential Fusion (`DirichletEvidentialFusion` in [`reports/05` L110-L157](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/05_IMPLEMENTATION_BLUEPRINT.md#L110-L157)):** This is a textbook false transfer of `TMC` (Han et al., ICLR 2021 / TPAMI 2023) and `EDL` (Sensoy et al., NeurIPS 2018):
   - In `EDL`/`TMC`, `alpha = e + 1` parameterizes a second-order belief over a **categorical class simplex** (`K` mutually exclusive classes), supervised against known one-hot classification labels `y_c`.
   - `CARP-S2T` replaces `K` classes with `M=3` **complementary continuous modalities** (`vision, pose, fingerspelling`) and copies Sensoy's exact MSE + variance + KL loss:

$$
\mathcal{L}_{\text{evidential}} = \sum_{m=1}^{M} \left[\left(y_m - \frac{\alpha_m}{S}\right)^2 + \frac{\alpha_m (S - \alpha_m)}{S^2(S+1)}\right] + \lambda_{\text{KL}}\,\mathrm{KL}\!\left(\mathrm{Dir}(\alpha)\,\|\,\mathrm{Dir}(\mathbf{1})\right)
$$

   - **Three fatal mathematical errors follow:** (i) RGB appearance and skeletal pose are **not mutually exclusive** (`y_vis = 1` does NOT imply `y_pose = 0`); (ii) no real sign dataset (*iSign*, *ISLTranslate*) contains frame-level `ground_truth_reliability` `y_m`, making the loss uncomputable as written; and (iii) if `evidence_net` simply predicts unconstrained non-negative scalars normalized by `S`, it provides zero theoretical degradation guarantee (unlike `QMF` in Zhang et al., ICML 2023, which derives sample generalization bounds using predictive modality loss/uncertainty).
4. **Bidirectional `Mamba` (`Bi-S6`):** Already explored in sign processing (`EvSLT` `2408.10488`, `Sign-Mamba` ICASSP 2025, `SignMamba` *Neural Networks* 2026). On continuous ISL utterances (`T <= 180`, reduced via `coarse_stride=2` to `<= 90` tokens), self-attention on `T_c <= 90` tokens is lightweight and preserves arbitrary pairwise frame comparisons (critical for brief 2–3 frame fingerspelling/handshape transitions that recurrent state compression risks smoothing), while pure-PyTorch selective scans run substantially slower than attention on non-CUDA hardware ([`07_HARDWARE_AND_PERFORMANCE_REPORT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/07_HARDWARE_AND_PERFORMANCE_REPORT.md)).

### 4.2 Is Counterfactual Reliability Learning Genuinely Differentiated?

The codebase and reports conflate **two unrelated mechanisms** under this heading, neither of which yet achieves true causal stream-reliability invariance:

1. **Implemented Post-Fusion Visual Sensitivity Loss ([`decoder.py#L135-L173`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py#L135-L173)):** During training, `T5VisualDecoder` creates a *detached* counterfactual sequence by batch-rolling `[visual; plan]` embeddings (`embeddings.roll(1, dims=0)`, falling back to a single zero token if `B == 1` or targets match), executes a second `no_grad` T5 forward pass, and optimizes two hinge terms pushing *real* decoder NLL lower than `cf_nll` by margin `0.5` and pushing `KL(p_real || p_cf) >= 0.2`.
   - **Differentiation vs. Existing Literature:** Unlike **CF-VQA** (Niu et al., CVPR 2021), which formulates a structural causal model to subtract the Total Direct Language Effect at inference, `decoder.py` acts purely as an auxiliary **batch-contrastive hinge regularizer against unconditional/mismatched visual conditioning**. It serves a sensible anti-shortcut purpose (aligned with `Beyond BLEU`, `arXiv:2609.03734`), **but it never intervenes on individual streams** (pose vs. RGB vs. fingerspelling) and doubles decoder pass overhead.
2. **Implemented Log-Odds Stream Quality Gate ([`fusion.py#L76-L85`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L76-L85), [`L142-L154`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L142-L154)):** Adding `log(sigmoid(reliability_net(q)))` (computed from 5 non-differentiable stream descriptors: `presence, area, sharpness/motion/spotting, conf, valid`) to `quality_gate` utility logits before softmax is mathematically an **input-conditioned log-odds bias on soft sensor fusion**.
   - **Differentiation vs. Modality Dropout / QMF / Heteroscedastic Gating:** Without stream-level counterfactual supervision or explicit epistemic/aleatoric calibration, adding `log(r_s(t))` behaves like standard **attention gating + confidence prior (`Uni-Sign` score-aware gating; `Ma et al.` missing masking)**—and in `ISL-MEGA` is actively undermined by [`compute_fusion_entropy_loss`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/losses.py#L105-L125), which penalizes deviation from uniform `[1/S, ..., 1/S]` weights even when corrupted streams *should* drop to `~0`!
   - **Where Genuine Novelty *Could* Live:** Connecting Mechanism 1 and Mechanism 2 into **stream-level counterfactual reliability invariance**—e.g., training `reliability_net(q)` and cross-stream reconstruction under asymmetric single-modality counterfactual degradation (clean pose + degraded RGB vs. clean RGB + degraded pose) while enforcing `confidence_known` provenance separation—would genuinely go beyond static modality dropout (`Ma et al.`), classification-only `QMF`, and whole-clip `roll(1)` contrastive probes.

### 4.3 Continuous-Time & `Neural CDE` Assessment

Earlier brainstorming (`EvoHARE-Net`) invoked continuous-time regional state evolution, and `ISL-MEGA` names modules [`ContinuousTimeLagBias`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py#L20-L53) and [`RelativeTimeBias`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py#L48-L62). As verified in code:

- [`align_pose_to_video`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/data.py#L77-L122) performs **discrete piecewise-linear interpolation** of pose landmarks onto decoded video timestamps whenever bracketing pose gaps are `<= 0.25s`, setting larger gaps invalid.
- `ContinuousTimeLagBias` maps normalized elapsed seconds `tau = dt / W` through `[tau, tau^2, |tau|, sin(pi*tau), cos(pi*tau), sin(2*pi*tau), cos(2*pi*tau)]` plus learned per-stream temporal slopes to bias local neighbor cross-attention; `RelativeTimeBias` applies Gaussian RBF binning on `log(1 + |dt|)`; `AdaptiveEventConsolidator` feeds `log1p(dt)` into boundary transition energy.
- **Verdict:** There is **zero Neural CDE / RDE differential-equation solver**, zero rough-path log-signature, and zero continuous-time adjoint state integration (`Kidger et al., NeurIPS 2020`; `Morrill et al., ICML 2021`). Describing this module as "Neural CDE" would draw an instant factual rejection. Describing it accurately as **timestamp-parameterized asynchronous cross-stream attention** (handling variable-FPS YouTube sign videos without expensive ODE solvers) is both honest and engineering-sound—even if conceptually adjacent to continuous relative position encodings.

### 4.4 Is an Evolutionary / Dynamically Plastic Architecture (`EvoHARE-Net`) Justified for CSLT?

We independently re-evaluate `EvoHARE-Net` (*prediction-error-driven module growth + dynamic structural plasticity*) against primary 2022–2025 continual-learning literature:

1. **Mismatch Between Offline CSLT & Plasticity Theory:** Primary plasticity results—such as **Dohare et al. (*Nature*, 2024)** on Continual Backpropagation and **NISPA (Gurbuz & Dovrolis, 2022)**—solve loss of unit responsiveness and forgetting across **non-stationary multi-task / sequential domain shifts**. Standard CSLT benchmarks (`iSign`, `ISLTranslate`) are **stationary i.i.d. single-language training distributions**. Spawning sub-modules when mini-batch translation loss spikes simply allocates dedicated capacity to memorize noisy YouTube captions, atypical lighting, or ambiguous alignments.
2. **Evidence from Continual SLT Specifically:** **Yazdani et al. (NAACL 2025)** explicitly tested continual learning algorithms (*incremental multilingual SLT*) and demonstrated that even across **four distinct sign languages**, standard fine-tuning outperforms existing continual-learning topologies—directly falsifying any prior assumption that SLT needs complex topological plasticity before mastering cross-modal alignment.
3. **Encoder–Decoder Manifold Fragility & Systems Penalty:** Pretrained seq2seq decoders (`FLAN-T5`) condition on a stationary encoder manifold via cross-attention. Injecting uninitialized expert branches midway through training injects distribution shift into `inputs_embeds` (worsened by `T5`'s dual encoder–decoder coupling in [`decoder.py#L93`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/decoder.py#L93)) while breaking static computational graphs (`torch.compile`, CUDA graphs, DDP bucket layout) on memory-constrained academic GPUs.

**Reviewer Verdict on `EvoHARE-Net`:** **Firmly unjustified** for continuous single-language SLT. Keep rejected.

---

## 5. Five-Category Novelty Matrix

| Category | Components / Designs Across `ISL-MEGA` & `CARP-S2T` | Primary Verification & Grounding |
| :--- | :--- | :--- |
| **1. Established Methods** | - Pretrained DINOv2 / ViT-B/L visual encoders (`timm`) + LoRA (`r=16`)<br>- Multi-scale 1D temporal convolution stem + Conformer / Bidirectional S6 (Mamba)<br>- Anatomical 4-region ST-GCN (hand chains, upper body, face)<br>- Character-level BiGRU-CTC fingerspelling head<br>- FLAN-T5 seq2seq text decoder + Cross-entropy + InfoNCE contrastive alignment<br>- Timestamp-relative attention biases (`sin/cos(tau)`, log-distance RBF) | `Camgoz` CVPR 2020; `SignBERT` ICCV 2021; `CoSign` ICCV 2023; `Sign2GPT` ICLR 2024; `SignDINO` CVPR 2026; `ISLiM` WSLP 2026; `ConSignformer` / `EvSLT` 2024. |
| **2. Known Combinations** | - 4-ROI visual crops (L-hand, R-hand, face, body) + global teacher / masked local student SSL<br>- Multi-expert streams (continuous + fingerspelling CTC + face/mouth) fused into T5/LLM<br>- Skeletal pose + RGB regional features fused via detector score/quality gating<br>- Latent query bottleneck (`K=32` ordered/diverse queries) + Entropic Sinkhorn OT + CTC alignment | `SignMusketeers` ACL 2025; `SignBind-LLM` ECCV 2026; `Uni-Sign` ICLR 2025 (`PGF` + score-aware); `TransAlign-SLT` WSLP 2026 (`MOT` + CTC); `Think in Latent Thoughts` ACL 2026; `AttnSign` EMNLP 2026. |
| **3. Potential New Mechanisms** *(Strict Caveats)* | - **Provenance-aware quality calibration + log-odds stream attention** (`confidence_known` distinguishing genuine detector scores from synthetic `1.0` fill + 5-descriptor ROI quality feeding `log(r_s(t))` + continuous-lag windowing across variable-FPS RGB/pose)<br>- **Kinematic salience-driven reversible event slot consolidation & unpooling** (`AdaptiveEventConsolidator` + transpose-normalized `AdaptiveDetailRetrieval`, preserving batch-padding invariance)<br>- **Stream-targeted counterfactual causal invariance** *(if extended from whole-clip `roll(1)` in `decoder.py` to single-stream pose/RGB ablation under `QMF`-style calibration)* | Partially coded in [`pose.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/pose.py), [`fusion.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/fusion.py), and [`temporal.py`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/src/isl_mega/temporal.py), **but currently unvalidated empirically and undermined by conflicting entropy loss / 9-loss bloat**. |
| **4. Closest Competing Work** | - **`ISLiM` (WSLP 2026):** Direct ISL keypoint-to-T5 competitor on *iSign* (`5.85` BLEU-4).<br>- **`Uni-Sign` (ICLR 2025):** Closest prior on score-aware RGB + pose fusion.<br>- **`SignBind-LLM` (ECCV 2026):** Closest multi-expert continuous + fingerspelling + facial CTC-to-LLM.<br>- **`TransAlign-SLT` (WSLP 2026) & `Think in Latent Thoughts` (ACL 2026):** Closest OT alignment and latent-plan bottlenecks.<br>- **`QMF` (ICML 2023):** Closest provable quality-aware dynamic fusion formulation. | [`2026.wslp-1.3`](https://aclanthology.org/2026.wslp-1.3/), [`arXiv:2501.15187`](https://arxiv.org/abs/2501.15187), [`arXiv:2509.00030`](https://arxiv.org/abs/2509.00030), [`2026.wslp-1.21`](https://aclanthology.org/2026.wslp-1.21/), [`2026.acl-long.454`](https://aclanthology.org/2026.acl-long.454/), [`arXiv:2306.02050`](https://arxiv.org/abs/2306.02050). |
| **5. Unresolved Novelty Risks** | 1. **Zero empirical results & missing 543->66 adapter:** Everything remains unverified on real ISL data.<br>2. **Mathematical invalidity of `CARP-S2T` Dirichlet ERF:** Copying TMC classification loss onto unlabelled modality reliability has no target `y_m` and fails theoretical review.<br>3. **Loss & architectural self-sabotage in code:** `compute_fusion_entropy_loss` fights stream reliability specialization; dual T5-encoder encoding wastes FLOPs; `pose.py` slice layout (`0:21` L, `21:42` R, `42:48` Body) diverges from `0..5` convention.<br>4. **Dataset leakage & metric skepticism:** *iSign* / *ISLTranslate* / *GloSL-ISL* / *WSLP-2025 FS* share overlapping YouTube source channels; `Beyond BLEU` (`2609.03734`) exposes BLEU-4 language-prior illusions.<br>5. **Scooped components:** Semantic OT (`TransAlign-SLT`), ordered latent planning (`Think in Latent Thoughts`), and local-global SSL (`SignDINO`) are already published in 2026 top venues. | [`02_TEST_FAILURE_ROOT_CAUSE.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/02_TEST_FAILURE_ROOT_CAUSE.md), [`04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/04_MATHEMATICAL_AND_ANATOMICAL_AUDIT.md), [`05_DATASET_AND_REAL_DATA_READINESS.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/05_DATASET_AND_REAL_DATA_READINESS.md), [`06_TRAINING_AND_DECODER_AUDIT.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/06_TRAINING_AND_DECODER_AUDIT.md). |

---

## 6. One Defensible Research Hypothesis & Required Evidence

To pivot from an over-claimed, un-benchmarked synthesis (`ISL-MEGA`) or a mathematically flawed proposal (`CARP-S2T`) toward a genuinely publishable, falsifiable Q1 research contribution, the project must focus on **one precision-scoped hypothesis** that directly addresses both an open gap in ISL translation (`ISLiM` / `iSign` / `ISLTranslate`) and a theoretically coherent mechanism compatible with the implemented codebase.

### The Defensible Research Hypothesis

> **Core Falsifiable Hypothesis (Provenance-Calibrated Quality-Aware Multimodal Reliability for Degraded Gloss-Free ISL Translation):**
> *"In unconstrained gloss-free Indian Sign Language translation (`iSign` and `ISLTranslate`), explicitly modulating cross-articulator 66-point pose graphs and region-cropped visual representations (`DINOv2`) via **observable, provenance-aware stream quality calibration** (`QMF`-consistent energy/loss weighting rather than uniform-entropy regularization or ungrounded simplex-Dirichlet equations) coupled with **asymmetric single-modality counterfactual degradation training** preserves sentence translation fidelity (`chrF++`, semantic QA accuracy, and `BLEU-4` retention ratio) under severe tracking jitter, unilateral hand occlusion, and variable-rate visual blur significantly better than both pose-only translators (`ISLiM`) and unweighted or raw-confidence multimodal fusion—without requiring gloss annotations or language-prior shortcut memorization."*

Notice why this hypothesis clears Q1 conceptual scrutiny while remaining realistic:
1. It **abandons** the indefensible claims (`EvoHARE-Net`, `CARP-S2T`'s broken TMC Dirichlet copy, `Bi-S6` magic speedups, and `Neural CDE` misnomers).
2. It targets a real, unsolved vulnerability confirmed across 2025–2026 literature: pose-only models (`ISLiM`, `5.85` BLEU-4) collapse under MediaPipe occlusion, while `SignMusketeers` drops pose entirely because raw MediaPipe confidence is brittle, and `Uni-Sign` requires 1,985h pretraining—leaving **calibrated provenance-aware multimodal graceful degradation on low-resource ISL** unsolved.
3. It strips `ISL-MEGA`'s 9 conflicting loss terms down to an identifiable, theoretically sound core aligned with **Quality-Aware Multimodal Fusion (`QMF`, ICML 2023)** and **Beyond-BLEU visual grounding validation (`arXiv:2609.03734`)**.

### Concrete Evidence Required to Support This Hypothesis

A Q1 submission defending this hypothesis requires **seven non-negotiable pieces of evidence** before drafting a manuscript:

1. **Codebase Integrity & Coordinate Repair (Pre-Experimental Prerequisite):**
   - Resolve the `0:21` vs. `42:48` canonical 66-point MediaPipe extraction/mapping contract via a verified 543->66 adapter, enforce explicit dominant/active-hand side indicators rather than unconditioned mirror reflection, and fix the `test_compact_v3_config_and_motion_disabled_forward_backward` temporal-gradient / test-expectation bug audited in [`02_TEST_FAILURE_ROOT_CAUSE.md`](file:///Users/likhith/Codes%20-%20Github./ISLTranslate-ideas/DINOv2-PoseGraph-3DCNN-Conformer/reports/overnight/02_TEST_FAILURE_ROOT_CAUSE.md).
2. **Theoretical & Objective Cleanup (Replacing Invalid Dirichlet & Entropy Interference):**
   - Disable `compute_fusion_entropy_loss` (which fights stream downweighting under corruption) and reformulate `reliability_net` calibration via **QMF-style predictive quality consistency / cross-modal counterfactual stream dropout**, proving mathematically and via calibration curves (ECE / risk–coverage AURC) that predicted stream weights `r_s(t)` correlate negatively with actual modality corruption without needing unobtainable `y_m` labels.
3. **Real-Data Signer- & Source-Disjoint Splits on `iSign` & `ISLTranslate`:**
   - Construct leakage-verified train/val/test partitions across physical `iSign` and `ISLTranslate` assets using DSU connected-component statistics (`source_id` + `signer_id` + channel hash), ensuring zero cross-benchmark YouTube overlap with any fingerspelling subsets (`2025.wslp-main.6`).
4. **Controlled Head-to-Head Baselines on Identical Data & Decoders:**
   - Benchmark under matched split, tokenizer, and beam settings against:
     - **Pose-only SOTA:** Reproduced `ISLiM` (`WSLP 2026`, baseline reference `5.85` BLEU-4 on `iSign`) and `ISLTranslate Pose-SLT` (`Findings ACL 2023`, `6.09`).
     - **RGB-only Baseline:** Frozen/LoRA `DINOv2-Base` crops -> matched Conformer-6 -> matched T5 decoder.
     - **Multimodal Controls (Isolated Ablations):** Unweighted additive fusion, raw MediaPipe scalar confidence gating (`Uni-Sign`-style score prior without full 5-descriptor provenance), `dropout`-only fusion, and the proposed provenance-calibrated + counterfactual stream-ablated fusion.
5. **Systematic Clean + Corrupted Degradation Curves (Zero Retraining at Test Time):**
   - Measure Retention Ratio `BLEU-4_corrupt / BLEU-4_clean` and `chrF++_corrupt / chrF++_clean` across C1 Gaussian landmark jitter (`sigma in {0.02, 0.05, 0.10}`), C2 temporal pose chunk dropout (10%–50%), C3 unilateral dominant-hand landmark occlusion, C4 RGB Gaussian motion blur (`k=15`), and C5 combined degradation—demonstrating statistically significant separation (`p < 0.01`, paired bootstrap resampling across `>=3` seeds) where weighted streams shift smoothly between RGB and pose instead of collapsing.
6. **Beyond-BLEU Grounding & Causal Attribution Verification:**
   - Following Ranum et al. (`arXiv:2609.03734`), augment SacreBLEU with `chrF++`, ROUGE-L, open-weight LLM QA salient-content scoring, fluent Deaf ISL signer evaluation (meaning preservation, fingerspelled entity accuracy, hallucination rate), **plus zeroed-visual / mismatched-visual test-time ablation (`CF-drop` gap)** proving gains do not come from `FLAN-T5` spoken-language priors.
7. **Compute & Ablation Parity Profiling:**
   - Profile wall-clock latency, peak VRAM, and convergence on single academic GPU hardware (`16–24 GB`) using frozen `DINOv2-Base` + Conformer-6 (proving experimentally whether Conformer-6 vs. `Bi-S6` even differs at `T <= 180` rather than assuming `O(T)` speedup).

---

*End of Scientific Novelty and Prior-Art Investigation (`08_SCIENTIFIC_NOVELTY_REVIEW.md`).*
