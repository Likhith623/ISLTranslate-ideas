# Literature review and novelty audit

Search cutoff: **13 November 2026**. Scope: SLT and adjacent gesture/action modeling, concentrating on 2024–2026 while checking older mathematical and architectural antecedents. This is a targeted research review, not a systematic-review claim or a complete leaderboard. Publication information comes from the primary pages below. Authors' SOTA assertions are not independently reproduced results.

## 2024–2026 research landscape

| ID | Primary source and status | Relevant contribution | Consequence for PCAT |
|---|---|---|---|
| R1 | [Gong et al., LLMs are Good Sign Language Translators, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gong_LLMs_are_Good_Sign_Language_Translators_CVPR_2024_paper.html) | Pretrained language models are an important SLT direction. | A new decoder attachment alone is an inadequate novelty argument. |
| R2 | [Zhang, Tanzer and Firat, Scaling Sign Language Translation, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ced76a666704e381c3039871ffe558ee-Abstract-Conference.html) | Scales multilingual data, model size and translation directions with pretrained text models. | Match data and pretraining budgets before attributing gains to an operator. |
| R3 | [Gan et al., SignGraph: A Sign Sequence is Worth Graphs of Nodes, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gan_SignGraph_A_Sign_Sequence_is_Worth_Graphs_of_Nodes_CVPR_2024_paper.html) | Graph-level cross-region sign representation. | Cross-region structure is established prior art. |
| R4 | [Jiang et al., SignCLIP, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.518/) | Multilingual sign-video/text contrastive representation learning. | Pretraining and transfer baselines matter beyond narrow-domain translation scores. |
| R5 | [Jang et al., Lost in Translation, Found in Context, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.html) | Uses contextual textual cues alongside signing features. | Separate contextual information gains from visual encoder improvements. |
| R6 | [Kim et al., Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Leveraging_the_Power_of_MLLMs_for_Gloss-Free_Sign_Language_Translation_ICCV_2025_paper.html) | Adapts off-the-shelf multimodal language modeling to SLT. | A small scratch decoder is a controlled baseline, not automatically competitive with large pretrained systems. |
| R7 | [Gan et al., Learning Effective Sign Features without Text for Gloss-free Sign Language Translation, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html) | SignDINO uses sign-aware global/local visual self-supervision. | Local hand/face information and pretraining remain active bottlenecks. |
| R8 | [Thomas, Fish and Bowden, SignBind-LLM, arXiv:2509.00030v4](https://arxiv.org/abs/2509.00030v4), revised 2 September 2026 | Dedicated sign, fingerspelling and lipreading experts with learned temporal fusion. | Specialization and asynchronous fusion are not new by themselves. Treat scores as paper-reported and protocol-specific. |
| R9 | [Rao et al., SignRAG, arXiv:2610.11371v1](https://arxiv.org/abs/2610.11371v1), 8 October 2026 preprint | Hierarchical pretraining, retrieval augmentation and retrieval-aware reinforcement fine-tuning. | A late-2026 comparison must consider retrieval and external information, not just encoder architecture. |
| R15 | [Jiang et al., GCN-DevLSTM, arXiv:2403.15212](https://arxiv.org/abs/2403.15212), 2024 | Incorporates Lie-group path development into skeleton sequence modeling. | Temporal compression through geometric dynamics is a close antecedent, not a novel general idea here. |
| R16 | [Shi et al., Adaptive Global Gesture Paths and Signature Features for Skeleton-based Gesture Recognition, ICPR 2024 manuscript](https://discovery.ucl.ac.uk/id/eprint/10207332/1/ICPR2024.pdf) | Adaptive global gesture paths and signature features, evaluated on gesture/sign recognition. | Learned motion paths and signature-based sign representations already exist. |
| R18 | [Sincan et al., Gloss-free Sign Language Translation: An unbiased evaluation of progress in the field, CVIU 2025](https://www.sciencedirect.com/science/article/pii/S1077314225002218) | Unified evaluation probes the contribution of architectural versus training/preprocessing choices. | Use matched training, preprocessing and metric definitions; avoid a headline-score collage. |
| R19 | [Bridging Sign and Spoken Languages: Pseudo Gloss Generation for Sign Language Translation, NeurIPS 2025 author publication page](https://research.google/pubs/bridging-sign-and-spoken-languages-pseudo-glossgeneration-for-sign-language-translation/) | Pseudo-gloss supervision is another important route to stronger SLT. | Supervision regimes must be declared; this implementation uses text targets only. |

R7's title/venue and abstract were available through the primary CVF search index; direct page retrieval returned HTTP 403. Several other CVF and publisher pages similarly blocked a later direct fetch despite indexed primary abstracts. This review does not claim full-text inspection of those inaccessible papers or infer detailed numerical superiority from their abstracts. Accessible primary PDFs and author manuscripts below provided the most important novelty checks. Search-engine crawl dates were not used as publication dates.

## Older and adjacent work that prevents broad novelty claims

| ID | Primary source | Direct overlap |
|---|---|---|
| R10 | [Li, Zhang and Jin, LPSNet, ICCV Workshops 2017](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w11/Li_LPSNet_A_Novel_ICCV_2017_paper.pdf) | Uses path/log-path signatures for hand gesture recognition. “Signatures for gestures” is not new. |
| R11 | [Li et al., Skeleton-Based Gesture Recognition Using Several Fully Connected Layers with Path Signature Features and Temporal Transformer Module, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4878) | Spatial/temporal signatures, local/global temporal structure and learned temporal shifting. |
| R12 | [Kidger et al., Deep Signature Transforms, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/d2cdf047a6674cef251d56544a3cf029-Abstract.html) | Differentiable learned signature layers and stream-preserving lifts. End-to-end signature learning is established. |
| R13 | [Cheng et al., Skeleton-Based Gesture Recognition With Learnable Paths and Signature Features, IEEE TMM accepted manuscript, 2023](https://discovery.ucl.ac.uk/id/eprint/10178174/) | Learnable paths and plug-in spatial/temporal signature modules. |
| R14 | [Lou, Li and Ni, Path Development Network with Finite-dimensional Lie Group Representation, TMLR August 2024; arXiv v2 September 2024](https://arxiv.org/abs/2204.00740v2) | Trainable finite-dimensional Lie-group representations of paths. Group-valued trajectory layers and their stability motivations are established. |
| R17 | [Camgoz et al., Multi-channel Transformers for Multi-articulatory Sign Language Translation, ECCV 2020 Workshops](https://arxiv.org/abs/2009.00299) | Explicit asynchronous multi-articulator modeling. The linguistic motivation predates this project. |

## Novelty claim ledger

**Established ingredients, not claimed new:** anatomical reference coordinates, positive-definite metrics, arclength parameterization, interval overlap clipping, antisymmetric second-order signatures, Cayley transforms, Transformer decoding and byte tokenization. The classical invariance and orthogonality proofs are supplied for correctness, not as discoveries.

**Candidate contribution:** use a *shared trainable progress measure* to define exactly clipped local signed-area operators, then use these operators as *bounded orthogonal cross-articulator feature transport*, retaining the anatomical feature state and explicit order statistics. Test this single operation against identity-transport and classical signature controls with the same decoder. It is more specific than simply concatenating pretrained model outputs. Whether that specificity constitutes sufficient scientific novelty depends on a fuller prior-art review and experimental findings.

**Not defensible from this deliverable:** first-ever multi-articulator model; first gesture signature network; first invariant sequence layer; exact recovery of unobserved 3D motion; proof of superior BLEU; solved fingerspelling; universal sign-language equivalence under speed changes; or a guaranteed Q1/CVPR/ACL paper.

Exact-phrase searches for holonomy/area/Cayley transport in sign-language modeling and searches for learned metric progress/signature gesture modeling did not reveal an identical formulation in this targeted review. Absence in a search is not evidence of absence. R11–R16 must be compared in detail during an actual submission's related-work and baseline phase. In particular, Cayley transport is only a rational action of a level-two statistic here; it is not an exact replacement for the recurrent Lie-group development in R14/R15.

## Why there is no single “2026 SOTA BLEU” in this report

Different papers use different sign languages, dataset variants, external training corpora, language-model sizes, gloss/pseudo-gloss supervision, context, retrieval galleries and tokenization. A score from one regime is not a controlled architectural comparison with another. This repository has no trained real-corpus checkpoint and makes no leaderboard claim. See `EXPERIMENTS.md` for the experiment required to test the actual hypothesis.

## Additional v0.2 checks

[R20] [Gueuwou et al., SignMusketeers, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.1157/) studies efficient representations of relevant hand, face and posture information and explicitly discusses limitations of off-the-shelf pose coordinates. It supports keeping the landmark-input boundary visible; preserving all supplied coordinates does not recover visual information absent from them.

[R21] [Jang et al., Lost in Translation, Found in Embeddings, arXiv:2512.08040](https://arxiv.org/abs/2512.08040) describes unified translation/alignment using keypoints and lip-region images. This abstract-level check reinforces that adding mouthing observations is a separate sensory extension, not a theorem about landmark-only input.

The v0.2 additions remain established mathematical tools combined in a specific research construction. Prefix interval extraction follows the classical ordered-signature concatenation identity. Progress/time augmentation, multiscale descriptors and bypassing an aggressive bottleneck are not claimed as individually unprecedented. The strict UTF-8 inference mask is an implementation-correctness change based on [RFC 3629](https://www.rfc-editor.org/rfc/rfc3629), not an SLT research novelty claim.
