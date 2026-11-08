# Research references consulted

These works motivate design directions. They do not prove that ISL-MEGA is novel or effective, and their reported benchmark results should not be compared directly unless data, splits, preprocessing and metric signatures match.

1. Gan et al. (2026). **Learning Effective Sign Features without Text for Gloss-free Sign Language Translation.** CVPR 2026. [Official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html). Global teacher and masked hand/face local student views for text-free sign visual pretraining.
2. **SignMusketeers.** Findings of ACL 2025. [ACL Anthology](https://aclanthology.org/2025.findings-acl.1157/). Specialist visual features for hands/face/body in sign translation.
3. **Uni-Sign.** ICLR 2025. [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html). Prior-guided RGB-pose representation learning for sign understanding; note its task/data context differs from ISL.
4. **SignBind-LLM.** ECCV 2026. [Project/paper page](https://cogvis-cvssp.github.io/papers/signbind/). Specialist expert idea for continuous signing, fingerspelling and lipreading; its evaluated datasets are not ISLTranslate.
5. **AttnSign.** Findings of EMNLP 2026. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1435/). Attention steering and visual grounding for sign translation.
6. **Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation.** ACL 2026. [ACL Anthology](https://aclanthology.org/2026.acl-long.454/). Latent planning/grounding direction.
7. Ahmed and Mishra (2026). **ISLiM: A Compact End-to-End Sign Language Translation Framework.** WSLP 2026. [ACL Anthology](https://aclanthology.org/2026.wslp-1.3/). Relevant ISL pose CNN/Transformer/T5 baseline.
8. **ISLTranslate.** Findings of ACL 2023. [ACL Anthology](https://aclanthology.org/2023.findings-acl.665/). Dataset and original benchmark context.
9. **Continuous Fingerspelling Dataset for Indian Sign Language.** WSLP 2025. [ACL Anthology](https://aclanthology.org/2025.wslp-main.6/). ISL character/fingerspelling data resource.

## Design attribution

ISL-MEGA does not claim to reproduce SignDINO, SignMusketeers, Uni-Sign, SignBind-LLM or AttnSign. It implements its own software synthesis of selected high-level ideas. Any future publication should cite original methods, compare against available implementations where possible, and clearly separate reproduced baselines from the proposed contribution.
