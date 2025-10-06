## Relevant reference papers & datasets

> A curated and opinionated reference papers and datasets across pose estimation, 2D→3D lifting, court geometry, racquet‑sport datasets, anticipation/prediction, simulation/generation, pose⇄language, and pose‑centric action recognition.

### Pose estimation (2D & on‑device / 3D)
- **OpenPose — Realtime multi‑person 2D keypoints (PAFs)** — Cao et al., CVPR 2017 · [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.pdf) · [extended arXiv](https://arxiv.org/abs/1812.08008)
- **HRNet — high‑resolution representations for pose** — Sun et al., CVPR 2019 · [arXiv](https://arxiv.org/abs/1902.09212) · [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)
- **AlphaPose / RMPE — top‑down multi‑person** — Fang et al., ICCV 2017 · [arXiv](https://arxiv.org/abs/1612.00137) · [code](https://github.com/MVIG-SJTU/AlphaPose)
- **MoveNet — ultra‑fast 17‑kp (TF/TFLite/TFJS)** — Google, 2021 · [blog](https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html)
- **BlazePose — on‑device real‑time 2D/3D** — Bazarevsky et al., 2020 · [arXiv](https://arxiv.org/abs/2006.10204) · **BlazePose GHUM Holistic (3D)** — Grishchenko et al., 2022 · [arXiv](https://arxiv.org/abs/2206.11678)

### 2D→3D lifting & motion representations
- **Simple Baseline for 3D from 2D** — Martinez et al., ICCV 2017 · [arXiv](https://arxiv.org/abs/1705.03098) · [paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Martinez_A_Simple_yet_ICCV_2017_paper.pdf)
- **VideoPose3D — temporal convs over 2D keypoints** — Pavllo et al., CVPR 2019 · [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.pdf) · [code](https://github.com/facebookresearch/VideoPose3D)
- **PoseFormer — transformer for 3D pose** — Zheng et al., ICCV 2021 · [arXiv](https://arxiv.org/abs/2103.10455)
- **MotionBERT — unified motion pretraining** — Zhu et., ICCV 2023 · [arXiv](https://arxiv.org/abs/2210.06551) · [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.pdf)

### Court calibration & geometry (using field/line markings)
- **Robust camera calibration using court models** — Farin et al., 2004 · [paper](https://dirk-farin.net/publications/data/Farin2004b.pdf)
- **TVCalib — sports field registration as calibration** — Theiner et al., WACV 2023 · [paper](https://openaccess.thecvf.com/content/WACV2023/papers/Theiner_TVCalib_Camera_Calibration_for_Sports_Field_Registration_in_Soccer_WACV_2023_paper.pdf)
- **Badminton court auto‑calibration** — Ma et al., 2022 · [overview](https://scispace.com/papers/robust-automatic-camera-calibration-in-badminton-court-8rjcl4in)
- **Sports field registration via points & lines (multi‑view)** — 2024 · [arXiv](https://arxiv.org/abs/2404.08401)
- **Accurate tennis court line detection for amateur video** — 2024 · [arXiv](https://arxiv.org/pdf/2404.06977)

### Racquet‑sport datasets (badminton‑first) & related resources
- **ShuttleSet (KDD’23)** — stroke‑level singles dataset, 36k strokes · [arXiv](https://arxiv.org/abs/2306.04948) · [project](https://github.com/wywyWang/CoachAI-Projects)
- **VideoBadminton (2024)** — action recognition benchmark · [arXiv](https://arxiv.org/abs/2403.12385)
- **BadmintonDB (ACM ’22)** — player‑specific match analysis · [paper](https://dl.acm.org/doi/10.1145/3552437.3555696)
- **MultiSenseBadminton (Sci Data ’24)** — wearable multi‑sensor strokes · [paper](https://www.nature.com/articles/s41597-024-03144-z)
- **Kaggle: badminton stroke video** · [dataset](https://www.kaggle.com/datasets/shenhuichang/badminton-storke-video)
- **Kaggle: motion data for badminton shots** · [dataset](https://www.kaggle.com/datasets/dylanyves/motion-data-for-badminton-shots)
- **Kaggle: badminton pose‑estimation** · [dataset](https://www.kaggle.com/datasets/sonnig/badminton-pose-estimation)

_Table tennis (closely related fine‑grained stroke tasks):_
- **TTStroke‑21 / MediaEval Sports Video task (’21–’22)** · [2021 overview](https://arxiv.org/pdf/2112.11384) · [2022 task](https://multimediaeval.github.io/editions/2022/tasks/sportsvideo/) · [baseline 2022](https://arxiv.org/pdf/2302.02752)

### Anticipation / prediction (pre‑impact, trajectory)
- **Tennis: predict future shot direction from pose + position** — Shimizu et al., 2019 · [ACM DOI](https://dl.acm.org/doi/10.1145/3347318.3355523)
- **Badminton: predict shuttle trajectory** — Nokihara et al., 2023 · [open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC10219238/)
- **Tennis ball localization/trajectory with pose priors** — Xiao et al., 2024 · [arXiv](https://arxiv.org/abs/2401.17185)

### Simulation & video‑realistic players
- **Vid2Player — controllable video sprites of pro tennis players** — Zhang et al., TOG 2021 · [paper](https://dl.acm.org/doi/abs/10.1145/3448978) · [project](https://cs.stanford.edu/~haotianz/vid2player/)
- **Learning physically simulated tennis skills from broadcast videos** — Zhang et al., SIGGRAPH 2023 · [paper](https://dl.acm.org/doi/abs/10.1145/3592408) · [project](https://cs.stanford.edu/~haotianz/vid2player3d/)
- **Vid2Actor — animatable person from video** — Weng et al., 2020 · [arXiv](https://arxiv.org/abs/2012.12884)

### Pose ⇄ language & LLM/LMM motion interfaces
- **PoseScript — 3D poses paired with language; pose→text** — Delmas et al., ECCV 2022 · [arXiv](https://arxiv.org/abs/2210.11795) · [paper PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660340.pdf) · [code](https://github.com/naver/posescript)
- **PoseGPT — quantization‑based motion generation/forecasting** — Lucas et al., ECCV 2022 · [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660409.pdf)
- **MotionGPT — treating motion as a language** — NeurIPS 2023 · [openreview](https://openreview.net/forum?id=WqiZJGNkjn) · [code](https://github.com/OpenMotionLab/MotionGPT)

### Sports action recognition with pose (fine‑grained)
- **Two‑stream RGB+Pose w/ attention (TTStroke‑21)** — Hacker et al., 2023 · [arXiv](https://arxiv.org/abs/2302.02755)
- **Three‑stream 3D/1D CNN (TTStroke‑21)** — Martin et al., 2021 · [arXiv](https://arxiv.org/abs/2109.14306)
- **Continuous video → stroke signals (swim/tennis)** — Victor et al., 2017 · [arXiv](https://arxiv.org/abs/1705.09894)
- **Tennis strokes from generated stick‑figure poses** — Bačić et al., 2022 · [PDF](https://www.scitepress.org/Papers/2022/108273/108273.pdf)

### Industry uses of pose for sports reporting (context)
- **NYT + wrnch: 3D pose for gymnastics coverage** — NVIDIA Dev Blog, 2020 · [article](https://developer.nvidia.com/blog/nyt-wrnch-ai-3d-pose-estimation/)
- **MLB Statcast: pose‑tracking visualizations** — MLB.com, 2020 · [post](https://www.mlb.com/news/hawk-eye-statcast-pose-tracking-the-best-2020-postseason-moments)
