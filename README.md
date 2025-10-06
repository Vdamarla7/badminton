# Towards AI Coaches — Pose‑Centric and Explainable ML Experiments for Badminton

> Exploring AI‑assisted coaching for physical activities, starting with **badminton** and expanding to **other physical activites**. This repo contains pose datasets, code, and an LLM shot‑classification prototypes.

---

## Table of Contents
1. [Introduction](#1-introduction)  
2. [Why start with badminton?](#2-why-start-with-badminton)  
3. [Why poses?](#3-why-poses)  
4. [Open‑sourced pose releases](#4-open-sourced-poses)  
5. [Replicating prior findings (CNN & LSTM)](#5-replicated-prior-findings-cnn--lstm)  
6. [LMM/LLM‑based evaluation: motivation](#6-lmm-based-evaluation-why)  
7. [System A — PoseScript ➜ LLM shot classification](#7-system-a--posescript--llm-shot-classification)  
8. [System B — Custom text generator ➜ LLM shot classification](#8-system-b--custom-text-generator--llm-shot-classification)  
9. [Pointers: papers & datasets](#9-pointers-papers--datasets-i-find-relevant)  

— [Quickstart](#quickstart) • [Repo Structure](#repo-structure) • [Roadmap](#roadmap) • [Contributing](#contributing) • [License](#license) • [Acknowledgements](#acknowledgements)

---

## 1/ Introduction
I’m interested in building **AI‑based coaches for physical activities**. The long‑term goal is a system that can (a) **observe** your movement, (b) **explain** what’s happening in human terms, and (c) **suggest** targeted adjustments—like a good coach would.

I started work on this project because my math and programming skills improved when I started to used ChatGPT and other tools. For me, they were always available and infinitely patient coaches that explained in detail how they solved problems by sharing their chain-of-thoughts. I **wondered** if similar AI coaches were possible for physical activites as well. 

I’m starting with badminton, because I play varsity badminton and have the necessary domain knowledge. I considered guitar, which I also play, but opted against it becasue that would require processing both video and audio signals.

The primary signals for badminton coaching are **visual** (pose, court geometry, footwork, shot mechanics). 

Once the visual pipeline is solid, I’ll extend the framework to **guitar**, where we’ll fuse **audio + pose** for timing, fingering, and technique feedback.

In this repo, I will share datasets, code samples, and ML models that can be used to statistically analyze badminton games. All the data sets, code samples, and ML models are free so anyone can freely build their own methods on these datasets.

---

## 2/ Badminton

Badminton is as much a game of anticipation and preparation as it is a game of strength and speed. Understanding your opponent's playing style and favorite shots can give you the split-second advantage you need to win the point. The traditional way of getting this advantage is to observe your opponent's shot selection throughout a tournament and prepare for your match. However, today, data on every professional player is freely available on YouTube. By using modern computer vision techniques, machine learning, and statistics, players can drastically improve their preparation and training. I believe that machine learning will have a significant impact on every sport and can be used to help us reach our physical potential.

Racquet sports stress an AI system in multiple ways:

- **Speed & latency:** Shuttle/ball speeds are high; actionable coaching needs low‑latency understanding.
- **Fine‑grained actions:** Small changes in **arm angle**, **contact point**, or **stance** produce big differences in outcomes.
- **Clear segmentation:** Rallies are delimited by impact events; shot taxonomies are well‑defined.
- **Court geometry:** Lines allow approximate **free 3D calibration** from monocular video.
- **Consistency at the top:** Elite players exhibit **low intra‑class variance**, aiding robust recognition and comparison.

If we can make it work here, we should be able to generalize to other sports/activities as well.

---

## 3/ Why poses?

**Pose** provides a compact, largely appearance‑invariant representation of human movement:

- **Explainability:** Poses map directly to **kinematics** (joint angles, contact points) that coaches talk about.
- **Data efficiency:** Smaller input dimensionality vs. raw video; easier to build **sequence models**.
- **Privacy:** Poses can be shared without faces/clothing.
- **Generalization:** Less sensitive to lighting, jersey colors, or camera noise.
- **3D from 2D:** With court lines and known dimensions, we can lift to **approximate 3D** for better coaching signals.

---

## 4/ Open‑sourced poses
I’m releasing pose sequences extracted from common badminton video datasets so others can reproduce my results and have a baseline they can build with:

- **Video Badminton Dataset — Poses**  
VideoBadminton is a dataset of badminton clips that can be used to train ML models for shot classification. This data set contains over 7000 videos and contains hundreds of videos of 14 different shots. More information about this dataset may be found here: https://arxiv.org/html/2403.12385v1

I decided to create a derived dataset by extracting the poses and the bounding boxes of the players in this dataset. These poses can then be used to train ML models that utilize poses to classify shots. I am publishing this dataset for free so that anyone can work with these poses without having to extract them themselves.

Here is the link to the CSV files for the poses: [extracted poses](https://drive.google.com/file/d/14Ktq68uIm1I6CGAd1xHeOqgWw66stNo4/view?usp=sharing)

## How I created the data set:
The code is available here: [extract_poses_with_sapiens.py](https://github.com/Vdamarla7/badminton/blob/main/badminton/extract_poses_with_sapiens.py)
. I used the YOLO model to find the bounding boxes of every person in every single frame. Then I take the two biggest bounding boxes to run the Sapiens Pose Estimation model on, and I draw these onto the frame. The technique of using the two biggest bounding boxes works well for the VideoBadminton dataset, but may not work well with others, as one of the non-players may have a big bounding box.

Unfortunately, this pipeline gave me a few problems as I found that some videos in the dataset could not be read, and some videos had something odd going on with their frame count. These problems are listed in the exceptions.txt file.

Future Plans on VideoBadminton: I want to clean up some of the dataset, as there are files that I believe to be misclassified, and I want to figure out the exceptions. Furthermore, I plan on training various shot classification models using this data.

<img width="584" alt="Screenshot 2025-06-12 at 2 57 15 PM" src="https://github.com/user-attachments/assets/ba6224d1-72e0-4d8d-8294-a016e3f938cb" />

---

## 5/ Replicated prior findings (CNN & LSTM)
For the **Video Badminton Dataset**, I replicated baseline results using simple temporal models:

- **CNN baseline (frame features ➜ temporal pooling)**  
  Notebook: `notebooks/baselines/cnn_baseline.ipynb`

- **LSTM baseline (pose sequences ➜ shot label)**  
  Notebook: `notebooks/baselines/lstm_baseline.ipynb`

These aren’t state‑of‑the‑art; they’re **solid reference points** that help quantify progress when we add strong priors (poses, court geometry) and explainability.

---

## 6/ LMM‑based evaluation: why?
The deep baselines above make **correct** predictions reasonably often, but they’re **opaque**:

- It’s hard to know **what they learned** or **why** any specific prediction was made.  
- **SHAP** or saliency helps, but tends to be **cognitive‑heavy** for humans.

So I’m testing **LLM/LMM** pipelines that convert pose sequences into **natural‑language descriptions**, then ask a model to **classify shots** or **explain** mechanics in **human terms**. The aim is not just accuracy, but **useful explanations** that feel like a coach.

---

## 7/ System A — PoseScript → LLM shot classification
**Idea:** Use a pose‑to‑text generator (e.g., *PoseScript‑style* descriptions) to narrate short clips, then classify with an LLM.

**Pipeline**
```
video → pose extraction → (optional) 2D→3D lift → PoseScript-style description
      → LLM prompt (description + taxonomy) → shot label (+ rationale)
```

**Notebook:** `notebooks/llm/posescript_llm_shot_classification.ipynb`  
**Config:** `configs/llm/posescript.yaml`

**Prompt skeleton (illustrative):**
```text
You are a badminton coach. Given a pose-derived description of a short clip,
classify the shot as one of: {clear, drop, drive, lift, net, smash, serve}.
Also provide a one-sentence rationale tied to body/arm/stance.

Description:
- Player moves from base to forehand front.
- Racket shoulder abducts quickly; elbow extends; contact high and in front.
- Follow-through downward across body; opponent is midcourt.

Answer:
label: smash
rationale: ...
```

---

## 8/ System B — Custom text generator → LLM shot classification
**Idea:** Instead of generic pose narration, generate **task‑specific** text with **hand‑crafted features** (joint angles, velocities, contact height, footwork patterns, stance).

**Feature → Text rules (examples):**
- `contact_z > threshold` and `shoulder_abduction ↑` and `elbow_extension ↑` → “**high forehand contact with fast overhead swing**”
- `racket_path ~ horizontal` and `contact_z ~ chest` → “**flat drive motion**”
- `split_step → crossover → lunge` toward front court → “**aggressive approach to net**”

**Pipeline**
```
pose seq → engineered features → compact coaching phrases
         → LLM prompt (phrases + taxonomy) → shot label (+ rationale)
```

**Notebook:** `notebooks/llm/customtext_llm_shot_classification.ipynb`  
**Config:** `configs/llm/customtext.yaml`

This version yields **short, information‑dense** prompts and typically **clearer rationales**, making it easier to debug and iterate.

---

## Quickstart

### 1) Environment
```bash
# Python >= 3.10 recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you plan to run the LLM notebooks:
```bash
export OPENAI_API_KEY=YOUR_KEY   # or set in your shell/profile
```

### 2) Data
```
datasets/
  raw/
    video-badminton/      # original videos/labels (not included)
  poses/
    video-badminton/      # provided JSONL pose splits
```

### 3) Baselines
Open the notebooks:
- `notebooks/baselines/cnn_baseline.ipynb`
- `notebooks/baselines/lstm_baseline.ipynb`

### 4) LLM/LMM experiments
- `notebooks/llm/posescript_llm_shot_classification.ipynb`
- `notebooks/llm/customtext_llm_shot_classification.ipynb`

---

## Repo Structure
```
.
├── configs/
│   └── llm/
│       ├── posescript.yaml
│       └── customtext.yaml
├── datasets/
│   └── poses/
│       ├── video-badminton/
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   ├── test.jsonl
│       │   └── README.md
│       └── kaggle-badminton/          # (coming soon)
├── notebooks/
│   ├── baselines/
│   │   ├── cnn_baseline.ipynb
│   │   └── lstm_baseline.ipynb
│   └── llm/
│       ├── posescript_llm_shot_classification.ipynb
│       └── customtext_llm_shot_classification.ipynb
├── scripts/
│   ├── extract_poses.py
│   ├── lift_2d_to_3d.py
│   └── generate_text_descriptions.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Contributing
PRs are welcome! If you:
- Have additional pose datasets to share,
- Want to improve baselines,
- Or have better feature→text rules,

please open an issue or PR with a short description.

---

## License
- Code: MIT
- Pose annotations: CC BY‑NC 4.0

---

## Acknowledgements
I was inspired by Paul Liu’s work found here: https://cs.stanford.edu/people/paulliu/badminton/.

If you spot errors or have ideas for better taxonomies, I’d love to hear them—coaching is a team sport!

---

### Notes
- **Placeholders:** Any file paths or notebooks marked as *(coming soon)* or shown above as examples are intended as **placeholders** until the corresponding assets are added to the repo.  
- **Terminology:** I use **LMM** to mean large (multi)modal models and **LLM** for text‑only models. In practice, the shot‑classification prototypes here use **LLMs** fed with **pose‑derived text**.