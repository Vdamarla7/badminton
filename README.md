# AI Coaches for Physical Activities

> Exploring AI-assisted coaching for physical activities, starting with **badminton** and expanding to **other physical activities** over time. This repo contains pose datasets, code, and an LLM shot-classification prototype.

---

## Table of Contents
1. [Introduction](#1-introduction)  
2. [Why badminton?](#2-why-badminton)  
3. [Why poses?](#3-why-poses)
4. [Open sourced pose releases](#4-open-sourced-pose-releases)
5. [Replicating prior findings (CNN & LSTM)](#5-replicating-prior-findings-cnn--lstm)  
6. [LMM/LLM-based evaluation: motivation](#6-lmmllm-based-evaluation-motivation)  
7. [System A — PoseScript ➜ LLM shot classification](#7-system-a--posescript--llm-shot-classification)  
8. [System B — Custom text generator ➜ LLM shot classification](#8-system-b--custom-text-generator--llm-shot-classification)  
9. [Pointers: papers & datasets](references.md)  

— [Quickstart](#quickstart) • [Repo Structure](#repo-structure) • [Contributing](#contributing) • [License](#license) • [Acknowledgements](#acknowledgements)

---

## 1: Introduction
I’m interested in building **AI-based coaches for physical activities**. The long-term goal is a system that can (a) **observe** your movement, (b) **explain** what’s happening in human terms, and (c) **suggest** targeted adjustments—like a good coach would.

I started this project because my math and programming skills improved when I used ChatGPT and other tools. They felt like always-available, infinitely patient coaches that explained solutions step by step. I **wondered** if similar AI coaches were possible for physical activities as well.

I’m starting with badminton because I play varsity badminton and have the necessary domain knowledge. I considered guitar, which I also play, but opted against it because that would require processing both video and audio signals.

The primary signals for badminton coaching are **visual** (pose, court geometry, footwork, shot mechanics). Once the visual pipeline is solid, I’ll extend the framework to **guitar**, where we’ll fuse **audio + pose** for timing, fingering, and technique feedback.

In this repo, I share datasets, code samples, and ML models for statistical analysis of badminton games. Everything is free so anyone can build their own methods on these datasets.

---

## 2: Why badminton?

Badminton is as much a game of anticipation and preparation as it is a game of strength and speed. Understanding your opponent’s playing style and favorite shots can provide the split-second advantage you need to win a point. Traditionally, players observe opponents’ shot selection throughout a tournament and prepare for their match. Today, match footage is widely available online; with modern computer vision, ML, and statistics, players can drastically improve preparation and training.

Racquet sports stress an AI system in multiple ways:

- **Speed & latency:** Shuttle/ball speeds are high; actionable coaching needs low-latency understanding.  
- **Fine-grained actions:** Small changes in **arm angle**, **contact point**, or **stance** produce big outcome differences.  
- **Clear segmentation:** Rallies are delimited by impact events; shot taxonomies are well-defined.  
- **Court geometry:** Lines allow approximate **free 3D calibration** from monocular video.  
- **Consistency at the top:** Elite players exhibit **low intra-class variance**, aiding robust recognition and comparison.

If we can make it work here, we should be able to generalize to other sports/activities.

---

## 3: Why poses?

**Pose** provides a compact, largely appearance-invariant representation of human movement:

- **Explainability:** Poses map to **kinematics** (joint angles, contact points) that coaches discuss.  
- **Data efficiency:** Smaller dimensionality than raw video; easier to build **sequence models**.  
- **Privacy:** Poses can be shared without faces/clothing.  
- **Generalization:** Less sensitive to lighting, jersey colors, or camera noise.  
- **3D from 2D:** With court lines and known dimensions, we can lift to **approximate 3D** for better coaching signals.

---

## 4: Open sourced pose releases
I’m releasing pose sequences extracted from common badminton video datasets so others can reproduce my results and have a baseline to build on.

- **VideoBadminton — Poses**  
  VideoBadminton is a dataset of badminton clips for shot classification (7k+ clips; 14 shot classes). More info: https://arxiv.org/html/2403.12385v1

I created a derived dataset by extracting player poses and bounding boxes from each clip. These pose sequences can train pose-aware shot classifiers. I’m publishing them so anyone can work with poses without re-extracting.

**CSV files for the poses:** [extracted poses](https://drive.google.com/file/d/14Ktq68uIm1I6CGAd1xHeOqgWw66stNo4/view?usp=sharing)

More details [here](badminton/extractor.md).

---

## 5: Replicating prior findings (CNN & LSTM)
For **VideoBadminton**, I replicated simple temporal baselines:

- **CNN baseline (frame features ➜ temporal pooling)**  
  Notebook: `notebooks/baselines/cnn_baseline.ipynb`

- **LSTM baseline (pose sequences ➜ shot label)**  
  Notebook: `notebooks/baselines/lstm_baseline.ipynb`

They’re not SOTA; they’re **solid reference points** to quantify progress when adding poses, court geometry, and explainability.

---

## 6: LMM/LLM-based evaluation: motivation
The deep baselines above make **correct** predictions reasonably often, but they’re **opaque**:

- It’s hard to know **what they learned** or **why** a specific prediction was made.  
- **SHAP** or saliency helps, but is often **cognitively heavy** for humans.

So I’m testing **LLM/LMM** pipelines that convert pose sequences into **natural-language descriptions**, then classify shots or **explain mechanics** in human terms. The goal isn’t just accuracy; it’s **useful, coach-like explanations**.

---

## 7: System A — PoseScript → LLM shot classification
**Idea:** Use a pose-to-text generator (e.g., PoseScript-style descriptions) to narrate short clips, then classify with an LLM.

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

## 8: System B — Custom text generator → LLM shot classification
**Idea:** Instead of generic pose narration, generate **task-specific** text with **hand-crafted features** (joint angles, velocities, contact height, footwork patterns, stance).

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

This version yields **short, information-dense** prompts and typically **clearer rationales**, making it easier to debug and iterate.

---

## 9: Pointers: papers & datasets
A living, opinionated list of references relevant to this repo (pose estimation, 2D→3D, court calibration, racquet-sport datasets, anticipation/prediction, simulation, pose↔language, pose-centric action recognition).  
See: `REFERENCES.md` (or open an issue/PR to suggest additions).

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
- Pose annotations: CC BY-NC 4.0

---

## Acknowledgements
Inspired by Paul Liu’s work: https://cs.stanford.edu/people/paulliu/badminton/  
If you spot errors or have ideas for better taxonomies, I’d love to hear them—coaching is a team sport!

---

### Notes
- **Placeholders:** Any file paths or notebooks marked as *(coming soon)* are placeholders until assets are added.  
- **Terminology:** I use **LMM** to mean large (multi)modal models and **LLM** for text-only models. In practice, the shot-classification prototypes here use **LLMs** fed with **pose-derived text**.
