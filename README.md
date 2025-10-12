# QTopoQA
QTopoQA - Quantum Topological Protein Complex Interface QA. Quantum paper for PhD in progress.

Based on the original 2024 work of "TopoQA: a topological deep learning-based approach for protein complex structure interface quality assessment" - Han, Xia, et al.

QTopoQA is an experimental research codebase that explores topological data analysis (TDA) and quantum-inspired operators to assess the quality of protein complex interfaces (Interface QA / EMA).

The project aims to combine persistent homology & graph learning (GNN) with quantum-native constructions of the boundary operator to study when topological signals improve ranking and error detection in predicted complexes.

Github Repo structure (as of now): datasets/, qtdaqa/, scripts/, topoqa/, topoqa_train/ plus .gitignore and README.md.

## 📊 Traffic & Popularity
![Clones](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FQTopoQA-clones.json%3Fv%3D2)
![Views](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F9b749f24de62343dc995f8d524027c39%2Fraw%2FQTopoQA-views.json%3Fv%3D2)
![Stars](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FQTopoQA-stars.json%3Fv%3D2)
![Commits](https://img.shields.io/endpoint?cacheSeconds=300&url=https%3A%2F%2Fgist.githubusercontent.com%2Fkugendran-naidoo%2F2b0de4f9f92a605b780e986e6d48ffcc%2Fraw%2FQTopoQA-commits.json%3Fv%3D2)

> Auto-updated daily at 14:00 UTC via GitHub Actions.

## 📈 Metrics
![Activity (last 4 weeks)](https://raw.githubusercontent.com/kugendran-naidoo/QTopoQA/main/metrics/activity_4w.png)

> Auto-updated daily at 14:00 UTC via GitHub Actions.

# Table of contents

What is this?
Features
Repository layout
Installation
Data
Quick start
Reproducing experiments
How it works (conceptual)
Figures to add
Roadmap / help wanted
Citing
License
Acknowledgments

**What is this?**

Protein complex structure predictors (e.g., AF-Multimer / AF3) still benefit from quality assessment (QA) models that can rank candidate complexes without native structures.
Recent work (e.g., TopoQA) shows that persistent homology features coupled with GNNs can improve interface QA and ranking on standard benchmarks by capturing higher-order structure at interfaces.

**QTopoQA extends this direction and provides a sandbox to:**

build topological descriptors at residue/atom/interface level,

plug them into geometric/graph models,

and experiment with quantum/fermionic encodings of boundary operators (e.g., Projection-Friendly Tree Mapping, PFTM) for efficiency and new inductive biases.

This repo is under active development; interfaces and paths may change.

**Features**

Topological interface features. Hooks for PH-based summaries of local neighborhoods and interface graphs.

Graph & geometric models. A place to prototype GNN baselines and topological augmentations.

Quantum-inspired operators. Utilities for building/logging parity-tree (PFTM-style) boundary operators that keep projection-friendliness while reducing operator depth compared to Jordan–Wigner.

Scriptable training/eval. Reproducible CLI entry points and job scripts (local or cluster).

Dataset stubs. Folder conventions for placing or linking interface QA datasets.

**Installation**

Python: 3.9–3.11 recommended.

**Data**

This project expects a datasets/ directory with subfolders for the benchmarks you use.
Popular public sets in the literature include DBM55-AF2, HAF2, and new AF3-based sets. See the TopoQA paper for references and acquisition details.

**Repository Layout**
```text
QTopoQA/
├─ datasets/         # Put or symlink datasets here (see “Data”)
├─ qtdaqa/           # Quantum/Topological operator utilities (PFTM, parity trees, etc.)
├─ scripts/          # Helper scripts: preprocessing, training, evaluation, plotting
├─ topoqa/           # Core library code (data loaders, models, losses, metrics)
├─ topoqa_train/     # Train / eval entry points and experiment configs
├─ .gitignore
└─ README.md
