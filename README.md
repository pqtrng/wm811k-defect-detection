# WM-811K Wafer Defect Detection Pipeline

End-to-end pipeline for classifying wafer map defect patterns from the WM-811K dataset, built with a
data-engineering-first design: reproducible preprocessing, batch-orchestrated inference, and production-style packaging
rather than a one-off notebook.

## Motivation

Semiconductor fabs generate large volumes of wafer inspection data. Defect-pattern recognition (Center, Donut, Edge-Loc,
Edge-Ring, Loc, Scratch, Random, Near-full) is a key signal for yield analysis and equipment health monitoring. This
project treats the ML model as one component inside a maintainable data pipeline — the same way it would run in a fab
environment.

## Dataset

- **Source:** WM-811K (`LSWMD.pkl`) — ~811k wafer maps, ~172k labeled across 9 classes
- Heavily imbalanced (majority `none`); variable wafer-map dimensions
- Place `LSWMD.pkl` under `data/` (git-ignored)

## Pipeline Stages

1. **EDA** — class distribution, wafer-map dimensions, per-defect visualization
2. **Preprocessing** — resize to fixed grid, handle imbalance (augmentation / resampling), stratified train/val/test
   split
3. **Modeling** — baseline CNN → transfer learning (ResNet)
4. **Pipeline packaging** — preprocessing + inference scripted into a re-runnable batch job (Airflow-DAG-style
   orchestration)

## Stack

Python, PyTorch (CUDA), pandas, scikit-learn, matplotlib. Designed to run on WSL2 with GPU passthrough.

## Why this matters for streaming/batch infra

The wafer-map classifier is wrapped in a reproducible ingest → preprocess → infer → store pipeline. This mirrors how
defect-detection and equipment-health-monitoring research integrates into a fab's data backbone: high-volume inspection
data ingested into batch/streaming pipelines, with ML inference as a stage rather than an isolated experiment.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
├── data/         # dataset (git-ignored)
├── notebooks/    # 01_eda.ipynb, 02_preprocessing.ipynb
├── src/          # pipeline modules
├── models/       # trained checkpoints
└── requirements.txt
```
 