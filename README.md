# English-Telugu Neural Machine Translation (NMT)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&logo=Jupyter&logoColor=white)

Welcome to the final submission repository for our **English-Telugu Neural Machine Translation (NMT)** project! This repository contains all the deliverables, including notebooks for training, benchmarking, and visualizing our Transformer-based NMT models.

## Deliverables Overview

| File / Directory | Purpose |
|---|---|
| `nmt_multi_size_final.ipynb` | Trains and saves final Transformer NMT checkpoints for `10M`, `30M`, `50M`, and `60M` parameter targets. |
| `benchmarks.ipynb` | Benchmarks checkpoints on the **FLORES-200** (English-Telugu) dataset and reports BLEU, chrF, TER, and inference speed metrics. |
| `visualizations/visualizations_combined.ipynb` | Generates embedding alignment and attention-focused visualizations from trained models. |
| `requirement.txt` | Python dependency list required to run the notebooks in this submission. |
| `visualizations/*.png` | Pre-exported visualization figures included as static outputs for quick review. |

## Recommended Review Order

We recommend reviewing the files in the following order to understand the complete pipeline:
1. **Training & Architecture**: Open `nmt_multi_size_final.ipynb` to review the training setup, Transformer model definitions, and checkpoint generation process.
2. **Evaluation & Metrics**: Open `benchmarks.ipynb` to review the evaluation pipeline and the reported benchmark metrics across different model sizes.
3. **Qualitative Analysis**: Open `visualizations/visualizations_combined.ipynb` (and accompanying PNGs in the `visualizations/` folder) to review attention maps and embedding alignments.

## Environment Setup (Optional for Reproducibility)

If you wish to rerun the notebooks locally, please set up your Python environment using the following steps:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirement.txt
```

**On Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

Once dependencies are installed, you can launch Jupyter Notebook or JupyterLab to interact with the `.ipynb` files.

## Submission Scope

This submission folder intentionally focuses on the final notebooks and generated visualization outputs. Large intermediate artifacts (e.g., full training caches, massive datasets, and large checkpoint collections) are not duplicated here to keep the submission lightweight, unless specifically required for the final review.


