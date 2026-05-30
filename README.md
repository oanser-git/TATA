# TATA: Benchmark NIDS Test Sets Assessment and Targeted Augmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Paper:** *"TATA: Benchmark NIDS Test Sets Assessment and Targeted Augmentation"* — accepted in the **Main Track** of *ESORICS 2025*.

**TATA** is a model-agnostic framework that evaluates benchmark Network-Intrusion-Detection-System (NIDS) test sets via three complementary, dataset-centric metrics — **diversity**, **proximity**, and **scarcity** — and then closes uncovered gaps by guiding the generation of real network flows through offline deep reinforcement learning.

---

## Table of Contents

- [Quick Start](#quick-start)
- [What TATA Does](#what-tata-does)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Reproducing the Paper](#reproducing-the-paper)
  - [Main Pipeline](#main-pipeline)
  - [Individual Sections](#individual-sections)
- [Standalone Scripts](#standalone-scripts)
- [Hardware & Runtime Estimates](#hardware--runtime-estimates)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# 1. Clone and enter the repo
cd tata

# 2. Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Run the full reproduction pipeline with a test dataset
python scripts/reproduce_paper.py --dataset ids2017 --no-run-rl --ablation-epochs 10 --ablation-splits 2
```

---

## What TATA Does

1. **Preliminary Phase** — Trains a contrastive autoencoder on the training split to construct a structured latent space where flows cluster by traffic type.
2. **Phase 1: Assessment** — Embeds the test split and computes three metrics cluster-wise:
   - **Diversity** (Vendi score) — how varied are the test points?
   - **Proximity** (one-sided KS test) — how borderline are they vs. the training set?
   - **Scarcity** (1 − Gini coefficient) — how evenly distributed across negative clusters?
3. **Phase 2: Augmentation** — An offline RL agent (CQL / TD3+BC) directs a configurable traffic testbed (loss, jitter, delay, etc.) to generate additional flows that maximize the D/P/S reward.

---

## Repository Structure

```
configs/              # YAML hyperparameters for all modules
data/
  ids2017/            # Synthetic test data (bundled)
  ids2018/            # Synthetic test data (bundled)
  artifacts/          # Precomputed outputs (.gitkeep only)
  processed/          # Preprocessed splits (.gitkeep only)
  offline_transitions/  # RL transition data (.gitkeep only)
scripts/
  reproduce_paper.py              # Main orchestrator (recommended entry point)
  run_preliminary.py              # Phase 0 only (standalone)
  run_assessment.py               # Phase 1 only (standalone)
  collect_offline_data.py         # Phase 2a: collect RL transitions
  train_rl_agent.py               # Phase 2b: train CQL/TD3+BC
  evaluate_rl_agent.py            # Phase 2c: evaluate trained policy
  preprocess_data.py              # Data prep (standalone)
  experiments/
    run_ablation.py               # Section 4.2
    run_correlation.py            # Section 4.3
    run_augmentation_eval.py      # Section 4.4
    run_comparative_analysis.py   # Section 4.5
    run_split_select_retrain.py   # Appendix B
tata/                 # Main package (was src/)
  datasets/           # Generic CSV loaders, splits, preprocessing
  models/             # Contrastive AE, NIDS (RF/SVM/DNN), training, HPO
  embedding/          # Encoder wrapper, clustering
  metrics/            # Diversity, proximity, scarcity, assessment
  rl/                 # Environment, reward, offline dataset, FQE, HPO
  testbed/            # Mock testbed, QEMU testbed, scenarios
  experiments/        # Section 4.2–4.5 implementations
```

---

## Installation

**Requirements:** Python 3.10+, Linux/macOS (Windows untested). QEMU testbed requires Linux with KVM.

```bash
pip install -e ".[dev]"    # install package + dev dependencies (black, flake8, mypy)
```

Or with `uv`:

```bash
uv sync
```

---

## Datasets

TATA expects **CSV datasets** placed in `data/<dataset_name>/` directories.

**Supported datasets** (configured in `tata/datasets/loaders.py`):

| Dataset | Expected Path | Label Column | Notes |
|---------|--------------|--------------|-------|
| IDS2017 | `data/ids2017/` | `"Label"` | Refined CIC-IDS2017 CSV files |
| IDS2018 | `data/ids2018/` | `"Label"` | Refined CIC-IDS2018 CSV files |
| NSL-KDD | `data/nsl-kdd/` | `"label"` | `KDDTrain+.csv` (no header, 41 features + label + difficulty) |
| UNSW-NB15 | `data/unsw-nb15/` | `"label"` | `UNSW_NB15_training-set.csv` |
| CIC-DDoS2019 | `data/cic-ddos2019/` | `" Label"` | Note leading space |
| Bot-IoT, ToN-IoT, CTU-13, ISCX-IDS2012, ISCX-Tor, VPN-NonVPN, CIC-UNSW | `data/<name>/` | `"label"` or `"Label"` | Directory of CSVs |
| MNIST | — | — | Fetched via `sklearn.datasets.fetch_openml` |
| CIFAR-10 | `data/cifar10/` | — | Fetched via `torchvision` |

**Important:**
- All features must be numeric. If your dataset contains categorical columns, apply one-hot encoding before placing the CSVs in `data/<dataset_name>/`.

---

## Reproducing the Paper

### Main Pipeline

The recommended entry point is `scripts/reproduce_paper.py`, which runs the full pipeline sequentially:

```bash
python scripts/reproduce_paper.py \
    --dataset ids2017 \
    --data-dir data \
    --artifacts-dir data/artifacts \
    --device cpu \
    --random-state 42
```

**Phase toggles** (all enabled by default, disable with `--no-run-*`):
- `--run-ablation` — Section 4.2
- `--run-correlation` — Section 4.3
- `--run-rl` — Phase 2 + Appendix B (requires significant time)
- `--run-augmentation` — Section 4.4 (disabled by default; requires trained agents)
- `--run-comparative` — Section 4.5 (disabled by default; requires multiple datasets)

**Fast test run** (reduced epochs, skips RL):
```bash
python scripts/reproduce_paper.py \
    --dataset ids2017 \
    --no-run-rl \
    --ablation-epochs 10 \
    --ablation-splits 2 \
    --correlation-variants 2
```

### Individual Sections

Each paper section can also be run standalone:

| Section | Script | Command |
|---------|--------|---------|
| 4.2 Ablation | `scripts/experiments/run_ablation.py` | `python scripts/experiments/run_ablation.py --dataset ids2017 --n-splits 10` |
| 4.3 Correlation | `scripts/experiments/run_correlation.py` | `python scripts/experiments/run_correlation.py --dataset ids2017` |
| 4.4 Augmentation | `scripts/experiments/run_augmentation_eval.py` | Requires trained agents in `d3rlpy_logs/` |
| 4.5 Comparative | `scripts/experiments/run_comparative_analysis.py` | `python scripts/experiments/run_comparative_analysis.py --datasets ids2017 ids2018` |
| Appendix B (SSR) | `scripts/experiments/run_split_select_retrain.py` | `python scripts/experiments/run_split_select_retrain.py --dataset ids2017 --agent-path ...` |

---

## Standalone Scripts

In addition to the main `reproduce_paper.py` orchestrator, the following standalone scripts are provided for modular execution:

| Script | Purpose |
|--------|---------|
| `scripts/run_preliminary.py` | Phase 0: train autoencoder + clustering |
| `scripts/run_assessment.py` | Phase 1: compute D/P/S (loads pre-trained artifacts) |
| `scripts/collect_offline_data.py` | Phase 2a: collect RL transitions with random agent |
| `scripts/train_rl_agent.py` | Phase 2b: train CQL or TD3+BC offline RL agent |
| `scripts/evaluate_rl_agent.py` | Phase 2c: evaluate trained policy vs. baseline |
| `scripts/preprocess_data.py` | Standalone data prep and split saving |

All standalone scripts accept `--dataset` and `--data-dir` arguments and default to `ids2017` / `data`.

---

## Citation

If you use this code or the TATA framework, please cite:

```bibtex
@inproceedings{tata2025,
  title={TATA: Benchmark NIDS Test Sets Assessment and Targeted Augmentation},
  booktitle={Proceedings of the 30th European Symposium on Research in Computer Security (ESORICS)},
  year={2025},
  organization={Springer},
  note={Main Track}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact & Issues

For questions, bug reports, or dataset availability, please open an issue on the repository.

**Augmented test set:** The augmented test sets referenced in the paper will be released alongside the final artifact. A download link will be added here when available.
