# A Multi-Scale Machine Learning Framework for Coupled Chemical, Spin, and Structural Disorder in Alloys

## Overview

The project has two main pipelines:

**Model Training** — trains graph neural network (GNN) model and machine learning interatomic potential (MLIP) model to predict total energies, site-resolved magnetic moments, force, stress. 

**Monte Carlo Simulation** — performs Metropolis MC over chemical configurations (element swaps) and spin configurations (spin flips) at a sequence of temperatures to study order-disorder transitions. The energy evaluator can operate in three modes:
- `gnn_only` — direct GNN evaluation
- `relaxation` — BFGS structural relaxation via CHGNet, then GNN evaluation
- `md` — short NVT/NPT MD trajectory via CHGNet, then GNN averaging over snapshots

## Installation

```bash
git clone <repo-url>
cd <project-root>
cp .env_example .env          # edit PROJECT_ROOT to your local path
cp configs/logging/wandb.yaml_example configs/logging/wandb.yaml   # edit entity/project for wandb logging
pip install -r requirements.txt
```

## Usage

```bash
# Train GNN
python scripts/train_GNN_model.py          
python scripts/train_GNN_model_optuna.py    # allow optuna hyperparameter search

# Train MLIP
python scripts/train_CHGNET_optuna.py

# Run Monte Carlo simulation
python scripts/run_MC.py
```

## Configuration

Key MC settings in `configs/mc/mc.yaml`:

| Key | Description |
|-----|-------------|
| `system` | Parent structure, disorder configuration, initial MAGMOM |
| `calculator` | `null`, `relaxation`, or `md` |
| `mc` | `Tmax`, `Tmin`, `dT` or `num`; `linear` or `log` spacing |

## Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@misc{fang2026multiscalemachinelearningframework,
    title={A Multi-Scale Machine Learning Framework for Coupled Chemical, Spin, and Structural
Disorder in Alloys},
    author={Zhenyao Fang and Qimin Yan},
    year={2026},
    eprint={2607.07456},
    archivePrefix={arXiv},
    primaryClass={cond-mat.mtrl-sci},
    url={https://arxiv.org/abs/2607.07456},
}
```
