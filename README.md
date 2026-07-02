# GNN-driven Monte Carlo for FeCoC Magnetic Alloys

A framework for studying chemical ordering and magnetic properties in FeCoC alloys using graph neural networks (GNNs) and Monte Carlo (MC) simulations.

## Overview

The project has two main pipelines:

**GNN Training** — trains a crystal graph neural network to predict total energy, magnetic anisotropy energy, and per-site magnetic moments from atomic configurations. Two backbone architectures are supported: an E3NN-based equivariant network and a Transformer-based network.

**Monte Carlo Simulation** — performs Metropolis MC over chemical configurations (species swaps) and spin configurations (spin flips) at a sequence of temperatures to study order-disorder transitions. The energy evaluator can operate in three modes:
- `gnn_only` — direct GNN evaluation
- `relaxation` — BFGS structural relaxation via CHGNet, then GNN evaluation
- `md` — short NVT/NPT MD trajectory via CHGNet, then GNN averaging over snapshots

## Installation

```bash
git clone <repo-url>
cd GNN_and_MC
cp .env_example .env          # edit PROJECT_ROOT to your local path
cp configs/logging/wandb.yaml_example configs/logging/wandb.yaml   # edit entity/project
pip install -r requirements.txt
```

## Usage

All scripts are run from the repository root. Configuration is managed by [Hydra](https://hydra.cc).

```bash
# Train GNN
python scripts/train_GNN_model.py

# Run Monte Carlo simulation
python scripts/run_MC.py

# Override config keys at runtime
python scripts/run_MC.py mc.name=my_run mc.mc.steps_per_T=1E4

# Delete a config key (e.g. remove interstitial vacancy group)
python scripts/run_MC.py ~mc.system.disorder_cfg.Og

# Analyze MC results (plots saved to results_and_figs/MC/<name>/figs/)
python scripts/analyze_MC.py

# Compute long- and short-range order parameters
python scripts/analyze_order_parameters.py
```

## Configuration

The root config is `configs/configs.yaml`, which composes sub-configs:

```
configs/
├── configs.yaml
├── dataset/dataset.yaml
├── logging/wandb.yaml         # personal — copy from wandb.yaml_example
├── mc/mc.yaml
├── mlip/mlip.yaml
├── model/
│   ├── model.yaml
│   └── backbone/              # E3NN.yaml, Transformer.yaml
└── train/
    ├── train.yaml
    └── optuna/                # per-backbone Optuna search spaces
```

Key MC settings in `configs/mc/mc.yaml`:

| Key | Description |
|-----|-------------|
| `mc.system` | Parent structure, disorder configuration, initial MAGMOM |
| `mc.calculator.structural_distortion.mode` | `null`, `relaxation`, or `md` |
| `mc.mc.temperature` | `Tmax`, `Tmin`, `dT` or `num`; `linear` or `log` spacing |
| `mc.mc.steps_per_T` | MC steps per temperature |
| `mc.mc.save_results_path` | Output directory for `.pkl` result files |

## Repository Structure

```
configs/        Hydra configuration files
scripts/        Entry-point scripts
src/
  MC/           Monte Carlo engine, calculators, state, moves
  dataset/      Dataset and graph builder
  model/        GNN model definitions (E3NN, Transformer backbones)
  utils/        Shared utilities
mc_structures/  Reference VASP structures for MC
```

## Results

- GNN checkpoints: `results_and_figs/GNN/<run>/best-model.ckpt`
- MC results (per temperature): `results_and_figs/MC/<name>/results/`
- MC figures: `results_and_figs/MC/<name>/figs/`
- Order parameters: `results_and_figs/MC/<name>/order_parameters/`
- Hydra logs: `logs_and_ckpts/hydra/`
