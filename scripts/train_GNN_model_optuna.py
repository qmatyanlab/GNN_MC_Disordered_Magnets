import optuna
import json
import pickle as pkl
from pathlib import Path

import torch

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from train_GNN_model import run as train_gnn

from utils.train import suggest_hparams, apply_hparams
from utils.env_variables import CONFIG_PATH, CONFIG_FILENAME

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_FILENAME, version_base="1.3")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    if cfg.model.backbone.name != cfg.train.optuna.name:
        raise ValueError(f"Model backbone {cfg.model.backbone.name} does not match optuna search space {cfg.train.optuna.name}.")

    optuna_cfg = cfg.train.optuna
    search_space = optuna_cfg.search_space
    run_name = cfg.logging.wandb.name

    def objective(trial):
        hparams = suggest_hparams(trial, search_space)
        trial_cfg = apply_hparams(cfg, search_space, hparams)
        trial_cfg.logging.wandb.name = f"{run_name}_trial_{trial.number}"

        try:
            results = train_gnn(trial_cfg)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            trial.set_user_attr("failure_reason", "cuda_oom")
            raise optuna.TrialPruned("CUDA OOM")

        return results["test_loss_total_mae"]

    study = optuna.create_study(direction=optuna_cfg.direction)
    study.optimize(objective, n_trials=optuna_cfg.num_trials)

    # ==================================================================
    # save Optuna study
    # ==================================================================
    path = Path(f'logs_and_ckpts/optuna/{run_name}')
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "optuna_study.pkl", "wb") as f:
        pkl.dump(study, f)
    print(
        f"Saved Optuna study to {path / 'optuna_study.pkl'}."
    )

    # ==================================================================
    # save Optuna best trial
    # ==================================================================
    print("Best trial:")
    print("  Number:", study.best_trial.number)
    print("  Value:", study.best_trial.value)
    print("  Params:", study.best_trial.params)

    print(f"Best trial info can be found in logs_and_ckpts/summary/{run_name}_trial_{study.best_trial.number}")

if __name__ == "__main__":
    main()