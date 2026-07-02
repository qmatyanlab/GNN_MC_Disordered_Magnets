import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning.loggers import WandbLogger

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from utils.data import denormalize
from utils.env_variables import PROJECT_ROOT, CONFIG_FILENAME

def split_train_val_test(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Dict[str, DataLoader]:
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    train_loader = DataLoader(dataset[:train_size], batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(dataset[train_size:train_size + val_size], batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(dataset[train_size + val_size:], batch_size=2, num_workers=4)
    return {"train": train_loader, "val": val_loader, "test": test_loader}

def denormalize_test_result(result: dict, datamodule):
    if not datamodule.metadata:
        return result

    new_result = dict(result)
    target_names = sorted({
        k.removeprefix("test_loss_").removesuffix("_mae")
        for k in new_result.keys()
        if k.endswith("_mae") and "total" not in k
    })
    metadata = datamodule.metadata

    total_mae_denorm = 0
    for tgt in target_names:
        if tgt not in metadata:
            loss = new_result[f'test_loss_{tgt}_mae']
            new_result[f'test_loss_{tgt}_mae_denorm'] = loss
            total_mae_denorm += loss
        else:
            stats = metadata[tgt]
            denorm_loss = denormalize(new_result[f"test_loss_{tgt}_mae"], stats).item()
            new_result[f"test_loss_{tgt}_mae_denorm"] = denorm_loss
            total_mae_denorm += denorm_loss

    new_result["test_loss_total_mae_denorm"] = total_mae_denorm
    return new_result

def save_training_results(
        wandb_logger: WandbLogger, trainer: pytorch_lightning.Trainer,
        results_root_dir: Optional[str | Path]='logs_and_ckpts/summary', test_result: Optional[dict] = None
):
    if isinstance(results_root_dir, str):
        results_root_dir = Path(results_root_dir)
    results_root_dir = Path('.').resolve() / results_root_dir / wandb_logger.experiment.name
    results_root_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt_location = trainer.checkpoint_callback.best_model_path
    if model_ckpt_location:
        shutil.copy(model_ckpt_location, results_root_dir)

    results = {
        "id": wandb_logger.experiment.id,
        "name": wandb_logger.experiment.name,
        "model_ckpt_location": str(model_ckpt_location),
        "config": dict(wandb_logger.experiment.config),
        "summary": {k: v for k, v in dict(wandb_logger.experiment.summary).items() if 'loss' in k},
    }
    if test_result is not None:
        results["test_result"] = test_result

    with open(results_root_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model checkpoints and training results to {str(results_root_dir)}.")

def suggest_hparams(trial, search_space):
    hparams = {}
    for name, spec in search_space.items():
        if spec["type"] == "int":
            hparams[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            hparams[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "categorical":
            hparams[name] = trial.suggest_categorical(name, spec["values"])
        else:
            raise ValueError(f"Unknown hyperparameter type: {spec['type']}")
    return hparams

def apply_hparams(cfg: DictConfig, search_space: DictConfig, hparams: dict):
    cfg = cfg.copy()

    for name, value in hparams.items():
        spec = search_space[name]
        if "path" in spec:
            OmegaConf.update(cfg, spec["path"], value, merge=True)
        elif "paths" in spec:
            for path in spec["paths"]:
                OmegaConf.update(cfg, path, value, merge=True)
        else:
            raise ValueError(f"Hyperparameter {name} has no path or paths")

    return cfg