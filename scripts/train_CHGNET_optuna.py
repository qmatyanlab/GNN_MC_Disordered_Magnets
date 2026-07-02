import sys
import logging
import optuna
import pickle as pkl
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import hydra
from omegaconf import DictConfig

import torch
from pytorch_lightning import seed_everything

from chgnet.model.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import get_train_val_test_loader

from utils.env_variables import CONFIG_PATH, CONFIG_FILENAME
from utils.train import suggest_hparams
from utils.mlip import (
    load_data,
    choose_targets, build_loss_weights, apply_trainable_modules,
    evaluate_model, compute_metrics, compute_weighted_loss,
    to_serializable, save_json, plot_results
)

logger = logging.getLogger(__name__)

def setup_trainer(cfg, dataset, targets, hparams, device):
    trainer_targets = choose_targets(targets)

    loss_weights_dict = build_loss_weights(
        targets,
        energy_loss_ratio=cfg.train.energy_loss_ratio,
        force_loss_ratio=cfg.train.force_loss_ratio,
        stress_loss_ratio=cfg.train.stress_loss_ratio,
    )

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset,
        batch_size=hparams["batch_size"],
        train_ratio=cfg.train.train_ratio,
        val_ratio=cfg.train.val_ratio,
    )

    model = CHGNet.load(use_device=device)
    apply_trainable_modules(model, hparams["trainable_modules"])

    trainer = Trainer(
        model=model,
        targets=trainer_targets,
        optimizer=cfg.train.optimizer,
        scheduler=cfg.train.scheduler,
        criterion=cfg.train.criterion,
        learning_rate=hparams["lr"],
        epochs=cfg.train.epochs,
        use_device=device,
        print_freq=10,
        torch_seed=cfg.seed,
        data_seed=cfg.seed,
        **loss_weights_dict,
    )
    return trainer, trainer_targets, train_loader, val_loader, test_loader

def train_and_evaluate(cfg, trainer, loaders, trainer_targets, save_dir, device):
    train_loader, val_loader, test_loader = loaders
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=str(save_dir),
        save_test_result=True,
    )
    model = trainer.best_model
    results = evaluate_model(model, test_loader, trainer_targets, device=device)
    metrics = compute_metrics(results)
    return model, results, metrics

def run_trial(trial, cfg, dataset, targets, save_dir, device):
    seed_everything(cfg.seed)

    trial_dir = save_dir / f"trial_{trial.number}"
    trial_dir.mkdir(exist_ok=True, parents=True)

    hparams = suggest_hparams(trial, cfg.train_optuna.search_space)

    trainer, trainer_targets, *loaders = setup_trainer(
        cfg, dataset, targets, hparams, device
    )

    _, _, metrics = train_and_evaluate(
        cfg, trainer, loaders, trainer_targets, trial_dir, device
    )
    loss = compute_weighted_loss(metrics, targets)
    with open(trial_dir / "summary.pkl", "wb") as f:
        pkl.dump(
            {
                'loss': loss,
                'metrics': metrics,
                'history': trainer.training_history
            }, f
        )

    logger.info(f"[Trial {trial.number}] loss={loss:.6f}, params={trial.params}")
    return loss

def train_best_model(cfg, best_hparams, dataset, targets, save_dir, device):
    seed_everything(cfg.seed)

    best_dir = save_dir / "best-model"
    best_dir.mkdir(exist_ok=True, parents=True)

    trainer, trainer_targets, *loaders = setup_trainer(
        cfg, dataset, targets, best_hparams, device
    )

    model, results, metrics = train_and_evaluate(
        cfg, trainer, loaders, trainer_targets, best_dir, device
    )

    loss = compute_weighted_loss(metrics, targets)
    with open(best_dir / "summary.pkl", "wb") as f:
        pkl.dump(
            {
                'loss': loss,
                'metrics': metrics,
                'history': trainer.training_history
            }, f
        )

    logger.info(f"[Best trial] loss={loss:.6f}")
    return model, results, metrics

def run(cfg: DictConfig):
    cfg = cfg.mlip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path("results_and_figs") / "MLIP" / cfg.name
    save_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = save_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    dataset = load_data(cfg.dataset.root)
    targets = cfg.train.targets

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: run_trial(trial, cfg, dataset, targets, save_dir, device),
        n_trials=cfg.train_optuna.n_trials,
    )
    logger.info(f"Best params: {study.best_trial.params}")
    best_params = to_serializable(study.best_trial.params)
    save_json(best_params, save_dir / "best_params.json")

    best_model, results, metrics = train_best_model(
        cfg, study.best_trial.params, dataset, targets, save_dir, device
    )

    save_json(metrics, save_dir / "metrics_final.json")
    torch.save(best_model, save_dir / "best-model.pt")
    # with open(save_dir / "best_model.pkl", "wb") as f:
    #     pkl.dump(best_model, f)
    plot_results(results, fig_dir, prefix="best")

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_FILENAME, version_base="1.3")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.train.seed, workers=True)

    if cfg.train.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    run(cfg)

if __name__ == '__main__':
    main()