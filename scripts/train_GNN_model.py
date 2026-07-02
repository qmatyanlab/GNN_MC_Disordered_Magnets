import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import wandb
import hydra
from omegaconf import DictConfig

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils.env_variables import CONFIG_PATH, CONFIG_FILENAME
from utils.train import save_training_results, denormalize_test_result

def run(cfg: DictConfig):
    datamodule = hydra.utils.instantiate(cfg.dataset, seed=cfg.train.seed)

    model = hydra.utils.instantiate(cfg.model, train=cfg.train, dataset=cfg.dataset, _recursive_=False)

    wandb_logger = WandbLogger(
        project=cfg.logging.wandb.project,
        name=cfg.logging.wandb.name,
        entity=cfg.logging.wandb.entity,
        save_code=cfg.logging.wandb.save_code,
        # log_model=cfg.logging.wandb.log_model,
        save_dir='logs_and_ckpts/wandb_logs',
        offline=True,
    )
    # wandb_logger.watch(model)

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu" if cfg.train.use_gpu else "cpu",
        devices=cfg.train.num_gpus if cfg.train.use_gpu else 1,
        strategy=cfg.train.strategy if cfg.train.num_gpus > 1 else 'auto',
        callbacks=[
            EarlyStopping(
                monitor="val_loss_total_mse",
                mode="min",
                patience=cfg.train.patience
            ),
            ModelCheckpoint(
                monitor="val_loss_total_mse",
                mode="min",
                save_top_k=1,
                filename="best-model",
            )
        ],
        default_root_dir='logs_and_ckpts/lightning_logs',
        **cfg.train.pl_trainer
    )

    trainer.fit(model, datamodule=datamodule)

    datamodule.setup()
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    pred = model(batch)
    print(pred)
    test_result = trainer.test(model, datamodule=datamodule)[0]
    test_result = denormalize_test_result(test_result, datamodule)

    save_training_results(wandb_logger, trainer)
    wandb_logger.experiment.finish()

    return test_result

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