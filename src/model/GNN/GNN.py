import logging

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import hydra

from utils.model import initialization_params, to_numpy, compute_grad_stats, compute_param_stats

logger = logging.getLogger(__name__)

class GNN(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # ------- Targets  -------
        self.target_cfg = self.hparams.targets
        self.target_names = list(self.target_cfg.keys())

        # ------- Backbone -------
        self.backbone = hydra.utils.instantiate(
            self.hparams.backbone,
            elemental_feature_type=self.hparams.dataset.graph.node_features.elemental_feature_type,
            target_cfg=self.target_cfg,
        )
        initialization_params(self.backbone.parameters(), self.hparams.initialization)

    def forward(self, batch):
        return self.backbone(batch)

    def on_after_backward(self):
        grad_norm = compute_grad_stats(self.named_parameters())
        self.log_dict(
            grad_norm,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )

        param_norm = compute_param_stats(self.named_parameters())
        self.log_dict(
            param_norm,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )

    def compute_loss(self, batch, stage):
        outputs = self(batch)
        out_dict = {}

        total_loss_mse = 0
        total_loss_mae = 0

        for tgt_name, tgt_cfg in self.target_cfg.items():
            pred = outputs[tgt_name]
            target = batch[tgt_name]

            pred_var = pred.var(dim=0)

            loss_ratio = tgt_cfg["loss_ratio"]
            if loss_ratio == 0:
                continue

            loss_mse = F.mse_loss(pred, target) * loss_ratio
            loss_mae = F.l1_loss(pred, target) * loss_ratio

            out_dict[f"{stage}_loss_{tgt_name}_mse"] = loss_mse
            out_dict[f"{stage}_loss_{tgt_name}_mae"] = loss_mae
            out_dict[f"{stage}_var_{tgt_name}"] = pred_var.item()

            total_loss_mse += loss_mse
            total_loss_mae += loss_mae

        out_dict[f"{stage}_loss_total_mse"] = total_loss_mse
        out_dict[f"{stage}_loss_total_mae"] = total_loss_mae

        return out_dict, total_loss_mse

    def training_step(self, batch, batch_idx):
        out, loss = self.compute_loss(batch, "train")
        self.log_dict(out, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self.compute_loss(batch, "val")
        self.log_dict(out, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch, batch_idx):
        out, _ = self.compute_loss(batch, "test")
        self.log_dict(out, prog_bar=True, batch_size=batch.num_graphs)
        return out

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.train.optimizer, params=self.parameters())

        if "lr_scheduler" in self.hparams.train:
            scheduler = hydra.utils.instantiate(self.hparams.train.lr_scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss_total_mse",
            }

        return optimizer