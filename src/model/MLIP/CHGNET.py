import logging
import torch

from chgnet.model.model import CHGNet

logger = logging.getLogger(__name__)

class CHGNET:
    def __init__(self, device):
        self.device = device
        self.model = CHGNet.load(use_device=device)

    def set_trainable_modules(self, trainable_modules):
        for name, param in self.model.named_parameters():
            param.requires_grad = any(p in name for p in trainable_modules)

        # always freeze composition model
        for name, param in self.model.named_parameters():
            if "composition_model" in name:
                param.requires_grad = False

        total, trainable = 0, 0
        logger.info("\n=== Trainable Parameters ===")
        for name, param in self.model.named_parameters():
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
                logger.info(name)
        logger.info("----------------------------")
        logger.info(f"Trainable params: {trainable:,}")
        logger.info(f"Total params:     {total:,}")
        logger.info(f"Fraction:         {trainable / total:.4f}\n")

    def predict_structure(self, structure, tgts):
        return self.model.predict_structure(structure=structure, task=tgts,  batch_size=1)

    def predict_graph(self, graph, tgts):
        return self.model.predict_graph(graph=graph, task=tgts, batch_size=1)