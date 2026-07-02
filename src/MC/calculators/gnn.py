import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from MC.state.configuration import Configuration
from MC.calculators.base import BaseCalculator

from dataset.graph_builder import GraphBuilder

from utils.data import denormalize_outputs
from utils.model import to_numpy

logger = logging.getLogger(__name__)

class GNNCalculator(BaseCalculator):
    """
    Evaluates a configuration using a trained GNN model.

    Returns denormalized values in a dictionary, e.g.: {"energy": float, "final_MAGMOM": np.ndarray, ...}.
    """
    def __init__(
        self,
        model,
        graph_builder: GraphBuilder,
        metadata: dict,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters
        ----------
        model:
            Trained GNN (pytorch_lightning.LightningModule).
        graph_builder:
            GraphBuilder instance configured to match what the model was trained on.
        metadata:
            Normalization statistics loaded from the dataset (metadata.pt).
            Used to denormalize model outputs.
        device:
            Torch device for inference. Defaults to CUDA if available.
        """
        self.model = model
        self.graph_builder = graph_builder
        self.metadata = metadata
        self.use_MAGMOM = getattr(model.backbone, "use_initial_MAGMOM", False)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        graph_builder: GraphBuilder,
        metadata_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "GNNCalculator":
        """Convenience constructor that loads the model from a checkpoint file."""
        from model.GNN.GNN import GNN

        model = GNN.load_from_checkpoint(str(ckpt_path), weights_only=False)
        metadata = torch.load(str(metadata_path))
        return cls(
            model=model,
            graph_builder=graph_builder,
            metadata=metadata,
            device=device,
        )

    def _build_graph(self, configuration: Configuration):
        # Build the graph from real atoms only; vacancy/interstitial placeholders are excluded.
        active_structure, active_indices = configuration.get_active_structure()
        graph = self.graph_builder.build_from_structure(active_structure)

        # MAGMOM is a dynamic node feature (changes with each move) so it is
        # injected here rather than through the graph builder.  Slice to the
        # active indices so the tensor length matches the number of graph nodes.
        if self.use_MAGMOM:
            active_MAGMOM = configuration.MAGMOM[active_indices]
            graph.initial_MAGMOM = torch.tensor(
                active_MAGMOM, dtype=torch.float
            ).unsqueeze(1)  # (N_active, 1)

        graph.batch = torch.zeros(len(active_structure), dtype=torch.long)

        return graph

    def __call__(self, configuration: Configuration) -> Dict[str, Any]:
        graph = self._build_graph(configuration)
        graph = graph.to(self.device)

        with torch.no_grad():
            raw_outputs = self.model(graph)

        outputs = {k: to_numpy(v) for k, v in raw_outputs.items()}
        outputs = denormalize_outputs(outputs, self.metadata)
        return outputs
