import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model.GNN.GNN import GNN
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator

from MC.state.configuration import Configuration
from MC.state.state import MCState
from MC.state.moves import Proposal
from MC.calculators.base import BaseCalculator
from MC.calculators.gnn import GNNCalculator
from MC.calculators.mlip import MLIPRelaxCalculator, MLIPMDCalculator, RelaxFailedError, MDFailedError
from dataset.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class MCCalculator(BaseCalculator):
    """
    Unified calculator for MC simulations.

    Three modes, selected via the config field structural_distortion.mode:

        null / absent  — GNN only. Evaluates the proposal configuration
                         directly without any structural distortion.

        relaxation     — MLIP relaxes atomic positions (and optionally the
                         cell), then GNN evaluates the relaxed structure.

        md             — MLIP runs a short NVT or NPT MD trajectory; GNN
                         evaluates each snapshot and results are averaged.
    """

    def __init__(
        self,
        model,
        graph_builder: GraphBuilder,
        metadata: dict,
        mlip_relax: Optional[MLIPRelaxCalculator] = None,
        mlip_md: Optional[MLIPMDCalculator] = None,
        device: Optional[torch.device] = None,
        warm_start: bool = False,
    ):
        self._gnn = GNNCalculator(
            model=model,
            graph_builder=graph_builder,
            metadata=metadata,
            device=device,
        )
        self._mlip_relax = mlip_relax
        self._mlip_md = mlip_md
        self._dataset_metadata = metadata
        self._warm_start = warm_start

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        if self._mlip_relax is not None:
            return "relaxation"
        if self._mlip_md is not None:
            return "md"
        return "gnn_only"

    @property
    def dataset_metadata(self) -> dict:
        return self._dataset_metadata

    def set_temperature(self, temperature: float) -> None:
        """Update the MD temperature. Called by the engine at each MC temperature."""
        if self._mlip_md is not None:
            self._mlip_md.set_temperature(temperature)

    @classmethod
    def from_cfg(cls, cfg, device: Optional[torch.device] = None) -> "MCCalculator":
        """
        Construct an MCCalculator from a Hydra config node (cfg.calculator).

        Expected config fields
        ----------------------
        GNN.model_path, GNN.model_ckpt_name
        graph                        — GraphBuilder config
        dataset_metadata.dataset_path, dataset_metadata.signature_hash
        MLIP (optional)              — fine-tuned MLIP weights config
        structural_distortion.mode   — null | relaxation | md
        structural_distortion.relaxation.*  — relaxation hyperparams
        structural_distortion.md.*          — MD hyperparams
        """
        # --- GNN ---
        ckpt_path = Path(cfg.GNN.model_path) / cfg.GNN.model_ckpt_name / "best-model.ckpt"
        model = GNN.load_from_checkpoint(str(ckpt_path), weights_only=False)
        logger.info(f"Loaded GNN from {ckpt_path}.")

        # --- graph builder and metadata ---
        graph_builder = GraphBuilder.from_cfg(cfg.graph)
        metadata_path = Path(cfg.dataset_metadata.dataset_path) / cfg.dataset_metadata.signature_hash / "processed" / "metadata.pt"
        metadata = torch.load(str(metadata_path), weights_only=False)
        logger.info(f"Loaded metadata from {metadata_path}.")

        # --- structural distortion mode ---
        distortion_cfg = cfg.get("structural_distortion", {})
        mode = distortion_cfg.get("mode", None)

        mlip_relax = None
        mlip_md = None

        if mode in ("relaxation", "md"):
            ase_calculator = cls._load_ase_calculator(cfg, device)

            if mode == "relaxation":
                relax_cfg = distortion_cfg.get("relaxation", {})
                mlip_relax = MLIPRelaxCalculator(
                    calculator=ase_calculator,
                    fmax=relax_cfg.get("fmax", 0.05),
                    max_steps=relax_cfg.get("max_steps", 300),
                    relax_cell=relax_cfg.get("relax_cell", False),
                )

            elif mode == "md":
                md_cfg = distortion_cfg.get("md", {})
                mlip_md = MLIPMDCalculator(
                    ase_calculator=ase_calculator,
                    temperature=md_cfg.get("temperature", 300.0),
                    timestep=md_cfg.get("timestep", 2.0),
                    burn_in_steps=md_cfg.get("burn_in_steps", 100),
                    n_md_steps=md_cfg.get("n_md_steps", 200),
                    n_snapshots=md_cfg.get("n_snapshots", 10),
                    relax_cell=md_cfg.get("relax_cell", False),
                    friction=md_cfg.get("friction", 0.05),
                    pressure=md_cfg.get("pressure", 0.0),
                    taut=md_cfg.get("taut", 100.0),
                    taup=md_cfg.get("taup", 1000.0),
                    tchain=md_cfg.get("tchain", 3),
                    pchain=md_cfg.get("pchain", 3),
                    tloop=md_cfg.get("tloop", 1),
                    ploop=md_cfg.get("ploop", 1),
                    force_threshold=md_cfg.get("force_threshold", 50.0),
                    dist_threshold=md_cfg.get("dist_threshold", 1.0),
                    debug_traj_path=md_cfg.get("debug_traj_path", None),
                )

        warm_start = bool(distortion_cfg.get("warm_start", False))

        return cls(
            model=model,
            graph_builder=graph_builder,
            metadata=metadata,
            mlip_relax=mlip_relax,
            mlip_md=mlip_md,
            device=device,
            warm_start=warm_start,
        )

    @staticmethod
    def _load_ase_calculator(cfg, device):
        """Load the CHGNet model (fine-tuned or pretrained) and wrap as ASE calculator."""
        mlip_cfg = cfg.get("MLIP", None)
        if mlip_cfg is not None:
            mlip_path = Path(mlip_cfg.mlip_model_path) / mlip_cfg.mlip_ckpt_name / "best_model.pt"
            if mlip_path.exists():
                mlip_model = torch.load(str(mlip_path), weights_only=False)
                logger.info(f"Loaded fine-tuned MLIP from {mlip_path}.")
            else:
                logger.warning(
                    f"MLIP model path '{mlip_path}' not found; "
                    f"falling back to pretrained model."
                )
                mlip_model = CHGNet.load()
                logger.info("Loaded pretrained CHGNet.")
        else:
            mlip_model = CHGNet.load()
            logger.info("No MLIP config provided; loaded pretrained CHGNet.")

        return CHGNetCalculator(model=mlip_model, use_device=str(device))

    # ------------------------------------------------------------------
    # Warm-start helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_warm_start_atoms(
        configuration: Configuration,
        warm_start_configuration: Configuration,
        changed_indices: Tuple[int, ...],
    ):
        """
        Build an ASE Atoms object whose positions blend the last accepted relaxed
        geometry (warm_start_configuration) with the reference lattice (configuration),
        using the reference only at sites whose element identity changed.
        """
        active_atoms_new, active_indices_new = configuration.get_active_structure()
        active_atoms_prev, active_indices_prev = warm_start_configuration.get_active_structure()

        prev_pos_map = {
            global_idx: pos
            for global_idx, pos in zip(active_indices_prev, active_atoms_prev.get_positions())
        }
        changed_set = set(changed_indices)

        warm_positions = active_atoms_new.get_positions().copy()  # reference fallback
        for local_idx, global_idx in enumerate(active_indices_new):
            if global_idx in prev_pos_map and global_idx not in changed_set:
                warm_positions[local_idx] = prev_pos_map[global_idx]

        warm_atoms = active_atoms_new.copy()
        warm_atoms.set_positions(warm_positions)
        warm_atoms.set_cell(warm_start_configuration.structure.get_cell(), scale_atoms=False)
        return warm_atoms

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(
        self,
        state: MCState,
        proposal: Optional[Proposal] = None,
    ) -> Dict[str, Any]:
        # Read warm-start info before apply() clears it from state.
        warm_start_configuration = state.relaxed_configuration
        cached_snapshots         = state.md_snapshots

        if proposal is not None:
            configuration   = proposal.new_configuration
            changed_indices = proposal.changed_indices
            move_name       = proposal.info["move"]
            state.apply(proposal)
        else:
            configuration   = state.configuration
            changed_indices = ()
            move_name       = ""

        extras = {"total_initial_magnetization": configuration.total_magnetization()}

        is_spin_flip = (move_name == "flip_spin")

        if self._mlip_relax is not None:
            # Spin flips don't change species/positions, so MLIP forces/stresses
            # are identical to the last accepted step.  Reuse the cached relaxed
            # geometry and only re-run the (cheap) GNN with the new spins.
            if is_spin_flip and warm_start_configuration is not None:
                relaxed = warm_start_configuration.copy()
                relaxed.MAGMOM = configuration.MAGMOM.copy()
                gnn_results = self._gnn(relaxed)
                extras["lattice_parameters"] = relaxed.structure.cell.cellpar()
                if "final_MAGMOM" in gnn_results:
                    extras["total_final_magnetization"] = float(gnn_results.pop("final_MAGMOM").sum())
                results = {
                    "relaxed_configuration": relaxed,
                    "mlip_energy": None,
                    "converged": True,
                    **gnn_results,
                    **extras,
                }
            else:
                warm_atoms = None
                if self._warm_start and warm_start_configuration is not None:
                    warm_atoms = self._build_warm_start_atoms(
                        configuration, warm_start_configuration, changed_indices
                    )
                try:
                    mlip_results = self._mlip_relax(configuration, warm_start_atoms=warm_atoms)
                except RelaxFailedError as exc:
                    logger.warning(f"Relaxation failed — proposal rejected (energy set to NaN). Reason: {exc}")
                    results = {"energy": float("nan"), **extras}
                    state.set_results(results)
                    return results
                relaxed = mlip_results["relaxed_configuration"]
                gnn_results = self._gnn(relaxed)
                extras["lattice_parameters"] = relaxed.structure.cell.cellpar()
                if "final_MAGMOM" in gnn_results:
                    extras["total_final_magnetization"] = float(gnn_results.pop("final_MAGMOM").sum())
                results = {**mlip_results, **gnn_results, **extras}

        elif self._mlip_md is not None:
            # Same rationale for MD: spin flips don't change MLIP forces, so the
            # MD trajectory is statistically equivalent.  Reuse the cached snapshots
            # and re-evaluate the GNN on each with the new spin configuration.
            if is_spin_flip and cached_snapshots is not None:
                spin_snapshots = []
                for snapshot in cached_snapshots:
                    s = snapshot.copy()
                    s.MAGMOM = configuration.MAGMOM.copy()
                    spin_snapshots.append(s)
                all_gnn = [self._gnn(s) for s in spin_snapshots]
                del spin_snapshots
                averaged = {
                    key: np.mean([r[key] for r in all_gnn], axis=0)
                    for key in all_gnn[0]
                }
                del all_gnn
                extras["lattice_parameters"] = np.mean(
                    [s.structure.cell.cellpar() for s in cached_snapshots], axis=0
                )
                if "final_MAGMOM" in averaged:
                    extras["total_final_magnetization"] = float(averaged.pop("final_MAGMOM").sum())
                results = {**averaged, **extras, "md_snapshots": cached_snapshots}
            else:
                warm_atoms = None
                if self._warm_start and warm_start_configuration is not None:
                    warm_atoms = self._build_warm_start_atoms(
                        configuration, warm_start_configuration, changed_indices
                    )
                try:
                    md_results = self._mlip_md(configuration, warm_start_atoms=warm_atoms)
                except MDFailedError as exc:
                    logger.warning(f"MD failed — proposal rejected (energy set to NaN). Reason: {exc}")
                    results = {"energy": float("nan"), **extras}
                    state.set_results(results)
                    return results
                snapshots = md_results["snapshots"]
                all_gnn = [self._gnn(s) for s in snapshots]
                averaged = {
                    key: np.mean([r[key] for r in all_gnn], axis=0)
                    for key in all_gnn[0]
                }
                del all_gnn
                extras["lattice_parameters"] = np.mean(
                    [s.structure.cell.cellpar() for s in snapshots], axis=0
                )
                if "final_MAGMOM" in averaged:
                    extras["total_final_magnetization"] = float(averaged.pop("final_MAGMOM").sum())
                results = {
                    **averaged, **extras,
                    "md_snapshots":          snapshots,
                    "snapshot_temperatures": md_results.get("snapshot_temperatures"),
                    "snapshot_pressures":    md_results.get("snapshot_pressures"),
                }

        else:
            # GNN-only: lattice is fixed; report current cell parameters.
            gnn_results = self._gnn(configuration)
            extras["lattice_parameters"] = configuration.structure.cell.cellpar()
            if "final_MAGMOM" in gnn_results:
                extras["total_final_magnetization"] = float(gnn_results.pop("final_MAGMOM").sum())
            results = {**gnn_results, **extras}

        state.set_results(results)
        return results
