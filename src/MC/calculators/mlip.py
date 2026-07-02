import logging
from typing import Any, Dict, List, Type, Optional

import numpy as np

from ase import units
from ase.neighborlist import neighbor_list as ase_neighbor_list
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from MC.state.configuration import Configuration
from MC.calculators.base import BaseCalculator

logger = logging.getLogger(__name__)


class RelaxFailedError(RuntimeError):
    """Raised when an MLIP relaxation fails.

    Causes: isolated atoms (CHGNet bond-graph failure), NaN/inf forces or
    stresses (e.g. FrechetCellFilter logm on degenerate cell), or any
    other ValueError/RuntimeError from the MLIP during geometry optimisation.
    """
    pass


class MDFailedError(RuntimeError):
    """Raised when an MLIP MD run cannot produce a valid trajectory.

    Causes: isolated atoms (cell expanded too far in NPT), NaN/inf
    positions, or bond-graph construction failure.
    """
    pass


class MLIPRelaxCalculator(BaseCalculator):
    """
    Relaxes a configuration using an ASE-compatible MLIP calculator.

    Returns
    -------
    dict:
        "relaxed_configuration" : Configuration
        "mlip_energy"           : float  — potential energy of the relaxed structure (eV)
        "converged"             : bool   — whether fmax criterion was reached
    """

    def __init__(
        self,
        calculator,
        fmax: float = 0.05,
        max_steps: int = 300,
        optimizer_cls: Type[Optimizer] = BFGS,
        relax_cell: bool = False,
    ):
        """
        Parameters
        ----------
        calculator:
            ASE-compatible MLIP calculator.
        fmax:
            Force convergence criterion in eV/Å.
        max_steps:
            Maximum optimizer steps before giving up.
        optimizer_cls:
            ASE Optimizer class. Defaults to BFGS.
        relax_cell:
            If True, also relax the unit cell using ASE's ExpCellFilter.
        """
        self.calculator = calculator
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer_cls = optimizer_cls
        self.relax_cell = relax_cell

    def __call__(
        self,
        configuration: Configuration,
        warm_start_atoms=None,
    ) -> Dict[str, Any]:
        active_atoms, active_indices = configuration.get_active_structure()
        atoms = active_atoms.copy()
        if warm_start_atoms is not None:
            atoms.set_cell(warm_start_atoms.get_cell(), scale_atoms=False)
            atoms.set_positions(warm_start_atoms.get_positions())
        atoms.calc = self.calculator

        if self.relax_cell:
            from ase.filters import FrechetCellFilter
            opt_target = FrechetCellFilter(atoms)
        else:
            opt_target = atoms

        dyn = self.optimizer_cls(opt_target, logfile=None)
        try:
            converged = dyn.run(fmax=self.fmax, steps=self.max_steps)
        except (ValueError, RuntimeError) as exc:
            raise RelaxFailedError(str(exc)) from exc

        if not converged:
            logger.warning(
                f"MLIP relaxation did not converge within {self.max_steps} steps "
                f"(final fmax > {self.fmax} eV/Å)."
            )

        relaxed_atoms = opt_target.atoms if self.relax_cell else atoms

        relaxed_configuration = configuration.copy()
        full_positions = relaxed_configuration.structure.get_positions()
        full_positions[active_indices] = relaxed_atoms.get_positions()
        relaxed_configuration.structure.set_positions(full_positions)

        if self.relax_cell:
            relaxed_configuration.structure.set_cell(
                relaxed_atoms.get_cell(), scale_atoms=False
            )

        return {
            "relaxed_configuration": relaxed_configuration,
            "mlip_energy": float(relaxed_atoms.get_potential_energy()),
            "converged": bool(converged),
        }

class MLIPMDCalculator(BaseCalculator):
    """
    Runs a short MD trajectory using an ASE-compatible MLIP and returns
    a set of snapshots for downstream GNN evaluation.

    Two ensembles are supported:
        NVT (relax_cell=False): Langevin thermostat, fixed cell.
        NPT (relax_cell=True):  MTK (Martyna-Tobias-Klein) thermostat/barostat
                                via MTKNPT from ase.md.nose_hoover_chain.

    The trajectory is split into two phases:
        1. Burn-in  (burn_in_steps): equilibration, snapshots discarded.
        2. Production (n_md_steps):  n_snapshots collected at equal intervals.

    Returns
    -------
    dict:
        "snapshots" : List[Configuration]
            Copies of the original configuration with positions (and cell for
            NPT) updated from the MD trajectory. Species and MAGMOM unchanged.
    """

    def __init__(
        self,
        ase_calculator,
        temperature: float,
        timestep: float = 2.0,
        burn_in_steps: int = 100,
        n_md_steps: int = 200,
        n_snapshots: int = 10,
        relax_cell: bool = False,
        # NVT parameters
        friction: float = 0.05,
        # NPT parameters (MTKNPT)
        pressure: float = 0.0,
        taut: float = 100.0,
        taup: float = 1000.0,
        tchain: int = 3,
        pchain: int = 3,
        tloop: int = 1,
        ploop: int = 1,
        # Safety thresholds (None = disabled)
        force_threshold: Optional[float] = 50.0,
        dist_threshold: Optional[float] = 1.0,
        # Debug trajectory
        debug_traj_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        ase_calculator:
            ASE-compatible MLIP calculator.
        temperature:
            MD temperature in Kelvin. Updated by set_temperature() at each
            MC temperature step.
        timestep:
            MD timestep in femtoseconds.
        burn_in_steps:
            Equilibration steps (discarded before collecting snapshots).
        n_md_steps:
            Production steps from which snapshots are drawn.
        n_snapshots:
            Number of snapshots, evenly spaced over the production trajectory.
        relax_cell:
            If False (default), run NVT (Langevin, fixed cell).
            If True, run NPT (MTKNPT, variable cell).
        friction:
            Langevin friction coefficient in fs^-1 (NVT only).
        pressure:
            External pressure in GPa (NPT only). Default 0 (ambient).
        taut:
            MTK thermostat coupling time in fs (NPT only, maps to tdamp).
        taup:
            MTK barostat coupling time in fs (NPT only, maps to pdamp).
        tchain:
            Number of thermostat Nosé-Hoover chain links (NPT only).
        pchain:
            Number of barostat Nosé-Hoover chain links (NPT only).
        tloop:
            Number of thermostat sub-steps per MD step (NPT only).
        ploop:
            Number of barostat sub-steps per MD step (NPT only).
        force_threshold:
            If set, raise MDFailedError after any MD step where the maximum
            atomic force exceeds this value (eV/Å). None disables the check.
        dist_threshold:
            If set, raise MDFailedError after any MD step where any pair of
            atoms is closer than this distance (Å). None disables the check.
        debug_traj_path:
            If set, write a debug trajectory to this path (e.g. "debug_md.traj").
            The initial structure is also saved alongside as "<stem>_initial.vasp".
            Frames are flushed to disk at every MD step so the file is intact
            even if the process crashes mid-run.  Default None (disabled).
        """
        self.ase_calculator = ase_calculator
        self.temperature = temperature
        self.timestep = timestep
        self.burn_in_steps = burn_in_steps
        self.n_md_steps = n_md_steps
        self.n_snapshots = n_snapshots
        self.relax_cell = relax_cell
        self.friction = friction
        self.pressure = pressure
        self.taut = taut
        self.taup = taup
        self.tchain = tchain
        self.pchain = pchain
        self.tloop = tloop
        self.ploop = ploop
        self.force_threshold = force_threshold
        self.dist_threshold = dist_threshold
        self.debug_traj_path = debug_traj_path

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def _make_dynamics(self, atoms):
        if self.relax_cell:
            from ase.md.nose_hoover_chain import MTKNPT
            return MTKNPT(
                atoms,
                timestep=self.timestep * units.fs,
                temperature_K=self.temperature,
                pressure_au=self.pressure * units.GPa,
                tdamp=self.taut * units.fs,
                pdamp=self.taup * units.fs,
                tchain=self.tchain,
                pchain=self.pchain,
                tloop=self.tloop,
                ploop=self.ploop,
            )
        else:
            from ase.md.langevin import Langevin
            return Langevin(
                atoms,
                timestep=self.timestep * units.fs,
                temperature_K=self.temperature,
                friction=self.friction / units.fs,
            )

    def __call__(
        self,
        configuration: Configuration,
        warm_start_atoms=None,
    ) -> Dict[str, Any]:
        active_atoms, active_indices = configuration.get_active_structure()
        atoms = active_atoms.copy()
        if warm_start_atoms is not None:
            atoms.set_cell(warm_start_atoms.get_cell(), scale_atoms=False)
            atoms.set_positions(warm_start_atoms.get_positions())
        atoms.calc = self.ase_calculator

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        Stationary(atoms, preserve_temperature=True)
        dyn = self._make_dynamics(atoms)

        snapshot_interval = max(1, self.n_md_steps // self.n_snapshots)
        snapshots:             List[Configuration] = []
        snapshot_temperatures: List[float]         = []
        snapshot_pressures:    List[float]         = []

        # ── Per-step safety checks ────────────────────────────────────────
        if self.force_threshold is not None or self.dist_threshold is not None:
            _force_threshold = self.force_threshold
            _dist_threshold  = self.dist_threshold

            def _safety_check():
                if _force_threshold is not None:
                    forces = atoms.get_forces()
                    max_f  = np.abs(forces).max() if np.isfinite(forces).all() else np.inf
                    if max_f > _force_threshold:
                        raise MDFailedError(
                            f"Force threshold exceeded at MD step "
                            f"{dyn.get_number_of_steps()}: "
                            f"max |F| = {max_f:.1f} eV/Å > {_force_threshold} eV/Å."
                        )
                if _dist_threshold is not None:
                    _, _, d = ase_neighbor_list("ijd", atoms, _dist_threshold)
                    if len(d) > 0:
                        raise MDFailedError(
                            f"Distance threshold exceeded at MD step "
                            f"{dyn.get_number_of_steps()}: "
                            f"min dist = {d.min():.3f} Å < {_dist_threshold} Å."
                        )

            dyn.attach(_safety_check, interval=1)
        # ─────────────────────────────────────────────────────────────────

        # ── Debug trajectory ──────────────────────────────────────────────
        _traj_obj = None
        if self.debug_traj_path is not None:
            from pathlib import Path as _Path
            from ase.io import write as _ase_write
            from ase.io.trajectory import Trajectory as _Trajectory

            _traj_p = _Path(self.debug_traj_path)
            _traj_p.parent.mkdir(parents=True, exist_ok=True)

            # Save the structure that is about to enter MD.
            _initial_path = _traj_p.with_name(_traj_p.stem + "_initial.vasp")
            _ase_write(str(_initial_path), atoms)
            logger.info(f"Debug: initial MD structure written to {_initial_path}")

            # Open trajectory; overwrite previous run's file.
            _traj_obj = _Trajectory(str(_traj_p), "w", atoms)
            _traj_obj.write()                   # frame 0 = initial (pre-dynamics)
            dyn.attach(_traj_obj.write, interval=1)   # flush every step
            logger.info(f"Debug: MD trajectory will be written to {_traj_p}")
        # ─────────────────────────────────────────────────────────────────

        try:
            dyn.run(self.burn_in_steps)
            logger.debug(f"MD burn-in complete ({self.burn_in_steps} steps).")

            if not np.isfinite(atoms.get_positions()).all():
                raise MDFailedError("Non-finite positions after burn-in.")

            for snap_i in range(self.n_snapshots):
                dyn.run(snapshot_interval)

                if not np.isfinite(atoms.get_positions()).all():
                    raise MDFailedError(
                        f"Non-finite positions after production interval {snap_i + 1}."
                    )

                snapshot = configuration.copy()
                full_positions = snapshot.structure.get_positions()
                full_positions[active_indices] = atoms.get_positions()
                snapshot.structure.set_positions(full_positions)
                if self.relax_cell:
                    snapshot.structure.set_cell(atoms.get_cell(), scale_atoms=False)
                snapshots.append(snapshot)

                snapshot_temperatures.append(float(atoms.get_temperature()))
                stress = atoms.calc.results.get("stress", None)
                snapshot_pressures.append(
                    float(-stress[:3].mean() / units.GPa) if stress is not None else float("nan")
                )

        except MDFailedError:
            raise
        except (ValueError, RuntimeError) as exc:
            # ValueError  — CHGNet CrystalGraphConverter: isolated atoms
            # RuntimeError — CHGNet CrystalGraphConverter: bond-graph failure
            raise MDFailedError(str(exc)) from exc
        finally:
            if _traj_obj is not None:
                _traj_obj.close()

        logger.debug(f"MD production complete: {len(snapshots)} snapshots collected.")
        return {
            "snapshots":             snapshots,
            "snapshot_temperatures": snapshot_temperatures,
            "snapshot_pressures":    snapshot_pressures,
        }
