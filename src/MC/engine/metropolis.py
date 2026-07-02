import gc
import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from MC.state.state import MCState
from MC.samplers.metropolis import MetropolisSampler
from MC.calculators.base import BaseCalculator

from utils.mc import get_temperature_schedule, save_mc_results, save_checkpoint

@dataclass
class MetropolisResult:
    energies: List[float]
    observables: Dict[str, List[Any]]
    accepted: List[bool]
    move_names: List[str]
    samples: List[Dict[str, Any]]
    n_steps: int       # actual MC steps (None proposals not counted)
    n_accepted: int

class MetropolisEngine:
    def __init__(
        self,
        state: MCState,
        moves,
        calculator: BaseCalculator,
        mc_cfg: Dict[str, Any],
    ):
        self.state = state
        self.moves = moves
        self.calculator = calculator
        self.mode = getattr(calculator, "mode", "gnn_only")
        self.dataset_metadata = getattr(calculator, "dataset_metadata", None)

        self._check_ensemble_consistency()

        # ------------------------------------------------------------------
        # Temperature configuration
        # ------------------------------------------------------------------
        if "temperature" not in mc_cfg:
            raise ValueError("mc.temperature must be specified in config.")

        self.temperature_cfg = mc_cfg["temperature"]

        # ------------------------------------------------------------------
        # MC step configuration
        # ------------------------------------------------------------------
        self.steps_per_T: int = int(mc_cfg.get("steps_per_T", 100))
        if self.steps_per_T <= 0:
            raise ValueError("steps_per_T must be > 0.")

        thin_ratio = mc_cfg.get("thin_ratio", 0.0)
        if not (0.0 <= thin_ratio < 1.0):
            raise ValueError("thin_ratio must be in [0, 1).")

        # thin is expressed in units of *actual* MC steps so that skipped
        # proposals (None) do not shift the sampling schedule.
        self.thin: int = max(1, int(self.steps_per_T * thin_ratio))

        # ------------------------------------------------------------------
        # Logging and storage
        # ------------------------------------------------------------------
        self.save_results_path = Path(mc_cfg.save_results_path)
        self.save_results_path.mkdir(parents=True, exist_ok=True)

        self.log_every: int = int(mc_cfg.get("log_every_n_steps", 100))
        if self.log_every <= 0:
            raise ValueError("log_every_n_steps must be > 0.")

        self.reject_nan   = mc_cfg.get("reject_nan",    True)
        self.store_samples = mc_cfg.get("store_samples", True)

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_ensemble_consistency(self):
        system_ensemble = self.state.configuration.ensemble
        move_ensemble   = self.moves.ensemble
        if system_ensemble != move_ensemble:
            raise ValueError(
                f"Move ensemble ({move_ensemble}) does not match "
                f"system ensemble ({system_ensemble})."
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self):
        Ts = get_temperature_schedule(**self.temperature_cfg)
        self.logger.info(
            f"Running Metropolis MC with {len(Ts)} temperatures, "
            f"from {Ts[0]:.1f} K to {Ts[-1]:.1f} K."
        )
        self.logger.info("=" * 100)

        mc_params = {
            "temperatures":     Ts,
            "steps_per_T":      self.steps_per_T,
            "thin":             self.thin,
            "mode":             self.mode,
            "dataset_metadata": self.dataset_metadata,
        }
        save_mc_results(mc_params, self.save_results_path, name="params")

        for T in Ts:
            result_path = self.save_results_path / f"{int(T)}K.pkl"
            if result_path.exists():
                self.logger.info(f"Skipping T = {T:.1f} K (result already exists at {result_path.name})")
                continue

            self.calculator.set_temperature(float(T))
            sampler = MetropolisSampler(
                temperature=float(T),
                reject_nan=self.reject_nan,
            )
            result = self.run_single_temperature(sampler=sampler, T=float(T))
            save_mc_results(result, self.save_results_path, name=f"{int(T)}K")
            checkpoint_path = self.save_results_path / "state_checkpoint.pkl"
            save_checkpoint(self.state, checkpoint_path)

        self.logger.info(f"Saved results to {str(self.save_results_path)}")

    def run_single_temperature(self, sampler: MetropolisSampler, T: float,) -> MetropolisResult:
        self.logger.info(f"Running Metropolis MC at T = {T:.1f} K")
        time_start = time.perf_counter()

        energies:       List[float]          = []
        observables:    Dict[str, List[Any]] = defaultdict(list)
        accepted_flags: List[bool]           = []
        move_names:     List[str]            = []
        samples:        List[Dict[str, Any]] = []

        n_accept = 0
        n_actual = 0   # actual MC steps; excludes iterations where propose() → None

        # Evaluate the initial configuration before the loop so that E_cur
        # is always defined and the first proposal has a valid reference energy.
        self.calculator(self.state)
        E_cur           = self.state.get_energy()
        observables_cur = self.state.get_observables()

        for _ in range(self.steps_per_T):
            proposal = self.moves.propose(self.state.configuration)
            if proposal is None:
                continue
            n_actual += 1

            self.calculator(self.state, proposal)
            E_new           = self.state.get_energy()
            observables_new = self.state.get_observables()
            accepted = sampler.accept(E_old=E_cur, E_new=E_new)
            if accepted:
                self.state.accept()
                E_cur           = E_new
                observables_cur = observables_new
                n_accept += 1
            else:
                self.state.revert()

            # --- record ---
            energies.append(E_cur)
            accepted_flags.append(bool(accepted))
            move_names.append(str(proposal.info["move"]))

            if observables_cur:
                for k, v in observables_cur.items():
                    observables[k].append(v)

            # Thinning is counted in actual steps so that skipped proposals
            # do not shift the sampling schedule.
            on_thin_step = (n_actual - 1) % self.thin == 0
            if self.store_samples and on_thin_step:
                relaxed      = self.state.relaxed_configuration
                md_snapshots = self.state.md_snapshots
                samples.append({
                    "configuration":         self.state.configuration.summary(),
                    "relaxed_configuration": (
                        relaxed.summary() if relaxed is not None else None
                    ),
                    "md_snapshots": (
                        [s.summary() for s in md_snapshots]
                        if md_snapshots is not None else None
                    ),
                    "energy":      E_cur,
                    "observables": observables_cur,
                })

            if n_actual % self.log_every == 0:
                self.logger.info(
                    f"T={T:.1f} K  step={n_actual}/{self.steps_per_T}"
                    f"  E={E_cur:.6f}  accepted={accepted}"
                )

        # --- end-of-temperature summary ---
        time_elapsed   = time.perf_counter() - time_start
        steps_per_sec  = n_actual / time_elapsed if time_elapsed > 0 else 0.0
        n_skipped      = self.steps_per_T - n_actual
        acc_rate       = n_accept / n_actual if n_actual > 0 else 0.0

        self.logger.info(
            f"T={T:.1f} K finished in {time_elapsed:.2f} s  |  "
            f"{n_actual} steps ({n_skipped} skipped)  |  "
            f"acceptance rate = {n_accept}/{n_actual} = {acc_rate:.3f}  |  "
            f"{steps_per_sec:.1f} steps/s"
        )
        self.logger.info("=" * 100)

        return MetropolisResult(
            energies=energies,
            observables=dict(observables),
            accepted=accepted_flags,
            move_names=move_names,
            samples=samples,
            n_steps=n_actual,
            n_accepted=n_accept,
        )
