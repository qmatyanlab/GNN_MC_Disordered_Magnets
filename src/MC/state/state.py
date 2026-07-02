from typing import Any, Dict, List, Optional

from MC.state.configuration import Configuration


class MCState:
    """
    Manages the current state of an MC simulation.

    Three objects are tracked:

        configuration          — proposal-space configuration (species and MAGMOM on
                                 the reference lattice).  Moves always operate on this.

        relaxed_configuration  — MLIP-relaxed geometry for the currently accepted
                                 configuration.  None in GNN-only runs.

        md_snapshots           — list of MLIP MD snapshots for the currently accepted
                                 configuration.  None in non-MD runs.
    """

    def __init__(self, configuration: Configuration):
        self.configuration: Configuration = configuration
        self.relaxed_configuration: Optional[Configuration] = None
        self.md_snapshots: Optional[List[Configuration]] = None

        self._outputs: Optional[Dict[str, Any]] = None

        # Saved state for revert — only valid between apply() and accept()/revert()
        self._previous_configuration: Optional[Configuration] = None
        self._previous_relaxed_configuration: Optional[Configuration] = None
        self._previous_md_snapshots: Optional[List[Configuration]] = None

    # ------------------------------------------------------------------
    # Results cache
    # ------------------------------------------------------------------

    def set_results(self, results: Dict[str, Any]) -> None:
        self._outputs = results
        if "relaxed_configuration" in results:
            self.relaxed_configuration = results["relaxed_configuration"]
        if "md_snapshots" in results:
            self.md_snapshots = results["md_snapshots"]

    def get_energy(self) -> float:
        if self._outputs is None:
            raise RuntimeError(
                "No results cached. Call set_results() after evaluating "
                "the configuration."
            )
        return float(self._outputs["energy"])

    def get_observables(self) -> Dict[str, Any]:
        """
        Return all cached outputs except infrastructure keys.
        """
        if self._outputs is None:
            raise RuntimeError(
                "No results cached. Call set_results() after evaluating "
                "the configuration."
            )
        _skip = {"energy", "relaxed_configuration", "mlip_energy", "converged", "md_snapshots"}
        return {k: v for k, v in self._outputs.items() if k not in _skip}

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def apply(self, proposal) -> None:
        """
        Apply a proposed move to the configuration.

        Saves the current state so it can be restored by revert().
        Raises RuntimeError if called again before the pending proposal is
        resolved via accept() or revert().
        """
        if self._previous_configuration is not None:
            raise RuntimeError(
                "apply() called before the previous proposal was resolved. "
                "Call accept() or revert() first."
            )
        self._previous_configuration = self.configuration
        self._previous_relaxed_configuration = self.relaxed_configuration
        self._previous_md_snapshots = self.md_snapshots

        self.configuration = proposal.new_configuration
        self.relaxed_configuration = None
        self.md_snapshots = None
        self._outputs = None

    def accept(self) -> None:
        """
        Finalise the current proposal.  Discards the saved previous state.
        """
        self._previous_configuration = None
        self._previous_relaxed_configuration = None
        self._previous_md_snapshots = None

    def revert(self) -> None:
        """
        Undo the last apply(), restoring configuration, relaxed_configuration,
        and md_snapshots to their state before the proposal.
        """
        if self._previous_configuration is None:
            raise RuntimeError("No previous configuration to revert to.")
        self.configuration = self._previous_configuration
        self.relaxed_configuration = self._previous_relaxed_configuration
        self.md_snapshots = self._previous_md_snapshots
        self._previous_configuration = None
        self._previous_relaxed_configuration = None
        self._previous_md_snapshots = None
        self._outputs = None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "configuration": self.configuration.summary(),
            "energy": self._outputs.get("energy") if self._outputs else None,
            "observables": self.get_observables() if self._outputs else {},
        }
