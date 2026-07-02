from abc import ABC, abstractmethod
from typing import Any, Dict

from MC.state.configuration import Configuration

class BaseCalculator(ABC):
    """
    Common interface for all calculators used in MC simulations.

    A calculator accepts a Configuration and returns a dict of results.
    For example:
        GNNCalculator          -> {"energy": float, "final_MAGMOM": array, ...}
        MLIPCalculator         -> {"relaxed_configuration": Configuration, "mlip_energy": float, "converged": bool}
        RelaxAndScoreCalculator -> merged dict of both of the above
    """

    @abstractmethod
    def __call__(self, configuration: Configuration) -> Dict[str, Any]:
        raise NotImplementedError
