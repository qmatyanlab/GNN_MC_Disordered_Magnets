import math
import random
from typing import Optional

from constants.physics import kB

class MetropolisSampler:

    name = "metropolis"

    def __init__(self, temperature, rng: Optional[random.Random] = None, reject_nan: bool = True):
        self.beta = 1 / (kB * temperature)
        self.rng = rng or random
        self.reject_nan = reject_nan

        self.n_proposed = 0
        self.n_accepted = 0

    def accept(self, E_old, E_new):
        self.n_proposed += 1

        if self.reject_nan:
            if not math.isfinite(E_new) or not math.isfinite(E_old):
                return False

        dE = E_new - E_old
        if dE < 0:
            self.n_accepted += 1
            return True

        p_accept = math.exp(-self.beta * dE)
        if self.rng.random() < p_accept:
            self.n_accepted += 1
            return True

        return False

    @property
    def acceptance_ratio(self):
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    def reset_statistics(self):
        self.n_proposed = 0
        self.n_accepted = 0