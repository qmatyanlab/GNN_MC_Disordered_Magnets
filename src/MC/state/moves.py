import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from MC.state.configuration import Configuration

logger = logging.getLogger(__name__)

@dataclass
class Proposal:
    new_configuration: Configuration
    changed_indices: Tuple[int, ...]
    info: Dict

class BaseMove:
    """Base class for MC moves."""

    name: str = "base"

    def propose(self, configuration: Configuration) -> Optional[Proposal]:
        raise NotImplementedError

def group_species(disorder_cfg: Dict[str, Dict]) -> Dict[str, str]:
    """Map each species name to its parent disorder group."""
    groups = {}
    for parent_elem, disorder_rule in disorder_cfg.items():
        for species in disorder_rule["species"]:
            if species in groups:
                raise ValueError(
                    f"Species {species} appears in multiple disorder groups."
                )
            groups[species] = parent_elem
    return groups

def _update_MAGMOM_for_element(MAGMOM: np.ndarray, idx: int, new_elem: str,
                                MAGMOM_dict: Dict[str, float]) -> None:
    """
    Update MAGMOM[idx] after the element at site idx has changed to new_elem.

    Preserves the spin sign at the site; updates the magnitude to the
    canonical moment for new_elem. If the current moment is zero (e.g. a
    non-magnetic element was previously there), the sign defaults to +1.
    """
    new_magnitude = MAGMOM_dict.get(new_elem, 0.0)
    sign = float(np.sign(MAGMOM[idx])) or 1.0
    MAGMOM[idx] = new_magnitude * sign

# ---------------------------------------------------------------------------
# Atomic moves
# ---------------------------------------------------------------------------

class SwapElementMove(BaseMove):
    """
    Swap two different elements within a disorder group (canonical ensemble),
    while preserving composition.
    """

    name = "swap_element"

    def __init__(
        self,
        disorder_cfg: Dict[str, Dict],
        rng: Optional[random.Random] = None,
    ):
        self.rng = rng or random
        self.disorder_cfg = disorder_cfg
        self.species_to_group = group_species(disorder_cfg)

    def propose(self, configuration: Configuration) -> Optional[Proposal]:
        new_configuration = configuration.copy()

        group = self.rng.choice(list(self.disorder_cfg.keys()))
        species_pool = self.disorder_cfg[group]["species"]

        candidate_species = [
            s for s in species_pool
            if s in new_configuration.element_site_dict
            and len(new_configuration.element_site_dict[s]) > 0
        ]
        if len(candidate_species) < 2:
            logger.debug(f"Not enough species to swap in group {group}; skipping.")
            return None

        species0, species1 = self.rng.sample(candidate_species, 2)
        idx0 = self.rng.choice(list(new_configuration.element_site_dict[species0]))
        idx1 = self.rng.choice(list(new_configuration.element_site_dict[species1]))

        # Swap species in structure
        structure = new_configuration.structure
        structure[idx0].symbol, structure[idx1].symbol = (
            structure[idx1].symbol,
            structure[idx0].symbol,
        )

        # Swap species in site-tracking dict
        new_configuration.element_site_dict[species0].remove(idx0)
        new_configuration.element_site_dict[species0].add(idx1)
        new_configuration.element_site_dict[species1].remove(idx1)
        new_configuration.element_site_dict[species1].add(idx0)

        # Update MAGMOM: idx0 now holds species1, idx1 now holds species0
        _update_MAGMOM_for_element(
            new_configuration.MAGMOM, idx0, species1, new_configuration.MAGMOM_dict
        )
        _update_MAGMOM_for_element(
            new_configuration.MAGMOM, idx1, species0, new_configuration.MAGMOM_dict
        )

        return Proposal(
            new_configuration=new_configuration,
            changed_indices=(idx0, idx1),
            info={
                "move": self.name,
                "group": group,
                "species": (species0, species1),
                "delta_N": {},
            },
        )


class FlipSpinMove(BaseMove):
    """
    Flip the magnetic moment sign at a randomly chosen site.

    Returns None if no site has a nonzero moment (no spin to flip).
    """

    name = "flip_spin"

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random

    def propose(self, configuration: Configuration) -> Optional[Proposal]:
        new_configuration = configuration.copy()
        MAGMOM = new_configuration.MAGMOM

        nonzero_indices = [i for i, m in enumerate(MAGMOM) if abs(m) > 1E-2]
        if not nonzero_indices:
            logger.debug("No nonzero magnetic moments to flip.")
            return None

        idx = self.rng.choice(nonzero_indices)
        new_configuration.MAGMOM[idx] *= -1

        return Proposal(
            new_configuration=new_configuration,
            changed_indices=(idx,),
            info={"move": self.name},
        )

class SubstituteElementMove(BaseMove):
    """
    Replace the species at a single site with a different species from the
    same disorder group (grand-canonical ensemble).
    """

    name = "substitute_element"

    def __init__(
        self,
        disorder_cfg: Dict[str, Dict],
        rng: Optional[random.Random] = None,
    ):
        self.rng = rng or random
        self.disorder_cfg = disorder_cfg
        self.species_to_group = group_species(disorder_cfg)

    def propose(self, configuration: Configuration) -> Optional[Proposal]:
        new_configuration = configuration.copy()

        group = self.rng.choice(list(self.disorder_cfg.keys()))
        species_pool = self.disorder_cfg[group]["species"]

        candidate_indices = [
            i for i, atom in enumerate(new_configuration.structure)
            if atom.symbol in species_pool
        ]
        if not candidate_indices:
            logger.debug(f"No sites available for substitution in group {group}.")
            return None

        idx = self.rng.choice(candidate_indices)
        old_elem = new_configuration.structure[idx].symbol
        new_elem = self.rng.choice([s for s in species_pool if s != old_elem])

        new_configuration.structure[idx].symbol = new_elem
        new_configuration.element_site_dict[old_elem].remove(idx)
        new_configuration.element_site_dict.setdefault(new_elem, set()).add(idx)

        # Update MAGMOM for the new element at this site
        _update_MAGMOM_for_element(
            new_configuration.MAGMOM, idx, new_elem, new_configuration.MAGMOM_dict
        )

        return Proposal(
            new_configuration=new_configuration,
            changed_indices=(idx,),
            info={
                "move": self.name,
                "group": group,
                "from": old_elem,
                "to": new_elem,
                "delta_N": {old_elem: -1, new_elem: 1},
            },
        )


# ---------------------------------------------------------------------------
# Move schedulers (canonical and grand-canonical)
# ---------------------------------------------------------------------------

class FixedCompositionMoves:
    """
    Canonical ensemble move scheduler.

    Randomly selects between element swaps and spin flips.
    At least one of fix_elem / fix_magmom must be False.
    """

    ensemble = "canonical"

    def __init__(
        self,
        disorder_cfg: Dict[str, Dict],
        fix_elem: bool,
        fix_magmom: bool,
        rng: Optional[random.Random] = None,
    ):
        if fix_elem and fix_magmom:
            raise ValueError(
                "Both fix_elem and fix_magmom are True; no move is possible."
            )
        self.rng = rng or random
        self.fix_elem = fix_elem
        self.fix_magmom = fix_magmom

        self._swap = SwapElementMove(disorder_cfg=disorder_cfg, rng=self.rng)
        self._flip = FlipSpinMove(rng=self.rng)

    def propose(self, configuration: Configuration) -> Optional[Proposal]:
        if self.fix_elem:
            return self._flip.propose(configuration)

        if self.fix_magmom:
            return self._swap.propose(configuration)

        # Both moves enabled: choose uniformly at random
        if self.rng.random() < 0.5:
            return self._flip.propose(configuration)
        return self._swap.propose(configuration)

class VariableCompositionMoves:
    """
    Grand-canonical ensemble move scheduler.

    Performs element substitutions (composition changes) and optionally
    spin flips.
    """

    ensemble = "grand_canonical"

    def __init__(
        self,
        disorder_cfg: Dict[str, Dict],
        fix_magmom: bool = False,
        rng: Optional[random.Random] = None,
    ):
        self.rng = rng or random
        self.fix_magmom = fix_magmom

        self._substitute = SubstituteElementMove(disorder_cfg=disorder_cfg, rng=self.rng)
        self._flip = FlipSpinMove(rng=self.rng)

    def propose(self, configuration: Configuration) -> Optional[Proposal]:
        if not self.fix_magmom and self.rng.random() < 0.5:
            return self._flip.propose(configuration)
        return self._substitute.propose(configuration)
