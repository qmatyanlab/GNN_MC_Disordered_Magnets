import random
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from ase import Atoms
from ase.io import read as ase_read

from utils.mc import get_site_indices, assign_sites_by_concentration, assign_sites_by_count

class Configuration:
    def __init__(
        self,
        parent_structure_path: Path,
        disorder_cfg: Dict[str, Dict[str, List]],
        MAGMOM_dict: Dict[str, float],
        chemical_potential_dict: Optional[Dict[str, float]],
        MAGMOM_init_mode: str = "random",
    ):
        structure = ase_read(parent_structure_path)
        self.structure: Atoms = structure.copy()

        self.disorder_cfg = disorder_cfg
        self.MAGMOM_dict = MAGMOM_dict  # stored so moves can look up element moments
        self.real_species: FrozenSet[str] = self._parse_real_species(disorder_cfg)
        self.element_site_dict = defaultdict(set)
        self.apply_disorder(disorder_cfg)

        if MAGMOM_init_mode not in ("random", "FM"):
            raise ValueError(f"MAGMOM_init_mode must be 'random' or 'FM', got '{MAGMOM_init_mode}'.")
        self.MAGMOM = self.init_MAGMOM(MAGMOM_dict, MAGMOM_init_mode)
        self.chemical_potential_dict = chemical_potential_dict

    @staticmethod
    def _parse_real_species(disorder_cfg: Dict) -> FrozenSet[str]:
        """
        Determine which species are physically real atoms (not vacancy/interstitial placeholders).
        """
        real: set = set()
        for _, rule in disorder_cfg.items():
            species_list = rule["species"]
            real_flags = rule.get("real", [True] * len(species_list))
            for species, is_real in zip(species_list, real_flags):
                if is_real:
                    real.add(species)
        return frozenset(real)

    def get_active_structure(self) -> Tuple[Atoms, List[int]]:
        """
        Return the physically present atoms and their indices in the full structure.

        Returns
        -------
        active_atoms : Atoms
            Subset of ``self.structure`` containing only real atoms.
        active_indices : list[int]
            Indices into ``self.structure`` that correspond to the rows of
            ``active_atoms``.  Use these to map positions back after MLIP
            relaxation.
        """
        active_indices = [
            i for i, atom in enumerate(self.structure)
            if atom.symbol in self.real_species
        ]
        active_atoms = self.structure[active_indices]
        return active_atoms, active_indices

    def apply_disorder(self, disorder: Dict[str, Dict[str, List]]):
        for parent_elem, disorder_rule in disorder.items():
            species = disorder_rule["species"]
            site_indices = get_site_indices(self.structure, parent_elem)

            if "count" in disorder_rule:
                site_element_assignment = assign_sites_by_count(
                    site_indices, disorder_rule["count"]
                )
            elif "concentration" in disorder_rule:
                site_element_assignment = assign_sites_by_concentration(
                    site_indices, disorder_rule["concentration"]
                )
            else:
                raise ValueError(
                    f"Disorder group '{parent_elem}' must specify either "
                    f"'count' or 'concentration'."
                )

            for site_idx, species_idx in zip(site_indices, site_element_assignment):
                elem = species[species_idx]
                self.structure[site_idx].symbol = elem
                self.element_site_dict[elem].add(site_idx)

    def init_MAGMOM(self, MAGMOM_dict: Dict[str, float], MAGMOM_init_mode: str = "random") -> np.ndarray:
        MAGMOM = np.zeros(len(self.structure), dtype=float)
        for elem, indices in self.element_site_dict.items():
            moment = MAGMOM_dict.get(elem, 0.0)
            for idx in indices:
                sign = random.choice([-1, 1]) if MAGMOM_init_mode == "random" else 1.0
                MAGMOM[idx] = moment * sign
        return MAGMOM

    def copy(self) -> "Configuration":
        return deepcopy(self)

    def get_structure(self) -> Atoms:
        return self.structure.copy()

    def get_element_site_dict(self) -> Dict[str, List[int]]:
        return {k: sorted(v) for k, v in self.element_site_dict.items()}

    def get_composition(self) -> Dict[str, int]:
        """Composition of real atoms only (vacancy/interstitial placeholders excluded)."""
        active_atoms, _ = self.get_active_structure()
        return active_atoms.symbols.formula.count()

    def total_magnetization(self) -> float:
        """Sum of magnetic moments over real atoms (vacancy/interstitial placeholders have moment=0)."""
        return float(self.MAGMOM.sum())

    def summary(self) -> Dict:
        active_atoms, active_indices = self.get_active_structure()
        return {
            "composition":         self.get_composition(),
            "total_magnetization": self.total_magnetization(),
            "MAGMOM":              self.MAGMOM[active_indices],
            "symbols":             active_atoms.get_chemical_symbols(),
            "positions":           active_atoms.get_positions(),
            "cell":                active_atoms.get_cell().array,
        }

class FixedCompositionConfiguration(Configuration):

    ensemble = "canonical"

    def __init__(self, *args, chemical_potential_dict=None, **kwargs):
        super().__init__(*args, chemical_potential_dict=None, **kwargs)


class VariableCompositionConfiguration(Configuration):

    ensemble = "grand_canonical"

    def __init__(
        self,
        *args,
        chemical_potential_dict: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        if chemical_potential_dict is None:
            raise ValueError(
                "chemical_potential_dict must be specified for variable composition configurations."
            )
        super().__init__(*args, chemical_potential_dict=chemical_potential_dict, **kwargs)
