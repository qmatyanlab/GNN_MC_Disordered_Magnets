import logging
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from typing import Iterable, List, Union, Optional

from ase import Atoms

def validate_GNN_model(model, graph_cfg):
    if getattr(model, "use_MAGMOM", False) and not graph_cfg.get("MAGMOM_features", False):
        raise ValueError("The GNN model uses MAGMOM features, but the graph builder configuration does not support MAGMOM features.")

    if getattr(model, "use_PH_features", False) and not graph_cfg.get("PH_features", False):
        raise ValueError("The GNN model uses PH features, but the graph builder configuration does not support PH features.")

    for tgt_model, tgt_graph_cfg in zip(model.target_names, graph_cfg.targets):
        if tgt_model != tgt_graph_cfg:
            raise ValueError(f"Target {tgt_model} in model does not match target {tgt_graph_cfg} in graph builder configuration.")

    if "energy" not in model.target_names:
        raise ValueError("The GNN model must predict an energy.")

    return True

def get_site_indices(
        atoms: Atoms,
        parent_element: str
) -> List[int]:
    symbols = atoms.get_chemical_symbols()
    if parent_element not in symbols:
        raise ValueError(f"Parent element {parent_element} not found in structure.")
    return [i for i, s in enumerate(symbols) if s == parent_element]

def assign_sites_by_count(
        site_indices: List[int],
        count: List[Optional[int]],
        *,
        rng: random.Random = random
) -> List[int]:
    '''
    Assign sites to species by explicit atom counts.

    Parameters
    ----------
    site_indices
        List of site indices being disordered.
    count
        Number of atoms for each species. Exactly one entry may be None,
        meaning "fill the remaining sites". Typically used for nonreal
        (vacancy/placeholder) species.
    '''
    n_sites = len(site_indices)
    n_null = sum(1 for c in count if c is None)
    if n_null > 1:
        raise ValueError(
            f"At most one entry in count may be None (fill-the-rest), got {count}."
        )

    specified = sum(c for c in count if c is not None)
    if n_null == 0 and specified != n_sites:
        raise ValueError(
            f"Counts {count} sum to {specified} but there are {n_sites} sites."
        )
    if n_null == 1 and specified > n_sites:
        raise ValueError(
            f"Specified counts {count} already exceed the number of sites ({n_sites})."
        )

    counts = [n_sites - specified if c is None else c for c in count]

    assignment = []
    for species_idx, n in enumerate(counts):
        assignment.extend([species_idx] * n)

    rng.shuffle(assignment)
    return assignment


def assign_sites_by_concentration(
        site_indices: List[int],
        concentration: List[float],
        *,
        rng: random.Random = random
) -> List[int]:
    '''
    Parameters
    ----------
    site_indices
        List of site indices being disordered.
    concentration
        Fractions for each species (not necessarily normalized).
    '''
    num_sites = len(site_indices)
    concentration = np.asarray(concentration, dtype=float)
    concentration = concentration / np.sum(concentration)

    expected = concentration * num_sites
    counts = np.round(expected).astype(int)

    while np.sum(counts) > num_sites:
        counts[np.argmax(counts - expected)] -= 1
    while np.sum(counts) < num_sites:
        counts[np.argmin(counts - expected)] += 1

    assignment = []
    for species_idx, count in enumerate(counts):
        assignment.extend([species_idx] * count)

    rng.shuffle(assignment)
    return assignment

def get_temperature_schedule(
        *,
        Tmin: float,
        Tmax: float,
        spacing: str = 'linear',
        num: Optional[int] = None,
        dT: Optional[float] = None,
) -> np.ndarray:
    if Tmin <= 0 or Tmax <= 0:
        raise ValueError("Temperatures must be positive.")

    if Tmin >= Tmax:
        raise ValueError(f"Tmin ({Tmin}) must be <= Tmax ({Tmax}).")

    if spacing == "linear":
        if dT is not None:
            if dT <= 0:
                raise ValueError("dT must be positive for linear spacing.")
            Tgrid = np.arange(Tmax, Tmin - dT, -dT)
            return Tgrid

        if num is None or num <= 0:
            raise ValueError("For linear spacing, either num > 0 or dT must be provided.")

        return np.linspace(Tmin, Tmax, num=num)

    if spacing == 'log':
        if dT is not None:
            raise ValueError("dT is not supported for log spacing.")

        if num is None or num <= 0:
            raise ValueError("For log spacing, num > 0 must be provided.")

        return np.logspace(np.log10(Tmin), np.log10(Tmax), num=num)

    raise ValueError(f"Unknown temperature spacing: {spacing}.")

def save_mc_results(results, path, name):
    path = Path(path)
    filename = Path(name)

    if filename.suffix != ".pkl":
        filename = filename.with_suffix(".pkl")

    full_path = path / filename
    with open(full_path, 'wb') as f:
        pickle.dump(results, f)

def load_mc_results(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results

def _config_diffs(d1, d2, prefix="") -> dict:
    """Recursively collect differing leaf values between two plain dicts."""
    diffs = {}
    all_keys = set(d1.keys() if isinstance(d1, dict) else []) | \
               set(d2.keys() if isinstance(d2, dict) else [])
    for k in sorted(all_keys):
        key = f"{prefix}.{k}" if prefix else str(k)
        v1 = d1.get(k, "<missing>") if isinstance(d1, dict) else "<missing>"
        v2 = d2.get(k, "<missing>") if isinstance(d2, dict) else "<missing>"
        if isinstance(v1, dict) and isinstance(v2, dict):
            diffs.update(_config_diffs(v1, v2, key))
        elif v1 != v2:
            diffs[key] = (v1, v2)
    return diffs


def check_config_consistency(current_cfg, saved_cfg_path: Path) -> None:
    """
    Compare the current Hydra config against the saved config.yaml from a
    previous run.  Raises ValueError listing every differing key if they do
    not match, so that an accidental parameter change is caught before the
    run produces inconsistent results.
    """
    from omegaconf import OmegaConf

    saved_cfg    = OmegaConf.load(saved_cfg_path)
    current_dict = OmegaConf.to_container(current_cfg, resolve=True)
    saved_dict   = OmegaConf.to_container(saved_cfg,   resolve=True)

    diffs = _config_diffs(current_dict, saved_dict)
    if diffs:
        lines = [
            f"  {k}:\n    saved:   {v[0]!r}\n    current: {v[1]!r}"
            for k, v in diffs.items()
        ]
        raise ValueError(
            f"Config mismatch with saved config at {saved_cfg_path}.\n"
            f"All parameters must match for a valid restart.\n"
            + "\n".join(lines)
        )


def save_checkpoint(state, path: Path) -> None:
    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def load_checkpoint(path: Path):
    path = Path(path)
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_logger(name, log_dir, filename):
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{filename}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.FileHandler(log_path, mode='w')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"mc_run_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def write_to_vasp(atoms, filename, format="vasp"):
    from ase.io import write
    from ase.build import sort

    write(filename, sort(atoms), format=format)

def test_set_up(state, moves, calculator):
    print("=" * 60)
    print("MC set-up test")
    print("=" * 60)

    # ── Initial evaluation ────────────────────────────────────────
    results = calculator(state.configuration)
    state.set_results(results)
    E0 = state.get_energy()
    observables0 = state.get_observables()
    print(f"[initial]  energy = {E0:.6f}")
    print(f"[initial]  observables = {observables0}")
    print(f"[initial]  composition = {state.configuration.get_composition()}")
    write_to_vasp(state.configuration.get_active_structure()[0], 'structure0.vasp')
    write_to_vasp(state.relaxed_configuration.get_active_structure()[0], 'structure0_relaxed.vasp')

    # ── Cycle 1: propose → apply → evaluate → accept ─────────────
    proposal = moves.propose(state.configuration)
    assert proposal is not None, "moves.propose() returned None on first attempt."

    conf_before_apply = state.configuration.summary()
    state.apply(proposal)
    conf_after_apply = state.configuration.summary()

    results_new = calculator(state.configuration)
    state.set_results(results_new)
    E1 = state.get_energy()
    print(f"\n[after apply, move={proposal.info['move']}]  energy = {E1:.6f}")
    print(f"[after apply]  composition = {state.configuration.get_composition()}")
    write_to_vasp(state.configuration.get_active_structure()[0], 'structure1.vasp')
    write_to_vasp(state.relaxed_configuration.get_active_structure()[0], 'structure1_relaxed.vasp')

    state.accept()
    assert state._previous_configuration is None, \
        "accept() did not clear _previous_configuration."
    E_accepted = state.get_energy()
    observables_accepted = state.get_observables()
    assert E_accepted == E1, "Energy changed unexpectedly after accept()."
    print(f"[after accept]  energy = {E_accepted:.6f}  ✓")
    print(f"[after accept]  observables = {observables_accepted}")
    print(f"[after accept]  composition = {state.configuration.get_composition()}")

    # ── Cycle 2: propose → apply → evaluate → revert ─────────────
    proposal2 = moves.propose(state.configuration)
    assert proposal2 is not None, "moves.propose() returned None on second attempt."

    conf_before_revert = state.configuration.summary()
    state.apply(proposal2)
    conf_after_revert = state.configuration.summary()

    results_new2 = calculator(state.configuration)
    state.set_results(results_new2)
    E2 = state.get_energy()
    print(f"\n[after apply, move={proposal2.info['move']}]  energy = {E2:.6f}")
    print(f"[after apply]  composition = {state.configuration.get_composition()}")
    write_to_vasp(state.configuration.get_active_structure()[0], 'structure2.vasp')
    write_to_vasp(state.relaxed_configuration.get_active_structure()[0], 'structure2_relaxed.vasp')

    state.revert()
    assert state._previous_configuration is None, \
        "revert() did not clear _previous_configuration."

    # After revert, outputs must be cleared — get_energy() should raise.
    try:
        state.get_energy()
        raise AssertionError("get_energy() should raise RuntimeError after revert().")
    except RuntimeError:
        pass  # expected

    # Configuration must be restored to what it was before apply.
    comp_after_revert = state.configuration.get_composition()
    comp_before_revert = conf_before_revert["composition"]
    assert comp_after_revert == comp_before_revert, (
        f"Composition changed after revert(): {comp_before_revert} → {comp_after_revert}"
    )

    print(f"[after revert]  composition restored = {comp_after_revert}  ✓")
    write_to_vasp(state.configuration.get_active_structure()[0], 'structure3.vasp')
    write_to_vasp(state.relaxed_configuration.get_active_structure()[0], 'structure3_relaxed.vasp')

    print("\nAll checks passed.")
    print("=" * 60)