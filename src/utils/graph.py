from __future__ import annotations

import yaml
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple
from enum import Enum

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from ripser import ripser

import torch

from ase import Atoms
from ase.atoms import Atom
from ase.neighborlist import neighbor_list

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from e3nn.math import soft_one_hot_linspace

from constants.model import EMBEDDING_DICT_PATH

# ============================================================
#                      ATOMIC FEATURES
# ============================================================

@dataclass(frozen=True)
class AtomicFeatureSpec:
    features: Dict[int, List[float]]
    dim: int
    feature_type: str

def get_atomic_feature_spec(feature_type: str) -> AtomicFeatureSpec:
    feature_type = feature_type.lower()
    embedding_dict_file = EMBEDDING_DICT_PATH / f"{feature_type}.yaml"

    if not embedding_dict_file.exists():
        raise FileNotFoundError(f"Atomic embedding file not found: {embedding_dict_file}")

    with open(embedding_dict_file, "r") as f:
        raw = yaml.safe_load(f)

    features = {int(z): list(map(float, vec)) for z, vec in raw.items()}
    dim = len(features[next(iter(features))])

    return AtomicFeatureSpec(
        features=features,
        dim=dim,
        feature_type=feature_type
    )

# ============================================================
#                  EDGE / DISTANCE ENCODINGS
# ============================================================

def smooth_cutoff(x: torch.Tensor) -> torch.Tensor:
    u = 2 * (x - 1)
    y = (torch.cos(np.pi * u) * -1 + 1) / 2
    y = torch.where(u > 0, torch.zeros_like(y), y)
    y = torch.where(u < -1, torch.ones_like(y), y)
    return y

def get_edge_attr_from_distance(dist: np.ndarray | torch.Tensor, r_cutoff: float, num_radial_basis: float) -> torch.Tensor:
    assert dist.ndim == 1, 'Dimension of input distance array must be 1.'

    if isinstance(dist, np.ndarray):
        dist = torch.from_numpy(dist)

    edge_attr = soft_one_hot_linspace(
        dist,
        start=0.0,
        end=r_cutoff,
        number=num_radial_basis,
        basis="gaussian",
        cutoff=False,
    ) * (num_radial_basis**0.5)

    edge_attr *= smooth_cutoff(dist / r_cutoff).unsqueeze(-1)
    return edge_attr.float()

def torch_pbc_neighbor_list(
    structure: Atoms,
    r_cutoff: float,
    device: torch.device = torch.device("cpu"),
    self_interaction: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch-only PBC neighbor list for a single structure.

    Args:
        structure: ASE Atoms
        r_cutoff: cutoff radius in Angstrom
        device: torch.device
        self_interaction: whether to include i→i self loops (same atom, same image)

    Returns:
        src: (E,) long
        dst: (E,) long
        dist: (E,) float
        disp: (E, 3) float (Cartesian displacement vectors)
    """

    lattice = torch.tensor(structure.cell.array, dtype=torch.float32, device=device)  # (3, 3)
    frac_coords = torch.tensor(
        structure.get_scaled_positions(), dtype=torch.float32, device=device
    )  # (N, 3)

    N = frac_coords.shape[0]

    # Determine image shifts from cutoff and cell lengths
    norms = torch.linalg.norm(lattice, dim=1)  # (3,)
    max_shifts = torch.ceil(r_cutoff / norms).to(torch.int64)
    ranges = [range(-int(n.item()), int(n.item()) + 1) for n in max_shifts]
    shifts = torch.tensor(list(product(*ranges)), dtype=torch.float32, device=device)  # (num_images, 3)

    # All (i, j) pairs
    idx_i, idx_j = torch.meshgrid(
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        indexing="ij",
    )
    idx_i = idx_i.reshape(-1)  # (N^2,)
    idx_j = idx_j.reshape(-1)  # (N^2,)

    coords_i = frac_coords[idx_i]  # (N^2, 3)
    coords_j = frac_coords[idx_j]  # (N^2, 3)

    # Apply image shifts to j
    coords_j_images = coords_j[None, :, :] + shifts[:, None, :]  # (K, N^2, 3)
    delta_frac = coords_j_images - coords_i[None, :, :]          # (K, N^2, 3)
    delta_cart = torch.einsum("kni,ij->knj", delta_frac, lattice)  # (K, N^2, 3)
    dists = torch.norm(delta_cart, dim=-1)                         # (K, N^2)

    # Mask by cutoff and self-interaction policy
    is_self = (idx_i == idx_j)[None, :]  # (1, N^2)
    if self_interaction:
        mask = dists < r_cutoff
    else:
        mask = (dists < r_cutoff) & (~is_self)

    k_idx, p_idx = torch.nonzero(mask, as_tuple=True)

    src = idx_i[p_idx]
    dst = idx_j[p_idx]
    disp = delta_cart[k_idx, p_idx]  # (E, 3)
    dist = dists[k_idx, p_idx]       # (E,)

    return src, dst, dist, disp

# ============================================================
#               DUMMY ELEMENT AND PH FEATURES
# ============================================================

def generate_parent_structure(atoms: Atoms, supercell_size=(3, 3, 3)):
    face_centers = [
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.5, 1.0], [0.5, 1.0, 0.5], [1.0, 0.5, 0.5],
    ]
    edge_centers = [
        [0.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.5], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0],
        [0.0, 1.0, 0.5], [0.0, 0.5, 1.0], [0.5, 1.0, 0.0],
        [0.5, 0.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 0.5],
    ]
    interstitials_frac_unitcell = face_centers + edge_centers

    supercell = atoms * supercell_size
    cell = supercell.get_cell()

    interstitials_frac_supercell = []
    for i in range(supercell_size[0]):
        for j in range(supercell_size[1]):
            for k in range(supercell_size[2]):
                shift = np.array([i, j, k], dtype=float)
                for coord in interstitials_frac_unitcell:
                    new_coord = (np.array(coord) + shift) / supercell_size
                    new_coord = np.mod(new_coord, 1.0)
                    interstitials_frac_supercell.append(
                        tuple(np.round(new_coord, 6))
                    )

    unique_frac_coords = sorted(set(interstitials_frac_supercell))

    for frac in unique_frac_coords:
        cart = frac @ cell
        supercell.append(Atom("Og", position=cart))

    return supercell

def generate_structure_with_dummy_element(structure: Atoms, pristine_structure: Atoms):
    assert pristine_structure is not None, "Pristine structure must be provided when using dummy element."
    pristine_structure_with_interstitial = generate_parent_structure(pristine_structure)
    pristine_syms = np.array(pristine_structure_with_interstitial.get_chemical_symbols())
    pristine_frac = pristine_structure_with_interstitial.get_scaled_positions()
    interstitial_frac_coords = pristine_frac[pristine_syms == "Og"]

    distorted_syms = np.array(structure.get_chemical_symbols())
    distorted_frac_coords = structure.get_scaled_positions()

    FeCo_mask = np.isin(distorted_syms, ["Fe", "Co"])
    FeCo_syms = distorted_syms[FeCo_mask]
    FeCo_frac_coords = distorted_frac_coords[FeCo_mask]
    C_frac_coords = distorted_frac_coords[distorted_syms == "C"]

    occupied_mask = np.zeros(len(interstitial_frac_coords), dtype=bool)
    tree = cKDTree(interstitial_frac_coords)
    if C_frac_coords.shape[0] > 0 and interstitial_frac_coords.shape[0] > 0:
        dists, inds = tree.query(C_frac_coords, distance_upper_bound=1e-2)
        valid = (inds < len(interstitial_frac_coords))
        occupied_mask[inds[valid]] = True

        if not np.all(valid):
            for i in np.where(~valid)[0]:
                print(f"Warning: C atom {i} could not be matched to any interstitial site.")

    cell = structure.get_cell()
    new_atoms = Atoms(cell=cell, pbc=True)
    new_atoms.extend(
        Atoms(
            symbols=FeCo_syms.tolist(),
            positions=FeCo_frac_coords @ cell,
        )
    )
    new_atoms.extend(
        Atoms(
            symbols=np.where(occupied_mask, "C", "Og").tolist(),
            positions=interstitial_frac_coords @ cell,
        )
    )
    return new_atoms

def calculate_persistent_homology_features(atoms: Atoms, r_cutoff: float):
    PH_features = []

    positions = atoms.get_positions()
    cell = atoms.get_cell()

    src, dst, shift = neighbor_list(
        "ijS", atoms, cutoff=r_cutoff, self_interaction=True
    )
    neighbors_by_src = [[] for _ in range(len(atoms))]
    for i, j, s in zip(src, dst, shift):
        rel_vec = positions[j] - positions[i] + s @ cell
        neighbors_by_src[i].append(rel_vec)

    for i in range(len(atoms)):
        coords = [np.zeros(3)]
        coords.extend(neighbors_by_src[i])
        coords = np.asarray(coords)
        dist_mat = squareform(pdist(coords, metric="euclidean"))

        dgms = ripser(
            dist_mat,
            maxdim=2,
            thresh=r_cutoff,
            distance_matrix=True,
        )["dgms"]

        for d in range(3):
            dgms[d] = dgms[d][np.all(np.isfinite(dgms[d]), axis=1)]

        PH_feature = []

        # ---- 0D ----
        PH_feature.extend([
            np.mean(dgms[0][:, 1]),
            np.min(dgms[0][:, 1]),
            np.max(dgms[0][:, 1]),
            np.std(dgms[0][:, 1]),
        ])

        # ---- 1D ----
        PH_feature.extend([
            np.mean(dgms[1][:, 0]),
            np.min(dgms[1][:, 0]),
            np.max(dgms[1][:, 0]),
            np.std(dgms[1][:, 0]),
            np.mean(dgms[1][:, 1]),
            np.min(dgms[1][:, 1]),
            np.max(dgms[1][:, 1]),
            np.std(dgms[1][:, 1]),
            np.mean(dgms[1][:, 1] - dgms[1][:, 0]),
            np.min(dgms[1][:, 1] - dgms[1][:, 0]),
            np.max(dgms[1][:, 1] - dgms[1][:, 0]),
            np.std(dgms[1][:, 1] - dgms[1][:, 0]),
        ])

        # ---- 2D ----
        if dgms[2].shape[0] == 0:
            PH_feature.extend([0.0] * 12)
        else:
            PH_feature.extend([
                np.mean(dgms[2][:, 0]),
                np.min(dgms[2][:, 0]),
                np.max(dgms[2][:, 0]),
                np.std(dgms[2][:, 0]),
                np.mean(dgms[2][:, 1]),
                np.min(dgms[2][:, 1]),
                np.max(dgms[2][:, 1]),
                np.std(dgms[2][:, 1]),
                np.mean(dgms[2][:, 1] - dgms[2][:, 0]),
                np.min(dgms[2][:, 1] - dgms[2][:, 0]),
                np.max(dgms[2][:, 1] - dgms[2][:, 0]),
                np.std(dgms[2][:, 1] - dgms[2][:, 0]),
            ])

        PH_features.append(PH_feature)

    return np.asarray(PH_features)

# ============================================================
#               Structure Conversion
# ============================================================

def ase_structure_to_pymatgen_structure(structure: Atoms | Structure) -> Structure:
    if not structure:
        return None

    if isinstance(structure, Structure):
        return structure
    return AseAtomsAdaptor.get_structure(structure)

def pymatgen_structure_to_ase_structure(structure: Structure | Atoms) -> Atoms:
    if not structure:
        return None

    if isinstance(structure, Atoms):
        return structure
    return AseAtomsAdaptor.get_atoms(structure)