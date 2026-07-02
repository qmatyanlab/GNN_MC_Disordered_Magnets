# utils/graph_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import logging

from omegaconf import OmegaConf

import torch
from torch_geometric.data import Data

from ase import Atoms
from ase.neighborlist import neighbor_list

from pymatgen.core import Structure

from utils.graph import (
    get_atomic_feature_spec,
    ase_structure_to_pymatgen_structure,
    pymatgen_structure_to_ase_structure,
    calculate_persistent_homology_features,
    torch_pbc_neighbor_list,
    get_edge_attr_from_distance,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NodeFeatureConfig:
    elemental_feature_type: str = 'one_hot'
    PH_features: bool = False

@dataclass(frozen=True)
class EdgeFeatureConfig:
    r_cutoff: float = 6.0
    num_radial_basis: int = 128

    displacement_vector: bool = True

@dataclass(frozen=True)
class GraphBuilderConfig:
    backend: str = 'ase'
    device: str = "cpu"

    node_features: NodeFeatureConfig = NodeFeatureConfig()
    edge_features: EdgeFeatureConfig = EdgeFeatureConfig()

    properties: Optional[Dict[str, List[str]]] = None

class GraphBuilder:
    def __init__(self, cfg: GraphBuilderConfig):
        self.cfg = cfg

        spec = get_atomic_feature_spec(
            feature_type=cfg.node_features.elemental_feature_type
        )
        self.atomic_features = spec.features
        self.atomic_feature_dim = spec.dim

        if cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available; use CPU graph builder.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)

    @classmethod
    def from_cfg(cls, graph_cfg):
        node_cfg = NodeFeatureConfig(
            elemental_feature_type=graph_cfg.node_features.elemental_feature_type,
            PH_features=graph_cfg.node_features.get('PH_features', True)
        )
        edge_cfg = EdgeFeatureConfig(
            r_cutoff=graph_cfg.edge_features.r_cutoff,
            num_radial_basis=graph_cfg.edge_features.num_radial_basis,
            displacement_vector=graph_cfg.edge_features.get('displacement_vector', True)
        )

        properties = {}
        if 'properties' in graph_cfg:
            properties = OmegaConf.to_container(graph_cfg.properties)

        cfg = GraphBuilderConfig(
            backend=graph_cfg.backend,
            node_features=node_cfg,
            edge_features=edge_cfg,
            properties=properties
        )
        return cls(cfg)

    def build_from_row(self, row) -> Data:
        structure = pymatgen_structure_to_ase_structure(row.structure)

        properties = None
        if self.cfg.properties:
            all_props = self.cfg.properties.get("graph", []) + self.cfg.properties.get("node", [])
            properties = {k: row[k] for k in all_props}

        return self.build_from_structure(structure, properties)

    def build_from_structure(
            self,
            structure: Atoms | Structure,
            properties: Optional[Dict[str, float]] = None
    ) -> Data:
        if self.cfg.backend == 'ase' or self.cfg.backend == 'torch_pbc':
            structure = pymatgen_structure_to_ase_structure(structure)
        elif self.cfg.backend == 'pymatgen':
            structure = ase_structure_to_pymatgen_structure(structure)

        if self.cfg.backend == 'ase':
            return self._build_graph_ase(structure=structure, properties=properties)
        elif self.cfg.backend == 'pymatgen':
            return self._build_graph_pymatgen(structure=structure, properties=properties)
        elif self.cfg.backend == 'torch_pbc':
            return self._build_graph_torch_pbc(structure=structure, properties=properties)
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

    def _attach_properties(self, data: Data, properties: Dict):
        prop_cfg = self.cfg.properties or {}
        graph_props = prop_cfg.get("graph", [])
        node_props = prop_cfg.get("node", [])

        N = data.x.shape[0] # number of nodes

        for name, value in properties.items():
            if name in graph_props:
                data[name] = torch.tensor([value], dtype=torch.float).unsqueeze(0)

            elif name in node_props:
                value = np.array(value)
                if value.ndim == 1:
                    value = value.reshape(-1, 1)
                assert value.shape[0] == N, f"{name}: expected ({N}, *), got {value.shape}."
                data[name] = torch.tensor(value, dtype=torch.float)

            else:
                raise ValueError(f"Property {name} not declared in config")
        return data

    def _build_graph_ase(self, structure: Atoms, properties: Optional[Dict[str, float]]) -> Data:
        """
        Convert ASE Atoms object → PyTorch Geometric Data graph using ASE neighbor list function.
        """
        node_Z = structure.get_atomic_numbers()
        node_x = np.array([self.atomic_features[int(z)] for z in node_Z])

        lattice = np.array(structure.get_cell())
        pos = structure.get_positions()
        frac_coords = structure.get_scaled_positions()

        edge_src, edge_tgt, edge_dist, edge_vec = neighbor_list(
            quantities='ijdD', a=structure, cutoff=self.cfg.edge_features.r_cutoff, self_interaction=True
        )
        edge_index = np.stack([edge_src, edge_tgt], axis=0)
        edge_attr = get_edge_attr_from_distance(edge_dist, r_cutoff=self.cfg.edge_features.r_cutoff, num_radial_basis=self.cfg.edge_features.num_radial_basis)

        data = Data(
            lattice=torch.tensor(lattice, dtype=torch.float).unsqueeze(0),
            x=torch.tensor(node_x, dtype=torch.float),
            species=torch.tensor(node_Z, dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=edge_attr,
            edge_dist=torch.tensor(edge_dist, dtype=torch.float),
            # cart_coords=torch.tensor(pos, dtype=torch.float),
            frac_coords=torch.tensor(frac_coords, dtype=torch.float),
        )

        if self.cfg.node_features.PH_features:
            data.PH_features = torch.tensor(
                calculate_persistent_homology_features(
                    structure, self.cfg.edge_features.r_cutoff
                ),
                dtype=torch.float
            )

        if self.cfg.edge_features.displacement_vector:
            data.edge_vec = torch.tensor(edge_vec, dtype=torch.float)

        if properties:
            data = self._attach_properties(data, properties)

        return data

    def _build_graph_pymatgen(self, structure: Structure, properties: Optional[Dict[str, float]]) -> Data:
        """
        Convert pymatgen structure to PyG Data using Structure.get_neighbor_list.
        """
        node_Z = np.array([site.specie.Z for site in structure])
        node_x = np.array([self.atomic_features[int(z)] for z in node_Z])

        lattice = np.array(structure.lattice.matrix)
        pos = np.array(structure.cart_coords)
        frac_coords = np.array(structure.frac_coords)

        center, neighbor, image, distance = structure.get_neighbor_list(
            r=self.cfg.edge_features.r_cutoff,
            sites=structure.sites,
            exclude_self=False,
        )
        edge_index = np.stack([center, neighbor], axis=0)
        edge_attr = get_edge_attr_from_distance(
            distance,
            r_cutoff=self.cfg.edge_features.r_cutoff,
            num_radial_basis=self.cfg.edge_features.num_radial_basis
        )

        data = Data(
            lattice=torch.tensor(lattice, dtype=torch.float).unsqueeze(0),
            x=torch.tensor(node_x, dtype=torch.float),
            species = torch.tensor(node_Z, dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=edge_attr,
            edge_dist=torch.tensor(distance, dtype=torch.float),
            frac_coords=torch.tensor(frac_coords, dtype=torch.float),
        )

        if self.cfg.node_features.PH_features:
            ase_structure = pymatgen_structure_to_ase_structure(structure)
            data.PH_features = torch.tensor(
                calculate_persistent_homology_features(
                    ase_structure, self.cfg.edge_features.r_cutoff
                ),
                dtype=torch.float
            )

        if self.cfg.edge_features.displacement_vector:
            shift_cart = image @ lattice  # (E, 3)
            edge_vec = pos[neighbor] + shift_cart - pos[center]  # (E, 3)

            data.edge_vec = torch.from_numpy(edge_vec).float()

        if properties:
            data = self._attach_properties(data, properties)

        return data

    def _build_graph_torch_pbc(self, structure: Atoms, properties: Optional[Dict[str, float]]) -> Data:
        node_Z = structure.get_atomic_numbers()
        node_x = np.array([self.atomic_features[int(z)] for z in node_Z])

        lattice = np.array(structure.get_cell())
        pos = structure.get_positions()
        frac_coords = structure.get_scaled_positions()

        edge_src, edge_dst, edge_dist, edge_disp_vec = torch_pbc_neighbor_list(
            structure=structure,
            r_cutoff=self.cfg.edge_features.r_cutoff,
            device=self.device,
            self_interaction=True,
        )
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_attr = get_edge_attr_from_distance(
            edge_dist,
            r_cutoff=self.cfg.edge_features.r_cutoff,
            num_radial_basis=self.cfg.edge_features.num_radial_basis,
        )

        data = Data(
            lattice=torch.tensor(lattice, dtype=torch.float).unsqueeze(0),
            x=torch.tensor(node_x, dtype=torch.float, device=self.device),
            species=torch.tensor(node_Z, dtype=torch.long, device=self.device),
            edge_index=edge_index.long(),
            edge_attr=edge_attr,
            edge_dist=edge_dist,
            frac_coords=torch.tensor(frac_coords, dtype=torch.float, device=self.device),
        )

        if self.cfg.node_features.PH_features:
            data.PH_features = torch.tensor(
                calculate_persistent_homology_features(
                    structure, self.cfg.edge_features.r_cutoff
                ),
                dtype=torch.float,
            )

        if self.cfg.edge_features.displacement_vector:
            data.edge_vec = edge_disp_vec

        if properties:
            data = self._attach_properties(data, properties)

        return data

    def _graph_to_ase_structure(self, graph: Data):
        lattice = graph.lattice.squeeze(0).numpy()
        atomic_numbers = graph.species.numpy()
        frac_coords = graph.frac_coords.numpy()

        return Atoms(numbers=atomic_numbers, scaled_positions=frac_coords, cell=lattice)

    def _graph_to_pymatgen_structure(self, graph: Data):
        lattice = graph.lattice.squeeze(0).numpy()
        atomic_numbers = graph.species.numpy()
        frac_coords = graph.frac_coords.numpy()

        return Structure(lattice=lattice, species=atomic_numbers, coords=frac_coords, coords_are_cartesian=False)