from __future__ import annotations

from typing import Literal, Optional
from pathlib import Path
from tqdm import tqdm
import json
import hashlib
import logging

import pandas as pd

from omegaconf import OmegaConf

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from ase.io import read

from dataset.graph_builder import GraphBuilder

from utils.data import normalize

logger = logging.getLogger(__name__)

class Dataset(Dataset):
    def __init__(
        self,
        graph_cfg,
        dataset_cfg,
        transform=None,
        pre_transform=None,
    ):
        self.graph_cfg = graph_cfg
        self.graph_builder = GraphBuilder.from_cfg(graph_cfg)
        self.properties = OmegaConf.to_container(graph_cfg.properties)

        self.normalization_cfg = dataset_cfg.get("normalization_cfg", {})

        signature = {
            "backend": graph_cfg.backend,
            "device": graph_cfg.device,
            "node_features": OmegaConf.to_container(graph_cfg.node_features),
            "edge_features": OmegaConf.to_container(graph_cfg.edge_features),
            "properties": self.properties,
            "normalization_cfg": OmegaConf.to_container(self.normalization_cfg) if OmegaConf.is_config(self.normalization_cfg) else None,
        }
        signature_str = json.dumps(signature, sort_keys=True)
        signature_hash = hashlib.sha1(signature_str.encode("utf-8")).hexdigest()[:8]

        self.signature = signature
        self.signature_hash = signature_hash
        logger.info(f"Dataset signature: {signature_hash}")

        self.raw_root = Path(dataset_cfg.dataset_root)
        self.dataset_filename = dataset_cfg.dataset_filename

        self.root = self.raw_root / self.signature_hash
        self.root.mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        self.graphs_dir = Path(self.processed_dir) / f"graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        # Record dataset signature
        signature_file = self.raw_root / "dataset_signatures.json"
        if signature_file.exists():
            with open(signature_file, "r") as f:
                all_signatures = json.load(f)
        else:
            all_signatures = {}

        if signature_hash not in all_signatures:
            all_signatures[signature_hash] = signature
            with open(signature_file, "w") as f:
                json.dump(all_signatures, f, indent=2, sort_keys=True)

        super().__init__(self.root, transform, pre_transform)
        # self.data, self.slices, self.metadata = torch.load(self.processed_paths[0], weights_only=False)

        metadata_path = Path(self.processed_dir) / "metadata.pt"
        self.metadata = torch.load(metadata_path, weights_only=False) if metadata_path.exists() else {}

        self.max_samples = dataset_cfg.get("max_samples", None)
        self.file_list = sorted(self.graphs_dir.glob("graph_*.pt"))
        if self.max_samples is not None:
            self.file_list = self.file_list[:self.max_samples]

    @property
    def raw_file_names(self):
        return ["results.pkl", "unitcell.vasp"]

    @property
    def processed_file_names(self):
        return [f"metadata.pt"]

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        data = torch.load(self.file_list[idx], weights_only=False)
        data.idx = idx
        return data

    def process(self):
        df = pd.read_pickle(Path(self.raw_root) / "raw" / f"{self.dataset_filename}.pkl")

        prop_cfg = self.properties or {}
        graph_props = prop_cfg.get("graph", [])
        node_props = prop_cfg.get("node", [])

        metadata = {}
        for prop in graph_props:
            if prop in self.normalization_cfg:
                norm, stats = normalize(
                    df[prop].to_list(), self.normalization_cfg[prop]
                )
                df[prop] = norm
                metadata[prop] = stats
        # did not normalize node level properties

        df = df[: 10]
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            data = self.graph_builder.build_from_row(row)
            torch.save(data.cpu(), self.graphs_dir / f"graph_{idx}.pt")

        metadata_path = Path(self.processed_dir) / f"metadata.pt"
        torch.save(metadata, metadata_path)
        self.metadata = metadata

        self.file_list = sorted(self.graphs_dir.glob("graph_*.pt"))

class CrystalGraphDataModule(LightningDataModule):
    def __init__(self, graph, dataset, seed):
        super().__init__()
        self.graph_cfg = graph
        self.dataset_cfg = dataset

        self.seed = seed

        self.dataset = None
        self.metadata = None

    def prepare_data(self):
        _ = Dataset(
            graph_cfg=self.graph_cfg,
            dataset_cfg=self.dataset_cfg,
        )

    def setup(self, stage: Optional[str] = None):
        dataset = Dataset(
            graph_cfg=self.graph_cfg,
            dataset_cfg=self.dataset_cfg,
        )

        self.dataset = dataset
        self.metadata = dataset.metadata

        n_total = len(dataset)
        n_train = int(n_total * self.dataset_cfg.train_ratio)
        n_val = int(n_total * self.dataset_cfg.val_ratio)
        n_test = n_total - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.dataset_cfg.batch_size.train,
            shuffle=True,
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.seed)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.dataset_cfg.batch_size.val,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.dataset_cfg.batch_size.test,
            shuffle=False,
            pin_memory=True
        )