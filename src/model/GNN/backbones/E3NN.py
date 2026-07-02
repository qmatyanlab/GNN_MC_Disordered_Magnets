import torch
import torch.nn as nn

from e3nn import o3
from model.GNN.backbones.E3NN_Network.Network import PeriodicNetwork

from utils.graph import get_atomic_feature_spec
from utils.model import build_mlp, get_graph_and_node_targets

from constants.model import NUM_PH_FEATURES

class E3NNBackbone(nn.Module):
    def __init__(
            self,
            num_hidden_layers,
            num_hidden_channels,
            r_cutoff,
            num_MLP_layers,
            num_radial_basis,
            target_cfg,
            use_initial_MAGMOM,
            use_PH_features,
            use_frac_coords,
            elemental_feature_type='one_hot',
            act_fn='relu',
            lmax=2,
            multiplicity=1,
            radial_mlp_layers=2,
            radial_mlp_hidden_dim=128,
            *args, **kwargs
    ):
        super().__init__()

        self.num_hidden_channels = num_hidden_channels

        # ------- Node features  -------
        atomic_feature_spec = get_atomic_feature_spec(elemental_feature_type)
        self.elemental_embedding = nn.Linear(atomic_feature_spec.dim, num_hidden_channels)

        self.use_initial_MAGMOM = use_initial_MAGMOM
        if use_initial_MAGMOM:
            self.initial_MAGMOM_embedding = nn.Linear(1, num_hidden_channels)

        self.use_PH_features = use_PH_features
        if use_PH_features:
            self.PH_embedding = nn.Linear(NUM_PH_FEATURES, num_hidden_channels)

        self.use_frac_coords = use_frac_coords
        if use_frac_coords:
            self.frac_coords_embedding = nn.Linear(3, num_hidden_channels)

        all_embedding_dim = (
            1 + int(self.use_initial_MAGMOM) + int(self.use_PH_features) + int(self.use_frac_coords)
        ) * num_hidden_channels
        self.embedding = nn.Linear(all_embedding_dim, num_hidden_channels)

        # ------- Encoder  -------
        self.encoder = PeriodicNetwork(
            irreps_in=o3.Irreps(f"{num_hidden_channels}x0e"),
            irreps_out=o3.Irreps(f"{num_hidden_channels}x0e"),
            num_layers=num_hidden_layers,
            r_cutoff=r_cutoff,
            num_radial_basis=num_radial_basis,
            lmax=lmax,
            multiplicity=multiplicity,
            radial_mlp_layers=radial_mlp_layers,
            radial_mlp_hidden_dim=radial_mlp_hidden_dim,
            num_neighbors=64,
        )

        # -------- Decoder  -------
        self.target_cfg = target_cfg
        graph_target_cfg, node_target_cfg = get_graph_and_node_targets(target_cfg)
        self.graph_target_names = graph_target_cfg['names']
        self.graph_target_lengths = graph_target_cfg['lengths']
        self.node_target_names = node_target_cfg['names']
        self.node_target_lengths = node_target_cfg['lengths']

        self.graph_decoders = nn.ModuleDict()
        for name, length in zip(self.graph_target_names, self.graph_target_lengths):
            self.graph_decoders[name] = build_mlp(
                in_dim=num_hidden_channels,
                out_dim=length,
                hidden_dim=num_hidden_channels,
                num_layers=num_MLP_layers,
                act_fn=act_fn,
            )

        self.node_decoders = nn.ModuleDict()
        for name, length in zip(self.node_target_names, self.node_target_lengths):
            self.node_decoders[name] = build_mlp(
                in_dim=num_hidden_channels,
                out_dim=length,
                hidden_dim=num_hidden_channels,
                num_layers=num_MLP_layers,
                act_fn=act_fn,
            )

    def forward(self, batch):
        x_element = self.elemental_embedding(batch.x)
        embeds = [x_element]

        if self.use_initial_MAGMOM:
            embeds.append(self.initial_MAGMOM_embedding(batch.initial_MAGMOM))

        if self.use_PH_features:
            embeds.append(self.PH_embedding(batch.PH_features))

        if self.use_frac_coords:
            embeds.append(self.frac_coords_embedding(batch.frac_coords))

        x = torch.cat(embeds, dim=-1)
        x = self.embedding(x)

        x, graph_feature = self.encoder(x, batch)

        output = {}

        for name in self.graph_target_names:
            value = self.graph_decoders[name](graph_feature)
            output[name] = value

        for name in self.node_target_names:
            value = self.node_decoders[name](x)
            output[name] = value

        return output