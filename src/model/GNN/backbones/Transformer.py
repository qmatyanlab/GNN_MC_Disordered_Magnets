import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_add_pool as pool
# from torch_geometric.nn import global_mean_pool as pool

from utils.graph import get_atomic_feature_spec
from utils.model import build_mlp, SinusoidsEmbedding, get_activation_fn, split_output, get_graph_and_node_targets

from constants.model import NUM_PH_FEATURES

class TransformerBackbone(nn.Module):
    def __init__(
            self,
            num_hidden_layers,
            num_hidden_channels,
            num_heads,
            num_MLP_layers,
            num_radial_basis,
            act_fn,
            target_cfg,
            use_initial_MAGMOM,
            use_PH_features,
            use_frac_coords,
            use_node_features_in_edge,
            use_displacement_vector,
            num_sinusoids=10,
            elemental_feature_type='one_hot',
            *args, **kwargs
    ):
        super().__init__()

        # ------- Node features  -------
        atomic_feature_spec = get_atomic_feature_spec(elemental_feature_type)
        self.element_embedding = nn.Linear(atomic_feature_spec.dim, num_hidden_channels)

        self.use_initial_MAGMOM = use_initial_MAGMOM
        if self.use_initial_MAGMOM:
            self.initial_MAGMOM_embedding = nn.Linear(1, num_hidden_channels)

        self.use_PH_features = use_PH_features
        if self.use_PH_features:
            self.PH_embedding = nn.Linear(NUM_PH_FEATURES, num_hidden_channels)

        self.use_frac_coords = use_frac_coords
        if self.use_frac_coords:
            self.frac_coords_embedding = nn.Linear(3, num_hidden_channels)

        all_embedding_dim = (
            1 + int(self.use_initial_MAGMOM) + int(self.use_PH_features) + int(self.use_frac_coords)
        ) * num_hidden_channels

        self.embedding = nn.Linear(all_embedding_dim, num_hidden_channels)

        # ------- Edge features  -------
        self.use_displacement_vector = use_displacement_vector
        self.use_node_features_in_edge = use_node_features_in_edge

        self.displacement_vector_embedding = SinusoidsEmbedding(
            n_frequencies=num_sinusoids,
        )

        edge_dim = num_radial_basis

        if self.use_displacement_vector:
            edge_dim += self.displacement_vector_embedding.dim

        if self.use_node_features_in_edge:
            edge_dim += 2 * num_hidden_channels

        # ------- Encoder  -------
        self.encoder = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.encoder.append(
                TransformerConv(
                    num_hidden_channels,
                    num_hidden_channels,
                    heads=num_heads,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(num_hidden_channels))

        # ------- Decoder  -------
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

        self.act_fn = get_activation_fn(act_fn)

    def forward(self, batch):
        x_element = self.element_embedding(batch.x)
        embeds = [x_element]

        if self.use_initial_MAGMOM:
            embeds.append(self.initial_MAGMOM_embedding(batch.initial_MAGMOM))

        if self.use_PH_features:
            embeds.append(self.PH_embedding(batch.PH_features))

        if self.use_frac_coords:
            embeds.append(self.frac_coords_embedding(batch.frac_coords))

        x = torch.cat(embeds, dim=-1)
        x = self.embedding(x)

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        edge_inputs = [edge_attr]

        if self.use_displacement_vector:
            if not hasattr(batch, "edge_vec"):
                raise AttributeError(
                    "use_displacement_vector=True but edge_vec missing."
                )
            edge_vec = batch.edge_vec
            edge_inputs.append(self.displacement_vector_embedding(edge_vec))

        if self.use_node_features_in_edge:
            edge_inputs.append(x[edge_index[0]])
            edge_inputs.append(x[edge_index[1]])

        edge_attr_all = torch.cat(edge_inputs, dim=-1)

        # for conv in self.encoder:
        #     residual = x
        #     x = conv(x, edge_index, edge_attr_all)
        #     x = self.act_fn(x)
        #     x = x + residual

        for conv, norm in zip(self.encoder, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr_all)
            x = norm(x)
            x = self.act_fn(x)
            x = x + residual

        graph_feature = pool(x, batch.batch)

        output = {}

        for name in self.graph_target_names:
            value = self.graph_decoders[name](graph_feature)
            output[name] = value

        for name in self.node_target_names:
            value = self.node_decoders[name](x)
            output[name] = value

        return output