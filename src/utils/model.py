import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

def print_tensor_stats(x, name='tensor'):
    x = x.flatten()
    print(f"--- {name} statistics ---")
    print("shape:", tuple(x.shape))
    print("dtype:", x.dtype)
    print("min:", x.min().item())
    print("max:", x.max().item())
    print("mean:", x.mean().item())
    print("std:", x.std().item())
    print("median:", x.median().item())
    print("quantiles:", torch.quantile(x, torch.tensor([0.25, 0.5, 0.75])).tolist())

def get_graph_and_node_targets(target_cfg):
    graph_target_names = [
        name for name, cfg in target_cfg.items()
        if cfg['level'] == 'graph'
    ]
    graph_target_lengths = [
        target_cfg[name]['length'] for name in graph_target_names
    ]
    graph_target_cfg = {
        'names': graph_target_names,
        'lengths': graph_target_lengths,
    }

    node_target_names = [
        name for name, cfg in target_cfg.items()
        if cfg['level'] == 'node'
    ]
    node_target_lengths = [
        target_cfg[name]['length'] for name in node_target_names
    ]
    node_target_cfg = {
        'names': node_target_names,
        'lengths': node_target_lengths,
    }
    return graph_target_cfg, node_target_cfg

def get_activation_fn(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif act_fn == 'prelu':
        return nn.PReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'selu':
        return nn.SELU()
    elif act_fn == 'silu' or act_fn == 'swish':
        return nn.SiLU()  # SiLU is the same as Swish
    elif act_fn == 'gelu':
        return nn.GELU()
    elif act_fn == 'tanh':
        return nn.Tanh()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'identity' or act_fn == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

def reset_layer_parameters(layer, act_fn='relu'):
    """
    Applies Xavier or He initialization to a layer (Linear or Conv).
    """
    nonlinearity = 'relu' if act_fn in ['relu', 'leaky_relu'] else 'linear'

    if isinstance(layer, nn.Linear):
        if act_fn in ['relu', 'leaky_relu']:
            init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        else:
            init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)

    # For TransformerConv or other convs, try to access submodules
    elif hasattr(layer, 'lin') and isinstance(layer.lin, nn.Linear):
        reset_layer_parameters(layer.lin, act_fn)
    elif hasattr(layer, 'lin_l') and isinstance(layer.lin_l, nn.Linear):
        reset_layer_parameters(layer.lin_l, act_fn)
    elif hasattr(layer, 'lin_r') and isinstance(layer.lin_r, nn.Linear):
        reset_layer_parameters(layer.lin_r, act_fn)

def initialization_params(parameters, initialization_method=None):
    if initialization_method is None:
        return

    for p in parameters:
        if p.dim() == 1:
            torch.nn.init.normal_(p, std=1e-2)
        else:
            if initialization_method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(p)
            elif initialization_method == 'xavier_normal':
                torch.nn.init.xavier_normal_(p)
            elif initialization_method == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(p, nonlinearity='relu')
            elif initialization_method == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(p, nonlinearity='relu')
            elif initialization_method == 'orthogonal':
                torch.nn.init.orthogonal_(p)
            elif initialization_method == 'normal':
                torch.nn.init.normal_(p, mean=0, std=1e-2)
            elif initialization_method == 'uniform':
                torch.nn.init.uniform_(p)
            else:
                raise ValueError(f"Unknown initialization method: {initialization_method}")

def build_mlp(in_dim, out_dim, hidden_dim, num_layers, act_fn):
    layers = []
    activation = get_activation_fn(act_fn)
    for i in range(num_layers):
        input_dim = in_dim if i == 0 else hidden_dim
        output_dim = out_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(input_dim, output_dim, bias=True))
        if i != num_layers - 1:
            # layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(activation)
    return nn.Sequential(*layers)

class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def scatter_mean(tensor, batch=None, dim=0):
    if batch is None:
        batch = torch.zeros(tensor.shape[0], dtype=torch.long, device=tensor.device)

    dim_size = int(batch.max()) + 1

    out = torch.zeros(dim_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    out.scatter_add_(dim, batch.unsqueeze(-1).expand_as(tensor), tensor)

    count = torch.zeros(dim_size, dtype=tensor.dtype, device=tensor.device)
    count.scatter_add_(0, batch, torch.ones_like(batch, dtype=tensor.dtype))
    count = count.clamp(min=1).unsqueeze(-1)
    return out / count

def scatter_add(tensor, batch=None, dim=0):
    if batch is None:
        batch = torch.zeros(tensor.shape[0], dtype=torch.long, device=tensor.device)

    dim_size = int(batch.max()) + 1
    out = torch.zeros(dim_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    out.scatter_add_(dim, batch.unsqueeze(-1).expand_as(tensor), tensor)
    return out

def split_output(x: torch.Tensor, sizes: list[int]):
    total_sizes = sum(sizes)
    assert x.shape[-1] == total_sizes, f"Sum of sizes ({total_sizes}) does not match x.shape[-1] = {x.shape[-1]}"

    output = []
    start = 0
    for s in sizes:
        output.append(x[..., start:start+s])
        start += s
    return output

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        x = np.squeeze(x)
        if x.ndim == 0:
            return float(x)
        return x

    return x

def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def get_param_group(name):
    if "embedding" in name:
        return "embedding"
    elif "encoder" in name or "norms" in name:
        return "encoder"
    elif "decoder" in name:
        return "decoders"
    else:
        return "others"

def compute_param_stats(named_params):
    stats = {}

    for name, p in named_params:
        if not p.requires_grad:
            continue

        group = get_param_group(name)

        if group not in stats:
            stats[group] = {
                "sum_sq": 0.0,      # for L2
                "sum_abs": 0.0,     # for L1
                "sum": 0.0,         # mean
                "sum_sq_raw": 0.0,  # variance
                "count": 0,
                "max_abs": 0.0,
            }

        g = stats[group]
        data = p.data

        g["sum_sq"] += torch.sum(data ** 2).item()
        g["sum_abs"] += torch.sum(data.abs()).item()
        g["sum"] += torch.sum(data).item()
        g["sum_sq_raw"] += torch.sum(data ** 2).item()
        g["count"] += data.numel()
        g["max_abs"] = max(g["max_abs"], data.abs().max().item())

    out = {}

    for group, g in stats.items():
        count = g["count"]

        mean = g["sum"] / count
        var = g["sum_sq_raw"] / count - mean ** 2
        std = var ** 0.5 if var > 0 else 0.0

        l2 = g["sum_sq"] ** 0.5
        l1 = g["sum_abs"]
        rms = (g["sum_sq"] / count) ** 0.5

        out[f"{group}_param_norm_l2"] = l2
        out[f"{group}_param_norm_l1"] = l1
        out[f"{group}_param_abs_max"] = g["max_abs"]

        out[f"{group}_param_mean"] = mean
        out[f"{group}_param_std"] = std
        out[f"{group}_param_rms"] = rms

        out[f"{group}_param_count"] = count

    return out

def compute_grad_stats(named_params):
    stats = {}

    for name, p in named_params:
        if not p.requires_grad:
            continue

        group = get_param_group(name)

        if group not in stats:
            stats[group] = {
                "sum_sq": 0.0,      # for L2
                "sum_abs": 0.0,     # for L1
                "sum": 0.0,         # mean
                "sum_sq_raw": 0.0,  # variance
                "count": 0,
                "max_abs": 0.0,
            }

        g = stats[group]
        grad = p.grad

        g["sum_sq"] += torch.sum(grad ** 2).item()
        g["sum_abs"] += torch.sum(grad.abs()).item()
        g["sum"] += torch.sum(grad).item()
        g["sum_sq_raw"] += torch.sum(grad ** 2).item()
        g["count"] += grad.numel()
        g["max_abs"] = max(g["max_abs"], grad.abs().max().item())

    # -------- finalize --------
    out = {}

    for group, g in stats.items():
        count = g["count"]

        mean = g["sum"] / count
        var = g["sum_sq_raw"] / count - mean ** 2
        std = var ** 0.5 if var > 0 else 0.0

        l2 = g["sum_sq"] ** 0.5
        l1 = g["sum_abs"]
        rms = (g["sum_sq"] / count) ** 0.5

        out[f"{group}_grad_norm_l2"] = l2
        out[f"{group}_grad_norm_l1"] = l1
        out[f"{group}_grad_abs_max"] = g["max_abs"]

        out[f"{group}_grad_mean"] = mean
        out[f"{group}_grad_std"] = std
        out[f"{group}_grad_rms"] = rms

        out[f"{group}_grad_count"] = count

    return out