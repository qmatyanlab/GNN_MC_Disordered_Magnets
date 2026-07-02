from pathlib import Path
import logging
import json
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import rcParams

import torch

from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

rcParams["font.family"] = "Arial"
rcParams["font.size"] = 16

def load_data(filename):
    filename = Path(filename)
    with open(filename, 'rb') as f:
        return pkl.load(f)

def flatten(x):
    return np.asarray(x).reshape(-1)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return [to_numpy(i) for i in x]
    else:
        return x

def choose_targets(requested_targets: str):
    # We always assume energy in the chosen targets.
    # If not trained, we set the training ratio to be zero.
    requested = "".join(sorted(set(requested_targets)))
    if "m" in requested:
        return "efsm"
    if "s" in requested:
        return "efs"
    return "ef"

def build_loss_weights(
    requested_targets: str,
    energy_loss_ratio: float = 0.0,
    force_loss_ratio: float = 1.0,
    stress_loss_ratio: float = 0.1,
    mag_loss_ratio: float = 0.0,
) -> dict[str, float]:
    requested = set(requested_targets)
    return {
        "energy_loss_ratio": energy_loss_ratio if "e" in requested else 0.0,
        "force_loss_ratio": force_loss_ratio if "f" in requested else 0.0,
        "stress_loss_ratio": stress_loss_ratio if "s" in requested else 0.0,
        "mag_loss_ratio": mag_loss_ratio if "m" in requested else 0.0,
    }

def print_model_modules(model, max_depth=2):
    def recurse(module, prefix="", depth=0):
        if depth > max_depth:
            return
        for name, child in module.named_children():
            logger.info(f"{prefix}{name}: {child.__class__.__name__}")
            recurse(child, prefix + "  ", depth + 1)

    logger.info("\n=== Model Modules ===")
    recurse(model)
    logger.info("====================\n")

def print_named_parameters(model):
    logger.info("\n=== Model Parameters ===")
    for name, param in model.named_parameters():
        logger.info(f"{name:60s} shape={tuple(param.shape)}")
    logger.info("========================\n")

def apply_trainable_modules(model, trainable_modules):
    for name, param in model.named_parameters():
        param.requires_grad = any(p in name for p in trainable_modules)

    # always freeze composition model
    for name, param in model.named_parameters():
        if "composition_model" in name:
            param.requires_grad = False

    total, trainable = 0, 0
    logger.info("\n=== Trainable Parameters ===")
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            logger.info(name)
    logger.info("----------------------------")
    logger.info(f"Trainable params: {trainable:,}")
    logger.info(f"Total params:     {total:,}")
    logger.info(f"Fraction:         {trainable/total:.4f}\n")

def to_serializable(obj):
    from omegaconf import ListConfig, DictConfig, OmegaConf

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (ListConfig, DictConfig)):
        return to_serializable(OmegaConf.to_container(obj, resolve=True))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    else:
        return obj

def save_json(obj: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def evaluate_model(model, loader, requested_targets, device):
    model.to(device)
    model.eval()

    results = {
        "energy": {"pred": [], "tgt": []},
        "forces": {"pred": [], "tgt": []},
        "stress": {"pred": [], "tgt": []},
    }
    with torch.enable_grad():
        for graphs, tgts in loader:
            tgts_np = {k: to_numpy(v) for k, v in tgts.items()}

            preds = model.predict_graph(graphs, task=requested_targets)
            for i, pred in enumerate(preds):
                if "e" in requested_targets:
                    results["energy"]["pred"].append(pred["e"])
                    results["energy"]["tgt"].append(tgts_np["e"][i])

                if "f" in requested_targets:
                    f_pred = np.array(pred["f"])  # (Ni, 3)
                    f_tgt = np.array(tgts_np["f"][i])  # (Ni, 3)

                    results["forces"]["pred"].extend(flatten(f_pred))
                    results["forces"]["tgt"].extend(flatten(f_tgt))

                if "s" in requested_targets:
                    s_pred = np.array(pred["s"])  # (3, 3)
                    s_tgt = np.array(tgts_np["s"][i])  # (3, 3)

                    results["stress"]["pred"].extend(flatten(s_pred))
                    results["stress"]["tgt"].extend(flatten(s_tgt))
    return results

def compute_metrics(results):
    metrics = {}
    logger.info("\n=== Test Metrics ===")
    for key, val in results.items():
        pred = np.array(val["pred"])
        tgt = np.array(val["tgt"])

        mae = np.mean(np.abs(pred - tgt))
        mse = np.mean((pred - tgt) ** 2)
        r2 = r2_score(tgt, pred)
        metrics[key] = {"mae": float(mae), "mse": float(mse), "r2": float(r2)}
        logger.info(f"{key:10s} mae={mae:.6f} mse={mse:.6f} r2={r2:.6f}")
    return metrics

def compute_weighted_loss(metrics, targets):
    loss = 0.0
    mapping = {
        "e": "energy",
        "f": "forces",
        "s": "stress",
        "m": "magmom",
    }
    for key, name in mapping.items():
        if key in targets and name in metrics:
            loss += metrics[name]["mae"]
    return loss

def plot_results(results, metrics, fig_dir, prefix=None):
    label_cfg = {
        "energy": ("E",       "eV"),
        "forces": ("F",       r"eV/$\mathrm{\AA}$"),
        "stress": (r"\sigma", "GPa"),
    }

    keys = list(results.keys())
    n = len(keys)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        val = results[key]
        x = np.array(val["tgt"])
        y = np.array(val["pred"])

        symbol, unit = label_cfg.get(key, (key, ""))

        ax.scatter(x, y, s=5, alpha=0.3)

        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "k--")

        ax.set_xlabel(f"${symbol}_{{\\mathrm{{DFT}}}}$ ({unit})")
        ax.set_ylabel(f"${symbol}_{{\\mathrm{{MLIP}}}}$ ({unit})")

        if key in metrics:
            mae = metrics[key]["mae"]
            r2  = metrics[key]["r2"]
            ax.text(
                0.95, 0.05,
                f"MAE = {mae:.2f}\nR$^2$ = {r2:.2f}",
                transform=ax.transAxes,
                va="bottom", ha="right",
                #
                # bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

    plt.tight_layout()

    fname = f"{prefix}_results" if prefix else "results"
    fig.savefig(fig_dir / f"{fname}.pdf")
    plt.close(fig)