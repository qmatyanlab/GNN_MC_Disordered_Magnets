from __future__ import annotations

from typing import Union

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.size'] = 16
rcParams['font.family'] = 'Arial'

# ============================================================
#                  NORMALIZATION HELPERS
# ============================================================

def normalize(
    unnorm: list,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """
    Normalize array and compute stats.
    """
    reduce = cfg.get("reduce", "min_max")
    axis = cfg.get("axis", "all")
    if isinstance(axis, str):
        axis_map = {"configuration": 0, "xgrid": 1, "all": None}
        if axis not in axis_map:
            raise ValueError("axis must be one of ['configuration', 'xgrid', 'all']")
        axis = axis_map[axis]

    stats = {"reduce": reduce, "axis": axis}

    unnorm = np.array(unnorm)
    if reduce == "min_max":
        min_val = np.min(unnorm, axis=axis, keepdims=True)
        max_val = np.max(unnorm, axis=axis, keepdims=True)
        norm = 2 * (unnorm - min_val) / (max_val - min_val) - 1
        stats.update({"min_val": min_val, "max_val": max_val})

    elif reduce == "mean":
        mean_val = np.mean(unnorm, axis=axis, keepdims=True)
        norm = unnorm / mean_val
        stats["mean_val"] = mean_val

    elif reduce == "median":
        median_val = np.median(np.max(unnorm, axis=1))
        norm = unnorm / median_val
        stats["median_val"] = median_val

    elif reduce == "gaussian":
        mean_val = np.mean(unnorm, axis=axis, keepdims=True)
        std_val = np.std(unnorm, axis=axis, keepdims=True)
        norm = (unnorm - mean_val) / std_val
        stats.update({"mean_val": mean_val, "std_val": std_val})

    elif reduce == "log":
        norm = np.log(unnorm)

    else:
        raise ValueError(f"Unknown normalization function: {reduce}")

    return norm, stats

def denormalize(norm: np.ndarray, stats: dict) -> np.ndarray:
    """
    Reverse normalization.
    """
    func = stats["reduce"]

    if func == "min_max":
        return (norm + 1) * (stats["max_val"] - stats["min_val"]) / 2 + stats["min_val"]

    if func == "mean":
        return norm * stats["mean_val"]

    if func == "median":
        return norm * stats["median_val"]

    if func == "gaussian":
        return norm * stats["std_val"] + stats["mean_val"]

    if func == "log":
        return np.exp(norm)

    raise ValueError(f"Unknown normalization function: {func}")

def denormalize_outputs(outputs, metadata):
    denorm_outputs = {}
    for tgt, val in outputs.items():
        if tgt in metadata:
            denorm_outputs[tgt] = denormalize(val, metadata[tgt]).squeeze()
        else:
            denorm_outputs[tgt] = val
    return denorm_outputs

def filter_outliers_iqr(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (x >= lower) & (x <= upper)
    return x[mask], mask

def fit_gaussian_and_plot(x, fig_path, title):
    x_filtered, mask = filter_outliers_iqr(x)

    mu = np.mean(x_filtered)
    sigma = np.std(x_filtered)

    print(f"\n{title}")
    print(f"Total samples      : {len(x)}")
    print(f"After filtering    : {len(x_filtered)}")
    print(f"Removed as outliers: {len(x) - len(x_filtered)}")
    print(f"Gaussian fit μ     : {mu:.6f}")
    print(f"Gaussian fit σ     : {sigma:.6f}")

    # Histogram
    plt.figure(figsize=(6, 4))
    count, bins, _ = plt.hist(
        x_filtered,
        bins=30,
        density=True,
        alpha=0.6,
        edgecolor="black"
    )

    # Gaussian curve
    x_axis = np.linspace(bins.min(), bins.max(), 300)
    pdf = stats.norm.pdf(x_axis, mu, sigma)

    plt.plot(x_axis, pdf, linewidth=2)
    plt.title(title)
    plt.xlabel("Normalized value")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return {
        "mu": mu,
        "sigma": sigma,
        "n_total": len(x),
        "n_filtered": len(x_filtered),
    }