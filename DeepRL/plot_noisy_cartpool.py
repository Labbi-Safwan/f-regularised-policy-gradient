
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import math
from typing import Dict, Any, Tuple, List, Optional
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def create_folder_if_not_exists(folder_path: str):
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

def save_figure(fig: plt.Figure, path: str):
    create_folder_if_not_exists(os.path.dirname(path))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def subsample_series(
    x: np.ndarray,
    mean: np.ndarray,
    std_or_se: np.ndarray,
    every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if every is None or every <= 1:
        return x, mean, std_or_se
    n = len(x)
    idx = np.arange(0, n, every)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return x[idx], mean[idx], std_or_se[idx]

def build_groupings(records):
    grouped = defaultdict(list)
    metadata = {
        "reg_alpha": set(),
        "param_alpha": set(),
        "learning_rate": set(),
        "entropy_coeff": set(),
    }
    for rec in records:
        rewards = np.asarray(rec["rewards"])
        key = (
            rec["reg_alpha"],
            rec["param_alpha"],
            rec["learning_rate"],
            rec["entropy_coeff"],
        )
        grouped[key].append(rewards)
        metadata["reg_alpha"].add(rec["reg_alpha"])
        metadata["param_alpha"].add(rec["param_alpha"])
        metadata["learning_rate"].add(rec["learning_rate"])
        metadata["entropy_coeff"].add(rec["entropy_coeff"])
    metadata = {k: sorted(v) for k, v in metadata.items()}
    return grouped, metadata

def summarize_runs(run_list):
    arr = np.stack(run_list)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        std = arr.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    se = std / math.sqrt(arr.shape[0]) if arr.shape[0] > 0 else std
    return mean, se

def alpha_key(alpha: float, decimals: int = 1) -> float:
    return float(np.round(float(alpha), decimals))

def get_style_for_alpha(
    alpha: float,
    STYLE_FPG: Dict[float, Dict[str, Any]],
    tol: float = 1e-8,
) -> Tuple[str, str, str]:
    a = alpha_key(alpha, decimals=1)
    if abs(a - 1.0) <= tol:
        label = "PPO (baseline)"
    else:
        label = rf"{a}-Tsallis PPO"
    if a not in STYLE_FPG:
        raise KeyError(f"alpha={a} not found in STYLE_FPG. Add it to STYLE_FPG.")
    style = STYLE_FPG[a]
    return label, style["color"], style["marker"]

def make_legend_figure(
    legend_handles: List[Line2D],
    fontsize: int,
    ncol: int = 6,
) -> plt.Figure:
    fig = plt.figure(figsize=(10, 1))
    fig.legend(
        handles=legend_handles,
        loc="center",
        fontsize=fontsize,
        frameon=False,
        handlelength=3.0,
        handletextpad=1.0,
        markerscale=1.4,
        ncol=ncol,
    )
    fig.tight_layout()
    return fig

def make_single_cartpole_noisy_plot(
    noise_value: float,
    results_folder: Path,
    file_pattern: str,
    scale_x: float,
    fontsize: int,
    font: Dict[str, Any],
    STYLE_FPG: Dict[float, Dict[str, Any]],
    fpg_alpha_tol: float,
    subsample_every: int,
    max_steps: Optional[int],
) -> Tuple[plt.Figure, List[Line2D]]:

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    filename = results_folder / file_pattern.format(noise=noise_value)
    if not filename.exists():
        raise FileNotFoundError(f"{filename} not found")

    with open(filename, "rb") as fp:
        raw_records = pickle.load(fp)
    print(f"Loaded {len(raw_records)} records from {filename}")

    filtered = []
    for r in raw_records:
        if "noise" in r:
            if np.isclose(float(r["noise"]), float(noise_value)):
                filtered.append(r)
        else:
            filtered.append(r)
    raw_records = filtered

    grouped, metadata = build_groupings(raw_records)

    best_curves: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for reg_alpha in metadata["reg_alpha"]:
        for param_alpha in metadata["param_alpha"]:
            pair_best_score = None
            pair_best_entry = None
            for lr in metadata["learning_rate"]:
                for entropy in metadata["entropy_coeff"]:
                    key = (reg_alpha, param_alpha, lr, entropy)
                    if key not in grouped:
                        continue
                    mean, se = summarize_runs(grouped[key])
                    x = np.arange(mean.shape[0])

                    tail = mean[-100:] if mean.shape[0] >= 100 else mean
                    final_score = tail.mean()

                    if pair_best_score is None or final_score > pair_best_score:
                        pair_best_score = final_score
                        pair_best_entry = {"mean": mean, "se": se, "x": x}
            if pair_best_entry is not None:
                best_curves[(reg_alpha, param_alpha)] = pair_best_entry

    diag_stats = [
        ((reg_alpha, param_alpha), stats)
        for (reg_alpha, param_alpha), stats in best_curves.items()
        if np.isclose(reg_alpha, param_alpha)
    ]
    if not diag_stats:
        raise RuntimeError(f"No diagonal entries found for noise={noise_value}")

    diag_stats.sort(key=lambda t: float(t[0][0]))  

    legend_handles: List[Line2D] = []

    for ((reg_a, _param_a), stats) in diag_stats:
        x = stats["x"].astype(np.float64)
        mean = stats["mean"]
        se = stats["se"]

        if max_steps is not None:
            mask = x <= max_steps
            x = x[mask]
            mean = mean[mask]
            se = se[mask]

        x = x / scale_x

        xs, ms, ss = subsample_series(x, mean, se, subsample_every)

        label, color, marker = get_style_for_alpha(
            reg_a, STYLE_FPG=STYLE_FPG, tol=fpg_alpha_tol
        )

        markevery = max(1, len(xs) // 10)

        ax.plot(
            xs,
            ms,
            linestyle="-",
            linewidth=1.4,
            color=color,
            marker=marker,
            markersize=5,
            markevery=markevery,
        )

        ax.fill_between(
            xs,
            ms - ss,
            ms + ss,
            color=color,
            alpha=0.10,
            edgecolor="none",
            linewidth=0,
        )

        legend_handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=1.4, label=label)
        )


    ax.set_xlabel(r"Steps ($\times 10^4$)", fontsize=fontsize, **font)
    ax.set_ylabel("Average return", fontsize=fontsize, **font)

    ax.grid(linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=fontsize - 3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if max_steps is not None:
        ax.set_xlim(0, max_steps / scale_x)

    fig.tight_layout()
    return fig, legend_handles

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    NOISE_VALUES = [0.0, 0.5, 2.0, 10.0]  # <-- edit these 4 values

    results_folder = Path("./")

    file_pattern = "cartpole_noisy_results_{noise}.pkl"

    scale_x = 1e4

    fontsize = 16
    font = {"family": "serif"}

    STYLE_FPG = {
        0.1: {"color": "#0173B2", "marker": "o"},
        0.3: {"color": "#029E73", "marker": "s"},
        0.5: {"color": "#F20A99", "marker": "D"},
        0.7: {"color": "#D55E00", "marker": "v"},
        0.9: {"color": "#CC78BC", "marker": "^"},
        1.0: {"color": "#56B4E9", "marker": "P"},
    }
    fpg_alpha_tol = 1e-8

    subsample_every = 1000

    MAX_STEPS = 20000  

    out_dir = "./plots"
    create_folder_if_not_exists(out_dir)

    legend_handles_final: Optional[List[Line2D]] = None

    for noise in NOISE_VALUES:
        fig, legend_handles = make_single_cartpole_noisy_plot(
            noise_value=noise,
            results_folder=results_folder,
            file_pattern=file_pattern,
            scale_x=scale_x,
            fontsize=fontsize,
            font=font,
            STYLE_FPG=STYLE_FPG,
            fpg_alpha_tol=fpg_alpha_tol,
            subsample_every=subsample_every,
            max_steps=MAX_STEPS,
        )
        legend_handles_final = legend_handles
        save_figure(fig, os.path.join(out_dir, f"cartpole_noisy_sigma_{noise}.pdf"))

    if legend_handles_final is None:
        raise RuntimeError("No legend handles were generated (no plots were created).")

    fig_leg = make_legend_figure(
        legend_handles=legend_handles_final,
        fontsize=fontsize - 4,
        ncol=len(legend_handles_final),
    )
    save_figure(fig_leg, os.path.join(out_dir, "legend_cartpole_noisy.pdf"))

    print(f"Saved {len(NOISE_VALUES)} plots to {out_dir}/ and legend_cartpole_noisy.pdf")
