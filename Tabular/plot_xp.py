import os
import pickle
from typing import List, Dict, Optional, Any, Tuple
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def load_results_from_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def create_folder_if_not_exists(folder_path: str):
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def subsample_series(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    every: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if every is None or every <= 1:
        return x, mean, std

    n = len(x)
    idx = np.arange(0, n, every)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    return x[idx], mean[idx], std[idx]


def parse_fpg(folder_name: str) -> Optional[Dict[str, float]]:
    parts = folder_name.split("_")
    if len(parts) != 6:
        return None
    if parts[0] != "step" or parts[2] != "alpha" or parts[4] != "temperature":
        return None
    try:
        return {"step": float(parts[1]), "alpha": float(parts[3]), "temp": float(parts[5])}
    except ValueError:
        return None


def parse_logbarrier(folder_name: str) -> Optional[Dict[str, float]]:
    parts = folder_name.split("_")
    if len(parts) != 4:
        return None
    if parts[0] != "step" or parts[2] != "temperature":
        return None
    try:
        return {"step": float(parts[1]), "temp": float(parts[3])}
    except ValueError:
        return None


def parse_escort(folder_name: str) -> Optional[Dict[str, float]]:
    parts = folder_name.split("_")
    if len(parts) != 4:
        return None
    if parts[0] != "step" or parts[2] != "alpha":
        return None
    try:
        return {"step": float(parts[1]), "alpha": float(parts[3])}
    except ValueError:
        return None


def parse_hadamard(folder_name: str) -> Optional[Dict[str, float]]:
    parts = folder_name.split("_")
    if len(parts) != 2:
        return None
    if parts[0] != "step":
        return None
    try:
        return {"step": float(parts[1])}
    except ValueError:
        return None


def parse_folder(algo_name: str, folder_name: str) -> Optional[Dict[str, Any]]:
    if algo_name == "fpg":
        return parse_fpg(folder_name)
    if algo_name == "logbarrier":
        return parse_logbarrier(folder_name)
    if algo_name == "escort":
        return parse_escort(folder_name)
    if algo_name == "hadamard":
        return parse_hadamard(folder_name)
    return None


def load_config_stats(
    base_dir: str,
    env_size: int,
    seeds: List[int],
    max_iter: Optional[int],
) -> Optional[Dict[str, np.ndarray]]:
    trajectories = []
    for seed in seeds:
        path = os.path.join(base_dir, f"size_{env_size}_seed_{seed}_true_objective.pkl")
        if not os.path.exists(path):
            return None
        trajectories.append(np.array(load_results_from_pickle(path)))

    if len(trajectories) == 0:
        return None

    min_len = min(len(t) for t in trajectories)
    if max_iter is not None:
        min_len = min(min_len, max_iter)

    mat = np.stack([t[:min_len] for t in trajectories], axis=0)
    mean = np.asarray(mat.mean(axis=0)).squeeze()
    std = np.asarray(mat.std(axis=0)).squeeze()

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(
            f"Expected 1D mean/std, got mean.shape={mean.shape}, std.shape={std.shape} "
            f"from base_dir={base_dir}"
        )

    return {"mean": mean, "std": std}


def alpha_key(alpha: float, tol: float = 1e-8) -> float:
    """Normalize float alpha by rounding (avoids 0.30000000000004 issues)."""
    return float(np.round(alpha, 1))


def get_style_for_entry(
    algo_name: str,
    alpha: Optional[float],
    fpg_alpha_tol: float,
    STYLE_FPG: Dict[float, Dict[str, Any]],
    STYLE_OTHER: Dict[str, Dict[str, Any]],
) -> Tuple[str, str, str]:
    """Returns (label, color, marker) for a plotted entry."""
    if algo_name == "fpg":
        a = alpha_key(alpha, tol=fpg_alpha_tol)
        if abs(a - 1.0) <= fpg_alpha_tol:
            label = "Softmax + Entropy"
        else:
            label = rf"fPG, $\alpha$ = {a}"
        style = STYLE_FPG[a]
        return label, style["color"], style["marker"]

    label_map = {
        "logbarrier": "Softmax + log-barrier",
        "escort": "Escort-transform",
        "hadamard": "Hadamard",
    }
    label = label_map[algo_name]
    style = STYLE_OTHER[algo_name]
    return label, style["color"], style["marker"]



def plot_best_algos_on_ax(
    ax,
    read_root: str,
    env_name: str,
    env_size: int,
    seeds: List[int],
    max_iter: Optional[int],
    algorithms: List[str],
    fontsize: int,
    font: Dict[str, Any],
    STYLE_FPG: Dict[float, Dict[str, Any]],
    STYLE_OTHER: Dict[str, Dict[str, Any]],
    fpg_alphas_to_plot: Optional[List[float]] = None,
    fpg_alpha_tol: float = 1e-8,
    subsample_every: int = 1,
) -> List[Line2D]:
    legend_handles: List[Line2D] = []

    for algo_name in algorithms:
        algo_root = os.path.join(read_root, algo_name, env_name, f"size_{env_size}")
        if not os.path.exists(algo_root):
            continue

        subfolders = [
            d for d in os.listdir(algo_root)
            if os.path.isdir(os.path.join(algo_root, d))
        ]

        if algo_name == "fpg":
            best_per_alpha: Dict[float, Dict[str, Any]] = {}

            for folder in subfolders:
                cfg = parse_folder(algo_name, folder)
                if cfg is None:
                    continue

                a = alpha_key(cfg["alpha"], tol=fpg_alpha_tol)
                stats = load_config_stats(os.path.join(algo_root, folder), env_size, seeds, max_iter)
                if stats is None:
                    continue

                final_val = float(stats["mean"][-1])
                if (a not in best_per_alpha) or (final_val > best_per_alpha[a]["final"]):
                    best_per_alpha[a] = {
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "final": final_val,
                        "step": cfg["step"],
                        "temp": cfg["temp"],
                    }

            want = None
            if fpg_alphas_to_plot is not None:
                want = set(alpha_key(x, tol=fpg_alpha_tol) for x in fpg_alphas_to_plot)

            for a in sorted(best_per_alpha):
                if want is not None and a not in want:
                    continue

                info = best_per_alpha[a]
                x = np.arange(len(info["mean"]))
                xs, ms, ss = subsample_series(x, info["mean"], info["std"], subsample_every)

                label, color, marker = get_style_for_entry(
                    "fpg", a, fpg_alpha_tol, STYLE_FPG, STYLE_OTHER
                )

                markevery = max(1, len(xs) // 10)

                ax.plot(
                    xs,
                    ms,
                    color=color,
                    marker=marker,
                    markersize=5,
                    markevery=markevery,
                    linewidth=1.4,
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

        else:
            best: Optional[Dict[str, Any]] = None

            for folder in subfolders:
                cfg = parse_folder(algo_name, folder)
                if cfg is None:
                    continue

                stats = load_config_stats(os.path.join(algo_root, folder), env_size, seeds, max_iter)
                if stats is None:
                    continue

                final_val = float(stats["mean"][-1])
                if best is None or final_val > best["final"]:
                    best = {
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "final": final_val,
                        **cfg,
                    }

            if best is None:
                continue

            x = np.arange(len(best["mean"]))
            xs, ms, ss = subsample_series(x, best["mean"], best["std"], subsample_every)

            label, color, marker = get_style_for_entry(
                algo_name, None, fpg_alpha_tol, STYLE_FPG, STYLE_OTHER
            )

            markevery = max(1, len(xs) // 10)

            ax.plot(
                xs,
                ms,
                color=color,
                marker=marker,
                markersize=5,
                markevery=markevery,
                linewidth=1.4,
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

    ax.grid(linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=fontsize - 3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    return legend_handles


def save_figure(fig: plt.Figure, path: str):
    create_folder_if_not_exists(os.path.dirname(path))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_single_plot(
    read_root: str,
    env_name: str,
    env_size: int,
    seeds: List[int],
    max_iter: Optional[int],
    algorithms: List[str],
    fontsize: int,
    font: Dict[str, Any],
    STYLE_FPG: Dict[float, Dict[str, Any]],
    STYLE_OTHER: Dict[str, Dict[str, Any]],
    fpg_alphas_to_plot: Optional[List[float]],
    fpg_alpha_tol: float,
    subsample_every: int,
) -> Tuple[plt.Figure, List[Line2D]]:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    legend_handles = plot_best_algos_on_ax(
        ax=ax,
        read_root=read_root,
        env_name=env_name,
        env_size=env_size,
        seeds=seeds,
        max_iter=max_iter,
        algorithms=algorithms,
        fontsize=fontsize,
        font=font,
        STYLE_FPG=STYLE_FPG,
        STYLE_OTHER=STYLE_OTHER,
        fpg_alphas_to_plot=fpg_alphas_to_plot,
        fpg_alpha_tol=fpg_alpha_tol,
        subsample_every=subsample_every,
    )
    ax.set_ylabel("Average return", fontsize=fontsize, **font)
    ax.set_xlabel("Iteration $t$", fontsize=fontsize, **font)
    fig.tight_layout()
    return fig, legend_handles


def make_legend_figure(
    legend_handles: List[Line2D],
    fontsize: int,
    ncol: int = 1,
) -> plt.Figure:
    fig = plt.figure(figsize=(4, 3))
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


if __name__ == "__main__":
    read_root = "./experiments"
    out_dir = "./plots"

    seeds = list(range(16)) 
    max_iter = None

    algorithms = ["fpg", "logbarrier",'escort','hadamard']

    nchain_sizes = [10, 15, 20]
    deepsea_sizes = [10, 15]

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
    STYLE_OTHER = {
        "logbarrier": {"color": "#E69F00", "marker": "X"},
        "escort": {"color": "#8044E0", "marker": "h"},
        "hadamard": {"color": "#F0E442", "marker": "^"},
    }

    fpg_alphas_to_plot = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    fpg_alpha_tol = 1e-8

    subsample_every = 300


    legend_handles_final: Optional[List[Line2D]] = None

    for sz in nchain_sizes:
        fig, legend_handles = make_single_plot(
            read_root=read_root,
            env_name="nchain",
            env_size=sz,
            seeds=seeds,
            max_iter=max_iter,
            algorithms=algorithms,
            fontsize=fontsize,
            font=font,
            STYLE_FPG=STYLE_FPG,
            STYLE_OTHER=STYLE_OTHER,
            fpg_alphas_to_plot=fpg_alphas_to_plot,
            fpg_alpha_tol=fpg_alpha_tol,
            subsample_every=subsample_every,
        )
        legend_handles_final = legend_handles
        save_figure(fig, os.path.join(out_dir, f"nchain_size_{sz}.pdf"))

    for sz in deepsea_sizes:
        fig, legend_handles = make_single_plot(
            read_root=read_root,
            env_name="deepsea",
            env_size=sz,
            seeds=seeds,
            max_iter=max_iter,
            algorithms=algorithms,
            fontsize=fontsize,
            font=font,
            STYLE_FPG=STYLE_FPG,
            STYLE_OTHER=STYLE_OTHER,
            fpg_alphas_to_plot=fpg_alphas_to_plot,
            fpg_alpha_tol=fpg_alpha_tol,
            subsample_every=subsample_every,
        )
        legend_handles_final = legend_handles
        save_figure(fig, os.path.join(out_dir, f"deepsea_size_{sz}.pdf"))


    if legend_handles_final is None:
        raise RuntimeError("No legend handles were generated (no plots were created).")

    fig_leg = make_legend_figure(
        legend_handles=legend_handles_final,
        fontsize=fontsize,
        ncol=1,  
    )
    save_figure(fig_leg, os.path.join(out_dir, "legend.pdf"))
