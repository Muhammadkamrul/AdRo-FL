#!/usr/bin/env python3

from __future__ import annotations

"""
#Last Run Command:
python adrofl_revision_runner_styled_figures_v2.py \
    --repo-root /home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT \
    --output-root revision_outputs_styled \
    --existing-output-root /home/kamrul/Documents/kamrul_files_Linux/OORT/adrofl_revision_runner_package_for_cifar10/revision_outputs_live_cifar10only
"""

"""
Styled AdRo-FL figure generator
===============================

Purpose
-------
Generate the same figure set produced by the revision runner, but with a cleaner,
consistent visual style inspired by the user's plotting notebooks:
- large readable fonts
- visible markers
- light dashed grids
- consistent AdRo-FL / Random / Oort color mapping
- higher-resolution exports

This script reads the same legacy result files used by the uploaded
`adrofl_revision_runner.py`, and it can also restyle the analytical and live
ablation figures when the corresponding CSV files already exist.

Main outputs
------------
Baseline figures (always attempted):
- figures/cluster_local_accuracy.png
- figures/cluster_local_loss.png
- figures/cluster_global_accuracy.png
- figures/cluster_global_loss.png
- figures/noncluster_vrf_accuracy.png
- figures/noncluster_vrf_loss.png
- figures/privacy_violation_first_100_rounds.png
- figures/svhn_avg_bits.png
- figures/svhn_avg_energy.png
- figures/svhn_bits_energy_tradeoff.png

Analytical figures (generated from simulation by default):
- reviewer_analysis/adaptive_targeting_rates.png
- reviewer_analysis/adaptive_targeting_frequency_profile.png

Live ablation figures (only if a round-log CSV exists):
- live_ablations/cluster_security_efficiency_ablation.png
- live_ablations/vrf_utility_loss_over_rounds.png

Example
-------
python adrofl_revision_runner_styled_figures.py \
    --repo-root /path/to/adroFL \
    --output-root revision_outputs_styled

If you already have live ablation CSVs inside an earlier revision output folder,
point `--existing-output-root` to that folder so the live-ablation figures are
restyled from the existing logs.
"""



import argparse
import ast
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants copied/adapted from the uploaded runner
# ---------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

DATASETS = ("mnist", "fmnist", "cifar10", "svhn")

DISPLAY_NAMES = {
    "custom": "AdRo-FL",
    "random": "Random",
    "oort": "Oort",
}

# Publication-friendly palette, still close to the notebook feel:
# blue = AdRo-FL, green = Random, crimson = Oort.
STYLE = {
    "custom": {"label": "AdRo-FL", "color": "#1f77b4", "linestyle": "-",  "marker": "o"},
    "random": {"label": "Random",  "color": "#2ca02c", "linestyle": "--", "marker": "^"},
    "oort":   {"label": "Oort",    "color": "#d62728", "linestyle": ":",  "marker": "s"},
}

PRIVACY_BAR_COLOR = "#8ecae6"
PRIVACY_LINE_COLOR = "#d62728"

PAPER_FILE_MAPS: Dict[str, Dict[str, Dict[str, str]]] = {
    "cluster_local": {
        "mnist": {
            "custom": "output_mnist_custom_local_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "random": "output_mnist_random_local_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "oort": "output_mnist_oort_local_dirichlet.txt",
        },
        "fmnist": {
            "custom": "output_fmnist_custom_local_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "random": "output_fmnist_random_local_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "oort": "output_fmnist_oort_local_dirichlet.txt",
        },
        "cifar10": {
            "custom": "output_cifar10_custom_local_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "random": "output_cifar10_random_local_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "oort": "output_cifar10_oort_local_dirichlet.txt",
        },
        "svhn": {
            "custom": "output_svhn_custom_local_dirichlet_Q8_lr0.01_ep1_w0.5_NEW_NEW.txt",
            "random": "output_svhn_random_local_dirichlet_Q8_lr0.01_ep1.txt",
            "oort": "output_svhn_oort_local_dirichlet_NEW.txt",
        },
    },
    "cluster_global": {
        "mnist": {
            "custom": "output_mnist_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "random": "output_mnist_random_global_dirichlet_Q8_lr0.01_ep1_w0.4_NEW.txt",
            "oort": "output_mnist_oort_global_dirichlet_NEW.txt",
        },
        "fmnist": {
            "custom": "output_fmnist_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "random": "output_fmnist_random_global_dirichlet_Q8_lr0.01_ep1_w0.4_NEW.txt",
            "oort": "output_fmnist_oort_global_dirichlet_NEW.txt",
        },
        "cifar10": {
            "custom": "output_cifar10_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "random": "output_cifar10_random_global_dirichlet_Q8_lr0.01_ep1_w0.4_NEW.txt",
            "oort": "output_cifar10_oort_global_dirichlet_NEW.txt",
        },
        "svhn": {
            "custom": "output_svhn_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4.txt",
            "random": "output_svhn_random_global_dirichlet_lr0.01_ep1.txt",
            "oort": "output_svhn_oort_global_dirichlet_NEW.txt",
        },
    },
    "noncluster_vrf": {
        "mnist": {
            "custom": "output_mnist_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4_vrf_UtilitySigned.txt",
            "random": "output_mnist_random_global_dirichlet_Q8_lr0.01_ep1_w0.4_NEW.txt",
            "oort": "output_mnist_oort_global_dirichlet_NEW_NonCluster.txt",
        },
        "fmnist": {
            "custom": "output_fmnist_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4_vrf_UtilitySigned.txt",
            "random": "output_fmnist_random_global_dirichlet_Q8_lr0.01_ep1_w0.4_NEW.txt",
            "oort": "output_fmnist_oort_global_dirichlet_NEW_NonCluster.txt",
        },
        "cifar10": {
            "custom": "output_cifar10_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4_vrf_UtilitySigned.txt",
            "random": "output_cifar10_random_global_dirichlet_Q8_lr0.01_ep1_w0.4_NEW.txt",
            "oort": "output_cifar10_oort_global_dirichlet_NEW_NonCluster.txt",
        },
        "svhn": {
            "custom": "output_svhn_custom_global_dirichlet_Q8_lr0.01_ep1_w0.4_vrf_UtilitySigned.txt",
            "random": "output_svhn_random_global_dirichlet_lr0.01_ep1.txt",
            "oort": "output_svhn_oort_global_dirichlet_NEW_NonCluster.txt",
        },
    },
}

PRIVACY_VIOLATION_FILE = "output_svhn_oort_global_violation.txt"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_")


def metric_key(method_key: str, metric: str) -> str:
    return f"{metric}_{method_key}"


def safe_mean(values: Sequence[Any]) -> Optional[float]:
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            if np.isnan(value):
                continue
        except Exception:
            pass
        clean.append(float(value))
    if not clean:
        return None
    return float(np.mean(clean))


def parse_legacy_result_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    data: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value
    return data


def estimate_energy_from_bits(bits: Sequence[Any], bandwidth: float = 1e6, power: float = 0.1, seed: int = SEED) -> List[float]:
    rng = np.random.default_rng(seed)
    energies: List[float] = []
    for value in bits:
        if value is None:
            energies.append(float("nan"))
            continue
        snr_db = rng.uniform(0, 30)
        snr = 10 ** (snr_db / 10)
        capacity = bandwidth * np.log2(1 + snr)
        transmission_time = float(value) / capacity
        energies.append(power * transmission_time)
    return energies


def detect_repo_root(user_supplied: Optional[str]) -> Path:
    if user_supplied:
        root = Path(user_supplied).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Repo root does not exist: {root}")
        return root

    candidates = [
        Path.cwd(),
        Path.cwd() / "adroFL",
        Path("/mnt/data/adroFL_work/adroFL"),
        Path("/mnt/data/adroFL_repo/adroFL"),
        Path("/mnt/data/adroFL_unzipped/adroFL"),
    ]
    for candidate in candidates:
        if (candidate / "results").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not auto-detect repo root. Please pass --repo-root.")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def apply_global_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "lines.linewidth": 2.6,
        "lines.markersize": 6.5,
    })


def stylize_axis(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None, title: Optional[str] = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_alpha(0.65)


def _markevery(length: int) -> int:
    if length <= 20:
        return 2
    return max(1, length // 10)


# ---------------------------------------------------------------------------
# Baseline plots
# ---------------------------------------------------------------------------


def load_series(results_dir: Path, file_map: Mapping[str, Mapping[str, str]], metric: str) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for dataset, methods in file_map.items():
        out[dataset] = {}
        for method_key, file_name in methods.items():
            blob = parse_legacy_result_file(results_dir / file_name)
            values = list(blob.get(metric_key(method_key, metric), []))
            clean = []
            for value in values:
                try:
                    clean.append(float(value))
                except Exception:
                    pass
            out[dataset][method_key] = clean
    return out


def save_metric_grid(
    series: Mapping[str, Mapping[str, Sequence[float]]],
    output_path: Path,
    ylabel: str,
    super_title: str,
    legend_loc: str = "upper center",
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=False)
    axes = axes.flatten()

    for ax, dataset in zip(axes, DATASETS):
        for method_key in ("custom", "random", "oort"):
            values = [float(v) for v in series[dataset][method_key] if v is not None]
            if not values:
                continue
            rounds = np.arange(1, len(values) + 1)
            st = STYLE[method_key]
            ax.plot(
                rounds,
                values,
                label=st["label"],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markevery=_markevery(len(values)),
                markerfacecolor="white",
                markeredgewidth=1.4,
            )

        stylize_axis(ax, xlabel="Rounds", ylabel=ylabel, title=dataset.upper())
        if ylabel.lower().startswith("accuracy"):
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(max(0, ymin), min(100, ymax + 2))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_loc, ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(super_title, y=0.995, fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_and_loss(results_dir: Path, file_map: Mapping[str, Mapping[str, str]], out_dir: Path, setting_name: str) -> None:
    accuracy_series = load_series(results_dir, file_map, "accuracy")
    loss_series = load_series(results_dir, file_map, "loss")
    save_metric_grid(
        accuracy_series,
        out_dir / f"{slugify(setting_name)}_accuracy.png",
        ylabel="Accuracy (%)",
        super_title=f"{setting_name.replace('_', ' ').title()}: Accuracy",
    )
    save_metric_grid(
        loss_series,
        out_dir / f"{slugify(setting_name)}_loss.png",
        ylabel="Loss",
        super_title=f"{setting_name.replace('_', ' ').title()}: Loss",
    )


def plot_privacy_violation(results_dir: Path, out_dir: Path, max_rounds: int = 100) -> Path:
    blob = parse_legacy_result_file(results_dir / PRIVACY_VIOLATION_FILE)
    violations = list(blob.get("violation_count", []))[:max_rounds]
    rounds = np.arange(1, len(violations) + 1)
    ensure_dir(out_dir)
    out_path = out_dir / "privacy_violation_first_100_rounds.png"

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(rounds, violations, width=0.85, color=PRIVACY_BAR_COLOR, edgecolor="black", linewidth=0.7, label="Oort")
    ax.axhline(0, color=PRIVACY_LINE_COLOR, linewidth=3.0, linestyle="--", label="AdRo-FL")
    stylize_axis(ax, xlabel="Rounds", ylabel="Privacy Violation Count", title="Cluster Privacy Violations in the First 100 Rounds")
    if violations:
        ax.set_yticks(np.arange(0, max(violations) + 1, 1))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_svhn_bits_energy_table(results_dir: Path) -> pd.DataFrame:
    custom_blob = parse_legacy_result_file(results_dir / PAPER_FILE_MAPS["cluster_global"]["svhn"]["custom"])
    random_blob = parse_legacy_result_file(results_dir / PAPER_FILE_MAPS["cluster_global"]["svhn"]["random"])
    oort_blob = parse_legacy_result_file(results_dir / PRIVACY_VIOLATION_FILE)

    rows = [
        {
            "method_key": "custom",
            "method": "AdRo-FL",
            "avg_bits": safe_mean(custom_blob.get("bits_custom", [])),
            "avg_energy": safe_mean(custom_blob.get("energy_custom", [])),
        },
        {
            "method_key": "random",
            "method": "Random",
            "avg_bits": safe_mean(random_blob.get("bits_random", [])),
            "avg_energy": safe_mean(random_blob.get("energy_random", [])),
        },
        {
            "method_key": "oort",
            "method": "Oort",
            "avg_bits": safe_mean(oort_blob.get("grad_size_bits", [])),
            "avg_energy": safe_mean(estimate_energy_from_bits(oort_blob.get("grad_size_bits", []))),
        },
    ]
    return pd.DataFrame(rows)


def plot_svhn_metric_bars(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ordered = df.copy().reset_index(drop=True)
    y_pos = np.arange(len(ordered))
    colors = [STYLE[key]["color"] for key in ordered["method_key"]]
    bars = ax.barh(y_pos, ordered[metric], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["method"])
    stylize_axis(ax, xlabel=ylabel, ylabel=None, title=f"SVHN {ylabel}")

    for idx, bar in enumerate(bars):
        value = ordered.loc[idx, metric]
        label = f"{value:.4f}" if metric == "avg_energy" else f"{value:.1f}"
        ax.text(bar.get_width() + max(ordered[metric]) * 0.015, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_svhn_tradeoff(df: pd.DataFrame, out_path: Path) -> Path:
    annotation_offsets = {
        "AdRo-FL": (16, 12),
        "Random": (16, -28),
        "Oort": (-96, 16),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        st = STYLE[row["method_key"]]
        ax.scatter(row["avg_bits"], row["avg_energy"], s=320, color=st["color"], edgecolor="black", linewidth=1.2, zorder=3)
        ax.axhline(row["avg_energy"], color=st["color"], linestyle="--", linewidth=1.0, alpha=0.35)
        ax.axvline(row["avg_bits"], color=st["color"], linestyle="--", linewidth=1.0, alpha=0.35)
        dx, dy = annotation_offsets[row["method"]]
        ax.annotate(
            f"{row['method']}\n({row['avg_bits']:.1f}, {row['avg_energy']:.4f})",
            xy=(row["avg_bits"], row["avg_energy"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": st["color"], "lw": 1.1},
            arrowprops={"arrowstyle": "->", "color": st["color"], "lw": 1.2, "alpha": 0.75},
        )

    stylize_axis(ax, xlabel="Average Bits Transmitted", ylabel="Average Energy (J)", title="SVHN Communication Efficiency Trade-off")
    x_values = df["avg_bits"].to_numpy()
    y_values = df["avg_energy"].to_numpy()
    x_margin = max(1.0, (x_values.max() - x_values.min()) * 0.28)
    y_margin = max(1e-5, (y_values.max() - y_values.min()) * 0.35)
    ax.set_xlim(x_values.min() - x_margin, x_values.max() + x_margin)
    ax.set_ylim(y_values.min() - y_margin, y_values.max() + y_margin)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_svhn_bits_energy(results_dir: Path, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    df = build_svhn_bits_energy_table(results_dir)
    df.to_csv(out_dir / "svhn_bits_energy_summary.csv", index=False)

    paths = [
        plot_svhn_metric_bars(df, "avg_bits", "Average Bits", out_dir / "svhn_avg_bits.png"),
        plot_svhn_metric_bars(df, "avg_energy", "Average Energy (J)", out_dir / "svhn_avg_energy.png"),
        plot_svhn_tradeoff(df, out_dir / "svhn_bits_energy_tradeoff.png"),
    ]
    return paths


def write_baseline_figures(results_dir: Path, output_root: Path) -> List[Path]:
    out_paths: List[Path] = []
    figures_dir = ensure_dir(output_root / "figures")
    for setting, file_map in PAPER_FILE_MAPS.items():
        plot_accuracy_and_loss(results_dir, file_map, figures_dir, setting)
        out_paths.append(figures_dir / f"{slugify(setting)}_accuracy.png")
        out_paths.append(figures_dir / f"{slugify(setting)}_loss.png")
    out_paths.append(plot_privacy_violation(results_dir, figures_dir))
    out_paths.extend(plot_svhn_bits_energy(results_dir, figures_dir))
    return out_paths


# ---------------------------------------------------------------------------
# Analytical figures
# ---------------------------------------------------------------------------


def find_k_minimum(num_clients: int, honest_fraction: float, dishonest_fraction: float, p_attack: float) -> Optional[int]:
    dishonest_count = max(1, int(round(dishonest_fraction * num_clients)))
    for k in range(1, dishonest_count + 2):
        numerator = honest_fraction * num_clients * math.comb(dishonest_count, k - 1)
        denominator = math.comb(num_clients, k)
        if denominator == 0:
            continue
        if numerator / denominator <= p_attack:
            return k
    return None


def simulate_adaptive_targeting(
    eligible_pool_size: int = 80,
    selected_capacity: int = 20,
    dishonest_fraction: float = 0.10,
    rounds: int = 3000,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_dishonest = max(1, int(round(eligible_pool_size * dishonest_fraction)))
    dishonest_ids = list(range(n_dishonest))
    honest_ids = list(range(n_dishonest, eligible_pool_size))
    target_id = honest_ids[0]

    insecure_counts = Counter()
    vrf_counts = Counter()

    for _ in range(rounds):
        fill_needed = max(0, selected_capacity - (1 + len(dishonest_ids)))
        other_honest = honest_ids[1:]
        fill_needed = min(fill_needed, len(other_honest))
        insecure_selected = [target_id] + dishonest_ids + rng.choice(other_honest, size=fill_needed, replace=False).tolist()
        insecure_counts.update(insecure_selected)

        vrf_selected = rng.choice(np.arange(eligible_pool_size), size=min(selected_capacity, eligible_pool_size), replace=False)
        vrf_counts.update(vrf_selected.tolist())

    freq_rows: List[Dict[str, Any]] = []
    for cid in range(eligible_pool_size):
        role = "dishonest" if cid in dishonest_ids else ("target_honest" if cid == target_id else "honest")
        freq_rows.append({
            "client_id": cid,
            "role": role,
            "insecure_selection_rate": insecure_counts[cid] / rounds,
            "vrf_selection_rate": vrf_counts[cid] / rounds,
        })
    freq_df = pd.DataFrame(freq_rows)

    def concentration_stats(column: str) -> Dict[str, float]:
        values = freq_df[column].to_numpy()
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = float(((2 * index - n - 1) * sorted_vals).sum() / (n * sorted_vals.sum() + 1e-12))
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
            "min": float(values.min()),
            "gini": gini,
            "target_rate": float(freq_df.loc[freq_df["role"] == "target_honest", column].iloc[0]),
            "dishonest_mean_rate": float(freq_df.loc[freq_df["role"] == "dishonest", column].mean()),
            "honest_mean_rate": float(freq_df.loc[freq_df["role"] == "honest", column].mean()),
        }

    summary_df = pd.DataFrame([
        {"setting": "insecure_global", **concentration_stats("insecure_selection_rate")},
        {"setting": "verified_VRF_final", **concentration_stats("vrf_selection_rate")},
    ])
    summary_df.insert(1, "eligible_pool_size", eligible_pool_size)
    summary_df.insert(2, "selected_capacity", selected_capacity)
    summary_df.insert(3, "dishonest_fraction", dishonest_fraction)
    summary_df.insert(4, "rounds", rounds)
    return summary_df, freq_df


def plot_adaptive_targeting(summary_df: pd.DataFrame, freq_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    paths: List[Path] = []

    # one figure, two panels
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    # -------------------------
    # Left panel: grouped bars
    # -------------------------
    ax = axes[0]
    plot_df = summary_df[["setting", "target_rate", "dishonest_mean_rate", "honest_mean_rate"]].copy()
    x = np.arange(len(plot_df))
    width = 0.24

    ax.bar(x - width, plot_df["target_rate"], width=width, color="#1f77b4", label="Target honest")
    ax.bar(x,         plot_df["dishonest_mean_rate"], width=width, color="#d62728", label="Dishonest mean")
    ax.bar(x + width, plot_df["honest_mean_rate"], width=width, color="#2ca02c", label="Other honest mean")

    ax.set_xticks(x)

    label_map = {
        "insecure_global": "Insecure Global",
        "verified_VRF_final": "Verified VRF Final",
    }
    ax.set_xticklabels([label_map[s] for s in plot_df["setting"]])

    #ax.set_xticklabels([s.replace("_", " ").title() for s in plot_df["setting"]])
    stylize_axis(
        ax,
        xlabel=None,
        ylabel="Per-round Selection Rate",
    )
    ax.legend(frameon=False)

    # -----------------------------------
    # Right panel: selection-rate profile
    # -----------------------------------
    ax = axes[1]
    ordered = freq_df.sort_values(["role", "client_id"]).reset_index(drop=True)
    ax.plot(
        ordered.index,
        ordered["insecure_selection_rate"],
        color="#d62728",
        linewidth=2.2,
        label="Insecure",
    )
    ax.plot(
        ordered.index,
        ordered["vrf_selection_rate"],
        color="#1f77b4",
        linewidth=2.2,
        label="VRF finalization",
    )
    stylize_axis(
        ax,
        xlabel="Eligible Clients",
        ylabel="Per-round Selection Rate",
    )
    ax.legend(frameon=False)

    fig.tight_layout()

    # save as one combined figure
    combined_path = out_dir / "adaptive_targeting_combined.png"
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(combined_path)

    summary_df.to_csv(out_dir / "adaptive_targeting_summary.csv", index=False)
    freq_df.to_csv(out_dir / "adaptive_targeting_client_frequencies.csv", index=False)
    return paths


def build_probability_bound_table(
    pool_sizes: Sequence[int] = (40, 60, 80, 100),
    dishonest_fraction: float = 0.10,
    honest_fraction: float = 0.90,
    p_attack: float = 0.001,
) -> pd.DataFrame:
    rows = []
    for n in pool_sizes:
        rows.append({
            "eligible_pool_size": n,
            "honest_fraction": honest_fraction,
            "dishonest_fraction": dishonest_fraction,
            "p_attack": p_attack,
            "k_minimum": find_k_minimum(n, honest_fraction, dishonest_fraction, p_attack),
        })
    return pd.DataFrame(rows)


def write_analytical_figures(output_root: Path) -> List[Path]:
    analysis_dir = ensure_dir(output_root / "reviewer_analysis")
    prob_df = build_probability_bound_table()
    prob_df.to_csv(analysis_dir / "vrf_probability_bound_table.csv", index=False)
    summary_df, freq_df = simulate_adaptive_targeting()
    return plot_adaptive_targeting(summary_df, freq_df, analysis_dir)


# ---------------------------------------------------------------------------
# Live ablation figures (restyle from existing logs)
# ---------------------------------------------------------------------------


def plot_live_ablation_figures(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    paths: List[Path] = []

    cluster_df = df[df["variant"].astype(str).str.startswith("cluster_global_")].copy()
    if not cluster_df.empty:
        summary = (
            cluster_df.groupby(["dataset", "variant"], dropna=False)
            .agg(final_accuracy=("accuracy", "last"), avg_utility_loss_ratio=("utility_loss_ratio", "mean"))
            .reset_index()
        )

        pretty_names = {
            "cluster_global_utility_only": "Utility only",
            "cluster_global_utility_plus_efficiency": "Utility + efficiency",
            "cluster_global_utility_plus_security": "Utility + security",
            "cluster_global_full_adrofl": "Full AdRo-FL",
        }
        order = list(pretty_names.keys())

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
        dataset_colors = {
            "mnist": "#1f77b4",
            "fmnist": "#ff7f0e",
            "cifar10": "#2ca02c",
            "svhn": "#9467bd",
        }
        for dataset in sorted(summary["dataset"].dropna().unique()):
            subset = summary[summary["dataset"] == dataset].copy()
            subset["variant"] = pd.Categorical(subset["variant"], categories=order, ordered=True)
            subset = subset.sort_values("variant")
            x = np.arange(len(subset))
            axes[0].plot(x, subset["final_accuracy"], marker="o", linewidth=2.4, color=dataset_colors.get(dataset, None), label=dataset.upper())
            axes[1].plot(x, subset["avg_utility_loss_ratio"], marker="o", linewidth=2.4, color=dataset_colors.get(dataset, None), label=dataset.upper())

        for ax, ylabel, title in zip(
            axes,
            ["Final Accuracy (%)", "Average Utility-loss Ratio"],
            ["Accuracy Under Security/Efficiency Ablations", "Utility Gap Under Security/Efficiency Ablations"],
        ):
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([pretty_names[v] for v in order], rotation=24, ha="right")
            stylize_axis(ax, ylabel=ylabel)
            ax.legend(frameon=False)

        fig.tight_layout()
        path = out_dir / "cluster_security_efficiency_ablation.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    vrf_df = df[df["variant"].astype(str).str.contains("noncluster_verified_pool_only", na=False)].copy()
    if not vrf_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        dataset_colors = {
            "mnist": "#1f77b4",
            "fmnist": "#ff7f0e",
            "cifar10": "#2ca02c",
            "svhn": "#9467bd",
        }
        for dataset in sorted(vrf_df["dataset"].dropna().unique()):
            subset = vrf_df[vrf_df["dataset"] == dataset].copy()
            ax.plot(subset["round"], subset["utility_loss_ratio"], linewidth=1.9, color=dataset_colors.get(dataset, None), label=dataset.upper())
        stylize_axis(ax, xlabel="FL Round", ylabel="VRF Utility-loss Ratio", title="Per-round Utility Gap Introduced by Verifiable Random Finalization")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / "vrf_utility_loss_over_rounds.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def _infer_dataset_variant_from_path(csv_path: Path) -> Tuple[Optional[str], Optional[str]]:
    parts = csv_path.parts
    if "live_ablations" not in parts:
        return None, None
    idx = parts.index("live_ablations")
    if len(parts) >= idx + 4:
        dataset = parts[idx + 1]
        variant = parts[idx + 2]
        return dataset, variant
    return None, None


def _load_live_ablation_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    dataset, variant = _infer_dataset_variant_from_path(csv_path)
    if "dataset" not in df.columns and dataset is not None:
        df["dataset"] = dataset
    if "variant" not in df.columns and variant is not None:
        df["variant"] = variant
    return df


def collect_live_ablation_frames(existing_output_roots: Sequence[Path]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    seen_csvs = set()

    for root in existing_output_roots:
        live_root = root / "live_ablations"
        if not live_root.exists():
            continue

        combined_csv = live_root / "all_live_ablation_round_logs.csv"
        if combined_csv.exists():
            resolved = combined_csv.resolve()
            if resolved not in seen_csvs:
                frames.append(_load_live_ablation_csv(combined_csv))
                seen_csvs.add(resolved)

        for csv_path in sorted(live_root.glob("*/*/*_round_log.csv")):
            resolved = csv_path.resolve()
            if resolved in seen_csvs:
                continue
            frames.append(_load_live_ablation_csv(csv_path))
            seen_csvs.add(resolved)

    return frames


def try_write_live_ablation_figures(existing_output_roots: Sequence[Path], output_root: Path) -> List[Path]:
    paths: List[Path] = []
    if not existing_output_roots:
        return paths

    frames = collect_live_ablation_frames(existing_output_roots)
    if not frames:
        return paths

    df = pd.concat(frames, ignore_index=True)

    dedupe_cols = [c for c in ["dataset", "variant", "round"] if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols, keep="last").copy()

    combined_csv = ensure_dir(output_root / "live_ablations") / "all_live_ablation_round_logs_combined.csv"
    df.to_csv(combined_csv, index=False)
    paths.append(combined_csv)

    paths.extend(plot_live_ablation_figures(df, ensure_dir(output_root / "live_ablations")))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Styled AdRo-FL figure generator")
    parser.add_argument("--repo-root", type=str, default=None, help="Path to the AdRo-FL repo root")
    parser.add_argument("--results-dir", type=str, default=None, help="Optional override for the legacy results directory")
    parser.add_argument("--output-root", type=str, default="revision_outputs_styled", help="Where styled figures are written")
    parser.add_argument(
        "--existing-output-root",
        type=str,
        action="append",
        default=[],
        help="Existing revision output folder. Repeat this flag to combine multiple dataset/output folders for live-ablation figures.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the adaptive-targeting analytical figures",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    apply_global_style()

    repo_root = detect_repo_root(args.repo_root)
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else (repo_root / "results" / "svhn2")
    if not results_dir.exists():
        raise FileNotFoundError(f"Expected results directory not found: {results_dir}")

    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())
    generated: List[Path] = []

    generated.extend(write_baseline_figures(results_dir, output_root))

    if not args.skip_analysis:
        generated.extend(write_analytical_figures(output_root))

    existing_output_roots = [Path(item).expanduser().resolve() for item in args.existing_output_root]
    generated.extend(try_write_live_ablation_figures(existing_output_roots, output_root))

    manifest = pd.DataFrame({"generated_path": [str(p) for p in generated]})
    manifest_path = output_root / "generated_files_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"[OK] Generated {len(generated)} styled figure files")
    print(f"[OK] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
