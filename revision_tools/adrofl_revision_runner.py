#!/usr/bin/env python3

"""
Last run command:
python adrofl_revision_runner.py 
--repo-root /home/kamrul/Documents/kamrul_files_Linux/OORT/ClusterFed_OORT 
--output-root revision_outputs_live_fmnistOnly 
--mode all 
--run-live-ablations 
--datasets fmnist 
--rounds 3000

"""


"""
AdRo-FL reviewer-revision runner
================================

Purpose
-------
This script is an all-in-one utility for your minor revision. It does four jobs:

1. Rebuild tidy CSV summaries and cleaner figures from the legacy `.txt` result files
   already stored in `results/svhn2/`.
2. Generate publication-ready tables for the three settings used in the paper:
   - cluster-local
   - cluster-global
   - distributed / non-cluster VRF
3. Run new reviewer-facing analyses:
   - security-vs-efficiency ablation summaries
   - utility-loss analysis for the C-threshold and VRF selection stage
   - adaptive targeting / concentration simulations for the distributed setting
4. Optionally rerun new live ablations on top of the original codebase by importing the
   checked-in `food101/` scripts, overriding the hard-coded globals, and logging round-wise
   CSV files.

Important note
--------------
This script does *not* implement a concrete secure aggregation protocol. It is intended for
re-running the same SA-compatible client-selection pipeline that your manuscript studies.
The generated metadata files make that assumption explicit so the revision stays honest.

Typical usage
-------------
Baseline summaries and figures only (fast):

    python adrofl_revision_runner.py \
        --repo-root /path/to/adroFL \
        --output-root revision_outputs \
        --mode summarize

Everything that does not require live retraining:

    python adrofl_revision_runner.py \
        --repo-root /path/to/adroFL \
        --output-root revision_outputs \
        --mode all

Run reviewer ablations live (heavy):

    python adrofl_revision_runner.py \
        --repo-root /path/to/adroFL \
        --output-root revision_outputs \
        --mode all \
        --run-live-ablations \
        --datasets mnist,fmnist,cifar10,svhn \
        --rounds 3000

Notes on live ablations
-----------------------
- They require the same Python / PyTorch / torchvision / pynacl / scipy / gurobipy stack that
  your original workstation used.
- The script lazily imports those modules only when live ablations are requested.
- The live ablation code is written to preserve the paper-level defaults:
  100 clients, 20 selected per round, 3000 rounds, batch size 64, Dirichlet alpha 0.1,
  0.01 LR for AdRo-FL/Random, 0.1 LR for Oort, 8-bit quantization, dataset-specific deadlines,
  C=2 in cluster mode, and 10% colluding clients with P_attack <= 0.001 in the distributed mode.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
import textwrap
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend for batch figure generation.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paper-compatible defaults and legacy file mappings
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASETS = ("mnist", "fmnist", "cifar10", "svhn")

DEADLINES = {
    "mnist": 0.50,
    "fmnist": 0.11,
    "cifar10": 0.14,
    "svhn": 0.13,
}

# Weight used in the paper. Some checked-in scripts drifted on SVHN, so the live
# runner forces 0.4 unless you explicitly override it.
UTILITY_WEIGHT = {
    "mnist": 0.4,
    "fmnist": 0.4,
    "cifar10": 0.4,
    "svhn": 0.4,
}

THRESHOLDS_BY_DATASET = {
    "mnist": [60, 65, 70, 75, 80, 85, 90],
    "fmnist": [60, 65, 70, 75, 80, 85],
    "cifar10": [35, 40, 45, 50, 55],
    "svhn": [60, 65, 70, 75, 80, 85],
}

DISPLAY_NAMES = {
    "custom": "AdRo-FL",
    "random": "Random",
    "oort": "Oort",
}

METHOD_COLORS = {
    "custom": "#9467bd",   # close to the notebook palette
    "random": "#c5b0d5",
    "oort": "#ff7f0e",
}

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
# Small data structures
# ---------------------------------------------------------------------------

@dataclass
class SummaryRow:
    setting: str
    dataset: str
    method_key: str
    method: str
    final_accuracy: Optional[float]
    best_accuracy: Optional[float]
    best_loss: Optional[float]
    avg_bits: Optional[float]
    avg_energy: Optional[float]
    avg_round_duration: Optional[float]
    rounds_recorded: int
    thresholds: Dict[str, Optional[int]]


@dataclass
class RunNote:
    note: str
    secure_aggregation_protocol_implemented: bool = False
    selection_layer_is_sa_compatible: bool = True


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def safe_mean(values: Sequence[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def safe_max(values: Sequence[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.max(clean))


def safe_min(values: Sequence[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.min(clean))


def first_round_reaching_threshold(values: Sequence[Any], threshold: float) -> Optional[int]:
    for i, value in enumerate(values, start=1):
        if value is None:
            continue
        try:
            if float(value) >= threshold:
                return i
        except Exception:
            continue
    return None


def parse_namespace_line(text: str) -> Dict[str, Any]:
    """Parse a line like `Run_configurations = Namespace(a=1, b='x')`.

    The result is intentionally shallow; it is only used in metadata files.
    """
    text = text.strip()
    if "Namespace(" not in text:
        return {"raw": text}
    start = text.find("Namespace(") + len("Namespace(")
    end = text.rfind(")")
    body = text[start:end]
    result: Dict[str, Any] = {}
    if not body:
        return result
    parts = []
    current = []
    depth = 0
    in_str = False
    quote = ""
    for ch in body:
        if in_str:
            current.append(ch)
            if ch == quote:
                in_str = False
            continue
        if ch in {'"', "'"}:
            in_str = True
            quote = ch
            current.append(ch)
            continue
        if ch in "([{" :
            depth += 1
        elif ch in ")]}" :
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            result[key] = ast.literal_eval(value)
        except Exception:
            result[key] = value
    return result


def parse_legacy_result_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy result file: {path}")
    data: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "Run_configurations":
                data[key] = parse_namespace_line(raw_line)
                continue
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value
    return data


def metric_key(method_key: str, metric: str) -> str:
    return f"{metric}_{method_key}"


def load_setting_metrics(results_dir: Path, file_map: Mapping[str, Mapping[str, str]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset, methods in file_map.items():
        out[dataset] = {}
        for method_key, file_name in methods.items():
            out[dataset][method_key] = parse_legacy_result_file(results_dir / file_name)
    return out


def build_summary_rows(setting: str, results_dir: Path, file_map: Mapping[str, Mapping[str, str]]) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    metrics = load_setting_metrics(results_dir, file_map)
    for dataset, method_map in metrics.items():
        thresholds = THRESHOLDS_BY_DATASET[dataset]
        for method_key, blob in method_map.items():
            acc = blob.get(metric_key(method_key, "accuracy"), [])
            loss = blob.get(metric_key(method_key, "loss"), [])
            bits = blob.get(metric_key(method_key, "bits"), [])
            energy = blob.get(metric_key(method_key, "energy"), [])
            if not bits and method_key == "oort" and dataset == "svhn" and setting in {"cluster_global", "noncluster_vrf"}:
                # For Oort energy/bits on SVHN the notebook reconstructs from the logging file.
                violation_blob = parse_legacy_result_file(results_dir / PRIVACY_VIOLATION_FILE)
                bits = violation_blob.get("grad_size_bits", [])
                energy = estimate_energy_from_bits(bits, seed=SEED)
            round_duration = blob.get("round_duration", []) or blob.get("total_round_duration", [])
            rows.append(
                SummaryRow(
                    setting=setting,
                    dataset=dataset,
                    method_key=method_key,
                    method=DISPLAY_NAMES[method_key],
                    final_accuracy=float(acc[-1]) if acc else None,
                    best_accuracy=safe_max(acc),
                    best_loss=safe_min(loss),
                    avg_bits=safe_mean(bits),
                    avg_energy=safe_mean(energy),
                    avg_round_duration=safe_mean(round_duration),
                    rounds_recorded=len(acc),
                    thresholds={f"@{thr}": first_round_reaching_threshold(acc, thr) for thr in thresholds},
                )
            )
    return rows


def summary_rows_to_dataframe(rows: Sequence[SummaryRow]) -> pd.DataFrame:
    flat: List[Dict[str, Any]] = []
    for row in rows:
        item = {
            "setting": row.setting,
            "dataset": row.dataset,
            "method_key": row.method_key,
            "method": row.method,
            "final_accuracy": row.final_accuracy,
            "best_accuracy": row.best_accuracy,
            "best_loss": row.best_loss,
            "avg_bits": row.avg_bits,
            "avg_energy": row.avg_energy,
            "avg_round_duration": row.avg_round_duration,
            "rounds_recorded": row.rounds_recorded,
        }
        item.update(row.thresholds)
        flat.append(item)
    return pd.DataFrame(flat)


def estimate_energy_from_bits(bits: Sequence[Any], bandwidth: float = 1e6, power: float = 0.1, seed: int = 42) -> List[float]:
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


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Baseline paper-style figure generation
# ---------------------------------------------------------------------------

def _load_series_for_plot(results_dir: Path, file_map: Mapping[str, Mapping[str, str]], metric: str) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for dataset, methods in file_map.items():
        out[dataset] = {}
        for method_key, file_name in methods.items():
            blob = parse_legacy_result_file(results_dir / file_name)
            out[dataset][method_key] = list(blob.get(metric_key(method_key, metric), []))
    return out


def _save_line_grid(
    series: Mapping[str, Mapping[str, Sequence[float]]],
    output_path: Path,
    ylabel: str,
    title_prefix: str,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    axes = axes.flatten()
    for ax, dataset in zip(axes, DATASETS):
        for method_key in ("custom", "random", "oort"):
            values = [v for v in series[dataset][method_key] if v is not None]
            if not values:
                continue
            x = np.arange(1, len(values) + 1)
            ax.plot(
                x,
                values,
                label=DISPLAY_NAMES[method_key],
                color=METHOD_COLORS[method_key],
                linewidth=1.8,
            )
        ax.set_title(dataset.upper(), fontsize=12)
        ax.set_xlabel("FL round")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle(title_prefix, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_and_loss(results_dir: Path, file_map: Mapping[str, Mapping[str, str]], out_dir: Path, setting_name: str) -> None:
    accuracy_series = _load_series_for_plot(results_dir, file_map, "accuracy")
    loss_series = _load_series_for_plot(results_dir, file_map, "loss")
    _save_line_grid(
        accuracy_series,
        out_dir / f"{slugify(setting_name)}_accuracy.png",
        ylabel="Accuracy (%)",
        title_prefix=f"{setting_name}: accuracy",
    )
    _save_line_grid(
        loss_series,
        out_dir / f"{slugify(setting_name)}_loss.png",
        ylabel="Loss",
        title_prefix=f"{setting_name}: loss",
    )


def plot_privacy_violation(results_dir: Path, out_dir: Path, max_rounds: int = 100) -> Path:
    blob = parse_legacy_result_file(results_dir / PRIVACY_VIOLATION_FILE)
    violations = list(blob.get("violation_count", []))[:max_rounds]
    rounds = np.arange(1, len(violations) + 1)
    ensure_dir(out_dir)
    out_path = out_dir / "privacy_violation_first_100_rounds.png"
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.bar(rounds, violations, color=METHOD_COLORS["oort"], width=0.8, label="Oort")
    ax.axhline(0, color=METHOD_COLORS["custom"], linewidth=2, linestyle="--", label="AdRo-FL (zero by design)")
    ax.set_xlabel("FL round")
    ax.set_ylabel("Clusters vulnerable to BSA")
    ax.set_title("Cluster privacy violations under insecure global selection")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
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
            "method": "AdRo-FL",
            "avg_bits": safe_mean(custom_blob.get("bits_custom", [])),
            "avg_energy": safe_mean(custom_blob.get("energy_custom", [])),
        },
        {
            "method": "Random",
            "avg_bits": safe_mean(random_blob.get("bits_random", [])),
            "avg_energy": safe_mean(random_blob.get("energy_random", [])),
        },
        {
            "method": "Oort",
            "avg_bits": safe_mean(oort_blob.get("grad_size_bits", [])),
            "avg_energy": safe_mean(estimate_energy_from_bits(oort_blob.get("grad_size_bits", []), seed=SEED)),
        },
    ]
    return pd.DataFrame(rows)


def plot_svhn_bits_energy(results_dir: Path, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    df = build_svhn_bits_energy_table(results_dir)
    out_paths: List[Path] = []

    # Bar charts
    for metric, ylabel in (("avg_bits", "Average transmitted bits"), ("avg_energy", "Average energy (J)")):
        fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
        bars = ax.bar(df["method"], df[metric], color=[METHOD_COLORS["custom"], METHOD_COLORS["random"], METHOD_COLORS["oort"]])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.4f}" if metric == "avg_energy" else f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9)
        fig.tight_layout()
        path = out_dir / f"svhn_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)

    # Trade-off scatter
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)
    for _, row in df.iterrows():
        method_key = next(k for k, v in DISPLAY_NAMES.items() if v == row["method"])
        ax.scatter(row["avg_bits"], row["avg_energy"], s=250, color=METHOD_COLORS[method_key], edgecolor="black")
        ax.annotate(
            row["method"],
            (row["avg_bits"], row["avg_energy"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    ax.set_xlabel("Average transmitted bits")
    ax.set_ylabel("Average energy (J)")
    ax.set_title("SVHN communication efficiency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    scatter_path = out_dir / "svhn_bits_energy_tradeoff.png"
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(scatter_path)

    df.to_csv(out_dir / "svhn_bits_energy_summary.csv", index=False)
    return out_paths


# ---------------------------------------------------------------------------
# Reviewer-focused analytical simulations
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
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate selection concentration under an adaptive malicious server.

    Interpretation:
    - `insecure_global`: the server can repeatedly force the same target and colluders into the final set.
    - `verified_vrf_final`: the attacker may still *want* to target the same pool, but the final decision is
      randomized and public, so the long-run selection frequencies flatten.
    """
    rng = np.random.default_rng(seed)
    n_dishonest = max(1, int(round(eligible_pool_size * dishonest_fraction)))
    dishonest_ids = list(range(n_dishonest))
    honest_ids = list(range(n_dishonest, eligible_pool_size))
    target_id = honest_ids[0]

    insecure_counts = Counter()
    vrf_counts = Counter()

    for _ in range(rounds):
        # Insecure adaptive attacker: always include target + all colluders, then fill the rest from honest clients
        fill_needed = max(0, selected_capacity - (1 + len(dishonest_ids)))
        other_honest = honest_ids[1:]
        if fill_needed > len(other_honest):
            fill_needed = len(other_honest)
        insecure_selected = [target_id] + dishonest_ids + rng.choice(other_honest, size=fill_needed, replace=False).tolist()
        insecure_counts.update(insecure_selected)

        # VRF finalization: attacker can prefer the same verified eligible pool, but final winners are effectively random.
        vrf_selected = rng.choice(np.arange(eligible_pool_size), size=min(selected_capacity, eligible_pool_size), replace=False)
        vrf_counts.update(vrf_selected.tolist())

    frequency_rows: List[Dict[str, Any]] = []
    for cid in range(eligible_pool_size):
        role = "dishonest" if cid in dishonest_ids else ("target_honest" if cid == target_id else "honest")
        frequency_rows.append(
            {
                "client_id": cid,
                "role": role,
                "insecure_selection_rate": insecure_counts[cid] / rounds,
                "vrf_selection_rate": vrf_counts[cid] / rounds,
            }
        )
    freq_df = pd.DataFrame(frequency_rows)

    def concentration_stats(column: str) -> Dict[str, float]:
        values = freq_df[column].to_numpy()
        # Simple Gini-like concentration measure.
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

    summary_df = pd.DataFrame(
        [
            {"setting": "insecure_global", **concentration_stats("insecure_selection_rate")},
            {"setting": "verified_vrf_final", **concentration_stats("vrf_selection_rate")},
        ]
    )
    summary_df.insert(1, "eligible_pool_size", eligible_pool_size)
    summary_df.insert(2, "selected_capacity", selected_capacity)
    summary_df.insert(3, "dishonest_fraction", dishonest_fraction)
    summary_df.insert(4, "rounds", rounds)
    return summary_df, freq_df


def plot_adaptive_targeting(summary_df: pd.DataFrame, freq_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    ensure_dir(out_dir)
    out_paths: List[Path] = []

    # Figure 1: target vs average client rate
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)
    plot_df = summary_df[["setting", "target_rate", "dishonest_mean_rate", "honest_mean_rate"]].copy()
    x = np.arange(len(plot_df))
    width = 0.24
    ax.bar(x - width, plot_df["target_rate"], width=width, label="Target honest")
    ax.bar(x, plot_df["dishonest_mean_rate"], width=width, label="Dishonest mean")
    ax.bar(x + width, plot_df["honest_mean_rate"], width=width, label="Other honest mean")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["setting"], rotation=0)
    ax.set_ylabel("Per-round selection rate")
    ax.set_title("Adaptive targeting concentration across rounds")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    path1 = out_dir / "adaptive_targeting_rates.png"
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(path1)

    # Figure 2: per-client frequency distribution
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
    ordered = freq_df.sort_values(["role", "client_id"]).reset_index(drop=True)
    ax.plot(ordered.index, ordered["insecure_selection_rate"], label="Insecure", linewidth=1.4)
    ax.plot(ordered.index, ordered["vrf_selection_rate"], label="VRF finalization", linewidth=1.4)
    ax.set_xlabel("Eligible clients (ordered by role)")
    ax.set_ylabel("Per-round selection rate")
    ax.set_title("Selection-frequency flattening under verifiable randomness")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    path2 = out_dir / "adaptive_targeting_frequency_profile.png"
    fig.savefig(path2, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(path2)

    summary_df.to_csv(out_dir / "adaptive_targeting_summary.csv", index=False)
    freq_df.to_csv(out_dir / "adaptive_targeting_client_frequencies.csv", index=False)
    return out_paths


def build_probability_bound_table(
    pool_sizes: Sequence[int] = (40, 60, 80, 100),
    dishonest_fraction: float = 0.10,
    honest_fraction: float = 0.90,
    p_attack: float = 0.001,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for n in pool_sizes:
        k_min = find_k_minimum(n, honest_fraction, dishonest_fraction, p_attack)
        rows.append(
            {
                "eligible_pool_size": n,
                "honest_fraction": honest_fraction,
                "dishonest_fraction": dishonest_fraction,
                "p_attack": p_attack,
                "k_minimum": k_min,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reproducibility / metadata writers
# ---------------------------------------------------------------------------

def collect_environment_manifest(repo_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append({"key": "python_version", "value": sys.version.replace("\n", " ")})
    rows.append({"key": "platform", "value": platform.platform()})
    rows.append({"key": "processor", "value": platform.processor()})
    rows.append({"key": "machine", "value": platform.machine()})
    rows.append({"key": "cpu_count", "value": os.cpu_count()})
    rows.append({"key": "repo_root", "value": str(repo_root)})
    rows.append({"key": "paper_rounds", "value": 3000})
    rows.append({"key": "paper_clients_total", "value": 100})
    rows.append({"key": "paper_clients_per_round", "value": 20})
    rows.append({"key": "paper_batch_size", "value": 64})
    rows.append({"key": "paper_learning_rate_adrofl_random", "value": 0.01})
    rows.append({"key": "paper_learning_rate_oort", "value": 0.1})
    rows.append({"key": "paper_dirichlet_alpha", "value": 0.1})
    rows.append({"key": "paper_quantization_bits", "value": 8})
    rows.append({"key": "paper_privacy_threshold_C", "value": 2})
    rows.append({"key": "paper_noncluster_dishonest_fraction", "value": 0.1})
    rows.append({"key": "paper_noncluster_p_attack", "value": 0.001})

    # Optional ML stack versions
    for pkg_name in ("torch", "torchvision", "numpy", "pandas", "matplotlib", "scipy", "nacl"):
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", "unknown")
        except Exception:
            version = "not_importable"
        rows.append({"key": f"package_{pkg_name}", "value": version})
    return pd.DataFrame(rows)


def write_revision_note(out_dir: Path) -> Path:
    ensure_dir(out_dir)
    note = RunNote(
        note=(
            "This revision runner reconstructs the published result summaries from the legacy text files, "
            "adds reviewer-facing ablations, and optionally re-runs the selection layer experiments. "
            "It does not instantiate a concrete secure aggregation protocol. The experimental layer is treated "
            "as compatible with secure aggregation, consistent with the manuscript's system model."
        )
    )
    path = out_dir / "revision_assumptions.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(note), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Live ablation runner (optional, heavy)
# ---------------------------------------------------------------------------

class LiveAblationRunner:
    def __init__(self, repo_root: Path, output_root: Path):
        self.repo_root = repo_root
        self.food101_dir = repo_root / "food101"
        self.output_root = output_root
        if str(self.food101_dir) not in sys.path:
            sys.path.insert(0, str(self.food101_dir))

    def _lazy_load_module(self, module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _cluster_script_name(dataset: str) -> str:
        return f"ClusterFed_{dataset}_custom_random_Global_local_param.py"

    @staticmethod
    def _vrf_script_name(dataset: str) -> str:
        return f"VRF_informed_selection_{dataset}.py"

    def _patch_cluster_selection(self, module, round_logs: List[Dict[str, Any]]) -> None:
        """Monkey patch Server.select_clients to log utility-loss information.

        The three server files (`server_custom_mnist.py`, `server_custom_fmnist.py`,
        `server_custom_cifar10_svhn.py`) share the same selection logic, so one patch works.
        """
        collections = __import__("collections")
        orig_select_clients = module.Server.select_clients

        def instrumented_select_clients(server_self, K, G, D, client_select_type, FL_round, weight,
                                        global_total_samples, num_classes, selection_scope,
                                        num_clients_to_select=None, VRF_scope=False):
            valid_clusters = []
            deadline_avg = []
            original_payload_sizes = []
            quantized_payload_sizes = []
            server_self.selected_clients.clear()
            all_valid_clients = []

            for cluster in server_self.clusters:
                valid_clients = []
                for client in cluster.clients:
                    orig_payload_size_bits = client.calculate_payload_size()
                    original_payload_sizes.append(orig_payload_size_bits / 8)

                    if client_select_type == 'custom':
                        if server_self.quantization_bit < 32:
                            quantized_gradients = client.quantized_gradient
                            quantized_payload_size_bits = sum(g.numel() * g.element_size() * 8 for g in quantized_gradients)
                            quantized_payload_sizes.append(quantized_payload_size_bits / 8)
                            transmission_time = client.calculate_transmission_time(quantized_payload_size_bits)
                        else:
                            transmission_time = client.calculate_transmission_time(orig_payload_size_bits)
                        deadline_avg.append(transmission_time)
                        client.calculate_energy_consumption(transmission_time)
                        if transmission_time <= D:
                            hybrid_score = (weight * client.local_loss) + ((1 - weight) * client.local_l2_norm * (client.get_num_samples() / max(global_total_samples, 1)))
                            client.metric = hybrid_score
                            valid_clients.append((client, hybrid_score, server_self.quantization_bit))
                    elif client_select_type == 'random':
                        transmission_time = client.calculate_transmission_time(orig_payload_size_bits)
                        client.calculate_energy_consumption(transmission_time)
                        valid_clients.append((client, 0.0, server_self.quantization_bit))

                if selection_scope == 'local' and len(valid_clients) >= K:
                    valid_clusters.append((cluster, valid_clients))
                if selection_scope == 'global':
                    all_valid_clients.extend(valid_clients)

            log_row = {
                "round": FL_round,
                "selection_scope": selection_scope,
                "client_select_type": client_select_type,
                "deadline": D,
                "quantization_bit": server_self.quantization_bit,
                "candidate_count": len(all_valid_clients),
                "pre_security_topk_utility": None,
                "selected_pre_filter_count": 0,
                "selected_post_filter_count": 0,
                "post_security_utility": None,
                "utility_loss_ratio": None,
                "violating_clusters": 0,
            }

            if selection_scope == 'local':
                if len(valid_clusters) == 0:
                    round_logs.append(log_row)
                    return None
                x = K
                selected_local = []
                for _, valid_clients in valid_clusters:
                    if client_select_type == 'custom':
                        valid_clients.sort(key=lambda item: item[1], reverse=True)
                        top_clients = valid_clients[:x]
                    else:
                        top_clients = random.sample(valid_clients, min(x, len(valid_clients)))
                    selected_local.extend(top_clients)
                server_self.selected_clients.extend(selected_local)
                log_row["selected_pre_filter_count"] = len(selected_local)
                log_row["selected_post_filter_count"] = len(selected_local)
                log_row["pre_security_topk_utility"] = float(sum(score for _, score, _ in selected_local))
                log_row["post_security_utility"] = float(sum(score for _, score, _ in selected_local))
                log_row["utility_loss_ratio"] = 0.0
                round_logs.append(log_row)
                return server_self.selected_clients

            # global selection
            if not all_valid_clients or not num_clients_to_select or num_clients_to_select <= 0:
                round_logs.append(log_row)
                return None

            if client_select_type == 'custom':
                ranked = sorted(all_valid_clients, key=lambda item: item[1], reverse=True)
                log_row["pre_security_topk_utility"] = float(sum(score for _, score, _ in ranked[:min(num_clients_to_select, len(ranked))]))
                top_80_percent = max(1, int(0.8 * len(ranked)))
                primary_pool = ranked[:top_80_percent]
                if VRF_scope:
                    log_row["selected_pre_filter_count"] = len(primary_pool)
                    log_row["selected_post_filter_count"] = len(primary_pool)
                    log_row["post_security_utility"] = float(sum(score for _, score, _ in primary_pool[:min(num_clients_to_select, len(primary_pool))]))
                    log_row["utility_loss_ratio"] = None
                    round_logs.append(log_row)
                    return primary_pool

                sampled = random.sample(primary_pool, min(num_clients_to_select, len(primary_pool)))
                log_row["selected_pre_filter_count"] = len(sampled)
                cluster_counts = collections.defaultdict(list)
                for item in sampled:
                    cluster_counts[item[0].cluster_id].append(item)
                violating = [cid for cid, items in cluster_counts.items() if len(items) < K]
                filtered = [entry for cid, entries in cluster_counts.items() if len(entries) >= K for entry in entries]
                server_self.selected_clients.extend(filtered)
                log_row["selected_post_filter_count"] = len(filtered)
                log_row["violating_clusters"] = len(violating)
                post_security = float(sum(score for _, score, _ in filtered)) if filtered else 0.0
                log_row["post_security_utility"] = post_security
                if log_row["pre_security_topk_utility"] not in (None, 0):
                    log_row["utility_loss_ratio"] = float(max(0.0, (log_row["pre_security_topk_utility"] - post_security) / log_row["pre_security_topk_utility"]))
                round_logs.append(log_row)
                return server_self.selected_clients

            # random global
            sampled = random.sample(all_valid_clients, min(num_clients_to_select, len(all_valid_clients)))
            server_self.selected_clients.extend(sampled)
            log_row["selected_pre_filter_count"] = len(sampled)
            log_row["selected_post_filter_count"] = len(sampled)
            log_row["pre_security_topk_utility"] = 0.0
            log_row["post_security_utility"] = 0.0
            log_row["utility_loss_ratio"] = 0.0
            round_logs.append(log_row)
            return server_self.selected_clients

        module.Server.select_clients = instrumented_select_clients  # type: ignore[attr-defined]
        module._original_select_clients = orig_select_clients

    def _patch_vrf_selector(self, module, vrf_logs: List[Dict[str, Any]]) -> None:
        original_vrf_client_select = module.VRF_client_select

        def instrumented_vrf_client_select(server_capacity, all_clients_objects, server):
            primary_pool_metrics = []
            for client, score, _ in all_clients_objects:
                if hasattr(client, "metric"):
                    primary_pool_metrics.append(float(client.metric))
                elif score is not None:
                    primary_pool_metrics.append(float(score))
            best_topk = float(sum(sorted(primary_pool_metrics, reverse=True)[:min(server_capacity, len(primary_pool_metrics))])) if primary_pool_metrics else 0.0
            selected_clients, duration = original_vrf_client_select(server_capacity, all_clients_objects, server)
            post_metrics = [float(getattr(client, "metric", 0.0)) for client, _, _ in selected_clients]
            post_sum = float(sum(sorted(post_metrics, reverse=True)[:min(server_capacity, len(post_metrics))])) if post_metrics else 0.0
            vrf_logs.append(
                {
                    "round": len(vrf_logs) + 1,
                    "primary_pool_size": len(all_clients_objects),
                    "selected_count_before_cap": len(selected_clients),
                    "pre_vrf_topk_utility": best_topk,
                    "post_vrf_utility": post_sum,
                    "utility_loss_ratio": float(max(0.0, (best_topk - post_sum) / best_topk)) if best_topk > 0 else None,
                    "vrf_duration": duration,
                }
            )
            return selected_clients, duration

        module.VRF_client_select = instrumented_vrf_client_select

    def _combine_round_logs(self, result_blob: Dict[str, Any], round_logs: List[Dict[str, Any]], method_key: str, dataset: str, variant: str) -> pd.DataFrame:
        accuracy = result_blob.get(metric_key(method_key, "accuracy"), [])
        loss = result_blob.get(metric_key(method_key, "loss"), [])
        bits = result_blob.get(metric_key(method_key, "bits"), [])
        energy = result_blob.get(metric_key(method_key, "energy"), [])
        train_loss = result_blob.get(metric_key(method_key, "train_loss"), [])
        round_duration = result_blob.get("round_duration", []) or result_blob.get("total_round_duration", [])

        max_len = max(len(round_logs), len(accuracy), len(loss), len(bits), len(energy), len(train_loss), len(round_duration), 0)
        rows: List[Dict[str, Any]] = []
        for idx in range(max_len):
            base = dict(round_logs[idx]) if idx < len(round_logs) else {"round": idx + 1}
            base.update(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "accuracy": accuracy[idx] if idx < len(accuracy) else None,
                    "loss": loss[idx] if idx < len(loss) else None,
                    "bits": bits[idx] if idx < len(bits) else None,
                    "energy": energy[idx] if idx < len(energy) else None,
                    "train_loss": train_loss[idx] if idx < len(train_loss) else None,
                    "round_duration": round_duration[idx] if idx < len(round_duration) else None,
                }
            )
            rows.append(base)
        return pd.DataFrame(rows)

    def run_cluster_variant(
        self,
        dataset: str,
        out_dir: Path,
        *,
        selection_scope: str,
        client_select_type: str,
        K: int,
        server_capacity: int,
        deadline: float,
        quantization_bit: int,
        rounds: int,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
        weight: Optional[float] = None,
        vrf_scope: bool = False,
        variant_name: str = "variant",
    ) -> Tuple[Path, Path]:
        ensure_dir(out_dir)
        module_path = self.food101_dir / self._cluster_script_name(dataset)
        module = self._lazy_load_module(f"revision_cluster_{dataset}_{slugify(variant_name)}", module_path)
        round_logs: List[Dict[str, Any]] = []
        self._patch_cluster_selection(module, round_logs)

        # Override checked-in globals.
        module.result_directory = str(out_dir) + os.sep
        module.ROUNDS = rounds
        module.learning_rate = learning_rate
        module.K = K
        module.SERVER_CAPACITY = server_capacity
        module.D = deadline
        module.LOCAL_EPOCHS = local_epochs
        module.QUANTIZATION_BIT = quantization_bit
        module.client_select_type = client_select_type
        module.selection_scope = selection_scope
        module.VRF_scope = vrf_scope
        module.weight = UTILITY_WEIGHT.get(dataset, 0.4) if weight is None else weight

        module.main()

        txt_files = sorted(out_dir.glob("output_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not txt_files:
            raise RuntimeError(f"No result file generated for {dataset} / {variant_name}")
        result_file = txt_files[0]
        blob = parse_legacy_result_file(result_file)
        csv_path = out_dir / f"{slugify(variant_name)}_round_log.csv"
        df = self._combine_round_logs(blob, round_logs, client_select_type, dataset, variant_name)
        df.to_csv(csv_path, index=False)
        return result_file, csv_path

    def run_vrf_variant(
        self,
        dataset: str,
        out_dir: Path,
        *,
        K: int,
        server_capacity: int,
        deadline: float,
        quantization_bit: int,
        rounds: int,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
        weight: Optional[float] = None,
        variant_name: str = "vrf_variant",
    ) -> Tuple[Path, Path]:
        ensure_dir(out_dir)
        module_path = self.food101_dir / self._vrf_script_name(dataset)
        module = self._lazy_load_module(f"revision_vrf_{dataset}_{slugify(variant_name)}", module_path)
        vrf_logs: List[Dict[str, Any]] = []
        self._patch_vrf_selector(module, vrf_logs)

        module.result_directory = str(out_dir) + os.sep
        module.ROUNDS = rounds
        module.learning_rate = learning_rate
        module.K = K
        module.SERVER_CAPACITY = server_capacity
        module.D = deadline
        module.LOCAL_EPOCHS = local_epochs
        module.QUANTIZATION_BIT = quantization_bit
        module.client_select_type = 'custom'
        module.selection_scope = 'global'
        module.VRF_scope = True
        module.weight = UTILITY_WEIGHT.get(dataset, 0.4) if weight is None else weight

        module.main()

        txt_files = sorted(out_dir.glob("output_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not txt_files:
            raise RuntimeError(f"No VRF result file generated for {dataset} / {variant_name}")
        result_file = txt_files[0]
        blob = parse_legacy_result_file(result_file)
        csv_path = out_dir / f"{slugify(variant_name)}_round_log.csv"
        df = self._combine_round_logs(blob, vrf_logs, "custom", dataset, variant_name)
        df.to_csv(csv_path, index=False)
        return result_file, csv_path


# ---------------------------------------------------------------------------
# High-level report builders
# ---------------------------------------------------------------------------

def write_setting_tables(results_dir: Path, output_root: Path) -> List[Path]:
    out_paths: List[Path] = []
    for setting, file_map in PAPER_FILE_MAPS.items():
        rows = build_summary_rows(setting, results_dir, file_map)
        df = summary_rows_to_dataframe(rows)
        out_path = output_root / "tables" / f"{setting}_summary.csv"
        save_dataframe(df, out_path)
        out_paths.append(out_path)
    return out_paths


def write_baseline_figures(results_dir: Path, output_root: Path) -> List[Path]:
    out_paths: List[Path] = []
    figures_dir = ensure_dir(output_root / "figures")
    for setting, file_map in PAPER_FILE_MAPS.items():
        plot_accuracy_and_loss(results_dir, file_map, figures_dir, setting_name=setting)
        out_paths.extend(
            [
                figures_dir / f"{slugify(setting)}_accuracy.png",
                figures_dir / f"{slugify(setting)}_loss.png",
            ]
        )
    out_paths.append(plot_privacy_violation(results_dir, figures_dir))
    out_paths.extend(plot_svhn_bits_energy(results_dir, figures_dir))
    return out_paths


def build_cross_setting_summary(results_dir: Path, output_root: Path) -> Path:
    frames = []
    for setting, file_map in PAPER_FILE_MAPS.items():
        rows = build_summary_rows(setting, results_dir, file_map)
        frames.append(summary_rows_to_dataframe(rows))
    full = pd.concat(frames, ignore_index=True)
    out_path = output_root / "tables" / "all_settings_summary.csv"
    save_dataframe(full, out_path)
    return out_path


def run_analytical_security_reports(output_root: Path) -> List[Path]:
    out_paths: List[Path] = []
    analysis_dir = ensure_dir(output_root / "reviewer_analysis")

    prob_df = build_probability_bound_table()
    prob_path = analysis_dir / "vrf_probability_bound_table.csv"
    prob_df.to_csv(prob_path, index=False)
    out_paths.append(prob_path)

    summary_df, freq_df = simulate_adaptive_targeting()
    out_paths.extend(plot_adaptive_targeting(summary_df, freq_df, analysis_dir))
    return out_paths


def run_live_reviewer_ablations(repo_root: Path, output_root: Path, datasets: Sequence[str], rounds: int) -> List[Path]:
    runner = LiveAblationRunner(repo_root, output_root)
    out_paths: List[Path] = []
    ablation_root = ensure_dir(output_root / "live_ablations")

    # Cluster-global ablation to disentangle security and efficiency features.
    cluster_variants = [
        {
            "name": "cluster_global_utility_only",
            "selection_scope": "global",
            "client_select_type": "custom",
            "K": 0,
            "server_capacity": 20,
            "deadline": 1e9,
            "quantization_bit": 32,
            "vrf_scope": False,
        },
        {
            "name": "cluster_global_utility_plus_efficiency",
            "selection_scope": "global",
            "client_select_type": "custom",
            "K": 0,
            "server_capacity": 20,
            "deadline": None,  # replaced per dataset
            "quantization_bit": 8,
            "vrf_scope": False,
        },
        {
            "name": "cluster_global_utility_plus_security",
            "selection_scope": "global",
            "client_select_type": "custom",
            "K": 2,
            "server_capacity": 20,
            "deadline": 1e9,
            "quantization_bit": 32,
            "vrf_scope": False,
        },
        {
            "name": "cluster_global_full_adrofl",
            "selection_scope": "global",
            "client_select_type": "custom",
            "K": 2,
            "server_capacity": 20,
            "deadline": None,  # replaced per dataset
            "quantization_bit": 8,
            "vrf_scope": False,
        },
    ]

    vrf_variants = [
        {
            "name": "noncluster_verified_pool_only",
            # implemented as the same first-level filtering with deadline + utility ranking, but without using this helper.
            # For the revision, the cleanest and most reproducible live run is the full VRF version.
            # The analytic simulation above covers adaptive targeting. Here we log utility loss from the full VRF stage.
            "K": 2,
            "server_capacity": 20,
            "deadline": None,
            "quantization_bit": 8,
        },
    ]

    combined_rows: List[pd.DataFrame] = []

    for dataset in datasets:
        dataset_root = ensure_dir(ablation_root / dataset)
        for variant in cluster_variants:
            variant_dir = ensure_dir(dataset_root / variant["name"])
            deadline = DEADLINES[dataset] if variant["deadline"] is None else variant["deadline"]
            try:
                _, csv_path = runner.run_cluster_variant(
                    dataset,
                    variant_dir,
                    selection_scope=variant["selection_scope"],
                    client_select_type=variant["client_select_type"],
                    K=variant["K"],
                    server_capacity=variant["server_capacity"],
                    deadline=deadline,
                    quantization_bit=variant["quantization_bit"],
                    rounds=rounds,
                    vrf_scope=variant["vrf_scope"],
                    variant_name=variant["name"],
                )
                out_paths.append(csv_path)
                combined_rows.append(pd.read_csv(csv_path))
            except Exception as exc:
                failure_path = variant_dir / "FAILED.txt"
                failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                out_paths.append(failure_path)
                print(f"[WARN] Live ablation failed for {dataset}/{variant['name']}: {exc}", file=sys.stderr)

        # Full distributed VRF run with utility-gap logging.
        for variant in vrf_variants:
            variant_dir = ensure_dir(dataset_root / variant["name"])
            deadline = DEADLINES[dataset] if variant["deadline"] is None else variant["deadline"]
            try:
                _, csv_path = runner.run_vrf_variant(
                    dataset,
                    variant_dir,
                    K=variant["K"],
                    server_capacity=variant["server_capacity"],
                    deadline=deadline,
                    quantization_bit=variant["quantization_bit"],
                    rounds=rounds,
                    variant_name=variant["name"],
                )
                out_paths.append(csv_path)
                combined_rows.append(pd.read_csv(csv_path))
            except Exception as exc:
                failure_path = variant_dir / "FAILED.txt"
                failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                out_paths.append(failure_path)
                print(f"[WARN] Live VRF run failed for {dataset}/{variant['name']}: {exc}", file=sys.stderr)

    if combined_rows:
        full = pd.concat(combined_rows, ignore_index=True)
        full_path = ablation_root / "all_live_ablation_round_logs.csv"
        full.to_csv(full_path, index=False)
        out_paths.append(full_path)
        summary_path = summarize_live_ablations(full, ablation_root)
        out_paths.append(summary_path)
        out_paths.extend(plot_live_ablation_figures(full, ablation_root))

    return out_paths


def summarize_live_ablations(df: pd.DataFrame, out_dir: Path) -> Path:
    summary = (
        df.groupby(["dataset", "variant"], dropna=False)
        .agg(
            final_accuracy=("accuracy", "last"),
            best_accuracy=("accuracy", "max"),
            avg_loss=("loss", "mean"),
            avg_bits=("bits", "mean"),
            avg_energy=("energy", "mean"),
            avg_round_duration=("round_duration", "mean"),
            avg_utility_loss_ratio=("utility_loss_ratio", "mean"),
            avg_selected_post_filter_count=("selected_post_filter_count", "mean"),
            avg_violating_clusters=("violating_clusters", "mean"),
        )
        .reset_index()
    )
    out_path = out_dir / "live_ablation_summary.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def plot_live_ablation_figures(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    paths: List[Path] = []

    # Figure 1: cluster-global security vs efficiency trade-off.
    cluster_df = df[df["variant"].astype(str).str.startswith("cluster_global_")].copy()
    if not cluster_df.empty:
        summary = (
            cluster_df.groupby(["dataset", "variant"], dropna=False)
            .agg(final_accuracy=("accuracy", "last"), avg_utility_loss_ratio=("utility_loss_ratio", "mean"))
            .reset_index()
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300)
        for dataset in sorted(summary["dataset"].dropna().unique()):
            subset = summary[summary["dataset"] == dataset]
            axes[0].plot(subset["variant"], subset["final_accuracy"], marker="o", label=dataset.upper())
            axes[1].plot(subset["variant"], subset["avg_utility_loss_ratio"], marker="o", label=dataset.upper())
        axes[0].set_ylabel("Final accuracy (%)")
        axes[1].set_ylabel("Average utility-loss ratio")
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(frameon=False)
        axes[0].set_title("Accuracy under security/efficiency ablations")
        axes[1].set_title("Utility loss introduced by the C-threshold")
        fig.tight_layout()
        path = out_dir / "cluster_security_efficiency_ablation.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Figure 2: VRF utility-loss over rounds.
    vrf_df = df[df["variant"].astype(str).str.contains("noncluster_verified_pool_only")].copy()
    if not vrf_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
        for dataset in sorted(vrf_df["dataset"].dropna().unique()):
            subset = vrf_df[vrf_df["dataset"] == dataset]
            ax.plot(subset["round"], subset["utility_loss_ratio"], linewidth=1.2, label=dataset.upper())
        ax.set_xlabel("FL round")
        ax.set_ylabel("VRF utility-loss ratio")
        ax.set_title("Per-round utility gap introduced by verifiable random finalization")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / "vrf_utility_loss_over_rounds.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

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
        if (candidate / "food101").exists() and (candidate / "results").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not auto-detect the AdRo-FL repo root. Please pass --repo-root /path/to/adroFL"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdRo-FL reviewer revision runner")
    parser.add_argument("--repo-root", type=str, default=None, help="Path to the extracted `adroFL/` repo root")
    parser.add_argument("--output-root", type=str, default="revision_outputs", help="Where generated CSVs/figures will be stored")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["summarize", "analysis", "live", "all"],
        help="summarize=paper tables/figures, analysis=reviewer simulations, live=heavy retraining ablations, all=everything",
    )
    parser.add_argument("--run-live-ablations", action="store_true", help="Actually run the heavy live ablations")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS), help="Comma-separated datasets for live runs")
    parser.add_argument("--rounds", type=int, default=3000, help="Rounds for live ablations")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = detect_repo_root(args.repo_root)
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())
    results_dir = repo_root / "results" / "svhn2"
    if not results_dir.exists():
        raise FileNotFoundError(f"Expected results directory not found: {results_dir}")

    generated: List[Path] = []

    # Metadata / reproducibility files are always helpful.
    metadata_dir = ensure_dir(output_root / "metadata")
    env_df = collect_environment_manifest(repo_root)
    env_path = metadata_dir / "environment_manifest.csv"
    env_df.to_csv(env_path, index=False)
    generated.append(env_path)
    generated.append(write_revision_note(metadata_dir))

    if args.mode in {"summarize", "all"}:
        generated.extend(write_setting_tables(results_dir, output_root))
        generated.append(build_cross_setting_summary(results_dir, output_root))
        generated.extend(write_baseline_figures(results_dir, output_root))

    if args.mode in {"analysis", "all"}:
        generated.extend(run_analytical_security_reports(output_root))

    if args.mode in {"live", "all"} and args.run_live_ablations:
        datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
        for dataset in datasets:
            if dataset not in DATASETS:
                raise ValueError(f"Unsupported dataset in --datasets: {dataset}")
        generated.extend(run_live_reviewer_ablations(repo_root, output_root, datasets, args.rounds))

    manifest = pd.DataFrame({"generated_path": [str(p) for p in generated]})
    manifest_path = output_root / "generated_files_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[OK] Generated {len(generated)} files")
    print(f"[OK] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
