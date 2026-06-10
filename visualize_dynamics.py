"""
Controller dynamics visualization suite.

Re-runs closed-loop PID vs LSTM simulations and generates publication-quality
figures explaining performance differences across disturbance scenarios.

Usage:
    python visualize_dynamics.py
    python visualize_dynamics.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.lstm_policy import LSTMPolicy
from preprocessing.normalization import FeatureNormalizer
from simulation_runner.closed_loop_eval import ClosedLoopEvaluator
from utils import load_config, set_seed

SCENARIOS = [
    "normal",
    "sensor_noise",
    "ec_drift",
    "temp_spike",
    "delayed_response",
    "actuator_saturation",
]
CONTROLLERS = ("pid", "lstm")
OUTPUT_SUBDIR = "controller_comparison"
DPI = 300

# Publication palette
COLORS = {"pid": "#2166ac", "lstm": "#b2182b", "target": "#1a1a1a"}


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
    })


def _time_axis(n_steps: int, dt: float) -> np.ndarray:
    return np.arange(n_steps) * dt / 60.0


def _dose_per_step(flowrate: np.ndarray, duration: np.ndarray) -> np.ndarray:
    return flowrate * duration / 60.0


def _cumulative_dose(flowrate: np.ndarray, duration: np.ndarray) -> np.ndarray:
    return np.cumsum(_dose_per_step(flowrate, duration))


def load_evaluation_results(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_trained_model(
    config: Dict[str, Any],
    checkpoint_dir: Path,
    input_size: int,
    device: str = "cpu",
) -> LSTMPolicy:
    mcfg = config.get("model", {})
    model = LSTMPolicy(
        input_size=input_size,
        hidden_size=mcfg.get("hidden_size", 128),
        num_layers=mcfg.get("num_layers", 2),
        dropout=mcfg.get("dropout", 0.2),
        output_size=mcfg.get("output_size", 2),
    )
    ckpt_path = checkpoint_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def collect_trajectories(
    config: Dict[str, Any],
    model: LSTMPolicy,
    normalizer: FeatureNormalizer,
    feature_columns: List[str],
    scenarios: List[str],
    device: str = "cpu",
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Re-run closed-loop episodes; return per-scenario per-controller trajectories."""
    evaluator = ClosedLoopEvaluator(config, seed=config.get("seed", 42))
    ec_target = config["simulation"]["ec_target"]
    data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for scenario in scenarios:
        data[scenario] = {}
        pid_res = evaluator.run_pid(scenario=scenario)
        lstm_res = evaluator.run_learned_policy(
            model, normalizer, feature_columns, scenario=scenario, device=device
        )
        for ctrl, res in [("pid", pid_res), ("lstm", lstm_res)]:
            traj = res["trajectory"]
            data[scenario][ctrl] = {
                "ec": np.asarray(traj["ec"], dtype=np.float64),
                "flowrate": np.asarray(traj["flowrate"], dtype=np.float64),
                "duration": np.asarray(traj["duration"], dtype=np.float64),
                "target_ec": np.full(len(traj["ec"]), ec_target),
                "metrics": res["metrics"],
            }
    return data


def plot_ec_response(
    trajectories: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    dt: float,
    out_dir: Path,
) -> List[Path]:
    saved = []
    for scenario in SCENARIOS:
        if scenario not in trajectories:
            continue
        pid_ec = trajectories[scenario]["pid"]["ec"]
        lstm_ec = trajectories[scenario]["lstm"]["ec"]
        target = trajectories[scenario]["pid"]["target_ec"]
        t = _time_axis(len(pid_ec), dt)

        y_all = np.concatenate([pid_ec, lstm_ec, target])
        y_pad = 0.05 * (y_all.max() - y_all.min() + 1e-6)
        ylim = (y_all.min() - y_pad, y_all.max() + y_pad)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t, pid_ec, color=COLORS["pid"], label="PID", alpha=0.9)
        ax.plot(t, lstm_ec, color=COLORS["lstm"], label="LSTM", alpha=0.9)
        ax.axhline(
            target[0], color=COLORS["target"], linestyle="--",
            linewidth=1.5, label="Target EC setpoint",
        )
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(ylim)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("EC (mS/cm)")
        ax.set_title(f"EC Control Response — {scenario}")
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        path = out_dir / f"ec_response_{scenario}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def plot_actions(
    trajectories: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    dt: float,
    out_dir: Path,
) -> List[Path]:
    saved = []
    for scenario in SCENARIOS:
        if scenario not in trajectories:
            continue
        n = len(trajectories[scenario]["pid"]["flowrate"])
        t = _time_axis(n, dt)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ctrl, label in [("pid", "PID"), ("lstm", "LSTM")]:
            fr = trajectories[scenario][ctrl]["flowrate"]
            dur = trajectories[scenario][ctrl]["duration"]
            axes[0].plot(t, fr, color=COLORS[ctrl], label=label, alpha=0.85)
            axes[1].plot(t, dur, color=COLORS[ctrl], label=label, alpha=0.85)

        axes[0].set_ylabel("Flowrate (mL/min)")
        axes[0].set_title(f"Control Actions — {scenario}")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.35)
        axes[1].set_ylabel("Duration (s)")
        axes[1].set_xlabel("Time (min)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.35)
        fig.tight_layout()
        path = out_dir / f"actions_{scenario}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def plot_cumulative_dose(
    trajectories: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    dt: float,
    out_dir: Path,
) -> List[Path]:
    saved = []
    for scenario in SCENARIOS:
        if scenario not in trajectories:
            continue
        n = len(trajectories[scenario]["pid"]["flowrate"])
        t = _time_axis(n, dt)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for ctrl, label in [("pid", "PID"), ("lstm", "LSTM")]:
            fr = trajectories[scenario][ctrl]["flowrate"]
            dur = trajectories[scenario][ctrl]["duration"]
            cum = _cumulative_dose(fr, dur)
            ax.plot(t, cum, color=COLORS[ctrl], label=label, alpha=0.9)

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Cumulative dose (flowrate × duration / 60)")
        ax.set_title(f"Cumulative Dosing — {scenario}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        path = out_dir / f"cumulative_dose_{scenario}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def plot_error_distribution(
    trajectories: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_dir: Path,
) -> Path:
    errors = {"pid": [], "lstm": []}
    for scenario in SCENARIOS:
        if scenario not in trajectories:
            continue
        for ctrl in CONTROLLERS:
            ec = trajectories[scenario][ctrl]["ec"]
            target = trajectories[scenario][ctrl]["target_ec"]
            errors[ctrl].extend(np.abs(ec - target).tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(max(errors["pid"]), max(errors["lstm"]), 0.01), 60)
    ax.hist(
        errors["pid"], bins=bins, density=True, alpha=0.45,
        color=COLORS["pid"], label="PID", edgecolor="white", linewidth=0.5,
    )
    ax.hist(
        errors["lstm"], bins=bins, density=True, alpha=0.45,
        color=COLORS["lstm"], label="LSTM", edgecolor="white", linewidth=0.5,
    )
    try:
        from scipy.stats import gaussian_kde
        x_grid = np.linspace(0, bins[-1], 300)
        for ctrl in CONTROLLERS:
            kde = gaussian_kde(errors[ctrl])
            ax.plot(x_grid, kde(x_grid), color=COLORS[ctrl], linewidth=2.0, linestyle="-")
    except ImportError:
        pass

    ax.set_xlabel("|EC − target EC|")
    ax.set_ylabel("Density")
    ax.set_title("EC Tracking Error Distribution (all scenarios)")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = out_dir / "error_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _metrics_table(eval_results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    return eval_results.get("closed_loop", {})


def plot_pareto_tradeoff(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for ctrl, color, marker in [
        ("pid", COLORS["pid"], "o"),
        ("lstm", COLORS["lstm"], "s"),
    ]:
        xs, ys, labels = [], [], []
        for scenario in SCENARIOS:
            if scenario not in metrics or ctrl not in metrics[scenario]:
                continue
            m = metrics[scenario][ctrl]
            xs.append(m["cumulative_dosing_cost"])
            ys.append(m["ec_mae"])
            labels.append(f"{ctrl.upper()}-{scenario}")
        ax.scatter(xs, ys, c=color, s=80, marker=marker, label=ctrl.upper(), zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(
                lab, (x, y), textcoords="offset points", xytext=(4, 4),
                fontsize=7, alpha=0.85,
            )

    ax.set_xlabel("Cumulative dosing cost")
    ax.set_ylabel("EC MAE")
    ax.set_title("EC Tracking vs Nutrient Cost Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = out_dir / "pareto_tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_robustness_heatmap(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str,
    title: str,
    filename: str,
    out_dir: Path,
) -> Path:
    scenarios_present = [s for s in SCENARIOS if s in metrics]
    mat = np.zeros((len(scenarios_present), len(CONTROLLERS)))
    for i, sc in enumerate(scenarios_present):
        for j, ctrl in enumerate(CONTROLLERS):
            mat[i, j] = metrics[sc].get(ctrl, {}).get(metric_key, np.nan)

    fig, ax = plt.subplots(figsize=(5.5, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(CONTROLLERS)))
    ax.set_xticklabels([c.upper() for c in CONTROLLERS])
    ax.set_yticks(range(len(scenarios_present)))
    ax.set_yticklabels(scenarios_present)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                color="black" if mat[i, j] < mat.max() * 0.7 else "white",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overshoot_comparison(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
) -> Path:
    scenarios_present = [s for s in SCENARIOS if s in metrics]
    x = np.arange(len(scenarios_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    pid_vals = [metrics[s]["pid"]["overshoot"] for s in scenarios_present]
    lstm_vals = [metrics[s]["lstm"]["overshoot"] for s in scenarios_present]
    ax.bar(x - width / 2, pid_vals, width, label="PID", color=COLORS["pid"])
    ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color=COLORS["lstm"])
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_present, rotation=20, ha="right")
    ax.set_ylabel("Overshoot (mS/cm above target)")
    ax.set_title("Overshoot Comparison by Scenario")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    path = out_dir / "overshoot_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_efficiency_frontier(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for ctrl, color in [("pid", COLORS["pid"]), ("lstm", COLORS["lstm"])]:
        xs, ys = [], []
        for scenario in SCENARIOS:
            if scenario not in metrics or ctrl not in metrics[scenario]:
                continue
            m = metrics[scenario][ctrl]
            xs.append(m["cumulative_dosing_cost"])
            ys.append(m["nutrient_efficiency"])
        ax.scatter(xs, ys, c=color, s=70, zorder=3, label=ctrl.upper())
        order = np.argsort(xs)
        ax.plot(
            np.array(xs)[order], np.array(ys)[order],
            color=color, linestyle="--", alpha=0.6, linewidth=1.2,
        )
        for x, y, sc in zip(xs, ys, [s for s in SCENARIOS if s in metrics]):
            ax.annotate(sc, (x, y), fontsize=7, alpha=0.8)

    ax.set_xlabel("Cumulative dosing cost")
    ax.set_ylabel("Nutrient efficiency (EC MAE / cumulative dose)")
    ax.set_title("Control Efficiency Frontier")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = out_dir / "efficiency_frontier.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """Report which controller dominates each aggregate metric."""
    keys = {
        "ec_mae": "lower",
        "cumulative_dosing_cost": "lower",
        "stability_variance": "lower",
        "overshoot": "lower",
        "nutrient_efficiency": "lower",
    }
    print("\n" + "=" * 60)
    print("CONTROLLER COMPARISON SUMMARY (PID vs LSTM)")
    print("=" * 60)

    for key, direction in keys.items():
        pid_wins = lstm_wins = ties = 0
        for scenario in SCENARIOS:
            if scenario not in metrics:
                continue
            pv = metrics[scenario]["pid"].get(key, np.nan)
            lv = metrics[scenario]["lstm"].get(key, np.nan)
            if np.isnan(pv) or np.isnan(lv):
                continue
            if direction == "lower":
                if pv < lv:
                    pid_wins += 1
                elif lv < pv:
                    lstm_wins += 1
                else:
                    ties += 1
        dominant = "PID" if pid_wins > lstm_wins else ("LSTM" if lstm_wins > pid_wins else "TIE")
        print(
            f"  {key:28s}  PID wins: {pid_wins:2d}  "
            f"LSTM wins: {lstm_wins:2d}  ties: {ties}  ->  {dominant}"
        )

    pid_mae = np.mean([metrics[s]["pid"]["ec_mae"] for s in SCENARIOS if s in metrics])
    lstm_mae = np.mean([metrics[s]["lstm"]["ec_mae"] for s in SCENARIOS if s in metrics])
    pid_cost = np.mean([metrics[s]["pid"]["cumulative_dosing_cost"] for s in SCENARIOS if s in metrics])
    lstm_cost = np.mean([metrics[s]["lstm"]["cumulative_dosing_cost"] for s in SCENARIOS if s in metrics])
    print("-" * 60)
    print(f"  Mean EC MAE           PID={pid_mae:.4f}  LSTM={lstm_mae:.4f}")
    print(f"  Mean dosing cost      PID={pid_cost:.2f}  LSTM={lstm_cost:.2f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controller dynamics visualization")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    _apply_style()
    config = load_config(ROOT / args.config)
    set_seed(config.get("seed", 42))

    paths = config.get("paths", {})
    figures_root = Path(paths.get("figures", "figures"))
    out_dir = figures_root / OUTPUT_SUBDIR
    os.makedirs(out_dir, exist_ok=True)

    eval_path = Path(paths.get("processed", "data/processed")) / "evaluation_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"{eval_path} not found — run: python main.py --stage evaluate"
        )
    eval_results = load_evaluation_results(eval_path)
    metrics = _metrics_table(eval_results)

    feat_path = Path(paths.get("processed", "data/processed")) / "feature_columns.json"
    with open(feat_path) as f:
        feature_cols = json.load(f)["features"]
    input_size = len(feature_cols)

    scaler_dir = paths.get("scalers", Path(paths.get("processed", "data/processed")) / "scalers")
    normalizer = FeatureNormalizer.load(Path(scaler_dir))
    checkpoint_dir = Path(paths.get("checkpoints", "checkpoints"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading trained LSTM policy...")
    model = load_trained_model(config, checkpoint_dir, input_size, device)

    dt = config["simulation"]["dt_seconds"]
    print(f"Re-running closed-loop simulations ({len(SCENARIOS)} scenarios)...")
    trajectories = collect_trajectories(
        config, model, normalizer, feature_cols, SCENARIOS, device=device
    )

    print("Generating plots...")
    saved: List[Path] = []
    saved.extend(plot_ec_response(trajectories, dt, out_dir))
    saved.extend(plot_actions(trajectories, dt, out_dir))
    saved.extend(plot_cumulative_dose(trajectories, dt, out_dir))
    saved.append(plot_error_distribution(trajectories, out_dir))
    saved.append(plot_pareto_tradeoff(metrics, out_dir))
    saved.append(plot_robustness_heatmap(
        metrics, "ec_mae", "EC MAE Robustness", "robustness_ec_mae.png", out_dir,
    ))
    saved.append(plot_robustness_heatmap(
        metrics, "stability_variance",
        "Stability Variance Robustness", "robustness_variance.png", out_dir,
    ))
    saved.append(plot_overshoot_comparison(metrics, out_dir))
    saved.append(plot_efficiency_frontier(metrics, out_dir))

    print(f"\nSaved {len(saved)} figures to {out_dir.resolve()}:")
    for p in sorted(saved):
        print(f"  {p.name}")

    print_summary(metrics)


if __name__ == "__main__":
    main()
