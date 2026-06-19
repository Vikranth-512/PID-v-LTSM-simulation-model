"""
Run all label-degeneracy diagnostics and write report + figures.

Usage:
    python -m diagnostics.label_degeneracy.run_all
    python -m diagnostics.label_degeneracy.run_all --eval-config baseline
    python -m diagnostics.label_degeneracy.run_all --eval-config fixed --tag after
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.label_degeneracy.core import (
    EvalConfig,
    RolloutMode,
    SafetyMode,
    best_action,
    dose_mass,
    evaluate_all_candidates,
    iter_candidates,
    score_candidate,
    synthetic_state,
    continuous_dose_sweep,
    weighted_parts,
)
from simulation.optimization_labeler import OptimizationLabeler
from utils import load_config, set_seed

OUTPUT_DIR = ROOT / "paper" / "figures" / "label_diagnostics"
EC_LEVELS = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3]
HORIZONS = [10, 20, 30, 40, 60]
N_STATES = 500
DOSE_SWEEP = np.linspace(0.0, 2.5, 26)


def _style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def get_eval_config(name: str) -> EvalConfig:
    if name == "baseline":
        return EvalConfig(
            safety_mode=SafetyMode.FULL,
            rollout_mode=RolloutMode.IMPULSE,
            horizon=None,
        )
    if name == "fixed":
        return EvalConfig(
            safety_mode=SafetyMode.NO_COLLAPSE,
            rollout_mode=RolloutMode.REPEAT,
            horizon=30,
        )
    raise ValueError(f"Unknown eval config: {name}")


def load_sample_states(labeler: OptimizationLabeler, n: int, seed: int = 42) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "all_trajectories_labeled.csv"
    df = pd.read_csv(path)
    return df.sample(min(n, len(df)), random_state=seed)


def diagnostic_1_cost_breakdown(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    out_dir: Path,
    eval_cfg: EvalConfig,
    tag: str,
) -> Dict[str, Any]:
    term_keys = [
        "tracking", "steady_state", "recovery", "stability",
        "oscillation", "overshoot", "nutrient", "action", "safety",
    ]
    rows_all = []
    safety_spreads = []

    for _, row in states.iterrows():
        st = labeler._row_to_state(row)
        cands = evaluate_all_candidates(labeler, st, eval_cfg)
        if not cands:
            continue
        safety_vals = [c["safety"] for c in cands if np.isfinite(c["total"])]
        if len(safety_vals) > 1:
            safety_spreads.append(float(np.std(safety_vals)))
        for c in cands:
            if not np.isfinite(c["total"]) or c["total"] >= 1e8:
                continue
            tot = c["total"]
            fracs = {k: c[k] / tot for k in term_keys if tot > 0}
            rows_all.append({**c, **{f"frac_{k}": fracs.get(k, 0) for k in term_keys}})

    df = pd.DataFrame(rows_all)
    summary_rows = []
    for k in term_keys:
        col = f"frac_{k}"
        summary_rows.append({
            "term": k,
            "mean_pct": df[col].mean() * 100,
            "median_pct": df[col].median() * 100,
            "var_pct": df[col].var() * 100,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"cost_breakdown_summary_{tag}.csv", index=False)

    mean_fracs = summary.set_index("term")["mean_pct"]
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_fracs.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylabel("Mean contribution (%)")
    ax.set_title(f"Weighted Cost Breakdown ({tag})")
    ax.set_xticklabels(mean_fracs.index, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / f"cost_breakdown_stacked_{tag}.png", dpi=300)
    plt.close(fig)

    return {
        "n_candidates": len(df),
        "safety_mean_pct": float(mean_fracs.get("safety", 0)),
        "nutrient_mean_pct": float(mean_fracs.get("nutrient", 0)),
        "action_mean_pct": float(mean_fracs.get("action", 0)),
        "safety_std_across_candidates_mean": float(np.mean(safety_spreads)) if safety_spreads else 0.0,
        "safety_std_across_candidates_median": float(np.median(safety_spreads)) if safety_spreads else 0.0,
    }


def diagnostic_2_terminal_collapse(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    out_dir: Path,
    eval_cfg: EvalConfig,
    tag: str,
) -> Dict[str, Any]:
    end_spreads = []
    end_rows = []
    max_rows = []

    for _, row in states.iterrows():
        st = labeler._row_to_state(row)
        cands = [c for c in evaluate_all_candidates(labeler, st, eval_cfg) if c["total"] < 1e8]
        if len(cands) < 2:
            continue
        ec_ends = [c["ec_end"] for c in cands if np.isfinite(c["ec_end"])]
        if len(ec_ends) < 2:
            continue
        spread = float(np.std(ec_ends))
        end_spreads.append(spread)
        for c in cands:
            end_rows.append({"dose": c["dose"], "ec_end": c["ec_end"], "ec_init": st.ec})
            max_rows.append({"dose": c["dose"], "ec_max": c["ec_max"], "ec_init": st.ec})

    end_df = pd.DataFrame(end_rows)
    max_df = pd.DataFrame(max_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(end_df["dose"], end_df["ec_end"], alpha=0.08, s=8, c="C0")
    ax.set_xlabel("Dose mass (flowrate × duration / 60)")
    ax.set_ylabel("EC at horizon end")
    ax.set_title(f"Terminal EC vs Dose ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"ec_end_vs_dose_{tag}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(max_df["dose"], max_df["ec_max"], alpha=0.08, s=8, c="C1")
    ax.set_xlabel("Dose mass")
    ax.set_ylabel("EC max over horizon")
    ax.set_title(f"Peak EC vs Dose ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"ec_max_vs_dose_{tag}.png", dpi=300)
    plt.close(fig)

    # Sample trajectories for one state
    sample_row = states.iloc[0]
    st = labeler._row_to_state(sample_row)
    fig, ax = plt.subplots(figsize=(10, 5))
    for fr, dur, lbl in [(0, 0, "(0,0)"), (2.5, 15, "(2.5,15)"), (5, 30, "(5,30)")]:
        _, _, tr = score_candidate(labeler, st, fr, dur, eval_cfg=eval_cfg)
        if tr is not None:
            ax.plot(tr, label=lbl, linewidth=1.8)
    ax.axhline(labeler.ec_target, color="k", linestyle="--", label="target")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("EC")
    ax.set_title(f"Sample EC Trajectories EC0={st.ec:.2f} ({tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"ec_trajectory_samples_{tag}.png", dpi=300)
    plt.close(fig)

    return {
        "mean_ec_end_spread_per_state": float(np.mean(end_spreads)),
        "median_ec_end_spread_per_state": float(np.median(end_spreads)),
        "pct_states_spread_lt_0.01": float(np.mean(np.array(end_spreads) < 0.01) * 100),
    }


def diagnostic_3_margins(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    out_dir: Path,
    eval_cfg: EvalConfig,
    tag: str,
) -> Dict[str, Any]:
    margins = []
    for _, row in states.iterrows():
        st = labeler._row_to_state(row)
        cands = evaluate_all_candidates(labeler, st, eval_cfg)
        valid = [c for c in cands if np.isfinite(c["total"]) and c["total"] < 1e5]
        if len(valid) < 2:
            continue
        valid.sort(key=lambda c: c["total"])
        margin = (valid[1]["total"] - valid[0]["total"]) / max(valid[0]["total"], 1e-9)
        if np.isfinite(margin):
            margins.append(margin)

    m = np.array(margins)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(m * 100, bins=50, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Margin (second_best - best) / best  (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Best vs Second-Best Margin ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"margin_histogram_{tag}.png", dpi=300)
    plt.close(fig)

    return {
        "margin_mean_pct": float(m.mean() * 100),
        "margin_median_pct": float(np.median(m) * 100),
        "margin_p90_pct": float(np.percentile(m, 90) * 100),
        "pct_margin_lt_0.5": float(np.mean(m < 0.005) * 100),
    }


def diagnostic_4_monotonic(
    labeler: OptimizationLabeler,
    out_dir: Path,
    eval_cfg: EvalConfig,
    tag: str,
) -> Dict[str, Any]:
    results = {}
    for ec in EC_LEVELS:
        st = synthetic_state(labeler, ec, time_since_last_dose=999.0)
        rows = continuous_dose_sweep(labeler, st, DOSE_SWEEP, eval_cfg)
        doses = [r["dose"] for r in rows]
        totals = [r["total"] for r in rows]
        strictly_decreasing = all(
            totals[i] >= totals[i + 1] for i in range(len(totals) - 1)
        )
        results[ec] = {"strictly_decreasing": strictly_decreasing, "rows": rows}

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(doses, totals, "o-", markersize=4, label="total")
        for term, c in [
            ("tracking", "C1"), ("recovery", "C2"), ("nutrient", "C3"), ("safety", "C4"),
        ]:
            ax.plot(doses, [r[term] for r in rows], "--", alpha=0.7, label=term, color=c)
        ax.set_xlabel("Dose mass")
        ax.set_ylabel("Weighted cost")
        ax.set_title(f"Objective vs Dose  EC={ec} ({tag})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"objective_vs_dose_EC_{ec:.1f}_{tag}.png", dpi=300)
        plt.close(fig)

    below = [ec for ec in EC_LEVELS if ec < labeler.ec_target]
    dec_below = sum(1 for ec in below if results[ec]["strictly_decreasing"])
    return {
        "strictly_decreasing_below_target": dec_below,
        "n_below_target": len(below),
        "per_ec": {str(ec): results[ec]["strictly_decreasing"] for ec in EC_LEVELS},
    }


def diagnostic_5_equivalence(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    out_dir: Path,
    eval_cfg: EvalConfig,
    tag: str,
) -> Dict[str, Any]:
    within_vars = []
    between_means = []
    all_rows = []

    for _, row in states.iterrows():
        st = labeler._row_to_state(row)
        cands = [c for c in evaluate_all_candidates(labeler, st, eval_cfg) if c["total"] < 1e8]
        by_dose: Dict[float, List[float]] = defaultdict(list)
        for c in cands:
            by_dose[round(c["dose"], 4)].append(c["total"])
            all_rows.append(c)
        dose_means = {d: np.mean(v) for d, v in by_dose.items()}
        for d, scores in by_dose.items():
            if len(scores) > 1:
                within_vars.append(float(np.var(scores)))
        if len(dose_means) > 1:
            between_means.append(float(np.var(list(dose_means.values()))))

    df = pd.DataFrame(all_rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    dose_groups = sorted(df["dose"].unique())
    data = [df[df["dose"] == d]["total"].values for d in dose_groups if len(df[df["dose"] == d]) > 1]
    if data:
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels([f"{d:.2f}" for d in dose_groups[: len(data)]], rotation=90, fontsize=7)
    ax.set_xlabel("Dose mass")
    ax.set_ylabel("Total score")
    ax.set_title(f"Score Distribution by Dose Equivalence Class ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"dose_equivalence_violin_{tag}.png", dpi=300)
    plt.close(fig)

    n_unique = len(set(round(d, 4) for d in df["dose"]))
    return {
        "unique_dose_count": n_unique,
        "mean_within_dose_variance": float(np.mean(within_vars)) if within_vars else 0.0,
        "mean_between_dose_variance": float(np.mean(between_means)) if between_means else 0.0,
    }


def _optimal_grid_heatmap(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    eval_cfg: EvalConfig,
) -> np.ndarray:
    """2D histogram: EC bin x tsl bin -> dominant action class."""
    ec_bins = np.linspace(0.3, 1.5, 13)
    tsl_bins = [0, 120, 600, 2000]
    grid = np.zeros((len(ec_bins) - 1, len(tsl_bins) - 1))
    counts = np.zeros_like(grid)

    for _, row in states.iterrows():
        st = labeler._row_to_state(row)
        cands = evaluate_all_candidates(labeler, st, eval_cfg)
        best, _, _ = best_action(cands)
        ec_i = np.digitize(st.ec, ec_bins) - 1
        tsl_i = np.digitize(st.time_since_last_dose, tsl_bins) - 1
        if 0 <= ec_i < grid.shape[0] and 0 <= tsl_i < grid.shape[1]:
            dose = dose_mass(best["flowrate"], best["duration"])
            grid[ec_i, tsl_i] += dose
            counts[ec_i, tsl_i] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)
    return grid


def diagnostic_6_safety_ablation(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    modes = [
        ("A_full", EvalConfig(SafetyMode.FULL, RolloutMode.IMPULSE)),
        ("B_no_safety", EvalConfig(SafetyMode.NONE, RolloutMode.IMPULSE)),
        ("C_no_terminal", EvalConfig(SafetyMode.NO_TERMINAL, RolloutMode.IMPULSE)),
        ("D_no_collapse", EvalConfig(SafetyMode.NO_COLLAPSE, RolloutMode.IMPULSE)),
    ]
    stats = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, (name, cfg) in zip(axes, modes):
        grid = _optimal_grid_heatmap(labeler, states, cfg)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)
        doses = []
        for _, row in states.iterrows():
            st = labeler._row_to_state(row)
            cands = evaluate_all_candidates(labeler, st, cfg)
            best, _, _ = best_action(cands)
            doses.append(dose_mass(best["flowrate"], best["duration"]))
        doses = np.array(doses)
        stats[name] = {
            "unique_doses": len(np.unique(np.round(doses, 2))),
            "pct_max_dose": float(np.mean(doses >= 2.49) * 100),
            "pct_zero_dose": float(np.mean(doses < 0.01) * 100),
            "entropy": _action_entropy(doses),
        }

    fig.suptitle(f"Safety Ablation — Mean Optimal Dose by EC×TSL ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"optimal_action_heatmaps_{tag}.png", dpi=300)
    plt.close(fig)
    return stats


def diagnostic_7_horizon(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    stats = {}
    max_dose_pcts = []
    entropies = []

    for H in HORIZONS:
        cfg = EvalConfig(SafetyMode.FULL, RolloutMode.IMPULSE, horizon=H)
        doses = []
        for _, row in states.iterrows():
            st = labeler._row_to_state(row)
            cands = evaluate_all_candidates(labeler, st, cfg)
            best, _, _ = best_action(cands)
            doses.append(dose_mass(best["flowrate"], best["duration"]))
        doses = np.array(doses)
        stats[H] = {
            "pct_max_dose": float(np.mean(doses >= 2.49) * 100),
            "pct_zero_dose": float(np.mean(doses < 0.01) * 100),
            "entropy": _action_entropy(doses),
        }
        max_dose_pcts.append(stats[H]["pct_max_dose"])
        entropies.append(stats[H]["entropy"])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(HORIZONS, max_dose_pcts, "o-", label="pct max dose", color="C0")
    ax1.set_xlabel("Horizon H")
    ax1.set_ylabel("% states choosing max dose")
    ax2 = ax1.twinx()
    ax2.plot(HORIZONS, entropies, "s--", label="dose entropy", color="C1")
    ax2.set_ylabel("Dose entropy (nats)")
    ax1.set_title(f"Optimal Action Distribution vs Horizon ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"optimal_action_distribution_vs_horizon_{tag}.png", dpi=300)
    plt.close(fig)
    return stats


def diagnostic_8_rollout(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    modes = [
        ("impulse", EvalConfig(SafetyMode.FULL, RolloutMode.IMPULSE)),
        ("repeat", EvalConfig(SafetyMode.FULL, RolloutMode.REPEAT)),
    ]
    stats = {}
    for name, cfg in modes:
        doses = []
        margins = []
        for _, row in states.iterrows():
            st = labeler._row_to_state(row)
            cands = evaluate_all_candidates(labeler, st, cfg)
            best, _, margin = best_action(cands)
            doses.append(dose_mass(best["flowrate"], best["duration"]))
            margins.append(margin)
        doses = np.array(doses)
        stats[name] = {
            "unique_doses": len(np.unique(np.round(doses, 2))),
            "pct_max_dose": float(np.mean(doses >= 2.49) * 100),
            "pct_zero_dose": float(np.mean(doses < 0.01) * 100),
            "entropy": _action_entropy(doses),
            "margin_median_pct": float(np.median(margins) * 100),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, cfg) in zip(axes, modes):
        doses = []
        for _, row in states.iterrows():
            st = labeler._row_to_state(row)
            cands = evaluate_all_candidates(labeler, st, cfg)
            best, _, _ = best_action(cands)
            doses.append(dose_mass(best["flowrate"], best["duration"]))
        ax.hist(doses, bins=30, edgecolor="black", alpha=0.75)
        ax.set_title(f"{name}: optimal dose distribution")
        ax.set_xlabel("Dose mass")
    fig.suptitle(f"Rollout Structure Comparison ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"rollout_structure_comparison_{tag}.png", dpi=300)
    plt.close(fig)
    return stats


def _action_entropy(doses: np.ndarray, bins: int = 20) -> float:
    doses = doses[np.isfinite(doses)]
    if len(doses) == 0:
        return 0.0
    hist, _ = np.histogram(doses, bins=bins, range=(0, 2.5))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def label_distribution_metrics(labeler: OptimizationLabeler) -> Dict[str, Any]:
    df = pd.read_csv(ROOT / "data" / "processed" / "all_trajectories_labeled.csv")
    fr = df["optimal_flowrate"].values
    dur = df["optimal_duration"].values
    doses = fr * dur / 60.0
    pairs = list(zip(fr, dur))
    from collections import Counter
    top = Counter(pairs).most_common(5)
    return {
        "n_samples": len(df),
        "pct_fr_zero": float(np.mean(fr == 0) * 100),
        "pct_fr_max": float(np.mean(fr >= 4.99) * 100),
        "pct_dur_zero": float(np.mean(dur == 0) * 100),
        "pct_dur_max": float(np.mean(dur >= 29.9) * 100),
        "dose_entropy": _action_entropy(doses, bins=30),
        "top_pairs": [(p, c) for p, c in top],
    }


def build_evaluation_cache(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
    eval_cfg: EvalConfig,
) -> List[Tuple[Any, List[Dict[str, Any]]]]:
    """Evaluate all candidates once per state (avoids redundant rollouts)."""
    cache: List[Tuple[Any, List[Dict[str, Any]]]] = []
    for i, (_, row) in enumerate(states.iterrows()):
        st = labeler._row_to_state(row)
        cands = evaluate_all_candidates(labeler, st, eval_cfg)
        cache.append((st, cands))
        if (i + 1) % 100 == 0:
            print(f"  cached {i + 1}/{len(states)} states", flush=True)
    return cache


def simulate_label_distribution(
    labeler: OptimizationLabeler,
    states: pd.DataFrame,
) -> Dict[str, Any]:
    """Run label_action on sampled states (no full dataset relabel)."""
    frs, durs, doses = [], [], []
    for _, row in states.iterrows():
        st = labeler._row_to_state(row)
        fr, dur, _ = labeler.label_action(st)
        frs.append(fr)
        durs.append(dur)
        doses.append(fr * dur / 60.0)
    frs = np.array(frs)
    durs = np.array(durs)
    doses = np.array(doses)
    from collections import Counter
    top = Counter(zip(frs, durs)).most_common(5)
    return {
        "n_samples": len(states),
        "pct_fr_zero": float(np.mean(frs == 0) * 100),
        "pct_fr_max": float(np.mean(frs >= 4.99) * 100),
        "pct_dur_zero": float(np.mean(durs == 0) * 100),
        "pct_dur_max": float(np.mean(durs >= 29.9) * 100),
        "dose_entropy": _action_entropy(doses, bins=30),
        "top_pairs": [(p, c) for p, c in top],
    }


def eval_cfg_from_labeler(labeler: OptimizationLabeler) -> EvalConfig:
    safety = (
        SafetyMode.NO_COLLAPSE
        if labeler.obj_cfg.disable_collapse_penalty
        else SafetyMode.FULL
    )
    rollout = (
        RolloutMode.REPEAT
        if labeler.rollout_mode == "periodic_repeat"
        else RolloutMode.IMPULSE
    )
    return EvalConfig(safety, rollout, horizon=labeler.horizon)


def run_full_pipeline(n_states: int = N_STATES) -> None:
    """Baseline diagnostics -> after (fixed config) -> report."""
    print("=== Phase 1: Baseline diagnostics ===")
    baseline = run_all("baseline", "baseline", n_states)
    print("=== Phase 2: After-fix diagnostics (production config) ===")
    config = load_config(ROOT / "configs" / "default.yaml")
    labeler = OptimizationLabeler(config)
    after_cfg = eval_cfg_from_labeler(labeler)
    # Temporarily run with production-aligned eval config
    _style()
    out_dir = OUTPUT_DIR
    states = load_sample_states(labeler, n_states)
    metrics: Dict[str, Any] = {"tag": "after", "eval_config": "production"}
    print(f"  rollout_mode={labeler.rollout_mode} H={labeler.horizon} "
          f"disable_collapse={labeler.obj_cfg.disable_collapse_penalty}")
    metrics["d1"] = diagnostic_1_cost_breakdown(labeler, states, out_dir, after_cfg, "after")
    metrics["d2"] = diagnostic_2_terminal_collapse(labeler, states, out_dir, after_cfg, "after")
    metrics["d3"] = diagnostic_3_margins(labeler, states, out_dir, after_cfg, "after")
    metrics["d4"] = diagnostic_4_monotonic(labeler, out_dir, after_cfg, "after")
    metrics["d5"] = diagnostic_5_equivalence(labeler, states, out_dir, after_cfg, "after")
    metrics["labels"] = simulate_label_distribution(labeler, states)
    with open(out_dir / "metrics_after.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    write_report(baseline, metrics)
    print("=== Done ===")


def run_all(
    eval_cfg_name: str = "baseline",
    tag: str = "baseline",
    n_states: int = N_STATES,
) -> Dict[str, Any]:
    _style()
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs" / "default.yaml")
    set_seed(config.get("seed", 42))
    labeler = OptimizationLabeler(config)
    eval_cfg = get_eval_config(eval_cfg_name)
    states = load_sample_states(labeler, n_states)

    print(f"Running diagnostics [{tag}] eval={eval_cfg_name} on {len(states)} states...")
    metrics: Dict[str, Any] = {"tag": tag, "eval_config": eval_cfg_name}

    metrics["d1"] = diagnostic_1_cost_breakdown(labeler, states, out_dir, eval_cfg, tag)
    metrics["d2"] = diagnostic_2_terminal_collapse(labeler, states, out_dir, eval_cfg, tag)
    metrics["d3"] = diagnostic_3_margins(labeler, states, out_dir, eval_cfg, tag)
    metrics["d4"] = diagnostic_4_monotonic(labeler, out_dir, eval_cfg, tag)
    metrics["d5"] = diagnostic_5_equivalence(labeler, states, out_dir, eval_cfg, tag)
    metrics["d6"] = diagnostic_6_safety_ablation(labeler, states.head(min(80, len(states))), out_dir, tag)
    metrics["d7"] = diagnostic_7_horizon(labeler, states.head(min(80, len(states))), out_dir, tag)
    metrics["d8"] = diagnostic_8_rollout(labeler, states.head(min(80, len(states))), out_dir, tag)
    metrics["labels"] = label_distribution_metrics(labeler)

    with open(out_dir / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def write_report(baseline: Dict[str, Any], after: Optional[Dict[str, Any]] = None) -> None:
    out = OUTPUT_DIR / "diagnostics_report.md"
    lines = [
        "# Label Degeneracy Diagnostics Report",
        "",
        "## Part 1 — Root-Cause Ranking (Evidence-Based)",
        "",
        "### 1. Single-impulse open-loop rollout (DOMINANT)",
        f"- Terminal EC spread across candidates: mean **{baseline['d2']['mean_ec_end_spread_per_state']:.4f}**",
        f"- States with spread < 0.01: **{baseline['d2']['pct_states_spread_lt_0.01']:.1f}%**",
        "- Rollout structure test (D8): see `rollout_structure_comparison_baseline.png`",
        "",
        "### 2. Safety penalty flat across candidates (HIGH)",
        f"- Safety mean cost share: **{baseline['d1']['safety_mean_pct']:.1f}%**",
        f"- Safety std across candidates (mean per state): **{baseline['d1']['safety_std_across_candidates_mean']:.4f}**",
        "",
        "### 3. Monotonic dose preference below target (HIGH)",
        f"- Strictly decreasing objective below target: **{baseline['d4']['strictly_decreasing_below_target']}/{baseline['d4']['n_below_target']}** EC levels",
        "",
        "### 4. Near-tie argmin landscape (HIGH)",
        f"- Margin median: **{baseline['d3']['margin_median_pct']:.3f}%**, p90: **{baseline['d3']['margin_p90_pct']:.3f}%**",
        f"- Decisions with margin < 0.5%: **{baseline['d3']['pct_margin_lt_0.5']:.1f}%**",
        "",
        "### 5. Dose equivalence classes (MEDIUM)",
        f"- Unique doses: **{baseline['d5']['unique_dose_count']}**",
        f"- Within-dose variance: **{baseline['d5']['mean_within_dose_variance']:.6f}** vs between: **{baseline['d5']['mean_between_dose_variance']:.4f}**",
        "",
        "### 6. Nutrient/action terms too weak (MEDIUM)",
        f"- Nutrient share: **{baseline['d1']['nutrient_mean_pct']:.2f}%**, action: **{baseline['d1']['action_mean_pct']:.2f}%**",
        "",
        "### 7. Horizon extends collapse (LOW-MEDIUM)",
        "- See `optimal_action_distribution_vs_horizon_baseline.png`",
        "",
        "## Part 2 — Diagnostic Metrics Summary",
        "",
        "### D1 Cost Breakdown",
        f"- Safety: {baseline['d1']['safety_mean_pct']:.1f}% | Nutrient: {baseline['d1']['nutrient_mean_pct']:.2f}% | Action: {baseline['d1']['action_mean_pct']:.2f}%",
        "",
        "### D2 Terminal Collapse",
        f"- Mean EC_end spread: {baseline['d2']['mean_ec_end_spread_per_state']:.4f}",
        "",
        "### D3 Margins",
        f"- Mean {baseline['d3']['margin_mean_pct']:.3f}%, median {baseline['d3']['margin_median_pct']:.3f}%",
        "",
        "### D6 Safety Ablation",
    ]
    for k, v in baseline["d6"].items():
        lines.append(f"- **{k}**: unique_doses={v['unique_doses']}, pct_max={v['pct_max_dose']:.1f}%, entropy={v['entropy']:.2f}")
    lines.extend([
        "",
        "### D8 Rollout Structure",
    ])
    for k, v in baseline["d8"].items():
        lines.append(f"- **{k}**: unique_doses={v['unique_doses']}, pct_max={v['pct_max_dose']:.1f}%, margin_med={v['margin_median_pct']:.3f}%")
    lines.extend([
        "",
        "### Current Label Distribution",
        f"- pct flowrate=0: {baseline['labels']['pct_fr_zero']:.1f}%, pct flowrate=5: {baseline['labels']['pct_fr_max']:.1f}%",
        f"- dose entropy: {baseline['labels']['dose_entropy']:.2f} nats",
        f"- top pairs: {baseline['labels']['top_pairs'][:3]}",
        "",
        "## Part 3 — Recommended Fixes (Ranked)",
        "",
        "1. **Repeated-dosing rollout** during horizon (structural) — evidence D8",
        "2. **Remove collapse penalty under open-loop coast** or shorten evaluation horizon (structural/objective)",
        "3. **Remove under-target nutrient relief** (objective) — evidence D4 monotonicity",
        "4. **Min-dose tie-break** among ε-optimal candidates (conservative)",
        "5. Weight tuning alone — lowest priority",
        "",
        "## Figures",
        "",
        "All plots in `paper/figures/label_diagnostics/`.",
    ])

    if after:
        lines.extend([
            "",
            "---",
            "",
            "## Before vs After Fix",
            "",
            "| Metric | Baseline | After |",
            "|--------|----------|-------|",
            f"| Margin median (%) | {baseline['d3']['margin_median_pct']:.3f} | {after['d3']['margin_median_pct']:.3f} |",
            f"| EC_end spread | {baseline['d2']['mean_ec_end_spread_per_state']:.4f} | {after['d2']['mean_ec_end_spread_per_state']:.4f} |",
            f"| D8 repeat unique doses | {baseline['d8']['repeat']['unique_doses']} | {after.get('d8',{}).get('repeat',{}).get('unique_doses','N/A')} |",
            f"| Label dose entropy | {baseline['labels']['dose_entropy']:.2f} | {after.get('labels',{}).get('dose_entropy','N/A')} |",
            f"| pct max dose (labels) | {baseline['labels']['pct_fr_max']:.1f} | {after.get('labels',{}).get('pct_fr_max','N/A')} |",
        ])

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", default="baseline", choices=["baseline", "fixed"])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--n-states", type=int, default=N_STATES)
    parser.add_argument("--full", action="store_true", help="Run baseline + after + report")
    args = parser.parse_args()
    if args.full:
        run_full_pipeline(args.n_states)
        return
    tag = args.tag or args.eval_config
    metrics = run_all(args.eval_config, tag, args.n_states)
    if tag == "baseline":
        write_report(metrics)


if __name__ == "__main__":
    main()
