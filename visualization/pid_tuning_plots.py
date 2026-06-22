"""
Publication plots for PID tuning experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class PIDTuningPlotter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    def plot_gain_heatmaps(self, coarse: List[dict], name: str = "gain_heatmaps") -> Path:
        if not coarse:
            return self.output_dir / f"{name}.png"
        kp = np.array([c["kp"] for c in coarse])
        ki = np.array([c["ki"] for c in coarse])
        kd = np.array([c["kd"] for c in coarse])
        sc = np.array([c["score"] for c in coarse])
        finite = np.isfinite(sc)
        if np.any(finite):
            plot_sc = np.where(finite, sc, np.nanmax(sc[finite]) * 1.5)
        else:
            plot_sc = sc

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc1 = axes[0].scatter(kp, ki, c=plot_sc, cmap="viridis_r", s=40, alpha=0.85)
        axes[0].set_xlabel("Kp")
        axes[0].set_ylabel("Ki")
        axes[0].set_title("Score landscape: Kp vs Ki")
        plt.colorbar(sc1, ax=axes[0], label="Composite score")

        sc2 = axes[1].scatter(kp, kd, c=plot_sc, cmap="viridis_r", s=40, alpha=0.85)
        axes[1].set_xlabel("Kp")
        axes[1].set_ylabel("Kd")
        axes[1].set_title("Score landscape: Kp vs Kd")
        plt.colorbar(sc2, ax=axes[1], label="Composite score")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_ec_trajectory(self, trace: Dict, name: str = "ec_trajectory") -> Path:
        if not trace or "ec" not in trace:
            return self.output_dir / f"{name}.png"
        ec = np.array(trace["ec"])
        dt = trace.get("dt", 60.0)
        target = trace.get("target", 1.2)
        t = np.arange(len(ec)) * dt / 60.0
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, ec, label="EC")
        ax.axhline(target, color="r", linestyle="--", label="Target")
        ax.fill_between(t, target, ec, where=ec > target, alpha=0.2, color="orange", label="Overshoot")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("EC (mS/cm)")
        ax.legend()
        ax.set_title("Tuned PID — closed-loop EC trajectory")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_dosing(self, trace: Dict, name: str = "dosing_behavior") -> Path:
        if not trace:
            return self.output_dir / f"{name}.png"
        dt = trace.get("dt", 60.0)
        t = np.arange(len(trace["flowrate"])) * dt / 60.0
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].step(t, trace["flowrate"], where="post", color="C3")
        axes[0].set_ylabel("Flowrate (mL/min)")
        axes[1].step(t, trace["duration"], where="post", color="C4")
        axes[1].set_ylabel("Duration (s)")
        axes[1].set_xlabel("Time (min)")
        fig.suptitle("Tuned PID — dosing actions")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_oscillation_analysis(self, trace: Dict, name: str = "oscillation") -> Path:
        if not trace or "ec" not in trace:
            return self.output_dir / f"{name}.png"
        ec = np.array(trace["ec"])
        target = trace.get("target", 1.2)
        err = ec - target
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(err, color="C0")
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_ylabel("EC error")
        axes[0].set_title("Error signal & oscillation")
        tail = err[-min(200, len(err)) :]
        axes[1].acorr(tail, maxlags=50)
        axes[1].set_xlabel("Lag")
        axes[1].set_title("Autocorrelation (tail segment)")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_ranking_table(self, results: Dict, top_n: int = 15, name: str = "rankings") -> Path:
        coarse = results.get("coarse", [])[:top_n]
        if not coarse:
            return self.output_dir / f"{name}.png"
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(coarse))))
        ax.axis("off")
        rows = [
            [f"{i+1}", f"{c['kp']:.3f}", f"{c['ki']:.4f}", f"{c['kd']:.3f}", f"{c['score']:.4f}"]
            for i, c in enumerate(coarse)
        ]
        table = ax.table(
            cellText=rows,
            colLabels=["Rank", "Kp", "Ki", "Kd", "Score"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax.set_title("Top coarse-search PID candidates (lower score is better)")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_disturbance_recovery(self, config: Dict, best: Dict, name: str = "disturbance_recovery") -> Path:
        from controllers.pid_tuner import evaluate_pid

        modes = ["normal", "heatwave", "nutrient_depletion"]
        fig, axes = plt.subplots(len(modes), 1, figsize=(11, 3 * len(modes)), sharex=False)
        if len(modes) == 1:
            axes = [axes]
        for ax, mode in zip(axes, modes):
            _, _, trace = evaluate_pid(
                best["kp"], best["ki"], best["kd"],
                config,
                n_episodes=1,
                episode_length=600,
                disturbance_mode=mode,
                seed=42,
                return_traces=True,
            )
            if trace:
                ec = np.array(trace["ec"])
                t = np.arange(len(ec)) * trace["dt"] / 60.0
                ax.plot(t, ec, label="EC")
                ax.axhline(trace["target"], color="r", linestyle="--")
            ax.set_ylabel("EC")
            ax.set_title(f"Recovery — {mode}")
        axes[-1].set_xlabel("Time (min)")
        fig.tight_layout()
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_all(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        self.plot_gain_heatmaps(results.get("coarse", []))
        self.plot_ranking_table(results)
        trace = results.get("reference_trace")
        if trace:
            self.plot_ec_trajectory(trace)
            self.plot_dosing(trace)
            self.plot_oscillation_analysis(trace)
        best = results.get("best")
        if best:
            self.plot_disturbance_recovery(config, best)
