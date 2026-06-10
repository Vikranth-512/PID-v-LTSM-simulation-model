"""
Publication diagnostics for precision-regulation label optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from simulation.labeling_objective import LabelingObjectiveConfig


class LabelingDiagnosticPlotter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    def plot_rollout_trajectory(
        self, sample: dict, cfg: LabelingObjectiveConfig, idx: int = 0
    ) -> Path:
        ec = np.array(sample.get("ec_trace", []))
        if len(ec) == 0:
            return self.output_dir / f"rollout_{idx}.png"
        t = np.arange(len(ec)) * 1.0
        band = cfg.target_band_pct * cfg.ec_target
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, ec, label="EC rollout")
        ax.axhline(cfg.ec_target, color="r", linestyle="--", label="Target")
        ax.fill_between(
            t,
            cfg.ec_target - band,
            cfg.ec_target + band,
            alpha=0.15,
            color="green",
            label="Target band",
        )
        ax.axhline(cfg.critical_threshold, color="orange", linestyle=":", label="Critical")
        opt = sample.get("optimal", {})
        ax.set_title(
            f"Optimal rollout (fr={opt.get('flowrate', 0):.1f}, "
            f"dur={opt.get('duration', 0):.0f}) — EC0={sample.get('ec_initial', 0):.2f}"
        )
        ax.set_xlabel("Horizon step")
        ax.set_ylabel("EC")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = self.output_dir / f"rollout_trajectory_{idx:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_cost_decomposition(self, sample: dict, idx: int = 0) -> Path:
        bd = sample.get("optimal", {}).get("breakdown", {})
        if not bd:
            return self.output_dir / f"cost_{idx}.png"
        keys = [
            "tracking",
            "steady_state",
            "recovery",
            "stability",
            "oscillation",
            "overshoot",
            "nutrient",
            "action",
        ]
        vals = [bd.get(k, 0) for k in keys]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(keys, vals, color="steelblue")
        ax.set_xlabel("Unweighted term value")
        ax.set_title("Cost decomposition — optimal candidate")
        fig.tight_layout()
        path = self.output_dir / f"cost_decomposition_{idx:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_candidate_comparison(self, sample: dict, idx: int = 0) -> Path:
        """Conservative vs aggressive vs optimal scores."""
        cons = sample.get("conservative")
        aggr = sample.get("aggressive")
        opt = sample.get("optimal")
        if not cons or not aggr:
            return self.output_dir / f"candidates_{idx}.png"
        labels = ["Conservative\n(min dose)", "Optimal", "Aggressive\n(max dose)"]
        scores = [cons["score"], opt["score"], aggr["score"]]
        doses = [
            cons["flowrate"] * cons["duration"],
            opt["flowrate"] * opt["duration"],
            aggr["flowrate"] * aggr["duration"],
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, scores, color=["gray", "green", "coral"])
        axes[0].set_ylabel("Total cost (lower better)")
        axes[0].set_title("Candidate scores")
        axes[1].bar(labels, doses, color=["gray", "green", "coral"])
        axes[1].set_ylabel("Dose proxy (fr×dur)")
        axes[1].set_title("Action magnitude")
        fig.tight_layout()
        path = self.output_dir / f"candidate_comparison_{idx:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_recovery_comparison(self, samples: List[dict], cfg: LabelingObjectiveConfig) -> Path:
        ttb = []
        in_band = []
        for s in samples:
            bd = s.get("optimal", {}).get("breakdown", {})
            ttb.append(bd.get("time_to_band", 1.0))
            in_band.append(1.0 if bd.get("in_band_at_end") else 0.0)
        if not ttb:
            return self.output_dir / "recovery.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(ttb))
        ax.bar(x, ttb, color="teal", alpha=0.8, label="Norm. time to band")
        ax2 = ax.twinx()
        ax2.plot(x, in_band, "ro-", label="In band at horizon end")
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel("In band (0/1)")
        ax.set_xlabel("Diagnostic sample")
        ax.set_ylabel("Normalized time-to-band")
        ax.set_title(f"Recovery metrics (band ±{cfg.target_band_pct*100:.0f}% of target)")
        fig.tight_layout()
        path = self.output_dir / "recovery_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_oscillation_terms(self, samples: List[dict]) -> Path:
        osc = [s.get("optimal", {}).get("breakdown", {}).get("oscillation", 0) for s in samples]
        stab = [s.get("optimal", {}).get("breakdown", {}).get("stability", 0) for s in samples]
        if not osc:
            return self.output_dir / "oscillation.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(osc))
        w = 0.35
        ax.bar(x - w / 2, osc, w, label="Oscillation term", color="coral")
        ax.bar(x + w / 2, stab, w, label="Stability term", color="steelblue")
        ax.set_xlabel("Sample")
        ax.legend()
        ax.set_title("Oscillation vs stability penalties (selected labels)")
        fig.tight_layout()
        path = self.output_dir / "oscillation_stability.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_top_candidates_table(self, sample: dict, idx: int = 0) -> Path:
        top = sample.get("candidates_top5", [])
        if not top:
            return self.output_dir / f"top5_{idx}.png"
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        rows = [
            [
                f"{c['flowrate']:.1f}",
                f"{c['duration']:.0f}",
                f"{c['score']:.3f}",
                f"{c['breakdown'].get('tracking', 0):.3f}",
                f"{c['breakdown'].get('recovery', 0):.3f}",
            ]
            for c in top
        ]
        table = ax.table(
            cellText=rows,
            colLabels=["fr", "dur", "total", "track", "recv"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.set_title("Top-5 candidate actions by total cost")
        fig.tight_layout()
        path = self.output_dir / f"top_candidates_{idx:02d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_all(self, samples: List[dict], cfg: LabelingObjectiveConfig) -> None:
        for i, s in enumerate(samples[:6]):
            self.plot_rollout_trajectory(s, cfg, i)
            self.plot_cost_decomposition(s, i)
            self.plot_candidate_comparison(s, i)
            self.plot_top_candidates_table(s, i)
        self.plot_recovery_comparison(samples, cfg)
        self.plot_oscillation_terms(samples)
