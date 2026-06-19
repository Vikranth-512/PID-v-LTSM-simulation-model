"""
Optimization-based pseudo-optimal control labels — Option B: Precision Regulation.

Long-horizon rollout with modular multi-objective scoring favoring:
  - tight setpoint tracking (time-weighted)
  - fast recovery into target band
  - controlled corrective aggression
  - bounded oscillation / nonlinear overshoot safety
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation.dynamics import TankDynamicsParams, TankState, step_dynamics
from simulation.labeling_objective import (
    CostBreakdown,
    LabelingObjectiveConfig,
    evaluate_rollout,
)
from utils.naming import (
    get_trajectory_index_padding,
    list_trajectory_files,
    parse_trajectory_index,
    trajectory_filename,
)

_ROLLOUT_PENALTY = 1e9


class OptimizationLabeler:
    """
    For each timestep, search (flowrate, duration) candidates via long-horizon
    rollout and precision-regulation weighted cost minimization.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        lab = config.get("labeling", {})
        sim = config.get("simulation", {})
        dyn = config.get("dynamics", {})

        self.obj_cfg = LabelingObjectiveConfig.from_yaml(lab, sim, dyn)
        self.horizon = self.obj_cfg.horizon
        self.debug_mode = self.obj_cfg.debug_mode

        self.candidate_flowrates = lab.get(
            "candidate_flowrates",
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        )
        self.candidate_durations = lab.get(
            "candidate_durations", [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        )

        self.dt = sim.get("dt_seconds", 60.0)
        self.ec_target = self.obj_cfg.ec_target
        self.ec_safe_min = sim.get("ec_safe_min", 0.4)
        self.ec_safe_max = sim.get("ec_safe_max", 2.5)
        self.flowrate_max = sim.get("flowrate_max", 5.0)
        self.duration_max = sim.get("duration_max", 30.0)
        self.min_time_between_doses = sim.get("min_time_between_doses", 120.0)
        self.rollout_mode = lab.get("rollout_mode", "impulse")
        self.tie_break_epsilon = float(lab.get("tie_break_epsilon", 0.0))

        self.base_params = TankDynamicsParams.from_config(
            dyn, ec_target=self.ec_target
        )
        self.trajectory_index_padding = get_trajectory_index_padding(config)
        self._diagnostic_samples: List[dict] = []
        self._max_diagnostic_samples = lab.get("max_diagnostic_samples", 8)
        self._failed_trajectories: List[dict] = []

    def _row_to_state(self, row: pd.Series) -> TankState:
        def _get(name: str, default: float) -> float:
            col = f"true_{name}" if f"true_{name}" in row.index else name
            return float(row[col]) if col in row.index and pd.notna(row[col]) else default

        n_delay = self.base_params.delay_steps
        queue = np.zeros(n_delay, dtype=np.float64)
        pending = _get("pending_absorption", 0.0)
        if pending > 0 and n_delay > 0:
            queue[0] = pending

        return TankState(
            water_temp=_get("water_temp", 22.0),
            ec=_get("ec", self.ec_target),
            turbidity=_get("turbidity", 50.0),
            prev_flowrate=float(row.get("prev_flowrate", row.get("flowrate", 0.0))),
            prev_duration=float(row.get("prev_duration", row.get("duration", 0.0))),
            time_since_last_dose=float(row.get("time_since_last_dose", 999.0)),
            ph=_get("ph", 7.2),
            dissolved_oxygen=_get("dissolved_oxygen", 8.0),
            ambient_temp=_get("ambient_temp", 22.0),
            cumulative_nutrients=_get("cumulative_nutrients", 0.0),
            step_index=int(row.get("timestep", 0)),
            absorption_queue=queue,
            algae_biomass=_get("algae_biomass", _get("turbidity", 80.0)),
            nutrient_memory=_get("nutrient_memory", 0.0),
            biomass_memory=_get("biomass_memory", 0.5),
            biomass_growth_drive=_get("biomass_growth_drive", 0.5),
            health_index=_get("health_index", 1.0),
            ec_velocity=_get("ec_velocity", 0.0),
            assimilation_pool=_get("assimilation_pool", 0.0),
        )

    @staticmethod
    def _state_finite(s: TankState) -> bool:
        return bool(
            np.isfinite(s.ec)
            and np.isfinite(s.turbidity)
            and np.isfinite(s.water_temp)
            and np.isfinite(s.nutrient_memory)
            and np.isfinite(s.algae_biomass)
        )

    def _rollout(
        self,
        state: TankState,
        flowrate: float,
        duration: float,
        params: TankDynamicsParams,
    ) -> Tuple[Optional[np.ndarray], Optional[TankState]]:
        ec_trace: List[float] = []
        s = state
        periodic = self.rollout_mode == "periodic_repeat"
        for k in range(self.horizon):
            if periodic:
                can_dose = s.time_since_last_dose >= self.min_time_between_doses
                fr = flowrate if can_dose and flowrate > 0 and duration > 0 else 0.0
                dur = duration if can_dose and flowrate > 0 and duration > 0 else 0.0
            else:
                fr = flowrate if k == 0 else 0.0
                dur = duration if k == 0 else 0.0
            s = step_dynamics(s, fr, dur, self.dt, params)
            if not self._state_finite(s):
                if self.debug_mode:
                    print(
                        f"[ROLL_OUT_FAIL] step={k} flow={flowrate} duration={duration} "
                        f"ec={s.ec} turbidity={s.turbidity} temp={s.water_temp}"
                    )
                return None, None
            ec_trace.append(s.ec)
        trace = np.array(ec_trace, dtype=np.float64)
        if not np.all(np.isfinite(trace)):
            return None, None
        return trace, s

    def score_action(
        self,
        state: TankState,
        flowrate: float,
        duration: float,
        params: Optional[TankDynamicsParams] = None,
        return_breakdown: bool = False,
    ) -> Tuple[float, Optional[CostBreakdown], Optional[np.ndarray]]:
        """
        Evaluate one candidate. Returns (total_score, breakdown, ec_trace).

        Lower score is better.
        """
        params = params or self.base_params

        if state.time_since_last_dose < self.min_time_between_doses and (
            flowrate > 0 or duration > 0
        ):
            bd = CostBreakdown(total=1e6, safety=1e6)
            return 1e6, bd if return_breakdown else None, None

        if (state.ec < self.ec_safe_min or state.ec > self.ec_safe_max) and (
            flowrate == 0 and duration == 0
        ):
            bd = CostBreakdown(total=5e5, safety=5e5)
            return 5e5, bd if return_breakdown else None, None

        ec_trace, final_s = self._rollout(state, flowrate, duration, params)
        if ec_trace is None or final_s is None:
            if self.debug_mode:
                print(
                    f"[ROLL_OUT_FAIL] flow={flowrate}, duration={duration}, "
                    f"ec_init={state.ec}, total_cost={_ROLLOUT_PENALTY}"
                )
            bd = CostBreakdown(total=_ROLLOUT_PENALTY, safety=_ROLLOUT_PENALTY)
            return _ROLLOUT_PENALTY, bd if return_breakdown else None, None

        bd = evaluate_rollout(
            ec_trace, final_s, state, flowrate, duration, self.obj_cfg
        )
        total = bd.total
        if not np.isfinite(total):
            total = _ROLLOUT_PENALTY
            bd.total = total
        if return_breakdown:
            return total, bd, ec_trace
        return total, None, None

    def label_action(
        self,
        state: TankState,
        params: Optional[TankDynamicsParams] = None,
        record_diagnostic: bool = False,
    ) -> Tuple[float, float, float]:
        """Return (optimal_flowrate, optimal_duration, best_score)."""
        params = params or self.base_params
        best_score = float("inf")
        best_fr, best_dur = 0.0, 0.0
        best_bd: Optional[CostBreakdown] = None
        best_trace: Optional[np.ndarray] = None
        candidate_results: List[dict] = []
        any_valid = False

        for fr in self.candidate_flowrates:
            for dur in self.candidate_durations:
                if fr > self.flowrate_max or dur > self.duration_max:
                    continue
                sc, bd, trace = self.score_action(
                    state, fr, dur, params, return_breakdown=True
                )
                if np.isfinite(sc) and sc < _ROLLOUT_PENALTY:
                    any_valid = True
                if bd is not None:
                    candidate_results.append({
                        "flowrate": fr,
                        "duration": dur,
                        "score": sc,
                        "breakdown": bd.to_dict(),
                    })
                if sc < best_score:
                    best_score = sc
                    best_fr, best_dur = fr, dur
                    best_bd, best_trace = bd, trace

        if self.tie_break_epsilon > 0 and candidate_results:
            threshold = best_score * (1.0 + self.tie_break_epsilon)
            tied = [c for c in candidate_results if c["score"] <= threshold]
            if tied:
                pick = min(tied, key=lambda c: c["flowrate"] * c["duration"])
                best_fr = pick["flowrate"]
                best_dur = pick["duration"]
                best_score = pick["score"]

        if not any_valid or not np.isfinite(best_score):
            best_fr, best_dur, best_score = 0.0, 0.0, _ROLLOUT_PENALTY

        if self.debug_mode and candidate_results:
            ranked = sorted(candidate_results, key=lambda c: c["score"])[:3]
            print(f"  [CANDIDATES] top3={ranked}")

        if record_diagnostic and len(self._diagnostic_samples) < self._max_diagnostic_samples:
            conservative = min(
                candidate_results,
                key=lambda c: c["flowrate"] * c["duration"],
                default=None,
            )
            aggressive = max(
                candidate_results,
                key=lambda c: c["flowrate"] * c["duration"],
                default=None,
            )
            self._diagnostic_samples.append({
                "ec_initial": state.ec,
                "optimal": {
                    "flowrate": best_fr,
                    "duration": best_dur,
                    "score": best_score,
                    "breakdown": best_bd.to_dict() if best_bd else {},
                },
                "ec_trace": best_trace.tolist() if best_trace is not None else [],
                "conservative": conservative,
                "aggressive": aggressive,
                "candidates_top5": sorted(candidate_results, key=lambda c: c["score"])[:5],
            })

        return best_fr, best_dur, best_score

    def label_trajectory(
        self,
        df: pd.DataFrame,
        sample_diagnostics: bool = False,
    ) -> pd.DataFrame:
        params = self.base_params
        opt_fr, opt_dur, opt_score = [], [], []
        n = len(df)
        diag_stride = max(1, n // max(self._max_diagnostic_samples, 1))

        for i, (_, row) in enumerate(df.iterrows()):
            state = self._row_to_state(row)
            record = sample_diagnostics and (i % diag_stride == 0)
            fr, dur, sc = self.label_action(state, params, record_diagnostic=record)
            opt_fr.append(fr)
            opt_dur.append(dur)
            opt_score.append(sc)

        out = df.copy()
        out["optimal_flowrate"] = opt_fr
        out["optimal_duration"] = opt_dur
        out["label_score"] = opt_score
        return out

    @staticmethod
    def validate_labeled_dataset(
        df: pd.DataFrame,
        expected_trajectories: int,
    ) -> None:
        """Raise if combined labeled dataset fails integrity checks."""
        if "trajectory_id" not in df.columns:
            raise ValueError("labeled dataset missing trajectory_id column")
        unique_ids = df["trajectory_id"].nunique()
        if unique_ids != expected_trajectories:
            raise RuntimeError(
                f"Labeling integrity check failed: expected {expected_trajectories} "
                f"unique trajectory_id values, got {unique_ids}"
            )
        for col in ("optimal_flowrate", "optimal_duration", "label_score"):
            if col not in df.columns:
                raise ValueError(f"labeled dataset missing column: {col}")
            vals = df[col].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(vals)):
                bad = int(np.sum(~np.isfinite(vals)))
                raise RuntimeError(
                    f"Labeling integrity check failed: {bad} non-finite values in {col}"
                )

    def label_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        run_diagnostics: bool = False,
        figures_dir: Optional[Path] = None,
        expected_trajectories: Optional[int] = None,
    ) -> Path:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if expected_trajectories is None:
            expected_trajectories = self.config.get("simulation", {}).get(
                "num_trajectories", 200
            )

        files = list_trajectory_files(input_dir, labeled=False)
        total = len(files)
        if total == 0:
            raise FileNotFoundError(f"No trajectory CSV files found in {input_dir}")

        labeled_dfs: List[pd.DataFrame] = []
        processed = 0
        failed_trajectories: List[dict] = []

        for file_idx, f in enumerate(files):
            idx = parse_trajectory_index(f.name)
            if idx is None:
                idx = file_idx

            print(f"[LABEL] {file_idx + 1}/{total} Processing trajectory {idx}", flush=True)

            try:
                df = pd.read_csv(f)
                labeled = self.label_trajectory(df, sample_diagnostics=run_diagnostics)

                if idx is not None:
                    out_path = output_dir / trajectory_filename(
                        idx, padding=self.trajectory_index_padding, labeled=True
                    )
                else:
                    out_path = output_dir / f.name.replace(".csv", "_labeled.csv")

                labeled.to_csv(out_path, index=False)
                labeled_dfs.append(labeled)
                processed += 1
                print(f"[LABEL] SUCCESS trajectory {idx}", flush=True)

            except Exception as e:
                print(f"[LABEL] FAILED trajectory {idx}", flush=True)
                print(repr(e), flush=True)
                traceback.print_exc()
                failed_trajectories.append({
                    "trajectory_id": idx,
                    "error": f"{type(e).__name__}: {e}",
                })

        print(f"[LABEL] Total processed: {processed}", flush=True)
        print(f"[LABEL] Total failed: {len(failed_trajectories)}", flush=True)

        failures_path = output_dir / "label_failures.json"
        with open(failures_path, "w") as ff:
            json.dump(failed_trajectories, ff, indent=2)
        if failed_trajectories:
            print(f"[LABEL] Failures saved to {failures_path}")

        self._failed_trajectories = failed_trajectories

        if processed != expected_trajectories:
            raise RuntimeError(
                f"Labeling incomplete: processed {processed}/{expected_trajectories} "
                f"trajectories ({len(failed_trajectories)} failed)"
            )

        if not labeled_dfs:
            raise RuntimeError("No trajectories were labeled successfully")

        combined = pd.concat(labeled_dfs, ignore_index=True)
        combined_path = output_dir / "all_trajectories_labeled.csv"
        combined.to_csv(combined_path, index=False)
        self.validate_labeled_dataset(combined, expected_trajectories)

        meta = {
            "labeling_mode": "precision_regulation_option_b",
            "horizon_steps": self.horizon,
            "trajectory_index_padding": self.trajectory_index_padding,
            "objective_config": self.obj_cfg.__dict__,
            "processed_count": processed,
            "failed_count": len(failed_trajectories),
            "expected_trajectories": expected_trajectories,
        }
        with open(output_dir / "labeling_metadata.json", "w") as mf:
            json.dump(meta, mf, indent=2, default=str)

        if run_diagnostics and figures_dir and self._diagnostic_samples:
            from visualization.labeling_diagnostics import LabelingDiagnosticPlotter

            plotter = LabelingDiagnosticPlotter(figures_dir)
            plotter.plot_all(self._diagnostic_samples, self.obj_cfg)
            with open(output_dir / "labeling_diagnostic_samples.json", "w") as df_out:
                json.dump(self._diagnostic_samples, df_out, indent=2)

        return combined_path

    def get_diagnostic_samples(self) -> List[dict]:
        return list(self._diagnostic_samples)

    def get_failed_trajectories(self) -> List[dict]:
        return list(self._failed_trajectories)
