"""
Core evaluation helpers for label degeneracy diagnostics.

Does NOT modify production modules; duplicates rollout/objective variants here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation.dynamics import TankDynamicsParams, TankState, step_dynamics
from simulation.labeling_objective import (
    CostBreakdown,
    LabelingObjectiveConfig,
    evaluate_rollout,
)
from simulation.optimization_labeler import OptimizationLabeler


class SafetyMode(str, Enum):
    FULL = "full"
    NONE = "none"
    NO_TERMINAL = "no_terminal"
    NO_COLLAPSE = "no_collapse"


class RolloutMode(str, Enum):
    IMPULSE = "impulse"
    REPEAT = "repeat"


@dataclass
class EvalConfig:
    safety_mode: SafetyMode = SafetyMode.FULL
    rollout_mode: RolloutMode = RolloutMode.IMPULSE
    horizon: Optional[int] = None


def iter_candidates(labeler: OptimizationLabeler) -> List[Tuple[float, float]]:
    out = []
    for fr in labeler.candidate_flowrates:
        for dur in labeler.candidate_durations:
            if fr <= labeler.flowrate_max and dur <= labeler.duration_max:
                out.append((float(fr), float(dur)))
    return out


def dose_mass(flowrate: float, duration: float) -> float:
    return flowrate * duration / 60.0


def _safety_ablated(
    ec_trace: np.ndarray,
    final_state: TankState,
    cfg: LabelingObjectiveConfig,
    mode: SafetyMode,
) -> float:
    if mode == SafetyMode.NONE:
        return 0.0
    pen = 0.0
    if mode != SafetyMode.NO_TERMINAL:
        if final_state.ec > cfg.ec_safe_max or final_state.ec < cfg.ec_safe_min:
            pen += cfg.unsafe_terminal_penalty
    if mode != SafetyMode.NO_COLLAPSE:
        if final_state.ec < cfg.ec_healthy_min:
            pen += cfg.collapse_penalty * (cfg.ec_healthy_min - final_state.ec)
    if np.any(ec_trace > cfg.critical_threshold + 0.2):
        pen += 2.0
    return pen


def rollout_trace(
    state: TankState,
    flowrate: float,
    duration: float,
    horizon: int,
    dt: float,
    params: TankDynamicsParams,
    mode: RolloutMode,
    min_time_between_doses: float,
) -> Tuple[Optional[np.ndarray], Optional[TankState]]:
    ec_trace: List[float] = []
    s = state
    for k in range(horizon):
        if mode == RolloutMode.IMPULSE:
            fr = flowrate if k == 0 else 0.0
            dur = duration if k == 0 else 0.0
        else:
            can_dose = s.time_since_last_dose >= min_time_between_doses
            fr = flowrate if can_dose and flowrate > 0 and duration > 0 else 0.0
            dur = duration if can_dose and flowrate > 0 and duration > 0 else 0.0
        s = step_dynamics(s, fr, dur, dt, params)
        if not (
            np.isfinite(s.ec)
            and np.isfinite(s.turbidity)
            and np.isfinite(s.water_temp)
        ):
            return None, None
        ec_trace.append(s.ec)
    trace = np.array(ec_trace, dtype=np.float64)
    if not np.all(np.isfinite(trace)):
        return None, None
    return trace, s


def score_candidate(
    labeler: OptimizationLabeler,
    state: TankState,
    flowrate: float,
    duration: float,
    eval_cfg: EvalConfig = EvalConfig(),
    params: Optional[TankDynamicsParams] = None,
) -> Tuple[float, Optional[CostBreakdown], Optional[np.ndarray]]:
    params = params or labeler.base_params
    obj_cfg = labeler.obj_cfg
    horizon = eval_cfg.horizon or labeler.horizon
    dt = labeler.dt

    if state.time_since_last_dose < labeler.min_time_between_doses and (
        flowrate > 0 or duration > 0
    ):
        bd = CostBreakdown(total=1e6, safety=1e6)
        return 1e6, bd, None

    if (state.ec < labeler.ec_safe_min or state.ec > labeler.ec_safe_max) and (
        flowrate == 0 and duration == 0
    ):
        bd = CostBreakdown(total=5e5, safety=5e5)
        return 5e5, bd, None

    ec_trace, final_s = rollout_trace(
        state,
        flowrate,
        duration,
        horizon,
        dt,
        params,
        eval_cfg.rollout_mode,
        labeler.min_time_between_doses,
    )
    if ec_trace is None or final_s is None:
        bd = CostBreakdown(total=1e9, safety=1e9)
        return 1e9, bd, None

    bd = evaluate_rollout(
        ec_trace, final_s, state, flowrate, duration, obj_cfg
    )
    if eval_cfg.safety_mode != SafetyMode.FULL:
        bd.safety = _safety_ablated(
            ec_trace, final_s, obj_cfg, eval_cfg.safety_mode
        )
        bd.total = bd.weighted_total(obj_cfg)
    return bd.total, bd, ec_trace


def weighted_parts(bd: CostBreakdown, cfg: LabelingObjectiveConfig) -> Dict[str, float]:
    return {
        "tracking": cfg.tracking * bd.tracking,
        "steady_state": cfg.steady_state * bd.steady_state,
        "recovery": cfg.recovery * bd.recovery,
        "stability": cfg.stability * bd.stability,
        "oscillation": cfg.oscillation * bd.oscillation,
        "overshoot": cfg.overshoot * bd.overshoot,
        "nutrient": cfg.nutrient_cost * bd.nutrient,
        "action": cfg.action_smoothness * bd.action,
        "safety": bd.safety,
        "total": bd.total,
    }


def evaluate_all_candidates(
    labeler: OptimizationLabeler,
    state: TankState,
    eval_cfg: EvalConfig = EvalConfig(),
) -> List[Dict[str, Any]]:
    rows = []
    for fr, dur in iter_candidates(labeler):
        total, bd, trace = score_candidate(
            labeler, state, fr, dur, eval_cfg=eval_cfg
        )
        if bd is None:
            continue
        parts = weighted_parts(bd, labeler.obj_cfg)
        row = {
            "flowrate": fr,
            "duration": dur,
            "dose": dose_mass(fr, dur),
            "total": total,
            **parts,
            "ec_end": float(trace[-1]) if trace is not None else np.nan,
            "ec_max": float(np.max(trace)) if trace is not None else np.nan,
            "ec_min": float(np.min(trace)) if trace is not None else np.nan,
        }
        rows.append(row)
    return rows


def best_action(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    valid = [r for r in rows if np.isfinite(r["total"]) and r["total"] < 1e8]
    valid.sort(key=lambda r: r["total"])
    if len(valid) < 2:
        return valid[0], valid[0], 0.0
    margin = (valid[1]["total"] - valid[0]["total"]) / max(valid[0]["total"], 1e-9)
    return valid[0], valid[1], margin


def synthetic_state(
    labeler: OptimizationLabeler,
    ec: float,
    time_since_last_dose: float = 999.0,
) -> TankState:
    s = TankState.create_initial(labeler.base_params)
    s.ec = ec
    s.time_since_last_dose = time_since_last_dose
    s.prev_flowrate = 0.0
    s.prev_duration = 0.0
    return s


def continuous_dose_sweep(
    labeler: OptimizationLabeler,
    state: TankState,
    doses: np.ndarray,
    eval_cfg: EvalConfig = EvalConfig(),
) -> List[Dict[str, Any]]:
    """Map dose mass to (flowrate=5, duration=dose*60/5) capped at duration_max."""
    rows = []
    for d in doses:
        if d <= 0:
            fr, dur = 0.0, 0.0
        else:
            fr = 5.0
            dur = min(labeler.duration_max, d * 60.0 / fr)
        total, bd, _ = score_candidate(
            labeler, state, fr, dur, eval_cfg=eval_cfg
        )
        if bd is None:
            continue
        parts = weighted_parts(bd, labeler.obj_cfg)
        rows.append({"dose": d, "flowrate": fr, "duration": dur, "total": total, **parts})
    return rows
