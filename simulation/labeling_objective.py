"""
Modular multi-objective cost for precision-regulation label generation (Option B).

J_total =
  λ_tracking  * J_tracking   (time-weighted setpoint error)
+ λ_recovery  * J_recovery   (band entry / persistence)
+ λ_stability * J_stability  (late-horizon variance, not transients)
+ λ_oscillation * J_oscillation (ringing / chatter)
+ λ_overshoot * J_overshoot  (nonlinear mild vs severe)
+ λ_cost * J_nutrient
+ λ_smoothness * J_action

Lower total is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np

from simulation.dynamics import TankDynamicsParams, TankState

_INVALID_COST = 1e6
_INVALID_TOTAL = 1e9


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Lag-1 style correlation with flat-sequence guard."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))


def _ensure_finite(value: float, name: str, debug: bool = False) -> float:
    if np.isfinite(value):
        return float(value)
    if debug:
        raise FloatingPointError(f"Invalid cost term: {name} = {value}")
    return _INVALID_COST


@dataclass
class LabelingObjectiveConfig:
    """Weights and thresholds loaded from YAML `labeling` section."""

    ec_target: float = 1.2
    ec_safe_min: float = 0.4
    ec_safe_max: float = 2.5
    ec_healthy_min: float = 0.75
    horizon: int = 60
    target_band_pct: float = 0.05
    time_weight_alpha: float = 2.0
    steady_state_tail_fraction: float = 0.35

    # λ weights
    tracking: float = 1.4
    recovery: float = 1.0
    steady_state: float = 0.8
    stability: float = 0.25
    oscillation: float = 0.6
    overshoot: float = 0.5
    nutrient_cost: float = 0.2
    action_smoothness: float = 0.15
    action_rate: float = 0.1

    transient_aggression_tolerance: float = 1.5
    under_target_cost_relief: float = 0.65

    mild_threshold_offset: float = 0.15
    critical_threshold: float = 1.7
    mild_overshoot_weight: float = 0.35
    severe_overshoot_weight: float = 4.0

    collapse_penalty: float = 8.0
    unsafe_terminal_penalty: float = 12.0
    debug_mode: bool = False

    @classmethod
    def from_yaml(cls, labeling: Dict[str, Any], sim: Dict[str, Any], dyn: Dict[str, Any]) -> "LabelingObjectiveConfig":
        w = labeling.get("weights", {})
        legacy = labeling.get("legacy_weights", labeling.get("weights_legacy", {}))
        os_cfg = labeling.get("overshoot", {})
        act = labeling.get("action_regularization", {})

        def _w(new_key: str, legacy_key: str, default: float) -> float:
            if new_key in w:
                return float(w[new_key])
            if legacy_key in legacy:
                return float(legacy[legacy_key])
            return default

        return cls(
            ec_target=sim.get("ec_target", 1.2),
            ec_safe_min=sim.get("ec_safe_min", 0.4),
            ec_safe_max=sim.get("ec_safe_max", 2.5),
            ec_healthy_min=dyn.get("ec_healthy_min", 0.75),
            horizon=labeling.get("horizon_steps", 60),
            target_band_pct=labeling.get("target_band_pct", 0.05),
            time_weight_alpha=labeling.get("time_weight_alpha", 2.0),
            steady_state_tail_fraction=labeling.get("steady_state_tail_fraction", 0.35),
            tracking=_w("tracking", "ec_error", 1.4),
            recovery=float(w.get("recovery", labeling.get("recovery_weight", 1.0))),
            steady_state=float(w.get("steady_state", labeling.get("steady_state_weight", 0.8))),
            stability=_w("stability", "instability", 0.25),
            oscillation=float(w.get("oscillation", labeling.get("oscillation_weight", 0.6))),
            overshoot=_w("overshoot", "overshoot", 0.5),
            nutrient_cost=_w("nutrient_cost", "nutrient_cost", 0.2),
            action_smoothness=float(act.get("smoothness_weight", w.get("action_smoothness", 0.15))),
            action_rate=float(act.get("rate_change_weight", w.get("action_rate", 0.1))),
            transient_aggression_tolerance=labeling.get("transient_aggression_tolerance", 1.5),
            under_target_cost_relief=labeling.get("under_target_cost_relief", 0.65),
            mild_threshold_offset=os_cfg.get("mild_threshold_offset", 0.15),
            critical_threshold=os_cfg.get("critical_threshold", 1.7),
            mild_overshoot_weight=os_cfg.get("mild_weight", 0.35),
            severe_overshoot_weight=os_cfg.get("severe_weight", 4.0),
            collapse_penalty=labeling.get("collapse_penalty", 8.0),
            unsafe_terminal_penalty=labeling.get("unsafe_terminal_penalty", 12.0),
            debug_mode=bool(labeling.get("debug_mode", False)),
        )


@dataclass
class CostBreakdown:
    """Per-candidate objective terms (unweighted)."""

    tracking: float = 0.0
    steady_state: float = 0.0
    recovery: float = 0.0
    stability: float = 0.0
    oscillation: float = 0.0
    overshoot: float = 0.0
    nutrient: float = 0.0
    action: float = 0.0
    safety: float = 0.0
    total: float = 0.0
    time_to_band: float = 0.0
    in_band_at_end: bool = False

    def weighted_total(self, cfg: LabelingObjectiveConfig) -> float:
        return (
            cfg.tracking * self.tracking
            + cfg.steady_state * self.steady_state
            + cfg.recovery * self.recovery
            + cfg.stability * self.stability
            + cfg.oscillation * self.oscillation
            + cfg.overshoot * self.overshoot
            + cfg.nutrient_cost * self.nutrient
            + cfg.action_smoothness * self.action
            + self.safety
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _time_weights(horizon: int, alpha: float) -> np.ndarray:
    """w_t = 1 + alpha * (t/H)^2 — late errors weighted more heavily."""
    if horizon <= 1:
        return np.ones(1)
    t = np.arange(horizon, dtype=np.float64)
    return 1.0 + alpha * (t / max(horizon - 1, 1)) ** 2


def _tracking_cost(ec_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> tuple[float, float]:
    """Time-weighted MAE + tail steady-state emphasis."""
    w = _time_weights(len(ec_trace), cfg.time_weight_alpha)
    errors = np.abs(ec_trace - cfg.ec_target)
    j_track = float(np.mean(w * errors))

    tail_n = max(3, int(len(ec_trace) * cfg.steady_state_tail_fraction))
    tail_err = float(np.mean(errors[-tail_n:]))
    j_ss = tail_err * (1.0 + cfg.time_weight_alpha * 0.5)
    return j_track, j_ss


def _recovery_cost(ec_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> tuple[float, float, bool]:
    """
    Penalize slow band entry and failure to remain in band.

    Returns (J_recovery, normalized_time_to_band, in_band_at_end).
    """
    band = cfg.target_band_pct * cfg.ec_target
    h = len(ec_trace)
    time_to_band = float(h)
    in_band = np.abs(ec_trace - cfg.ec_target) <= band
    for i, ok in enumerate(in_band):
        if ok:
            time_to_band = float(i)
            break

    norm_ttb = time_to_band / max(h, 1)
    persistence_fail = 0.0 if in_band[-1] else 1.0
    # Under-target bias at end — precision regulation should not settle low
    under_target_end = max(0.0, cfg.ec_target - ec_trace[-1]) / cfg.ec_target

    j_recovery = norm_ttb + 1.2 * persistence_fail + 0.8 * under_target_end
    return j_recovery, norm_ttb, bool(in_band[-1])


def _stability_cost(ec_trace: np.ndarray) -> float:
    """Late-horizon variance only — ignores early corrective transients."""
    h = len(ec_trace)
    start = h // 2
    tail = ec_trace[start:]
    if len(tail) < 3:
        return 0.0
    return float(np.var(tail))


def _oscillation_cost(ec_trace: np.ndarray) -> float:
    """
    Ringing / chatter: sign-change rate, rolling variance, lag-1 autocorrelation.
    """
    if len(ec_trace) < 4:
        return 0.0
    d = np.diff(ec_trace)
    signs = np.sign(d)
    sign_changes = float(np.sum(np.abs(np.diff(signs)) > 0)) / max(len(d) - 1, 1)

    win = min(8, len(ec_trace) // 2)
    rolling_var = 0.0
    if win >= 3:
        vars = [np.var(ec_trace[i : i + win]) for i in range(len(ec_trace) - win)]
        rolling_var = float(np.mean(vars))

    centered = ec_trace - np.mean(ec_trace)
    ac1 = _safe_corr(centered[:-1], centered[1:])
    ringing = max(0.0, ac1)

    cost = sign_changes + 0.5 * rolling_var + 0.8 * ringing
    return float(np.nan_to_num(cost, nan=0.0, posinf=_INVALID_COST, neginf=_INVALID_COST))


def _overshoot_cost(ec_trace: np.ndarray, cfg: LabelingObjectiveConfig) -> float:
    """
    Nonlinear: mild quadratic below critical; exponential above critical threshold.
    """
    target = cfg.ec_target
    mild_line = target + cfg.mild_threshold_offset
    critical = cfg.critical_threshold
    excess = np.maximum(0.0, ec_trace - target)

    mild_mask = (ec_trace > mild_line) & (ec_trace <= critical)
    severe_mask = ec_trace > critical

    j_mild = float(np.mean(excess[mild_mask] ** 2)) if np.any(mild_mask) else 0.0
    if np.any(severe_mask):
        sev_excess = np.clip(ec_trace[severe_mask] - critical, 0.0, 10.0)
        j_severe = float(np.mean(np.exp(sev_excess) - 1.0))
    else:
        j_severe = 0.0

    cost = cfg.mild_overshoot_weight * j_mild + cfg.severe_overshoot_weight * j_severe
    return float(np.nan_to_num(cost, nan=0.0, posinf=_INVALID_COST, neginf=_INVALID_COST))


def _action_cost(
    flowrate: float,
    duration: float,
    state: TankState,
    cfg: LabelingObjectiveConfig,
    ec_trace: np.ndarray,
) -> float:
    """
    Nutrient use + smoothness / rate limits.

    Under-target tracking error reduces effective nutrient penalty (allows corrective aggression).
    """
    dose = flowrate * duration / 60.0
    tracking_gap = max(0.0, cfg.ec_target - state.ec) / cfg.ec_target
    relief = 1.0 - cfg.under_target_cost_relief * min(1.0, tracking_gap)
    relief = max(0.35, relief)

    j_nutrient = dose * relief

    d_fr = abs(flowrate - state.prev_flowrate)
    d_dur = abs(duration - state.prev_duration)
    j_rate = 0.02 * d_fr + 0.002 * d_dur

    spike = max(0.0, dose - 3.0) ** 2 * 0.05
    tolerance = cfg.transient_aggression_tolerance
    if tracking_gap > 0.1:
        spike *= 1.0 / (1.0 + tolerance * tracking_gap)

    j_smooth = j_rate + spike
    return j_nutrient, j_smooth


def _safety_penalties(
    ec_trace: np.ndarray,
    final_state: TankState,
    cfg: LabelingObjectiveConfig,
    flowrate: float,
    duration: float,
) -> float:
    pen = 0.0
    if final_state.ec > cfg.ec_safe_max or final_state.ec < cfg.ec_safe_min:
        pen += cfg.unsafe_terminal_penalty
    if final_state.ec < cfg.ec_healthy_min:
        pen += cfg.collapse_penalty * (cfg.ec_healthy_min - final_state.ec)
    if np.any(ec_trace > cfg.critical_threshold + 0.2):
        pen += 2.0
    return pen


def evaluate_rollout(
    ec_trace: np.ndarray,
    final_state: TankState,
    initial_state: TankState,
    flowrate: float,
    duration: float,
    cfg: LabelingObjectiveConfig,
) -> CostBreakdown:
    """Compute full cost breakdown for one candidate action rollout."""
    dbg = cfg.debug_mode
    j_track, j_ss = _tracking_cost(ec_trace, cfg)
    j_rec, ttb, in_band = _recovery_cost(ec_trace, cfg)
    j_stab = _stability_cost(ec_trace)
    j_osc = _oscillation_cost(ec_trace)
    j_over = _overshoot_cost(ec_trace, cfg)
    j_nut, j_act = _action_cost(flowrate, duration, initial_state, cfg, ec_trace)
    j_safe = _safety_penalties(ec_trace, final_state, cfg, flowrate, duration)

    terms = {
        "tracking": j_track,
        "steady_state": j_ss,
        "recovery": j_rec,
        "stability": j_stab,
        "oscillation": j_osc,
        "overshoot": j_over,
        "nutrient": j_nut,
        "action": j_act,
        "safety": j_safe,
    }
    for name, val in terms.items():
        terms[name] = _ensure_finite(val, name, debug=dbg)

    if dbg:
        print(
            f"  [COST] flow={flowrate:.2f} dur={duration:.1f} "
            + " ".join(f"{k}={terms[k]:.4f}" for k in terms)
        )

    bd = CostBreakdown(
        tracking=terms["tracking"],
        steady_state=terms["steady_state"],
        recovery=terms["recovery"],
        stability=terms["stability"],
        oscillation=terms["oscillation"],
        overshoot=terms["overshoot"],
        nutrient=terms["nutrient"],
        action=terms["action"],
        safety=terms["safety"],
        time_to_band=ttb,
        in_band_at_end=in_band,
    )
    bd.total = bd.weighted_total(cfg)
    if not np.isfinite(bd.total):
        bd.total = _INVALID_TOTAL
    return bd
