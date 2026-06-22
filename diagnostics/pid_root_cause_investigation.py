"""
PID root-cause investigation: setpoint tracking failure analysis.
Run: python diagnostics/pid_root_cause_investigation.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.pid_controller import PIDConfig, PIDController, PIDGains
from controllers.pid_tuner import (
    EpisodeMetrics,
    _candidate_score,
    _efficiency_score,
    _metrics_from_trace,
    _regulatory_constraints,
    _settling_time,
    evaluate_pid,
)
from simulation.disturbances import DisturbanceConfig, DisturbanceGenerator
from simulation.dynamics import TankDynamicsParams, TankState, step_dynamics, thermal_efficiency
from simulation.environment import AlgaeTankEnvironment, EnvironmentConfig


TUNED_KP = 1.7320508075688774
TUNED_KI = 0.0
TUNED_KD = 0.9654893846056296


@dataclass
class PIDTrace:
    time: np.ndarray
    ec: np.ndarray
    target: np.ndarray
    error: np.ndarray
    p_term: np.ndarray
    i_term: np.ndarray
    d_term: np.ndarray
    u_unsat: np.ndarray
    u_sat: np.ndarray
    flowrate_raw: np.ndarray
    flowrate_clipped: np.ndarray
    duration_raw: np.ndarray
    duration_clipped: np.ndarray
    integral: np.ndarray
    saturated: np.ndarray
    deadband_active: np.ndarray


class InstrumentedPID(PIDController):
    """PID with full internal term logging."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._log: Dict[str, List[float]] = {
            k: []
            for k in [
                "p_term",
                "i_term",
                "d_term",
                "u_unsat",
                "u_sat",
                "flowrate_raw",
                "flowrate_clipped",
                "duration_raw",
                "duration_clipped",
                "integral",
                "saturated",
                "deadband_active",
                "error",
            ]
        }

    def reset(self) -> None:
        super().reset()
        for v in self._log.values():
            v.clear()

    def compute(self, ec: float) -> tuple[float, float]:
        cfg = self.config
        error = self.setpoint - ec
        self._log["error"].append(error)

        if abs(error) < cfg.deadband:
            self._log["deadband_active"].append(1.0)
            self._log["p_term"].append(0.0)
            self._log["i_term"].append(0.0)
            self._log["d_term"].append(0.0)
            self._log["u_unsat"].append(0.0)
            self._log["u_sat"].append(0.0)
            self._log["flowrate_raw"].append(0.0)
            self._log["flowrate_clipped"].append(0.0)
            self._log["duration_raw"].append(0.0)
            self._log["duration_clipped"].append(0.0)
            self._log["integral"].append(self._integral)
            self._log["saturated"].append(float(self._saturated))
            self._prev_error = error
            return 0.0, 0.0

        self._log["deadband_active"].append(0.0)
        raw_derivative = (error - self._prev_error) / max(self.dt, 1e-6)
        self._filtered_derivative = (
            cfg.derivative_alpha * self._filtered_derivative
            + (1.0 - cfg.derivative_alpha) * raw_derivative
        )
        self._prev_error = error

        u_p = self.gains.kp * error
        u_d = self.gains.kd * self._filtered_derivative
        u_i = self.gains.ki * self._integral
        u_unsat = u_p + u_i + u_d

        self._log["p_term"].append(u_p)
        self._log["i_term"].append(u_i)
        self._log["d_term"].append(u_d)
        self._log["u_unsat"].append(u_unsat)

        if u_unsat <= cfg.min_control_u:
            self._saturated = u_unsat <= 0
            self._log["u_sat"].append(0.0)
            self._log["flowrate_raw"].append(0.0)
            self._log["flowrate_clipped"].append(0.0)
            self._log["duration_raw"].append(0.0)
            self._log["duration_clipped"].append(0.0)
            self._log["integral"].append(self._integral)
            self._log["saturated"].append(float(self._saturated))
            self._prev_flowrate = 0.0
            self._prev_duration = 0.0
            return 0.0, 0.0

        flowrate = min(
            self.flowrate_max,
            max(cfg.min_flowrate_when_active, u_unsat * cfg.flowrate_scale),
        )
        duration = min(
            self.duration_max,
            max(cfg.min_duration_when_active, u_unsat * cfg.duration_scale),
        )
        flowrate_raw, duration_raw = flowrate, duration

        flowrate = self._rate_limit(flowrate, self._prev_flowrate, cfg.max_delta_flowrate)
        duration = self._rate_limit(duration, self._prev_duration, cfg.max_delta_duration)
        flowrate = float(np.clip(flowrate, self.flowrate_min, self.flowrate_max))
        duration = float(np.clip(duration, self.duration_min, self.duration_max))

        u_sat = flowrate / max(cfg.flowrate_scale, 1e-6)
        self._saturated = abs(u_unsat - u_sat) > 0.05

        if not self._saturated:
            self._integral += error * self.dt
            self._integral = float(
                np.clip(self._integral, cfg.integral_min, cfg.integral_max)
            )

        self._log["u_sat"].append(u_sat)
        self._log["flowrate_raw"].append(flowrate_raw)
        self._log["flowrate_clipped"].append(flowrate)
        self._log["duration_raw"].append(duration_raw)
        self._log["duration_clipped"].append(duration)
        self._log["integral"].append(self._integral)
        self._log["saturated"].append(float(self._saturated))

        self._prev_flowrate = flowrate
        self._prev_duration = duration
        return float(flowrate), float(duration)

    def build_trace(self, ec_history: List[float], dt: float, target: float) -> PIDTrace:
        n = len(ec_history)
        t = np.arange(n) * dt
        return PIDTrace(
            time=t,
            ec=np.array(ec_history),
            target=np.full(n, target),
            error=np.array(self._log["error"]),
            p_term=np.array(self._log["p_term"]),
            i_term=np.array(self._log["i_term"]),
            d_term=np.array(self._log["d_term"]),
            u_unsat=np.array(self._log["u_unsat"]),
            u_sat=np.array(self._log["u_sat"]),
            flowrate_raw=np.array(self._log["flowrate_raw"]),
            flowrate_clipped=np.array(self._log["flowrate_clipped"]),
            duration_raw=np.array(self._log["duration_raw"]),
            duration_clipped=np.array(self._log["duration_clipped"]),
            integral=np.array(self._log["integral"]),
            saturated=np.array(self._log["saturated"]),
            deadband_active=np.array(self._log["deadband_active"]),
        )


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def run_instrumented_episode(
    config: Dict[str, Any],
    kp: float,
    ki: float,
    kd: float,
    episode_length: int = 600,
    disturbance_mode: str = "normal",
    seed: int = 42,
    water_temp: Optional[float] = None,
    initial_ec: Optional[float] = None,
) -> Tuple[PIDTrace, Dict[str, float]]:
    sim = config.get("simulation", {})
    dyn = config.get("dynamics", {})
    tune = config.get("pid_tuning", {})
    weights = tune.get("weights", {})
    ec_target = sim.get("ec_target", 1.2)
    dt = sim.get("dt_seconds", 60.0)

    params = TankDynamicsParams.from_config(dyn, ec_target=ec_target)
    env_cfg = EnvironmentConfig(
        dt_seconds=dt,
        ec_target=ec_target,
        ec_safe_min=sim.get("ec_safe_min", 0.4),
        ec_safe_max=sim.get("ec_safe_max", 2.5),
        flowrate_min=sim.get("flowrate_min", 0.0),
        flowrate_max=sim.get("flowrate_max", 5.0),
        duration_min=sim.get("duration_min", 0.0),
        duration_max=sim.get("duration_max", 30.0),
        min_time_between_doses=sim.get("min_time_between_doses", 120.0),
        noise_std=sim.get("noise_std"),
    )
    dist_cfg = DisturbanceConfig.from_config(config.get("disturbances", {}))
    pid_cfg = PIDConfig(**tune.get("pid_behavior", {})) if tune.get("pid_behavior") else PIDConfig()

    rng = np.random.default_rng(seed)
    dist_gen = DisturbanceGenerator(dist_cfg, rng)
    schedule = dist_gen.build_schedule(disturbance_mode, episode_length)

    temp = water_temp if water_temp is not None else params.ambient_temp_mean
    ec0 = initial_ec if initial_ec is not None else ec_target
    s0 = TankState.create_initial(params, ec=ec0, water_temp=temp, rng=rng)

    env = AlgaeTankEnvironment(env_cfg, params, rng=rng, disturbance_generator=dist_gen)
    env.reset(initial_state=s0, disturbance_schedule=schedule)

    pid = InstrumentedPID(
        setpoint=ec_target,
        gains=PIDGains(kp=kp, ki=ki, kd=kd),
        config=pid_cfg,
        flowrate_max=env_cfg.flowrate_max,
        duration_max=env_cfg.duration_max,
        flowrate_min=env_cfg.flowrate_min,
        duration_min=env_cfg.duration_min,
        dt=dt,
    )

    ec_h, fr_h, dur_h = [], [], []
    for _ in range(episode_length):
        ec_val = env.state.ec if env.state else ec_target
        fr, dur = pid.compute(ec_val)
        ec_h.append(ec_val)
        fr_h.append(fr)
        dur_h.append(dur)
        env.step((fr, dur))

    trace = pid.build_trace(ec_h, dt, ec_target)
    metrics = _metrics_from_trace(
        np.array(ec_h),
        np.array(fr_h),
        np.array(dur_h),
        ec_target,
        dt,
        env_cfg.ec_safe_min,
        tune,
    )
    return trace, metrics.to_dict()


def tracking_stats(trace: PIDTrace) -> Dict[str, float]:
    err = trace.error  # setpoint - ec (positive = below setpoint)
    ec_err = trace.ec - trace.target[0]  # ec - target (negative = below setpoint)
    tail_n = max(60, len(trace.ec) // 5)
    tail = trace.ec[-tail_n:]
    return {
        "mean_error_signed": float(np.mean(err)),
        "mean_error_abs": float(np.mean(np.abs(err))),
        "median_error_abs": float(np.median(np.abs(err))),
        "steady_state_error": float(np.mean(trace.target[-tail_n:] - tail)),
        "final_error": float(trace.target[-1] - trace.ec[-1]),
        "max_error_abs": float(np.max(np.abs(err))),
        "mean_ec": float(np.mean(trace.ec)),
        "median_ec": float(np.median(trace.ec)),
        "std_ec": float(np.std(trace.ec)),
        "min_ec": float(np.min(trace.ec)),
        "max_ec": float(np.max(trace.ec)),
        "pct_time_within_5pct": float(np.mean(np.abs(ec_err) < 0.05 * trace.target[0]) * 100),
        "pct_time_within_deadband": float(np.mean(np.abs(err) < 0.04) * 100),
        "pct_time_below_setpoint": float(np.mean(trace.ec < trace.target[0]) * 100),
    }


def score_breakdown(metrics: EpisodeMetrics, tune_cfg: Dict[str, Any], horizon_seconds: float) -> Dict[str, float]:
    constraints = _regulatory_constraints(tune_cfg)
    eff_weights = tune_cfg.get("efficiency_weights", tune_cfg.get("weights", {}))
    score, feasible, eff = _candidate_score(metrics, tune_cfg, horizon_seconds)
    parts = {
        "regulatory_feasible": feasible,
        "tracking_ec_mae": metrics.ec_mae,
        "steady_state_error": metrics.steady_state_error,
        "time_in_band": metrics.time_in_band,
        "nutrient_usage": _efficiency_score(
            EpisodeMetrics(
                nutrient_usage=metrics.nutrient_usage,
                control_smoothness=0.0,
                actuator_aggressiveness=0.0,
            ),
            {"nutrient_usage": eff_weights.get("nutrient_usage", 0.4), "smoothness": 0.0, "aggressive_control": 0.0},
        ),
        "smoothness": eff_weights.get("smoothness", 0.2) * metrics.control_smoothness,
        "aggressive_control": eff_weights.get("aggressive_control", 0.3) * metrics.actuator_aggressiveness,
        "total": score,
    }
    return parts


def pid_term_stats(trace: PIDTrace) -> Dict[str, float]:
    abs_p = np.abs(trace.p_term)
    abs_i = np.abs(trace.i_term)
    abs_d = np.abs(trace.d_term)
    total = abs_p + abs_i + abs_d + 1e-12
    return {
        "mean_abs_P": float(np.mean(abs_p)),
        "mean_abs_I": float(np.mean(abs_i)),
        "mean_abs_D": float(np.mean(abs_d)),
        "pct_I_of_total": float(np.mean(abs_i / total) * 100),
        "max_integral": float(np.max(trace.integral)),
        "integral_nonzero_steps": float(np.sum(np.abs(np.diff(trace.integral, prepend=0)) > 1e-9)),
    }


def saturation_stats(trace: PIDTrace, env_cfg: EnvironmentConfig) -> Dict[str, float]:
    fr = trace.flowrate_clipped
    dur = trace.duration_clipped
    u = trace.u_unsat
    return {
        "frac_upper_flowrate": float(np.mean(fr >= env_cfg.flowrate_max - 1e-6)),
        "frac_upper_duration": float(np.mean(dur >= env_cfg.duration_max - 1e-6)),
        "frac_zero_dose": float(np.mean((fr == 0) & (dur == 0))),
        "frac_deadband": float(np.mean(trace.deadband_active > 0)),
        "frac_saturated_flag": float(np.mean(trace.saturated > 0)),
        "frac_u_below_min": float(np.mean((u <= 0.02) & (trace.deadband_active == 0))),
        "mean_dose_mass_per_step": float(np.mean(fr * dur / 60.0)),
        "max_dose_mass_per_step": float(np.max(fr * dur / 60.0)),
    }


def authority_analysis(config: Dict[str, Any], water_temp: float = 22.0) -> Dict[str, float]:
    sim = config.get("simulation", {})
    dyn = config.get("dynamics", {})
    ec_target = sim.get("ec_target", 1.2)
    dt = sim.get("dt_seconds", 60.0)
    params = TankDynamicsParams.from_config(dyn, ec_target=ec_target)
    dt_scale = dt / 60.0

    thermal = thermal_efficiency(water_temp, params)
    depletion_at_target = (
        params.baseline_ec_depletion + params.biological_uptake_rate * ec_target
    ) * thermal * dt_scale

    max_dose = sim.get("flowrate_max", 5.0) * sim.get("duration_max", 30.0) / 60.0
    gain = params.nutrient_to_ec_gain * thermal
    immediate = params.immediate_absorption_fraction
    max_influx_per_step = gain * immediate * max_dose

    # cooldown: max 1 dose per 120s = every 2 steps at dt=60
    cooldown_steps = max(1, int(np.ceil(sim.get("min_time_between_doses", 120.0) / dt)))
    max_sustained_influx = max_influx_per_step / cooldown_steps

    return {
        "ec_target": ec_target,
        "depletion_per_step_at_target": depletion_at_target,
        "depletion_per_hour_at_target": depletion_at_target * (3600 / dt),
        "max_dose_mass_per_step": max_dose,
        "max_influx_per_step_immediate": max_influx_per_step,
        "max_sustained_influx_per_step_with_cooldown": max_sustained_influx,
        "authority_ratio_sustained_vs_depletion": max_sustained_influx / (depletion_at_target + 1e-12),
        "authority_ratio_peak_vs_depletion": max_influx_per_step / (depletion_at_target + 1e-12),
    }


def ki_sensitivity(config: Dict[str, Any], ki_values: List[float]) -> List[Dict[str, Any]]:
    results = []
    for ki in ki_values:
        score, metrics, _ = evaluate_pid(
            TUNED_KP, ki, TUNED_KD,
            config,
            n_episodes=3,
            episode_length=600,
            disturbance_mode="normal",
            seed=42,
            water_temp=18.0,
            initial_ec=1.0,
        )
        trace, _ = run_instrumented_episode(
            config, TUNED_KP, ki, TUNED_KD,
            episode_length=600, seed=42, water_temp=18.0, initial_ec=1.0,
        )
        ts = tracking_stats(trace)
        results.append({
            "ki": ki,
            "score": score,
            "ec_mae": metrics["ec_mae"],
            "settling_time": metrics["settling_time"],
            "nutrient_usage": metrics["nutrient_usage"],
            "overshoot": metrics["overshoot"],
            "final_error": ts["final_error"],
            "steady_state_error": ts["steady_state_error"],
            "mean_ec": ts["mean_ec"],
        })
    return results


def compare_ki_scores(config: Dict[str, Any]) -> Dict[str, Any]:
    """Score decomposition: Ki=0 vs Ki=0.05 at same Kp,Kd."""
    tune = config.get("pid_tuning", {})
    dt = config.get("simulation", {}).get("dt_seconds", 60.0)
    episode_length = 600
    horizon = episode_length * dt
    out = {}
    for ki in [0.0, 0.05, 0.10, 0.20]:
        trace, mdict = run_instrumented_episode(
            config, TUNED_KP, ki, TUNED_KD,
            episode_length=episode_length, seed=42, water_temp=18.0, initial_ec=1.0,
        )
        m = EpisodeMetrics(**{k: mdict[k] for k in EpisodeMetrics.__dataclass_fields__ if k in mdict})
        bd = score_breakdown(m, tune, horizon)
        ts = tracking_stats(trace)
        out[f"ki={ki}"] = {**bd, **ts, "nutrient_usage_raw": m.nutrient_usage}
    return out


def settling_time_audit(ec: np.ndarray, target: float, dt: float) -> Dict[str, Any]:
    band = 0.08
    err = np.abs(ec - target)
    never_in_band = not np.any(err < band)
    first_in_band = int(np.argmax(err < band)) if np.any(err < band) else -1
    settling = _settling_time(ec, target, dt, band=band)
    horizon = len(ec) * dt
    return {
        "settling_band": band,
        "horizon_seconds": horizon,
        "reported_settling_time": settling,
        "is_timeout": settling >= horizon - 1e-6,
        "never_entered_band": never_in_band,
        "first_time_in_band_s": first_in_band * dt if first_in_band >= 0 else None,
        "min_abs_error": float(np.min(err)),
        "final_abs_error": float(err[-1]),
        "mean_abs_error": float(np.mean(err)),
    }


def main() -> None:
    config_path = ROOT / "configs" / "pid_tune_quick.yaml"
    if not (ROOT / "configs" / "default.yaml").exists():
        config_path = ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    # Merge dynamics from default if quick config is sparse
    default = load_config(ROOT / "configs" / "default.yaml")
    for key in ["dynamics", "disturbances"]:
        if key not in config or len(config.get(key, {})) < len(default.get(key, {})):
            config[key] = default[key]

    ec_target = config["simulation"]["ec_target"]
    weights = config.get("pid_tuning", {}).get("weights", {})

    print("=" * 72)
    print("PID ROOT CAUSE INVESTIGATION")
    print(f"Config: {config_path.name}  |  EC target: {ec_target}")
    print(f"Tuned gains: Kp={TUNED_KP:.4f}  Ki={TUNED_KI}  Kd={TUNED_KD:.4f}")
    print("=" * 72)

    scenarios = [
        ("normal", 18.0, 1.0, 42),
        ("normal", 22.0, 1.2, 42),
        ("normal", 26.0, 1.2, 42),
        ("heatwave", 26.0, 1.0, 42),
        ("nutrient_depletion", 22.0, 1.2, 123),
    ]

    print("\n## PART 1: Setpoint Tracking Performance")
    all_traces = {}
    for mode, temp, ec0, seed in scenarios:
        trace, metrics = run_instrumented_episode(
            config, TUNED_KP, TUNED_KI, TUNED_KD,
            episode_length=600, disturbance_mode=mode,
            seed=seed, water_temp=temp, initial_ec=ec0,
        )
        ts = tracking_stats(trace)
        key = f"{mode}_T{temp}_EC0{ec0}_s{seed}"
        all_traces[key] = trace
        print(f"\n--- {key} ---")
        print(f"  mean_error (signed, sp-ec): {ts['mean_error_signed']:+.4f}")
        print(f"  mean|error|: {ts['mean_error_abs']:.4f}  median: {ts['median_error_abs']:.4f}")
        print(f"  steady_state_error: {ts['steady_state_error']:+.4f}  final: {ts['final_error']:+.4f}")
        print(f"  max|error|: {ts['max_error_abs']:.4f}")
        print(f"  mean EC: {ts['mean_ec']:.4f}  (target {ec_target})  range [{ts['min_ec']:.3f}, {ts['max_ec']:.3f}]")
        print(f"  time within 5% band: {ts['pct_time_within_5pct']:.1f}%")
        print(f"  time below setpoint: {ts['pct_time_below_setpoint']:.1f}%")
        print(f"  ec_mae (tuner): {metrics['ec_mae']:.4f}  settling: {metrics['settling_time']:.0f}s")

    ref = all_traces["normal_T18.0_EC01.0_s42"]
    print("\n## PART 2: MAE Interpretation (reference scenario)")
    print(f"  Target EC: {ec_target}")
    print(f"  Mean EC achieved: {np.mean(ref.ec):.4f}")
    print(f"  Median EC: {np.median(ref.ec):.4f}")
    print(f"  Persistent offset (mean setpoint-ec): {np.mean(ref.target - ref.ec):+.4f}")
    print(f"  MAE = {np.mean(np.abs(ref.ec - ec_target)):.4f}  ->  {100*np.mean(np.abs(ref.ec-ec_target))/ec_target:.1f}% of target")
    print(f"  Interpretation: {'FAILURE TO REGULATE' if np.mean(np.abs(ref.ec-ec_target)) > 0.15 else 'minor tracking error'}")

    print("\n## PART 3: Plant Depletion vs PD Steady-State (theoretical)")
    auth = authority_analysis(config, water_temp=22.0)
    for k, v in auth.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  PD at zero error -> u=0 -> no dosing -> depletion continues -> offset required")
    print("  Integral action required for persistent disturbance (continuous nutrient loss)")

    print("\n## PART 4: Optimization Score Breakdown (Ki comparison)")
    ki_cmp = compare_ki_scores(config)
    for label, data in ki_cmp.items():
        print(f"\n  {label}:")
        print(f"    tracking contribution: {data['tracking_ec_mae']:.4f}")
        print(f"    nutrient contribution: {data['nutrient_usage']:.4f}")
        print(f"    total score: {data['total']:.4f}")
        print(f"    mean_ec: {data['mean_ec']:.4f}  steady_state_error: {data['steady_state_error']:+.4f}")

    print("\n## PART 6: P/I/D Term Contributions (reference)")
    terms = pid_term_stats(ref)
    for k, v in terms.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n## PART 7: Controller Authority")
    print(f"  Sustained authority ratio: {auth['authority_ratio_sustained_vs_depletion']:.1f}x depletion")
    print(f"  Peak authority ratio: {auth['authority_ratio_peak_vs_depletion']:.1f}x depletion")
    print(f"  Actuator CAN supply enough nutrients; bottleneck is control law, not hardware limit")

    sim = config["simulation"]
    env_cfg = EnvironmentConfig(
        flowrate_max=sim["flowrate_max"],
        duration_max=sim["duration_max"],
    )
    print("\n## PART 8: Saturation Analysis")
    sat = saturation_stats(ref, env_cfg)
    for k, v in sat.items():
        print(f"  {k}: {v:.4f}")

    print("\n## PART 9: Settling Time Audit")
    st = settling_time_audit(ref.ec, ec_target, config["simulation"]["dt_seconds"])
    for k, v in st.items():
        print(f"  {k}: {v}")

    print("\n## PART 10: Ki Sensitivity Study")
    ki_vals = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
    ki_results = ki_sensitivity(config, ki_vals)
    print(f"  {'Ki':>6} {'MAE':>8} {'Score':>8} {'SS err':>10} {'Final err':>10} {'Nutrient':>10} {'Settle(s)':>10}")
    for r in ki_results:
        print(
            f"  {r['ki']:6.2f} {r['ec_mae']:8.4f} {r['score']:8.4f} "
            f"{r['steady_state_error']:+10.4f} {r['final_error']:+10.4f} "
            f"{r['nutrient_usage']:10.1f} {r['settling_time']:10.0f}"
        )

    print("\n## PART 11: Baseline Validity Summary")
    ref_ts = tracking_stats(ref)
    checks = [
        ("Reaches setpoint?", ref_ts["pct_time_within_5pct"] > 50),
        ("Maintains setpoint?", ref_ts["steady_state_error"] < 0.05 and ref_ts["pct_time_within_5pct"] > 80),
        ("Eliminates SS error?", abs(ref_ts["steady_state_error"]) < 0.03),
        ("Acceptable regulation?", ref_ts["mean_error_abs"] < 0.06),
    ]
    for q, ok in checks:
        print(f"  {q} {'YES' if ok else 'NO'}")
    print(f"  Valid baseline? {'NO' if ref_ts['mean_error_abs'] > 0.15 else 'MARGINAL' if ref_ts['mean_error_abs'] > 0.06 else 'YES'}")

    # Save summary JSON
    out_path = ROOT / "data" / "processed" / "pid_root_cause_report.json"
    report = {
        "gains": {"kp": TUNED_KP, "ki": TUNED_KI, "kd": TUNED_KD},
        "ec_target": ec_target,
        "reference_scenario": {k: tracking_stats(ref)[k] for k in tracking_stats(ref)},
        "authority": auth,
        "saturation": sat,
        "settling_audit": st,
        "pid_terms": terms,
        "ki_sensitivity": ki_results,
        "ki_score_comparison": ki_cmp,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
