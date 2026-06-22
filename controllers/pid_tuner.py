"""
Systematic PID gain tuning for nonlinear algae-tank closed-loop control.

Multi-stage search: coarse grid → local refinement → stochastic validation.
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from controllers.pid_controller import PIDConfig, PIDController, PIDGains
from simulation.disturbances import DisturbanceConfig, DisturbanceGenerator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.environment import AlgaeTankEnvironment, EnvironmentConfig


@dataclass
class EpisodeMetrics:
    """Per-episode control quality metrics."""

    ec_mae: float = 0.0
    overshoot: float = 0.0
    settling_time: float = 0.0
    steady_state_error: float = 0.0
    time_in_band: float = 0.0
    oscillation_amplitude: float = 0.0
    nutrient_usage: float = 0.0
    actuator_aggressiveness: float = 0.0
    collapse_fraction: float = 0.0
    instability_penalty: float = 0.0
    control_smoothness: float = 0.0
    regulatory_feasible: bool = False
    efficiency_score: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PIDCandidateResult:
    kp: float
    ki: float
    kd: float
    mean_score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    stage: str = "coarse"

    def gains_dict(self) -> dict:
        return {"kp": self.kp, "ki": self.ki, "kd": self.kd}


def _settling_time(ec: np.ndarray, target: float, dt: float, band: float = 0.08) -> float:
    """First time index after which |ec - target| stays within band (simplified)."""
    err = np.abs(ec - target)
    for i in range(len(ec)):
        if np.all(err[i:] < band):
            return float(i * dt)
    return float(len(ec) * dt)


def _regulatory_constraints(config: Dict[str, Any]) -> Dict[str, float]:
    """Hard regulatory thresholds — tuning rejects candidates that violate any limit."""
    rc = config.get("regulatory_constraints", {})
    return {
        "max_ec_mae": rc.get("max_ec_mae", 0.08),
        "max_steady_state_error": rc.get("max_steady_state_error", 0.05),
        "min_time_in_band": rc.get("min_time_in_band", 0.90),
        "max_collapse_fraction": rc.get("max_collapse_fraction", 0.0),
        "reject_settling_timeout": rc.get("reject_settling_timeout", True),
        "target_band_pct": rc.get("target_band_pct", 0.05),
        "settling_band": rc.get("settling_band", 0.08),
        "steady_state_tail_fraction": rc.get("steady_state_tail_fraction", 0.35),
    }


def _passes_regulatory_constraints(
    metrics: EpisodeMetrics,
    constraints: Dict[str, float],
    horizon_seconds: float,
) -> bool:
    """Stage 1: candidate must regulate EC before efficiency is considered."""
    if metrics.ec_mae > constraints["max_ec_mae"]:
        return False
    if abs(metrics.steady_state_error) > constraints["max_steady_state_error"]:
        return False
    if metrics.time_in_band < constraints["min_time_in_band"]:
        return False
    if metrics.collapse_fraction > constraints["max_collapse_fraction"]:
        return False
    if constraints["reject_settling_timeout"] and metrics.settling_time >= horizon_seconds - 1e-6:
        return False
    return True


def _efficiency_score(metrics: EpisodeMetrics, weights: Dict[str, float]) -> float:
    """Stage 2: among valid regulators, minimize resource use and control effort."""
    return (
        weights.get("nutrient_usage", 0.4) * metrics.nutrient_usage * 0.01
        + weights.get("smoothness", 0.2) * metrics.control_smoothness
        + weights.get("aggressive_control", 0.3) * metrics.actuator_aggressiveness
    )


def _candidate_score(
    metrics: EpisodeMetrics,
    tune_cfg: Dict[str, Any],
    horizon_seconds: float,
) -> Tuple[float, bool, float]:
    """
    Two-stage score: regulatory feasibility gate, then efficiency objective.
    Returns (score, regulatory_feasible, efficiency_score).
    """
    constraints = _regulatory_constraints(tune_cfg)
    feasible = _passes_regulatory_constraints(metrics, constraints, horizon_seconds)
    if not feasible:
        return float("inf"), False, float("inf")
    eff_weights = tune_cfg.get("efficiency_weights", tune_cfg.get("weights", {}))
    eff = _efficiency_score(metrics, eff_weights)
    return eff, True, eff


def _metrics_from_trace(
    ec: np.ndarray,
    flowrate: np.ndarray,
    duration: np.ndarray,
    target: float,
    dt: float,
    ec_safe_min: float,
    tune_cfg: Dict[str, Any],
) -> EpisodeMetrics:
    constraints = _regulatory_constraints(tune_cfg)
    band = constraints["target_band_pct"] * target
    horizon_seconds = float(len(ec) * dt)

    errors = ec - target
    ec_mae = float(np.mean(np.abs(errors)))
    overshoot = float(np.max(np.maximum(0.0, ec - target)))
    settling = _settling_time(
        ec, target, dt, band=constraints["settling_band"]
    )
    tail_n = max(20, int(len(ec) * constraints["steady_state_tail_fraction"]))
    tail = ec[-tail_n:]
    tail_errors = tail - target
    steady_state_error = float(np.mean(target - tail))
    time_in_band = float(np.mean(np.abs(tail_errors) < band))
    oscillation = float(np.std(tail) + 0.5 * (np.max(tail) - np.min(tail)))
    nutrient = float(np.sum(flowrate * duration / 60.0))
    d_fr = np.diff(flowrate, prepend=flowrate[0])
    d_dur = np.diff(duration, prepend=duration[0])
    aggressiveness = float(np.mean(np.abs(d_fr)) + 0.1 * np.mean(np.abs(d_dur)))
    smoothness = float(np.mean(d_fr**2) + 0.05 * np.mean(d_dur**2))
    collapse_frac = float(np.mean(ec < ec_safe_min))
    instability = float(np.mean(np.maximum(0.0, ec - target - 0.35) ** 2))
    m = EpisodeMetrics(
        ec_mae=ec_mae,
        overshoot=overshoot,
        settling_time=settling,
        steady_state_error=steady_state_error,
        time_in_band=time_in_band,
        oscillation_amplitude=oscillation,
        nutrient_usage=nutrient,
        actuator_aggressiveness=aggressiveness,
        collapse_fraction=collapse_frac,
        instability_penalty=instability,
        control_smoothness=smoothness,
    )
    score, feasible, eff = _candidate_score(m, tune_cfg, horizon_seconds)
    m.regulatory_feasible = feasible
    m.efficiency_score = eff
    m.score = score
    return m


def evaluate_pid(
    kp: float,
    ki: float,
    kd: float,
    config: Dict[str, Any],
    n_episodes: int = 10,
    episode_length: int = 500,
    disturbance_mode: str = "normal",
    seed: int = 42,
    water_temp: Optional[float] = None,
    initial_ec: Optional[float] = None,
    return_traces: bool = False,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """
    Run closed-loop PID episodes; return (mean_score, aggregated_metrics, optional last trace).
    """
    sim = config.get("simulation", {})
    dyn = config.get("dynamics", {})
    tune = config.get("pid_tuning", {})
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
    episode_scores: List[float] = []
    agg: Dict[str, List[float]] = {
        k: []
        for k in [
            "ec_mae",
            "overshoot",
            "settling_time",
            "steady_state_error",
            "time_in_band",
            "oscillation_amplitude",
            "nutrient_usage",
            "actuator_aggressiveness",
            "collapse_fraction",
            "instability_penalty",
            "control_smoothness",
            "efficiency_score",
        ]
    }
    regulatory_pass: List[bool] = []
    last_trace = None

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        ep_rng = np.random.default_rng(ep_seed)
        dist_gen = DisturbanceGenerator(dist_cfg, ep_rng)
        schedule = dist_gen.build_schedule(disturbance_mode, episode_length)

        temp = water_temp if water_temp is not None else params.ambient_temp_mean + ep_rng.normal(0, 1.5)
        ec0 = initial_ec if initial_ec is not None else ec_target + ep_rng.normal(0, 0.1)
        s0 = TankState.create_initial(params, ec=ec0, water_temp=temp, rng=ep_rng)

        env = AlgaeTankEnvironment(
            env_cfg, params, rng=ep_rng, disturbance_generator=dist_gen
        )
        env.reset(initial_state=s0, disturbance_schedule=schedule)

        pid = PIDController(
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
        for t in range(episode_length):
            ec_val = env.state.ec if env.state else ec_target
            fr, dur = pid.compute(ec_val)
            ec_h.append(ec_val)
            fr_h.append(fr)
            dur_h.append(dur)
            env.step((fr, dur))

        ec_arr = np.array(ec_h)
        fr_arr = np.array(fr_h)
        dur_arr = np.array(dur_h)
        m = _metrics_from_trace(
            ec_arr, fr_arr, dur_arr, ec_target, dt, env_cfg.ec_safe_min, tune
        )
        episode_scores.append(m.score)
        regulatory_pass.append(m.regulatory_feasible)
        for k in agg:
            agg[k].append(getattr(m, k))
        if return_traces and ep == n_episodes - 1:
            last_trace = {"ec": ec_arr, "flowrate": fr_arr, "duration": dur_arr, "target": ec_target, "dt": dt}

    if not all(regulatory_pass):
        mean_score = float("inf")
    else:
        mean_score = float(np.mean(episode_scores))
    mean_metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    mean_metrics["score_std"] = float(np.std(episode_scores))
    mean_metrics["regulatory_feasible"] = all(regulatory_pass)
    mean_metrics["score"] = mean_score
    if return_traces:
        return mean_score, mean_metrics, last_trace
    return mean_score, mean_metrics, None


def _log_grid(low: float, high: float, n: int) -> np.ndarray:
    if low <= 0 and high > 0:
        low = max(low, 1e-4) if low == 0 else low
    if low > 0 and high > low:
        return np.geomspace(max(low, 1e-4), high, n)
    return np.linspace(low, high, n)


class PIDTuner:
    """Coarse grid → refinement → multi-scenario validation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.tune_cfg = config.get("pid_tuning", {})
        self.results: Dict[str, Any] = {
            "coarse": [],
            "refinement": [],
            "validation": [],
            "best": None,
        }

    def _coarse_candidates(self) -> List[Tuple[float, float, float]]:
        cs = self.tune_cfg.get("coarse_search", {})
        kp_r = cs.get("kp_range", [0.1, 10.0])
        ki_r = cs.get("ki_range", [0.05, 0.30])
        kd_r = cs.get("kd_range", [0.0, 5.0])
        n_kp = cs.get("n_kp", 8)
        n_ki = cs.get("n_ki", 6)
        n_kd = cs.get("n_kd", 6)
        max_eval = cs.get("max_evaluations", 80)

        kps = _log_grid(kp_r[0], kp_r[1], n_kp)
        kis = np.linspace(ki_r[0], ki_r[1], n_ki)
        kds = _log_grid(max(kd_r[0], 1e-4), kd_r[1], n_kd) if kd_r[1] > 0 else np.array([0.0])

        grid = list(itertools.product(kps, kis, kds))
        if len(grid) > max_eval:
            rng = np.random.default_rng(self.config.get("seed", 42))
            idx = rng.choice(len(grid), max_eval, replace=False)
            grid = [grid[i] for i in idx]
        return grid

    def run_coarse_search(self) -> List[PIDCandidateResult]:
        eval_cfg = self.tune_cfg.get("evaluation", {})
        n_ep = eval_cfg.get("n_episodes_coarse", 3)
        length = eval_cfg.get("episode_length_coarse", 400)
        modes = eval_cfg.get("disturbance_modes_coarse", ["normal", "heatwave"])

        candidates = self._coarse_candidates()
        results: List[PIDCandidateResult] = []
        print(f"Coarse search: {len(candidates)} gain combinations")

        for i, (kp, ki, kd) in enumerate(candidates):
            scores = []
            for mode in modes:
                sc, _, _ = evaluate_pid(
                    kp, ki, kd, self.config,
                    n_episodes=n_ep,
                    episode_length=length,
                    disturbance_mode=mode,
                    seed=self.config.get("seed", 42) + i,
                )
                scores.append(sc)
            mean_sc = float(np.mean(scores))
            results.append(
                PIDCandidateResult(
                    kp=float(kp), ki=float(ki), kd=float(kd),
                    mean_score=mean_sc, stage="coarse",
                )
            )
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(candidates)}] best so far: {min(results, key=lambda r: r.mean_score).mean_score:.4f}")

        results.sort(key=lambda r: r.mean_score)
        feasible = [r for r in results if np.isfinite(r.mean_score)]
        print(
            f"  Regulatory pass: {len(feasible)}/{len(results)} candidates "
            f"(Stage 1 feasibility filter)"
        )
        self.results["coarse"] = [
            r.gains_dict()
            | {
                "score": r.mean_score,
                "regulatory_feasible": np.isfinite(r.mean_score),
            }
            for r in results
        ]
        return results

    def run_refinement(
        self, top_from_coarse: List[PIDCandidateResult], top_n: int = 10
    ) -> List[PIDCandidateResult]:
        ref = self.tune_cfg.get("refinement", {})
        top_n = ref.get("top_candidates", top_n)
        step_kp = ref.get("step_kp", 0.4)
        step_ki = ref.get("step_ki", 0.02)
        step_kd = ref.get("step_kd", 0.15)
        eval_cfg = self.tune_cfg.get("evaluation", {})
        n_ep = eval_cfg.get("n_episodes", 5)
        length = eval_cfg.get("episode_length", 800)
        modes = eval_cfg.get("disturbance_modes", ["normal", "heatwave", "nutrient_depletion"])

        cs = self.tune_cfg.get("coarse_search", {})
        ki_min, ki_max = cs.get("ki_range", [0.05, 0.30])
        seeds = [r for r in top_from_coarse[:top_n] if np.isfinite(r.mean_score)]
        if not seeds:
            seeds = top_from_coarse[:top_n]
        refined: List[PIDCandidateResult] = []

        for base in seeds:
            for dk, di, dd in itertools.product([-1, 0, 1], repeat=3):
                if dk == di == dd == 0:
                    continue
                kp = max(0.05, base.kp + dk * step_kp)
                ki = float(np.clip(base.ki + di * step_ki, ki_min, ki_max))
                kd = max(0.0, base.kd + dd * step_kd)
                scores = []
                for mode in modes:
                    sc, _, _ = evaluate_pid(
                        kp, ki, kd, self.config,
                        n_episodes=max(2, n_ep // 2),
                        episode_length=length,
                        disturbance_mode=mode,
                        seed=42,
                    )
                    scores.append(sc)
                refined.append(
                    PIDCandidateResult(
                        kp=kp, ki=ki, kd=kd,
                        mean_score=float(np.mean(scores)),
                        stage="refinement",
                    )
                )

        refined.sort(key=lambda r: r.mean_score)
        self.results["refinement"] = [
            r.gains_dict()
            | {
                "score": r.mean_score,
                "regulatory_feasible": np.isfinite(r.mean_score),
            }
            for r in refined[:50]
        ]
        return refined

    def run_stochastic_validation(
        self, candidate: PIDCandidateResult
    ) -> Dict[str, Any]:
        val = self.tune_cfg.get("validation", {})
        n_ep = val.get("n_episodes", 8)
        length = val.get("episode_length", 2000)
        long_length = val.get("long_horizon_length", 3000)
        modes = val.get(
            "disturbance_modes",
            ["normal", "heatwave", "cold_shock", "nutrient_depletion", "actuator_failure", "sediment"],
        )
        temps = val.get("water_temps", [17.0, 22.0, 28.0, 30.0])
        initial_ecs = val.get("initial_ecs", [0.9, 1.1, 1.2, 1.35])
        seeds = val.get("seeds", [42, 123, 456, 789])

        all_scores: List[float] = []
        breakdown: List[dict] = []

        for mode in modes:
            for temp in temps:
                for ec0 in initial_ecs:
                    for seed in seeds:
                        sc, metrics, _ = evaluate_pid(
                            candidate.kp,
                            candidate.ki,
                            candidate.kd,
                            self.config,
                            n_episodes=1,
                            episode_length=length,
                            disturbance_mode=mode,
                            seed=seed,
                            water_temp=temp,
                            initial_ec=ec0,
                        )
                        all_scores.append(sc)
                        breakdown.append({
                            "mode": mode,
                            "temp": temp,
                            "initial_ec": ec0,
                            "seed": seed,
                            "score": sc,
                            **metrics,
                        })

        long_scores = []
        for seed in seeds[:3]:
            sc, _, _ = evaluate_pid(
                candidate.kp, candidate.ki, candidate.kd,
                self.config,
                n_episodes=1,
                episode_length=long_length,
                disturbance_mode="normal",
                seed=seed,
            )
            long_scores.append(sc)

        report = {
            "gains": candidate.gains_dict(),
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "worst_score": float(np.max(all_scores)),
            "long_horizon_mean": float(np.mean(long_scores)),
            "n_evaluations": len(all_scores),
            "breakdown_sample": breakdown[:30],
        }
        self.results["validation"] = report
        return report

    def _select_best(self, candidates: List[PIDCandidateResult]) -> PIDCandidateResult:
        feasible = [c for c in candidates if np.isfinite(c.mean_score)]
        if feasible:
            return feasible[0]
        print(
            "WARNING: No candidates passed regulatory constraints. "
            "Best infeasible candidate selected — relax constraints or extend horizon."
        )
        return candidates[0]

    def run_full_tuning(self) -> Dict[str, Any]:
        """Execute all stages and select best gains."""
        coarse = self.run_coarse_search()
        best_coarse = self._select_best(coarse)
        print(
            f"Coarse best: Kp={best_coarse.kp:.3f} Ki={best_coarse.ki:.4f} "
            f"Kd={best_coarse.kd:.3f} score={best_coarse.mean_score:.4f}"
        )

        refined = self.run_refinement(coarse)
        pool = sorted(coarse + refined, key=lambda r: r.mean_score)
        best_ref = self._select_best(pool)
        if refined:
            ref_best = self._select_best(refined)
            print(
                f"Refinement best: Kp={ref_best.kp:.3f} Ki={ref_best.ki:.4f} "
                f"Kd={ref_best.kd:.3f} score={ref_best.mean_score:.4f}"
            )
        else:
            print("Refinement: skipped")
        print(
            f"Global best:     Kp={best_ref.kp:.3f} Ki={best_ref.ki:.4f} "
            f"Kd={best_ref.kd:.3f} score={best_ref.mean_score:.4f}"
        )

        val_report = self.run_stochastic_validation(best_ref)
        print(f"Validation mean score: {val_report['mean_score']:.4f} (std={val_report['std_score']:.4f})")

        _, metrics, trace = evaluate_pid(
            best_ref.kp, best_ref.ki, best_ref.kd,
            self.config,
            n_episodes=1,
            episode_length=self.tune_cfg.get("evaluation", {}).get("episode_length", 500),
            disturbance_mode="normal",
            seed=42,
            return_traces=True,
        )

        self.results["best"] = {
            "kp": best_ref.kp,
            "ki": best_ref.ki,
            "kd": best_ref.kd,
            "coarse_score": best_coarse.mean_score,
            "refinement_score": best_ref.mean_score,
            "regulatory_feasible": np.isfinite(best_ref.mean_score),
            "validation": val_report,
            "metrics": metrics,
        }
        self.results["reference_trace"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in (trace or {}).items()
        }
        return self.results

    def save_results(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _json_safe(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(v) for v in obj]
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
                return None if np.isnan(obj) else "inf"
            return obj

        with open(path, "w") as f:
            json.dump(_json_safe(self.results), f, indent=2)

    def update_config_evaluation_pid(self) -> None:
        """Write best gains into config evaluation.pid (in-memory)."""
        if self.results.get("best"):
            b = self.results["best"]
            self.config.setdefault("evaluation", {}).setdefault("pid", {})
            self.config["evaluation"]["pid"] = {
                "kp": b["kp"],
                "ki": b["ki"],
                "kd": b["kd"],
            }


def run_pid_tuning(
    config: Dict[str, Any],
    output_json: Path,
    figures_dir: Path,
) -> Dict[str, Any]:
    """Entry point: tune, plot, save."""
    from visualization.pid_tuning_plots import PIDTuningPlotter

    tuner = PIDTuner(config)
    results = tuner.run_full_tuning()
    tuner.save_results(output_json)
    tuner.update_config_evaluation_pid()

    plotter = PIDTuningPlotter(figures_dir)
    plotter.plot_all(results, config)

    return results
