"""
Synthetic trajectory generation for policy learning.

Produces CSV trajectories with sensor states, hidden biological variables,
actuator history, and structured disturbances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from simulation.disturbances import DisturbanceConfig, DisturbanceGenerator
from simulation.dynamics import TankDynamicsParams, TankState
from simulation.environment import AlgaeTankEnvironment, EnvironmentConfig
from utils.naming import get_trajectory_index_padding, trajectory_filename


class SyntheticTrajectoryGenerator:
    """Generate diverse operational scenarios for supervised policy learning."""

    SCENARIO_TYPES = [
        "normal",
        "ec_drop",
        "temp_spike",
        "noisy_sensors",
        "delayed_response",
        "saturation",
        "nutrient_depletion",
        "actuator_failure",
        "heatwave",
        "cold_shock",
        "sediment",
    ]

    def __init__(self, config: Dict[str, Any], seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        sim = config.get("simulation", {})
        dyn = config.get("dynamics", {})
        dist_cfg = config.get("disturbances", {})

        self.ec_target = sim.get("ec_target", 1.2)
        self.env_config = EnvironmentConfig(
            dt_seconds=sim.get("dt_seconds", 60.0),
            ec_target=self.ec_target,
            ec_safe_min=sim.get("ec_safe_min", 0.4),
            ec_safe_max=sim.get("ec_safe_max", 2.5),
            flowrate_min=sim.get("flowrate_min", 0.0),
            flowrate_max=sim.get("flowrate_max", 5.0),
            duration_min=sim.get("duration_min", 0.0),
            duration_max=sim.get("duration_max", 30.0),
            min_time_between_doses=sim.get("min_time_between_doses", 120.0),
            noise_std=sim.get("noise_std"),
        )

        self.base_params = TankDynamicsParams.from_config(dyn, ec_target=self.ec_target)
        self.disturbance_config = DisturbanceConfig.from_config(dist_cfg)

        self.length_min = sim.get("trajectory_length_min", 100)
        self.length_max = sim.get("trajectory_length_max", 500)
        self.num_trajectories = sim.get("num_trajectories", 200)
        self.trajectory_index_padding = get_trajectory_index_padding(config)

    def _random_open_loop_actions(self, length: int) -> List[tuple]:
        """Exploratory actions for state coverage (not optimal labels)."""
        actions = []
        for t in range(length):
            if self.rng.random() < 0.22:
                fr = self.rng.uniform(0.5, self.env_config.flowrate_max)
                dur = self.rng.uniform(8, self.env_config.duration_max)
            else:
                fr, dur = 0.0, 0.0
            actions.append((fr, dur))
        return actions

    def generate_trajectory(
        self,
        traj_id: int,
        scenario: Optional[str] = None,
        length: Optional[int] = None,
    ) -> pd.DataFrame:
        scenario = scenario or self.rng.choice(self.SCENARIO_TYPES)
        length = length or int(self.rng.integers(self.length_min, self.length_max + 1))

        params = TankDynamicsParams.sample_random(self.base_params, self.rng)
        noise_mult = 3.0 if scenario == "noisy_sensors" else 1.0
        env_cfg = EnvironmentConfig(
            dt_seconds=self.env_config.dt_seconds,
            ec_target=self.env_config.ec_target,
            ec_safe_min=self.env_config.ec_safe_min,
            ec_safe_max=self.env_config.ec_safe_max,
            flowrate_min=self.env_config.flowrate_min,
            flowrate_max=self.env_config.flowrate_max,
            duration_min=self.env_config.duration_min,
            duration_max=self.env_config.duration_max,
            min_time_between_doses=self.env_config.min_time_between_doses,
            noise_std={
                k: v * noise_mult for k, v in self.env_config.noise_std.items()
            },
        )

        if scenario == "delayed_response":
            params = TankDynamicsParams(
                **{
                    **params.__dict__,
                    "immediate_absorption_fraction": 0.08,
                    "delay_kernel": (0.1, 0.25, 0.35, 0.30),
                }
            )

        dist_gen = DisturbanceGenerator(self.disturbance_config, self.rng)
        disturbance_schedule = dist_gen.build_schedule(scenario, length)

        env = AlgaeTankEnvironment(
            env_cfg, params, rng=self.rng, disturbance_generator=dist_gen
        )
        actions = self._random_open_loop_actions(length)

        rows: List[Dict] = []
        obs = env.reset(disturbance_schedule=disturbance_schedule)
        keys = env.observation_keys

        for t in range(length):
            true = env.state.as_dict() if env.state else {}
            fr, dur = actions[t]
            row = {
                "trajectory_id": traj_id,
                "timestep": t,
                "scenario": scenario,
                **{k: obs[i] for i, k in enumerate(keys)},
                **{f"true_{k}": v for k, v in true.items()},
                "flowrate": fr,
                "duration": dur,
            }
            rows.append(row)
            obs, _ = env.step((fr, dur))

        return pd.DataFrame(rows)

    def generate_dataset(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_dfs = []
        meta = {
            "trajectories": [],
            "config_snapshot": self.config,
            "dynamics_version": "v2_active_equilibrium",
            "trajectory_index_padding": self.trajectory_index_padding,
        }

        for i in range(self.num_trajectories):
            df = self.generate_trajectory(i)
            path = output_dir / trajectory_filename(
                i, padding=self.trajectory_index_padding
            )
            df.to_csv(path, index=False)
            all_dfs.append(df)
            meta["trajectories"].append(
                {
                    "id": i,
                    "file": path.name,
                    "length": len(df),
                    "scenario": df["scenario"].iloc[0],
                }
            )

        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / "all_trajectories.csv"
        combined.to_csv(combined_path, index=False)

        with open(output_dir / "dataset_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        return combined_path
