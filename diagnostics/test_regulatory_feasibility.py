"""Quick check: regulatory constraints vs episode length and Ki."""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.pid_tuner import evaluate_pid

config = yaml.safe_load(open(ROOT / "configs" / "pid_tune_quick.yaml"))
default = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
config["dynamics"] = default["dynamics"]
config["disturbances"] = default["disturbances"]

for length in [600, 1500, 2500]:
    print(f"=== episode_length={length} ===")
    for ki in [0.10, 0.15, 0.20, 0.25]:
        sc, m, _ = evaluate_pid(
            1.732, ki, 0.965, config,
            n_episodes=1,
            episode_length=length,
            disturbance_mode="normal",
            seed=42,
            water_temp=22.0,
            initial_ec=1.0,
        )
        ok = np.isfinite(sc)
        print(
            f"  ki={ki:.2f} feasible={ok} mae={m['ec_mae']:.3f} "
            f"ss={m['steady_state_error']:+.3f} band={m['time_in_band']:.2f} "
            f"settle={m['settling_time']:.0f}"
        )
