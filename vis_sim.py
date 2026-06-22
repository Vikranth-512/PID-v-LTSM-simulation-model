import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("figures/paper/label_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    "data/processed/all_trajectories_labeled.csv"
)

EC_TARGET = 1.2

# Consistent with feature engineering
df["ec_error"] = EC_TARGET - df["ec"]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    constrained_layout=True
)

# --------------------------------------------------
# Plot 1: EC Error vs Optimal Flowrate
# --------------------------------------------------

axes[0].scatter(
    df["ec_error"],
    df["optimal_flowrate"],
    alpha=0.08,
    s=8
)

axes[0].axvline(
    0,
    linestyle="--",
    linewidth=1
)

axes[0].set_title(
    "Expert Flowrate Selection vs EC Error",
    fontsize=13,
    weight="bold"
)

axes[0].set_xlabel(
    "EC Error (Target - Current EC)"
)

axes[0].set_ylabel(
    "Optimal Flowrate"
)

axes[0].grid(alpha=0.3)

# --------------------------------------------------
# Plot 2: EC Error vs Optimal Duration
# --------------------------------------------------

axes[1].scatter(
    df["ec_error"],
    df["optimal_duration"],
    alpha=0.08,
    s=8
)

axes[1].axvline(
    0,
    linestyle="--",
    linewidth=1
)

axes[1].set_title(
    "Expert Duration Selection vs EC Error",
    fontsize=13,
    weight="bold"
)

axes[1].set_xlabel(
    "EC Error (Target - Current EC)"
)

axes[1].set_ylabel(
    "Optimal Duration"
)

axes[1].grid(alpha=0.3)

plt.savefig(
    OUTPUT_DIR / "ec_error_vs_expert_actions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(OUTPUT_DIR / "ec_error_vs_expert_actions.png")