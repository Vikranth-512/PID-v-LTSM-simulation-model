from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde

# ============================================================
# Configuration
# ============================================================

CSV_PATH = "data/processed/all_trajectories_labeled.csv"

OUTPUT_DIR = Path("figures/paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EC = 1.2

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

sns.set_theme(style="whitegrid")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df):,} samples")
print(f"Trajectories: {df['trajectory_id'].nunique()}")

df["ec_error"] = TARGET_EC - df["ec"]

# ============================================================
# Select representative trajectory
# ============================================================

traj_lengths = df.groupby("trajectory_id").size()

representative_id = traj_lengths.idxmax()

traj = (
    df[df["trajectory_id"] == representative_id]
    .sort_values("timestep")
    .reset_index(drop=True)
)

print(f"Using trajectory {representative_id}")

# ============================================================
# FIGURE 1
# Example Multivariate Synthetic Trajectory
# ============================================================

fig, ax1 = plt.subplots(figsize=(11, 6))

ax1.plot(
    traj["timestep"],
    traj["ec"],
    label="EC"
)

ax1.set_xlabel("Timestep")
ax1.set_ylabel("EC")

ax2 = ax1.twinx()

ax2.plot(
    traj["timestep"],
    traj["water_temp"],
    linestyle="--",
    label="Water Temperature"
)

ax2.plot(
    traj["timestep"],
    traj["turbidity"],
    linestyle=":",
    label="Turbidity"
)

ax2.set_ylabel("Temperature / Turbidity")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left"
)

plt.title("Figure 1. Example Multivariate Synthetic Trajectory")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure1_multivariate_trajectory.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2
# EC Dynamics Under Nutrient Dosing
# ============================================================

fig, ax1 = plt.subplots(figsize=(11, 6))

ax1.plot(
    traj["timestep"],
    traj["ec"],
    linewidth=2,
    label="EC"
)

ax1.axhline(
    TARGET_EC,
    linestyle="--",
    label="Target EC"
)

ax1.set_xlabel("Timestep")
ax1.set_ylabel("EC")

dose = traj["flowrate"] * traj["duration"]

dose_scaled = (
    dose / dose.max() * traj["ec"].max()
    if dose.max() > 0
    else dose
)

ax1.bar(
    traj["timestep"],
    dose_scaled,
    alpha=0.25,
    width=1,
    label="Dosing Events"
)

ax1.legend()

plt.title("Figure 2. EC Dynamics Under Nutrient Dosing")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure2_ec_dosing_response.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 3
# Delayed Nutrient Assimilation Mechanism
# ============================================================

t = traj["timestep"].values

dose = (
    traj["flowrate"].values *
    traj["duration"].values
)

ec_response = traj["ec"].values

immediate_fraction = 0.28

delay_kernel = np.array([
    0.15,
    0.25,
    0.30,
    0.20,
    0.10
])

immediate = dose * immediate_fraction

delayed = np.convolve(
    dose * (1.0 - immediate_fraction),
    delay_kernel,
    mode="same"
)

dose_smooth = (
    pd.Series(dose)
    .rolling(window=7, center=True, min_periods=1)
    .mean()
)

immediate_smooth = (
    pd.Series(immediate)
    .rolling(window=7, center=True, min_periods=1)
    .mean()
)

delayed_smooth = (
    pd.Series(delayed)
    .rolling(window=7, center=True, min_periods=1)
    .mean()
)

delayed_norm = delayed_smooth / delayed_smooth.max()

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

ax1.plot(
    t,
    dose_smooth,
    linewidth=2.5,
    label="Applied Dose"
)

ax1.plot(
    t,
    immediate_smooth,
    linewidth=2,
    label="Immediate Absorption (28%)"
)

ax1.plot(
    t,
    delayed_norm,
    linewidth=2,
    linestyle="--",
    label="Delayed Assimilation (Normalized)"
)

ax1.set_ylabel("Dose Magnitude")

ax1.set_title(
    "Figure 3. Delayed Nutrient Assimilation Mechanism",
    fontsize=16,
    pad=15
)

ax1.grid(alpha=0.3)
ax1.legend(loc="upper right")

ax2.plot(
    t,
    ec_response,
    linewidth=2.5,
    label="EC Response"
)

ax2.axhline(
    y=1.2,
    linestyle="--",
    linewidth=2,
    label="Target EC (1.2 mS/cm)"
)

ax2.set_xlabel("Timestep")
ax2.set_ylabel("EC")

ax2.grid(alpha=0.3)
ax2.legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_3_delayed_assimilation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 5
# EC Distribution
# ============================================================

plt.figure(figsize=(10, 6))

ec_values = df["ec"].dropna()

kde = gaussian_kde(ec_values)

x = np.linspace(
    ec_values.min(),
    ec_values.max(),
    600
)

plt.hist(
    ec_values,
    bins=50,
    density=True,
    alpha=0.4,
    edgecolor="black",
    linewidth=0.5,
    label="Observed EC"
)

plt.plot(
    x,
    kde(x),
    linewidth=3,
    label="Kernel Density Estimate"
)

plt.axvline(
    1.2,
    linestyle="--",
    linewidth=2,
    label="Target EC"
)

plt.title("Figure 5. Electrical Conductivity Distribution")

plt.xlabel("Electrical Conductivity (mS/cm)")
plt.ylabel("Probability Density")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_5_ec_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 6
# Flowrate Labels
# ============================================================

plt.figure()

plt.hist(
    df["optimal_flowrate"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Optimal Flowrate")
plt.ylabel("Frequency")

plt.title("Figure 6. Optimal Flowrate Label Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure6_flowrate_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 7
# Duration Labels
# ============================================================

plt.figure()

plt.hist(
    df["optimal_duration"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Optimal Duration")
plt.ylabel("Frequency")

plt.title("Figure 7. Optimal Duration Label Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure7_duration_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 8
# Action Density Heatmap
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist2d(
    df["optimal_flowrate"],
    df["optimal_duration"],
    bins=30
)

plt.colorbar(label="Count")

plt.xlabel("Optimal Flowrate")
plt.ylabel("Optimal Duration")

plt.title(
    "Figure 8. Label Action Space Density"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure8_action_density_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 9
# Trajectory Length Distribution
# ============================================================

trajectory_lengths = (
    df.groupby("trajectory_id")
    .size()
)

plt.figure()

plt.hist(
    trajectory_lengths,
    bins=20,
    edgecolor="black"
)

plt.xlabel("Trajectory Length (timesteps)")
plt.ylabel("Frequency")

plt.title(
    "Figure 9. Trajectory Length Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure9_trajectory_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 10
# Expert Flowrate Selection vs EC Error
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

sns.scatterplot(
    data=df.sample(min(12000, len(df)), random_state=42),
    x="ec_error",
    y="optimal_flowrate",
    alpha=0.25,
    s=18,
    linewidth=0,
    ax=ax,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
)

ax.set_title(
    "Figure 10. Expert Flowrate Selection vs EC Error",
    fontsize=14,
    pad=12,
)

ax.set_xlabel(
    "EC Error (Target EC − Current EC)",
    fontsize=12,
)

ax.set_ylabel(
    "Optimal Flowrate (mL/min)",
    fontsize=12,
)

ax.set_ylim(-0.2, 5.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_10_flowrate_vs_ec_error.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ============================================================
# FIGURE 11
# Expert Duration Selection vs EC Error
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

sns.scatterplot(
    data=df.sample(min(12000, len(df)), random_state=42),
    x="ec_error",
    y="optimal_duration",
    alpha=0.25,
    s=18,
    linewidth=0,
    ax=ax,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
)

ax.set_title(
    "Figure 11. Expert Duration Selection vs EC Error",
    fontsize=14,
    pad=12,
)

ax.set_xlabel(
    "EC Error (Target EC − Current EC)",
    fontsize=12,
)

ax.set_ylabel(
    "Optimal Duration (s)",
    fontsize=12,
)

ax.set_ylim(-1, 31)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_11_duration_vs_ec_error.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved figures to:")
print(OUTPUT_DIR)