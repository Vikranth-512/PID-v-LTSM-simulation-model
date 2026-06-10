# visualize_controller_comparison.py

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("figures") / "controller_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open("C:/Users/retro/CascadeProjects/ml/data/processed/evaluation_results.json", "r") as f:
    results = json.load(f)

scenarios = []
pid_mae = []
lstm_mae = []

pid_dose = []
lstm_dose = []

pid_stability = []
lstm_stability = []

for scenario, data in results["closed_loop"].items():

    scenarios.append(scenario)

    pid_mae.append(data["pid"]["ec_mae"])
    lstm_mae.append(data["lstm"]["ec_mae"])

    pid_dose.append(data["pid"]["cumulative_dosing_cost"])
    lstm_dose.append(data["lstm"]["cumulative_dosing_cost"])

    pid_stability.append(data["pid"]["stability_variance"])
    lstm_stability.append(data["lstm"]["stability_variance"])

# =====================================================
# FIGURE 1
# EC MAE
# =====================================================

x = np.arange(len(scenarios))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x-width/2, pid_mae, width, label="PID")
plt.bar(x+width/2, lstm_mae, width, label="LSTM")

plt.xticks(x, scenarios, rotation=20)
plt.ylabel("EC MAE")
plt.title("EC Tracking Error Across Disturbance Scenarios")
plt.legend()
plt.tight_layout()
plt.savefig("figures/controller_comparison/ec_mae_comparison.png", dpi=300)
plt.close()

# =====================================================
# FIGURE 2
# Nutrient Consumption
# =====================================================

plt.figure(figsize=(10,5))
plt.bar(x-width/2, pid_dose, width, label="PID")
plt.bar(x+width/2, lstm_dose, width, label="LSTM")

plt.xticks(x, scenarios, rotation=20)
plt.ylabel("Cumulative Dosing Cost")
plt.title("Nutrient Consumption Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("figures/controller_comparison/dosing_cost_comparison.png", dpi=300)
plt.close()

# =====================================================
# FIGURE 3
# Stability
# =====================================================

plt.figure(figsize=(10,5))
plt.bar(x-width/2, pid_stability, width, label="PID")
plt.bar(x+width/2, lstm_stability, width, label="LSTM")

plt.xticks(x, scenarios, rotation=20)
plt.ylabel("Variance")
plt.title("Closed-Loop Stability Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("figures/controller_comparison/stability_comparison.png", dpi=300)
plt.close()

# =====================================================
# FIGURE 4
# Radar Chart
# =====================================================

pid_ec_mae = np.mean([
    0.3126, 0.2918, 0.3189,
    0.3049, 0.3667, 1.0091
])

lstm_ec_mae = np.mean([
    1.0133, 1.0258, 1.0124,
    1.0217, 1.0136, 1.0142
])

pid_variance = np.mean([
    0.0050, 0.0045, 0.0116,
    0.0063, 0.0078, 0.0543
])

lstm_variance = np.mean([
    0.0588, 0.0487, 0.0589,
    0.0573, 0.0564, 0.0615
])

pid_dose = np.mean([
    139.3, 126.9, 146.3,
    134.6, 174.1, 16.5
])

lstm_dose = np.mean([
    10.3, 10.3, 10.1,
    10.3, 10.2, 7.7
])

pid_eff = np.mean([
    0.00224, 0.00230, 0.00218,
    0.00227, 0.00211, 0.0610
])

lstm_eff = np.mean([
    0.09816, 0.09973, 0.10040,
    0.09951, 0.09977, 0.13093
])

pid_over = np.mean([
    0.0, 0.0, 0.0,
    0.0, 0.0006, 0.0
])

lstm_over = np.mean([
    0.0651, 0.0, 0.0230,
    0.0881, 0.0, 0.0941
])

# -----------------------------
# Engineering normalization
# higher score = better
# -----------------------------

def invert_metric(x, worst):
    return max(0, 1 - (x / worst))

pid_scores = [
    invert_metric(pid_ec_mae, 1.2),
    invert_metric(pid_over, 0.1),
    invert_metric(pid_variance, 0.07),
    invert_metric(pid_dose, 180),
    min(pid_eff / 0.15, 1.0)
]

lstm_scores = [
    invert_metric(lstm_ec_mae, 1.2),
    invert_metric(lstm_over, 0.1),
    invert_metric(lstm_variance, 0.07),
    invert_metric(lstm_dose, 180),
    min(lstm_eff / 0.15, 1.0)
]

labels = [
    "Tracking",
    "Overshoot",
    "Stability",
    "Dose Cost",
    "Efficiency"
]

# close polygon
pid_scores += pid_scores[:1]
lstm_scores += lstm_scores[:1]

angles = np.linspace(
    0,
    2*np.pi,
    len(labels),
    endpoint=False
).tolist()

angles += angles[:1]

# -----------------------------
# Radar plot
# -----------------------------

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

ax.plot(
    angles,
    pid_scores,
    linewidth=2,
    label="PID"
)

ax.fill(
    angles,
    pid_scores,
    alpha=0.25
)

ax.plot(
    angles,
    lstm_scores,
    linewidth=2,
    label="LSTM"
)

ax.fill(
    angles,
    lstm_scores,
    alpha=0.25
)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_ylim(0,1)

plt.title(
    "Controller Performance Tradeoff",
    pad=25,
    fontsize=16
)

plt.legend(loc="upper right")

plt.tight_layout()

output_file = OUTPUT_DIR / "radar_fixed.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {output_file.resolve()}")

plt.show()