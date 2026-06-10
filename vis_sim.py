import os
import glob
import random
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data/synthetic"

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if len(csv_files) < 5:
    raise ValueError("Need at least 5 trajectory CSV files.")

selected_files = random.sample(csv_files, 5)

fig, axes = plt.subplots(5, 4, figsize=(22, 18))

for row_idx, file_path in enumerate(selected_files):
    df = pd.read_csv(file_path)

    time_col = "timestamp" if "timestamp" in df.columns else df.index

    ec_col = "ec" if "ec" in df.columns else "true_ec"
    temp_col = "water_temp" if "water_temp" in df.columns else "true_water_temp"
    turb_col = "turbidity" if "turbidity" in df.columns else "true_turbidity"

    if "optimal_flowrate" in df.columns:
        flow_col = "optimal_flowrate"
    elif "prev_flowrate" in df.columns:
        flow_col = "prev_flowrate"
    else:
        raise ValueError(f"No flowrate column found in {file_path}")

    # EC
    axes[row_idx, 0].plot(time_col, df[ec_col])
    axes[row_idx, 0].set_title(f"Trajectory {row_idx+1} - EC")
    axes[row_idx, 0].set_xlabel("Time")
    axes[row_idx, 0].set_ylabel("EC")

    # Temperature
    axes[row_idx, 1].plot(time_col, df[temp_col])
    axes[row_idx, 1].set_title(f"Trajectory {row_idx+1} - Temp")
    axes[row_idx, 1].set_xlabel("Time")
    axes[row_idx, 1].set_ylabel("Temp")

    # Turbidity
    axes[row_idx, 2].plot(time_col, df[turb_col])
    axes[row_idx, 2].set_title(f"Trajectory {row_idx+1} - Turbidity")
    axes[row_idx, 2].set_xlabel("Time")
    axes[row_idx, 2].set_ylabel("Turbidity")

    # Flowrate
    axes[row_idx, 3].plot(time_col, df[flow_col])
    axes[row_idx, 3].set_title(f"Trajectory {row_idx+1} - Flowrate")
    axes[row_idx, 3].set_xlabel("Time")
    axes[row_idx, 3].set_ylabel("Flowrate")

plt.tight_layout()
plt.show()