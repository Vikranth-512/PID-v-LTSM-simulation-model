import numpy as np
import pandas as pd

data = np.load("data/processed/sequences.npz")

for k in data.files:
    print(k, data[k].shape)

print("Arrays found:")
print(data.files)

#for key in data.files:
#    print(f"{key}: {data[key].shape}")

df = pd.read_csv(
    "data/processed/all_trajectories_labeled.csv"
)

print(df["optimal_flowrate"].describe())
print()
print(df["optimal_duration"].describe())


print(
    (df["optimal_flowrate"] == 0).mean()
)

print(
    (df["optimal_duration"] == 0).mean()
)