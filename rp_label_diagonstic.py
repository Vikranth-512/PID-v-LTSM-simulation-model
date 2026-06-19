from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = (
    r"C:\Users\retro\CascadeProjects\ml\data\processed\all_trajectories_labeled.csv"
)

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATASET_PATH)

flow_col = "optimal_flowrate"
duration_col = "optimal_duration"


flow_counts = (
    df[flow_col]
    .value_counts()
    .sort_index()
)

duration_counts = (
    df[duration_col]
    .value_counts()
    .sort_index()
)

flow_pct = 100.0 * flow_counts / len(df)
duration_pct = 100.0 * duration_counts / len(df)

print("\nFLOWRATE DISTRIBUTION")
print("=" * 60)

for value, pct in flow_pct.items():
    print(f"{value:>4.1f} -> {pct:6.2f}%")

print("\nDURATION DISTRIBUTION")
print("=" * 60)

for value, pct in duration_pct.items():
    print(f"{value:>4.1f} -> {pct:6.2f}%")


print("\nTOP 10 FLOWRATE-DURATION PAIRS")
print("=" * 60)

pair_counts = (
    df.groupby([flow_col, duration_col])
    .size()
    .sort_values(ascending=False)
)

for (f, d), count in pair_counts.head(10).items():
    pct = 100.0 * count / len(df)
    print(
        f"Flow={f:>4.1f}  Dur={d:>4.1f}  "
        f"Count={count:>6d}  ({pct:5.2f}%)"
    )


plt.figure(figsize=(8, 5))

plt.bar(
    flow_counts.index.astype(str),
    flow_counts.values
)

plt.title("Optimal Flowrate Distribution")
plt.xlabel("Flowrate")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_flowrate_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(8, 5))

plt.bar(
    duration_counts.index.astype(str),
    duration_counts.values
)

plt.title("Optimal Duration Distribution")
plt.xlabel("Duration")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_duration_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


pair_pct = (
    100.0
    * pair_counts
    / len(df)
)

pair_pct.to_csv(
    OUTPUT_DIR / "flow_duration_pair_distribution.csv"
)

print("\nSaved:")
print(OUTPUT_DIR / "figure_flowrate_distribution.png")
print(OUTPUT_DIR / "figure_duration_distribution.png")
print(OUTPUT_DIR / "flow_duration_pair_distribution.csv")