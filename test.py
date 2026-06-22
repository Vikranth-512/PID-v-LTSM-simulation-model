import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_FILE = Path("data/processed/pid_tuning_results.json")
OUTPUT_DIR = Path("figures/paper")

# ============================================================
# Publication settings
# ============================================================

PLOT_HORIZON_STEPS = 1175  # 20 hours @ 60s timestep

FIG_DPI = 600

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_trace():
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    trace = results["reference_trace"]

    return (
        np.array(trace["ec"]),
        np.array(trace["flowrate"]),
        np.array(trace["duration"]),
        trace["target"],
        trace["dt"],
    )


def truncate_horizon(*arrays):
    return [a[:PLOT_HORIZON_STEPS] for a in arrays]


def save_figure(fig, stem):
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=FIG_DPI,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def figure_5_ec_tracking(ec, target, dt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ec = ec[:PLOT_HORIZON_STEPS]

    time_hours = np.arange(len(ec)) * dt / 3600.0

    fig, ax = plt.subplots(
        figsize=(9, 4.8),
        constrained_layout=True,
    )

    ax.axhspan(
        0.75,
        2.20,
        alpha=0.12,
        color="green",
        label="Healthy Operating Region",
    )

    ax.axhspan(
        target * 0.95,
        target * 1.05,
        alpha=0.15,
        color="orange",
        label="±5% Target Band",
    )

    ax.plot(
        time_hours,
        ec,
        linewidth=2.5,
        label="Measured EC",
    )

    ax.axhline(
        target,
        linestyle="--",
        linewidth=2.2,
        color="black",
        label=f"Target EC ({target:.2f})",
    )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Electrical Conductivity (mS/cm)")
    ax.set_title("Closed-Loop EC Regulation Using Tuned PID Controller")

    ax.set_xlim(0, time_hours[-1])

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.8,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    save_figure(fig, "figure_5_pid_ec_tracking")

    plt.close(fig)


def figure_6_control_actions(flowrate, duration, dt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flowrate = flowrate[:PLOT_HORIZON_STEPS]
    duration = duration[:PLOT_HORIZON_STEPS]

    time_hours = np.arange(len(flowrate)) * dt / 3600.0

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
        constrained_layout=True,
    )

    ax1.plot(
        time_hours,
        flowrate,
        linewidth=2.0,
    )

    ax1.set_ylabel("Flowrate (L/min)")
    ax1.set_title("PID Nutrient Dosing Actions")

    ax1.grid(
        True,
        alpha=0.25,
        linewidth=0.8,
    )

    ax2.plot(
        time_hours,
        duration,
        linewidth=2.0,
    )

    ax2.set_ylabel("Duration (s)")
    ax2.set_xlabel("Time (hours)")

    ax2.grid(
        True,
        alpha=0.25,
        linewidth=0.8,
    )

    save_figure(fig, "figure_6_pid_control_actions")

    plt.close(fig)


def main():
    ec, flowrate, duration, target, dt = load_trace()

    print(
        f"\nPlotting first {PLOT_HORIZON_STEPS} steps "
        f"({PLOT_HORIZON_STEPS * dt / 3600:.1f} hours)\n"
    )

    figure_5_ec_tracking(
        ec,
        target,
        dt,
    )

    figure_6_control_actions(
        flowrate,
        duration,
        dt,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()