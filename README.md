# Learned Nutrient Dosing Control for a nonlinear delayed aquatic ecosystem.

The pipeline includes:

nonlinear ecosystem simulation,
synthetic trajectory generation,
optimization-generated control labels,
LSTM sequence modeling,
PID auto-tuning,
disturbance robustness testing,
and publication-style evaluation/visualization.

The goal is not simply to outperform classical control, but to characterize when learned controllers become advantageous in delayed, nonlinear, multi-regime ecological systems.

This is **not** a forecasting model. It learns a control policy using optimization-generated pseudo-labels and LSTM sequence modeling.

## Overview

### The system continuously monitors multivariate aquatic ecosystem parameters such as:

Electrical Conductivity (EC)
Water Temperature
Turbidity
pH
Dissolved Oxygen

### and learns a control policy that outputs:

Nutrient Pump Flowrate
Pump Activation Duration

to maintain ecological stability and nutrient balance in a dynamic algae cultivation environment.

### Unlike conventional threshold-based systems, this project models the environment as a nonlinear delayed ecosystem with:

delayed nutrient absorption,
biological uptake,
algae growth dynamics,
disturbance-sensitive equilibrium,
oscillatory instability,
ecological collapse conditions,
and active closed-loop control requirements.
Core Research Goal

### This project investigates the tradeoff between:

Classical Stability-First Control
(PID / rule-based control)

vs

Learned Precision Regulation
(LSTM sequence policy learning)

The tuned PID controller establishes a robust, conservative ecological baseline.

## PID tuning (systematic baseline optimization)

```bash
# Full tuning (~72 coarse + refinement + validation)
python main.py --config configs/default.yaml --stage tune_pid

# Quick smoke test
python main.py --config configs/pid_tune_quick.yaml --stage tune_pid
```

Outputs: `data/processed/pid_tuning_results.json`, `figures/pid_tuning/`, `data/processed/pid_tuned_gains.yaml`

The enhanced PID includes anti-windup, derivative filtering, deadband, rate limiting, and output saturation. Tuning uses a weighted composite score over EC MAE, overshoot, oscillation, nutrient use, collapse fraction, and control smoothness.

Figures are written to `figures/dynamics_validation/`.

## Architecture

```
simulation/          → tank dynamics, synthetic trajectories, optimization labels
preprocessing/       → features, sliding windows, normalization
models/              → LSTM policy (PyTorch)
controllers/         → PID and rule-based baselines
training/            → losses, training loop, metrics
simulation_runner/   → closed-loop evaluation
visualization/       → publication plots
configs/             → YAML experiment config
main.py              → end-to-end orchestration
```

```
simulation/
│
├── dynamics.py
├── environment.py
├── synthetic_generator.py
├── optimization_labeler.py
├── disturbances.py
└── validate_dynamics.py

preprocessing/
│
├── feature_engineering.py
├── sequence_builder.py
└── scalers.py

models/
│
└── lstm_policy.py

controllers/
│
├── pid_controller.py
└── pid_tuner.py

training/
│
├── trainer.py
├── losses.py
└── metrics.py

simulation_runner/
│
└── closed_loop_eval.py

visualization/
│
└── plots.py

configs/
│
├── default.yaml
└── pid_tune_quick.yaml
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml --stage all
```

### Individual stages

```bash
python main.py --stage generate    # synthetic CSV trajectories
python main.py --stage label       # optimization-based labels
python main.py --stage preprocess  # features + sequences + scalers
python main.py --stage train       # LSTM policy training
python main.py --stage evaluate    # offline + closed-loop benchmarks
python main.py --stage export      # TorchScript / ONNX export
```

## Outputs

| Path | Description |
|------|-------------|
| `data/synthetic/` | Raw trajectory CSVs |
| `data/processed/` | Labeled data, sequences, scalers |
| `checkpoints/` | Best model weights, export artifacts |
| `figures/` | Training curves, EC trajectories, controller comparisons |
| `data/processed/evaluation_results.json` | Metrics summary |

## Control Policy

- **Input**: Window of engineered sensor features `(seq_len, n_features)`
- **Output**: `(flowrate, duration)`
- **Labels**: Short-horizon constrained search minimizing EC error, nutrient cost, instability, overshoot

## Generated artifacts include:

trajectory CSVs,
optimization labels,
trained policies,
controller metrics,
publication-quality plots,
TorchScript exports.
Research Contributions

## Metrics include:

EC tracking error,
overshoot,
oscillation amplitude,
nutrient efficiency,
control smoothness,
disturbance recovery,
collapse prevention.
Validation and Diagnostics

The project includes extensive visualization and diagnostics:

## Reproducibility

- Fixed seeds in `configs/default.yaml`
- Trajectory-level train/val/test splits (no window leakage)
- Train-only scaler fitting
- Experiment metadata JSON per run
