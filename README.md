# HydroControl: A Synthetic Benchmark Environment for Hydroponic Nutrient Regulation

HydroControl is a reproducible benchmark environment for evaluating control algorithms for hydroponic nutrient regulation under nonlinear dynamics, delayed nutrient assimilation, environmental disturbances, and actuator constraints.

The benchmark combines a physics-inspired simulation environment, optimization-derived expert demonstrations, standardized disturbance scenarios, and reproducible evaluation protocols to support fair comparison between classical and learning-based control strategies.

The repository includes:

* Nonlinear hydroponic process simulation
* Synthetic benchmark trajectory generation
* Optimization-derived expert control labels
* Standardized disturbance suite
* LSTM reference controller
* Tuned PID baseline controller
* Closed-loop benchmark evaluation
* Publication-quality visualization and diagnostics

Rather than proposing a single control algorithm, HydroControl provides a common experimental framework that can be used to evaluate, compare, and reproduce future research on intelligent process control.

---

# Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete benchmark pipeline:

```bash
python main.py --config configs/default.yaml --stage all
```

---

# Individual Pipeline Stages

```bash
python main.py --stage generate
```

Generate synthetic benchmark trajectories.

```bash
python main.py --stage label
```

Generate optimization-derived expert demonstrations.

```bash
python main.py --stage preprocess
```

Perform feature engineering, sequence generation, and dataset preparation.

```bash
python main.py --stage train
```

Train the reference LSTM controller.

```bash
python main.py --stage evaluate
```

Run closed-loop benchmark evaluation.

```bash
python main.py --stage export
```

Export trained models.

---

# Generated Outputs

| Path                                     | Description                                          |
| ---------------------------------------- | ---------------------------------------------------- |
| `data/synthetic/`                        | Generated benchmark trajectories                     |
| `data/processed/`                        | Benchmark dataset, expert labels, sequences, scalers |
| `checkpoints/`                           | Trained models and exported artifacts                |
| `figures/`                               | Benchmark figures, diagnostics, and evaluation plots |
| `data/processed/evaluation_results.json` | Benchmark evaluation metrics                         |

---

# Benchmark Overview

The benchmark models a closed-loop hydroponic nutrient regulation system in which controllers continuously observe multivariate process measurements and determine nutrient dosing actions.

## Observable Process Variables

* Electrical Conductivity (EC)
* Water Temperature
* Turbidity
* pH
* Dissolved Oxygen
* Ambient Temperature
* Previous Dosing Flowrate
* Previous Dosing Duration
* Time Since Last Dose

Additional engineered features capture temporal dynamics, historical control information, rolling statistics, and process trends, producing a 17-dimensional observation space suitable for sequential decision-making.

## Control Actions

Controllers produce two continuous control outputs:

* Nutrient Pump Flowrate
* Pump Activation Duration

These actions are constrained by realistic actuator limits and evaluated through closed-loop interaction with the benchmark environment.

---

# Benchmark Environment

Unlike many simplified control benchmarks that assume instantaneous actuator effects, HydroControl explicitly models delayed nutrient assimilation and nonlinear ecosystem dynamics.

The benchmark incorporates:

* Delayed nutrient assimilation through a multi-stage release process
* Nonlinear nutrient uptake
* Biological growth feedback
* Environmental coupling
* Nutrient saturation effects
* Long-term nutrient memory
* Stochastic process variability
* Closed-loop process dynamics

Together, these mechanisms create a challenging control problem that more closely resembles practical hydroponic nutrient management than stationary linear control environments.

# Benchmark Dataset

The benchmark dataset is generated directly from the simulation environment using optimization-derived expert demonstrations.

Current release:

* 200 simulation trajectories
* 59,042 optimization-labeled samples
* Variable trajectory lengths (100–500 timesteps)
* 17 engineered input features
* 2 continuous control targets
* Standardized train/validation/test partitions
* Sliding-window sequence generation (32 timesteps)

Expert labels are generated using long-horizon optimization rather than manually designed control rules, providing a consistent supervisory signal suitable for imitation learning, supervised policy learning, and benchmark evaluation.

---

# Optimization-Based Expert Demonstrations

Rather than relying on human annotations, HydroControl generates expert control actions using a long-horizon optimization framework.

The optimization objective simultaneously considers:

* Setpoint tracking accuracy
* Recovery performance
* Steady-state regulation
* Oscillation suppression
* Overshoot prevention
* Nutrient efficiency
* Action smoothness
* Safety constraints

The resulting expert trajectories provide standardized reference behaviour for controller development and evaluation.

---

# Reference Baselines

The repository includes representative implementations of both classical and learning-based control strategies.

## Enhanced PID Baseline

```bash
python main.py --config configs/default.yaml --stage tune_pid
```

Quick tuning:

```bash
python main.py --config configs/pid_tune_quick.yaml --stage tune_pid
```

The PID implementation includes:

* Anti-windup protection
* Derivative filtering
* Deadband compensation
* Rate limiting
* Output saturation

Controller gains are obtained through a multi-stage optimization procedure using a composite objective that balances tracking performance, robustness, oscillation suppression, nutrient consumption, and control smoothness.

---

## LSTM Reference Controller

The repository also includes a reference Long Short-Term Memory (LSTM) policy trained using optimization-derived expert demonstrations.

The model receives fixed-length sequences of engineered process features and predicts continuous nutrient dosing actions.

The implementation serves as a reproducible learning-based baseline for future benchmark comparisons rather than the primary contribution of the repository.

---

# Repository Structure

```text
simulation/          → benchmark dynamics and trajectory generation
preprocessing/       → feature engineering and sequence generation
models/              → reference LSTM controller
controllers/         → PID controller and tuning framework
training/            → training pipeline and evaluation metrics
simulation_runner/   → closed-loop benchmark evaluation
visualization/       → publication-quality figures
configs/             → experiment configuration
main.py              → end-to-end benchmark pipeline
```

```text
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

# Evaluation Metrics

The benchmark reports standardized controller performance using metrics including:

* EC tracking error
* Overshoot
* Oscillation
* Stability
* Nutrient consumption
* Control smoothness
* Disturbance recovery
* Long-horizon regulation performance

These metrics enable consistent comparison between classical controllers, learning-based policies, reinforcement learning methods, model predictive control, adaptive controllers, and future control approaches.

---

# Visualization and Diagnostics

The repository includes utilities for generating publication-quality figures covering:

* Environment dynamics
* Benchmark trajectories
* Dataset statistics
* Controller evaluation
* Disturbance robustness
* Optimization diagnostics
* Closed-loop performance

---

# Reproducibility

HydroControl is designed as a reproducible research benchmark.

The repository includes:

* Fixed experimental seeds
* Trajectory-level train/validation/test splits
* Train-only feature normalization
* Experiment metadata logging
* Optimization-derived expert labels
* Standardized disturbance scenarios
* Reference PID and LSTM implementations
* Publication-quality evaluation scripts

These components enable reproducible experimentation and facilitate fair comparison of future hydroponic nutrient regulation algorithms within a common benchmark framework.
