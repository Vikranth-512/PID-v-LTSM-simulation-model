# Pre-Training Verification Audit

**Generated from:** `C:\Users\retro\CascadeProjects\ml`

## Summary

| Section | Status |
|---------|--------|
| Section 1 — Dataset Integrity | **PASS** |
| Section 2 — Feature Leakage | **PASS** |
| Section 3 — Target Distribution | **PASS** |
| Section 4 — Scaling Consistency | **PASS** |
| Section 5 — Training Target Pipeline | **WARNING** |
| Section 6 — LSTM Architecture | **PASS** |
| Section 7 — Forward Pass Validation | **PASS** |
| Section 8 — Gradient Flow Audit | **PASS** |
| Section 9 — Overfit Sanity Test | **WARNING** |
| Section 10 — Inference Wrapper Audit | **WARNING** |
| Section 11 — Action Range Audit | **PASS** |
| Section 12 — Control-Aware Loss Audit | **PASS** |
| Section 13 — Checkpoint Export Audit | **PASS** |
| Section 14 — Closed-Loop Deployment Audit | **PASS** |

## Section Details

### Section 1 — Dataset Integrity — PASS

- All required arrays present: ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
- X_train.shape=(36556, 32, 17) (features=17)
- y_train.shape=(36556, 2)
- X_train: count_nan=0, count_inf=0
- y_train: count_nan=0, count_inf=0
- X_val: count_nan=0, count_inf=0
- y_val: count_nan=0, count_inf=0
- X_test: count_nan=0, count_inf=0
- y_test: count_nan=0, count_inf=0
- Split sizes: train=36556 (69.2%), val=7885 (14.9%), test=8401 (15.9%)

### Section 2 — Feature Leakage — PASS

- Feature count: 17
- Features: water_temp, ec, turbidity, prev_flowrate, prev_duration, time_since_last_dose, ph, dissolved_oxygen, ambient_temp, delta_ec, delta_temp, delta_turbidity, rolling_avg_ec, rolling_std_ec, ec_error, cumulative_nutrients, dosing_frequency
- Targets: optimal_flowrate, optimal_duration
- No label leakage detected; features are observable state/history derived

### Section 3 — Target Distribution — PASS

- optimal_flowrate: mean=2.4985 std=2.4848 min=0.0000 max=5.0000 median=1.5000
-   optimal_flowrate == 0: 49.1%
-   Plot saved: reports\hist_flowrate.png
- optimal_duration: mean=15.6869 std=14.3946 min=0.0000 max=30.0000 median=15.0000
-   optimal_duration == 0: 37.2%
-   Plot saved: reports\hist_duration.png
- Trajectories in labeled CSV: 200

### Section 4 — Scaling Consistency — PASS

- No fit_transform() in main.py or normalization.py
- main.py: normalizer.fit() on train, transform_features() on all splits
- Found: data\processed\scalers\normalizer_meta.json
- Found: data\processed\scalers\feature_scaler.pkl
- Found: data\processed\scalers\target_scaler.pkl
- Input scaler type: standard
- Output scaler type: standard (shared FeatureNormalizer)
- Feature columns in scaler: 17
- Target columns in scaler: ['optimal_flowrate', 'optimal_duration']

### Section 5 — Training Target Pipeline — WARNING

- Training loop: loss computed on scaled targets (yb from scaled splits)
- Validation metrics: inverse_transform_targets applied for RMSE/MAE only
- Loss during validation also uses scaled pred vs scaled target (consistent)
- y_batch (scaled): mean=-0.9609 std=0.1320 min=-1.1055 max=-0.7562
- pred (scaled): mean=0.0198 std=0.0699 min=-0.0557 max=0.0934
- WARNING: ControlAwareLoss dose_threshold=50.0 is in SCALED target space, not physical flowrate*duration

### Section 6 — LSTM Architecture — PASS

- input_size=17
- hidden_size=128
- output_size=2
- dropout=0.2
- total_trainable_parameters=215746
- model_summary.txt written (215746 params)

### Section 7 — Forward Pass Validation — PASS

- Input shape: (8, 32, 17)
- Output shape: (8, 2)
- Output finite: OK

### Section 8 — Gradient Flow Audit — PASS

- gradient_report.txt written

### Section 9 — Overfit Sanity Test — WARNING

- 128-sample overfit test, 80 epochs, final MSE=0.015360
- Initial MSE=0.827863
- overfit_test_loss.png saved
- Loss decreased but did not approach near-zero

### Section 10 — Inference Wrapper Audit — WARNING

- LearnedPolicyWrapper calls inverse_transform_targets() before returning
- Warmup: returns (0, 0) until buffer length == sequence_length
- Configured sequence_length=32
- Feature vector built in feature_columns.json order
- Post-warmup action: flowrate=2.4411, duration=18.9565
- WARNING: LearnedPolicyWrapper does NOT clip actions (ClosedLoopEvaluator does)

### Section 11 — Action Range Audit — PASS

- min predicted flowrate: 2.7455
- max predicted flowrate: 2.8018
- min predicted duration: 17.2381
- max predicted duration: 17.5296
- Physical limits: flowrate [0,5], duration [0,30]
- ClosedLoopEvaluator.run_learned_policy: action clipping IS implemented
- Recommendation: add clipping to LearnedPolicyWrapper for deployment parity

### Section 12 — Control-Aware Loss Audit — PASS

- Configured loss_type: control_aware
- Active loss class: ControlAwareLoss
- Control-aware loss is active (not silent MSE fallback)
- Loss weights: {'action': 1.0, 'instability': 0.2, 'aggressive_change': 0.15, 'excessive_dose': 0.1}
- loss_configuration_report.txt written

### Section 13 — Checkpoint Export Audit — PASS

- TorchScript max |diff|: 0.00e+00
- export_validation_report.txt written

### Section 14 — Closed-Loop Deployment Audit — PASS

- Closed-loop steps: 100
- Action stats — flowrate: mean=1.5784 min=0.0000 max=2.2967
- Action stats — duration: mean=11.2066 min=0.0000 max=16.5747
- EC stats: mean=0.5617 min=0.2875 max=1.1395

## Verdict

**TRAINING APPROVED**

Warnings present — review before full training:

- Section 5 — Training Target Pipeline
- Section 9 — Overfit Sanity Test
- Section 10 — Inference Wrapper Audit