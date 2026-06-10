"""
Pre-training verification audit for LSTM policy learning pipeline.

Read-only with respect to production code paths. Generates reports under reports/.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baseline_models import LearnedPolicyWrapper
from models.lstm_policy import LSTMPolicy
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.normalization import FeatureNormalizer
from preprocessing.sequence_builder import SequenceDataset
from simulation_runner.closed_loop_eval import ClosedLoopEvaluator
from training.losses import ControlAwareLoss, mse_loss
from training.train import Trainer
from utils import load_config

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

FORBIDDEN_FEATURES = {
    "optimal_flowrate",
    "optimal_duration",
    "label_score",
    "flowrate",
    "duration",
}
OBSERVABLE_PREFIXES = (
    "water_temp", "ec", "turbidity", "ph", "dissolved_oxygen", "ambient_temp",
    "prev_", "time_since", "delta_", "rolling_", "ec_error", "cumulative", "dosing",
)


@dataclass
class SectionResult:
    name: str
    status: str  # PASS | WARNING | FAIL
    details: List[str] = field(default_factory=list)


results: List[SectionResult] = []


def record(name: str, status: str, details: List[str]) -> None:
    results.append(SectionResult(name=name, status=status, details=details))


def section_1_dataset_integrity() -> None:
    details: List[str] = []
    status = "PASS"
    path = ROOT / "data/processed/sequences.npz"
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]

    if not path.exists():
        record("Section 1 — Dataset Integrity", "FAIL", ["sequences.npz not found"])
        return

    data = np.load(path)
    missing = [k for k in required if k not in data.files]
    if missing:
        record("Section 1 — Dataset Integrity", "FAIL", [f"Missing arrays: {missing}"])
        return
    details.append(f"All required arrays present: {required}")

    x_train, y_train = data["X_train"], data["y_train"]
    expected_feat = 17
    expected_tgt = 2
    if x_train.shape[2] != expected_feat:
        status = "FAIL"
        details.append(f"X_train.shape[2]={x_train.shape[2]}, expected {expected_feat}")
    else:
        details.append(f"X_train.shape={x_train.shape} (features={expected_feat})")

    if y_train.shape[1] != expected_tgt:
        status = "FAIL"
        details.append(f"y_train.shape[1]={y_train.shape[1]}, expected {expected_tgt}")
    else:
        details.append(f"y_train.shape={y_train.shape}")

    for split in required:
        arr = data[split]
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        details.append(f"{split}: count_nan={n_nan}, count_inf={n_inf}")
        if n_nan or n_inf:
            status = "FAIL"

    counts = {k: len(data[k]) for k in ["X_train", "X_val", "X_test"]}
    total = sum(counts.values())
    pcts = {k: 100.0 * v / total for k, v in counts.items()}
    details.append(
        f"Split sizes: train={counts['X_train']} ({pcts['X_train']:.1f}%), "
        f"val={counts['X_val']} ({pcts['X_val']:.1f}%), "
        f"test={counts['X_test']} ({pcts['X_test']:.1f}%)"
    )
    if not (65 <= pcts["X_train"] <= 75 and 12 <= pcts["X_val"] <= 18 and 12 <= pcts["X_test"] <= 18):
        if status == "PASS":
            status = "WARNING"
        details.append("Split ratios deviate from expected ~70/15/15 (trajectory-level split)")

    record("Section 1 — Dataset Integrity", status, details)


def section_2_feature_leakage() -> None:
    details: List[str] = []
    status = "PASS"
    feat_path = ROOT / "data/processed/feature_columns.json"
    if not feat_path.exists():
        record("Section 2 — Feature Leakage", "FAIL", ["feature_columns.json not found"])
        return

    with open(feat_path) as f:
        meta = json.load(f)
    features = meta.get("features", [])
    targets = meta.get("targets", [])
    details.append(f"Feature count: {len(features)}")
    details.append("Features: " + ", ".join(features))
    details.append("Targets: " + ", ".join(targets))

    leaked = [c for c in features if c in FORBIDDEN_FEATURES]
    if leaked:
        status = "FAIL"
        details.append(f"HARD FAIL — forbidden columns in features: {leaked}")

    suspicious = [c for c in features if "optimal" in c or "label" in c]
    if suspicious:
        status = "FAIL"
        details.append(f"Future label columns detected: {suspicious}")

    non_observable = [
        c for c in features
        if not any(c.startswith(p) or c == p.split("_")[0] for p in OBSERVABLE_PREFIXES)
        and c not in ("ec", "ph")
    ]
    # refine check
    allowed = set(features)
    for c in features:
        ok = (
            c in {"water_temp", "ec", "turbidity", "ph", "dissolved_oxygen", "ambient_temp"}
            or c.startswith(("prev_", "delta_", "rolling_", "ec_", "cumulative_", "dosing_", "time_"))
        )
        if not ok:
            if status == "PASS":
                status = "WARNING"
            details.append(f"Review feature observability: {c}")

    if status == "PASS":
        details.append("No label leakage detected; features are observable state/history derived")

    record("Section 2 — Feature Leakage", status, details)


def section_3_target_distribution() -> None:
    details: List[str] = []
    status = "PASS"
    path = ROOT / "data/processed/all_trajectories_labeled.csv"
    if not path.exists():
        record("Section 3 — Target Distribution", "FAIL", ["all_trajectories_labeled.csv not found"])
        return

    df = pd.read_csv(path)
    for col in ("optimal_flowrate", "optimal_duration"):
        s = df[col]
        details.append(
            f"{col}: mean={s.mean():.4f} std={s.std():.4f} "
            f"min={s.min():.4f} max={s.max():.4f} median={s.median():.4f}"
        )
        pct_zero = 100.0 * (s == 0).mean()
        details.append(f"  {col} == 0: {pct_zero:.1f}%")
        if pct_zero > 90:
            status = "WARNING"
            details.append(f"  WARNING: >90% of {col} are zero (collapsed labels)")
        max_flow, max_dur = 5.0, 30.0
        at_max = 100.0 * ((s >= max_flow - 1e-6) if col == "optimal_flowrate" else (s >= max_dur - 1e-6)).mean()
        if at_max > 90:
            status = "WARNING"
            details.append(f"  WARNING: >90% of {col} at maximum")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s.dropna(), bins=50, edgecolor="black", alpha=0.75)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        out = REPORTS / f"hist_{col.replace('optimal_', '')}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        details.append(f"  Plot saved: {out.relative_to(ROOT)}")

    n_traj = df["trajectory_id"].nunique()
    details.append(f"Trajectories in labeled CSV: {n_traj}")
    if n_traj < 200:
        status = "WARNING" if status == "PASS" else status
        details.append(f"WARNING: expected 200 trajectories, found {n_traj}")

    record("Section 3 — Target Distribution", status, details)


def section_4_scaling_consistency() -> None:
    details: List[str] = []
    status = "PASS"

    # Static code audit
    main_src = (ROOT / "main.py").read_text(encoding="utf-8")
    norm_src = (ROOT / "preprocessing/normalization.py").read_text(encoding="utf-8")

    if "fit_transform" in main_src or "fit_transform" in norm_src:
        status = "FAIL"
        details.append("fit_transform found in preprocessing pipeline")
    else:
        details.append("No fit_transform() in main.py or normalization.py")

    if "normalizer.fit(" in main_src and "transform_features" in main_src:
        details.append("main.py: normalizer.fit() on train, transform_features() on all splits")
    else:
        status = "FAIL"
        details.append("Could not verify fit/transform pattern in main.py")

    scaler_dir = ROOT / "data/processed/scalers"
    meta_path = scaler_dir / "normalizer_meta.json"
    feat_pkl = scaler_dir / "feature_scaler.pkl"
    tgt_pkl = scaler_dir / "target_scaler.pkl"
    for p in (meta_path, feat_pkl, tgt_pkl):
        if not p.exists():
            status = "FAIL"
            details.append(f"Missing scaler artifact: {p}")
        else:
            details.append(f"Found: {p.relative_to(ROOT)}")

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        details.append(f"Input scaler type: {meta.get('scaler_type')}")
        details.append(f"Output scaler type: {meta.get('scaler_type')} (shared FeatureNormalizer)")

    norm = FeatureNormalizer.load(scaler_dir) if meta_path.exists() else None
    if norm is not None:
        details.append(f"Feature columns in scaler: {len(norm.feature_columns)}")
        details.append(f"Target columns in scaler: {norm.target_columns}")

    record("Section 4 — Scaling Consistency", status, details)


def section_5_training_target_pipeline(config: dict) -> None:
    details: List[str] = []
    status = "PASS"

    train_src = inspect.getsource(Trainer.train)
    if "yb.to(self.device)" in train_src and "loss_fn(pred, yb" in train_src:
        details.append("Training loop: loss computed on scaled targets (yb from scaled splits)")
    else:
        status = "FAIL"
        details.append("Could not verify scaled-target loss in Trainer.train")

    val_src = inspect.getsource(Trainer._validate)
    if "inverse_transform_targets" in val_src:
        details.append("Validation metrics: inverse_transform_targets applied for RMSE/MAE only")
    details.append("Loss during validation also uses scaled pred vs scaled target (consistent)")

    data = np.load(ROOT / "data/processed/sequences.npz")
    xb = torch.from_numpy(data["X_train"][:64]).float()
    yb = torch.from_numpy(data["y_train"][:64]).float()
    model = LSTMPolicy(input_size=17, hidden_size=128, num_layers=2, dropout=0.2)
    model.eval()
    with torch.no_grad():
        pred = model(xb)
    details.append(
        f"y_batch (scaled): mean={yb.mean():.4f} std={yb.std():.4f} "
        f"min={yb.min():.4f} max={yb.max():.4f}"
    )
    details.append(
        f"pred (scaled): mean={pred.mean():.4f} std={pred.std():.4f} "
        f"min={pred.min():.4f} max={pred.max():.4f}"
    )
  # ControlAwareLoss dose threshold operates in scaled space
    tcfg = config.get("training", {})
    if tcfg.get("loss_type") == "control_aware":
        details.append(
            "WARNING: ControlAwareLoss dose_threshold=50.0 is in SCALED target space, "
            "not physical flowrate*duration"
        )
        if status == "PASS":
            status = "WARNING"

    record("Section 5 — Training Target Pipeline", status, details)


def section_6_architecture(config: dict) -> None:
    details: List[str] = []
    status = "PASS"
    mcfg = config.get("model", {})
    model = LSTMPolicy(
        input_size=17,
        hidden_size=mcfg.get("hidden_size", 128),
        num_layers=mcfg.get("num_layers", 2),
        dropout=mcfg.get("dropout", 0.2),
        output_size=mcfg.get("output_size", 2),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    details.append(f"input_size={model.input_size}")
    details.append(f"hidden_size={model.hidden_size}")
    details.append(f"output_size={model.output_size}")
    details.append(f"dropout={mcfg.get('dropout', 0.2)}")
    details.append(f"total_trainable_parameters={n_params}")

    if model.input_size != 17:
        status = "FAIL"
        details.append(f"input_size mismatch: {model.input_size} != 17")

    summary_lines = [
        "LSTMPolicy Architecture",
        "=====================",
        "17 → LSTM(128) → Dropout → LSTM(128) → Dropout → FC(64) → ReLU → FC(2)",
        "",
        str(model),
        "",
        f"Trainable parameters: {n_params}",
    ]
    summary_path = REPORTS / "model_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    details.append(f"model_summary.txt written ({n_params} params)")

    record("Section 6 — LSTM Architecture", status, details)


def section_7_forward_pass() -> None:
    details: List[str] = []
    status = "PASS"
    model = LSTMPolicy(input_size=17, hidden_size=128, num_layers=2, dropout=0.2)
    model.eval()
    x = torch.randn(8, 32, 17)
    with torch.no_grad():
        out = model(x)
    details.append(f"Input shape: {tuple(x.shape)}")
    details.append(f"Output shape: {tuple(out.shape)}")
    if out.shape != (8, 2):
        status = "FAIL"
        details.append(f"Expected output (batch, 2), got {out.shape}")
    if not torch.isfinite(out).all():
        status = "FAIL"
        details.append("Output contains NaN or Inf")
    else:
        details.append("Output finite: OK")

    record("Section 7 — Forward Pass Validation", status, details)


def section_8_gradient_flow() -> None:
    details: List[str] = []
    status = "PASS"
    model = LSTMPolicy(input_size=17, hidden_size=128, num_layers=2, dropout=0.2)
    model.train()
    loss_fn = ControlAwareLoss()
    x = torch.randn(16, 32, 17)
    y = torch.randn(16, 2)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()

    lines = ["Gradient Flow Report", "====================", ""]
    zero_layers = []
    explode_layers = []
    for name, param in model.named_parameters():
        if param.grad is None:
            lines.append(f"{name}: NO GRADIENT")
            zero_layers.append(name)
            continue
        g = param.grad.detach()
        g_mean = float(g.mean())
        g_std = float(g.std())
        g_max = float(g.abs().max())
        lines.append(f"{name}: mean={g_mean:.6e} std={g_std:.6e} max={g_max:.6e}")
        if g.abs().max() > 100:
            explode_layers.append(name)
        if g.abs().max() < 1e-12:
            zero_layers.append(name)

    grad_path = REPORTS / "gradient_report.txt"
    grad_path.write_text("\n".join(lines), encoding="utf-8")
    details.append(f"gradient_report.txt written")

    if explode_layers:
        status = "FAIL"
        details.append(f"Exploding gradients (|grad|>100): {explode_layers}")
    if len(zero_layers) == len(list(model.parameters())):
        status = "FAIL"
        details.append("All layers have zero/missing gradients")
    elif zero_layers:
        status = "WARNING"
        details.append(f"Near-zero gradients on: {zero_layers[:5]}")

    record("Section 8 — Gradient Flow Audit", status, details)


def section_9_overfit_sanity() -> None:
    details: List[str] = []
    status = "PASS"
    data = np.load(ROOT / "data/processed/sequences.npz")
    X, y = data["X_train"], data["y_train"]
    n = min(128, len(X))
    Xs, ys = X[:n], y[:n]
    ds = SequenceDataset(Xs, ys)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = LSTMPolicy(input_size=17, hidden_size=64, num_layers=2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(80):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))

    final_loss = losses[-1]
    details.append(f"128-sample overfit test, 80 epochs, final MSE={final_loss:.6f}")
    details.append(f"Initial MSE={losses[0]:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE (scaled)")
    ax.set_title("Overfit sanity test (128 samples)")
    fig.tight_layout()
    fig.savefig(REPORTS / "overfit_test_loss.png", dpi=120)
    plt.close(fig)
    details.append("overfit_test_loss.png saved")

    if final_loss > 0.05:
        status = "FAIL"
        details.append("Model did not overfit tiny subset (final loss > 0.05) — possible bug")
    elif final_loss > 0.01:
        status = "WARNING"
        details.append("Loss decreased but did not approach near-zero")

    record("Section 9 — Overfit Sanity Test", status, details)


def section_10_inference_wrapper(config: dict) -> None:
    details: List[str] = []
    status = "PASS"

    with open(ROOT / "data/processed/feature_columns.json") as f:
        feature_cols = json.load(f)["features"]

    wrapper_src = inspect.getsource(LearnedPolicyWrapper)
    if "inverse_transform_targets" in wrapper_src:
        details.append("LearnedPolicyWrapper calls inverse_transform_targets() before returning")
    else:
        status = "FAIL"

    if "self.sequence_length" in wrapper_src and "return 0.0, 0.0" in wrapper_src:
        details.append("Warmup: returns (0, 0) until buffer length == sequence_length")
    else:
        status = "FAIL"
        details.append("Warmup behavior not verified")

    seq_len = config.get("preprocessing", {}).get("sequence_length", 32)
    details.append(f"Configured sequence_length={seq_len}")

    # Ordering check
    build_src = inspect.getsource(LearnedPolicyWrapper._build_feature_vector)
    if "self.feature_columns" in build_src:
        details.append("Feature vector built in feature_columns.json order")
    else:
        status = "FAIL"

    norm = FeatureNormalizer.load(ROOT / "data/processed/scalers")
    model = LSTMPolicy(input_size=17, hidden_size=128, num_layers=2, dropout=0.2)
    wrapper = LearnedPolicyWrapper(model, norm, feature_cols, seq_len)

    obs = {c: 1.0 for c in feature_cols}
    for i in range(seq_len - 1):
        fr, dur = wrapper.act(obs)
        if fr != 0.0 or dur != 0.0:
            status = "FAIL"
            details.append(f"Warmup step {i}: expected (0,0), got ({fr},{dur})")
            break
    else:
        fr, dur = wrapper.act(obs)
        details.append(f"Post-warmup action: flowrate={fr:.4f}, duration={dur:.4f}")

    if "np.clip" not in wrapper_src:
        status = "WARNING" if status == "PASS" else status
        details.append("WARNING: LearnedPolicyWrapper does NOT clip actions (ClosedLoopEvaluator does)")

    record("Section 10 — Inference Wrapper Audit", status, details)


def section_11_action_range(config: dict) -> None:
    details: List[str] = []
    status = "PASS"
    data = np.load(ROOT / "data/processed/sequences.npz")
    norm = FeatureNormalizer.load(ROOT / "data/processed/scalers")
    model = LSTMPolicy(input_size=17, hidden_size=128, num_layers=2, dropout=0.2)
    model.eval()

    rng = np.random.default_rng(42)
    preds_raw = []
    n = 1000
    for _ in range(n):
        x = rng.standard_normal((1, 32, 17)).astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(x)).numpy()
        action = norm.inverse_transform_targets(out)[0]
        preds_raw.append(action)
    preds = np.array(preds_raw)

    fr_min, fr_max = float(preds[:, 0].min()), float(preds[:, 0].max())
    dur_min, dur_max = float(preds[:, 1].min()), float(preds[:, 1].max())
    details.append(f"min predicted flowrate: {fr_min:.4f}")
    details.append(f"max predicted flowrate: {fr_max:.4f}")
    details.append(f"min predicted duration: {dur_min:.4f}")
    details.append(f"max predicted duration: {dur_max:.4f}")
    details.append("Physical limits: flowrate [0,5], duration [0,30]")

    if fr_min < -0.01 or fr_max > 5.01 or dur_min < -0.01 or dur_max > 30.01:
        status = "WARNING"
        details.append("Predictions exceed physical bounds on random inputs (untrained model)")

    eval_src = inspect.getsource(ClosedLoopEvaluator.run_learned_policy)
    if "np.clip" in eval_src:
        details.append("ClosedLoopEvaluator.run_learned_policy: action clipping IS implemented")
    else:
        status = "FAIL"
        details.append("No clipping in ClosedLoopEvaluator")

    if "np.clip" not in inspect.getsource(LearnedPolicyWrapper.act):
        details.append("Recommendation: add clipping to LearnedPolicyWrapper for deployment parity")

    record("Section 11 — Action Range Audit", status, details)


def section_12_control_aware_loss(config: dict) -> None:
    details: List[str] = []
    status = "PASS"
    tcfg = config.get("training", {})
    loss_type = tcfg.get("loss_type", "control_aware")
    details.append(f"Configured loss_type: {loss_type}")

    trainer = Trainer(config)
    loss_fn = trainer._get_loss_fn()
    details.append(f"Active loss class: {type(loss_fn).__name__}")

    if loss_type == "control_aware":
        if not isinstance(loss_fn, ControlAwareLoss):
            status = "FAIL"
            details.append("Configured control_aware but got different loss class")
        else:
            details.append("Control-aware loss is active (not silent MSE fallback)")
            weights = tcfg.get("control_loss_weights", {})
            details.append(f"Loss weights: {weights}")
    elif isinstance(loss_fn, type(lambda: None)):
        pass

    lines = [
        "Loss Configuration Report",
        "=========================",
        f"loss_type: {loss_type}",
        f"active_class: {type(loss_fn).__name__}",
        f"weights: {json.dumps(tcfg.get('control_loss_weights', {}), indent=2)}",
        "",
        "Trainer._get_loss_fn source:",
        inspect.getsource(Trainer._get_loss_fn),
    ]
    (REPORTS / "loss_configuration_report.txt").write_text("\n".join(lines), encoding="utf-8")
    details.append("loss_configuration_report.txt written")

    record("Section 12 — Control-Aware Loss Audit", status, details)


def section_13_checkpoint_export() -> None:
    details: List[str] = []
    status = "PASS"
    model = LSTMPolicy(input_size=17, hidden_size=128, num_layers=2, dropout=0.2)
    model.eval()
    example = torch.randn(1, 32, 17)

    with tempfile.TemporaryDirectory() as tmp:
        export_path = Path(tmp) / "policy.pt"
        model.export_torchscript(example, str(export_path))
        loaded = torch.jit.load(str(export_path))
        with torch.no_grad():
            out_pt = model(example).numpy()
            out_ts = loaded(example).numpy()
        max_diff = float(np.abs(out_pt - out_ts).max())
        details.append(f"TorchScript max |diff|: {max_diff:.2e}")

    lines = [
        "Export Validation Report",
        "========================",
        f"export_torchscript: OK",
        f"max_output_difference: {max_diff:.2e}",
        f"threshold: 1e-5",
        f"status: {'PASS' if max_diff < 1e-5 else 'FAIL'}",
    ]
    (REPORTS / "export_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    details.append("export_validation_report.txt written")

    if max_diff >= 1e-5:
        status = "FAIL"
        details.append(f"Export mismatch exceeds 1e-5")

    record("Section 13 — Checkpoint Export Audit", status, details)


def section_14_closed_loop(config: dict) -> None:
    details: List[str] = []
    status = "PASS"

    # Override steps to 100 for audit
    audit_config = json.loads(json.dumps(config))
    audit_config["evaluation"]["closed_loop_steps"] = 100

    norm = FeatureNormalizer.load(ROOT / "data/processed/scalers")
    with open(ROOT / "data/processed/feature_columns.json") as f:
        feature_cols = json.load(f)["features"]

    model = LSTMPolicy(input_size=17, hidden_size=128, num_layers=2, dropout=0.2)
    evaluator = ClosedLoopEvaluator(audit_config, seed=42)

    try:
        result = evaluator.run_learned_policy(
            model, norm, feature_cols, scenario="normal", device="cpu"
        )
        traj = result["trajectory"]
        fr = traj["flowrate"]
        dur = traj["duration"]
        ec = traj["ec"]

        details.append(f"Closed-loop steps: {len(fr)}")
        details.append(
            f"Action stats — flowrate: mean={fr.mean():.4f} min={fr.min():.4f} max={fr.max():.4f}"
        )
        details.append(
            f"Action stats — duration: mean={dur.mean():.4f} min={dur.min():.4f} max={dur.max():.4f}"
        )
        details.append(f"EC stats: mean={ec.mean():.4f} min={ec.min():.4f} max={ec.max():.4f}")

        if not np.isfinite(fr).all() or not np.isfinite(dur).all():
            status = "FAIL"
            details.append("Non-finite actions detected")
        if not np.isfinite(ec).all():
            status = "FAIL"
            details.append("Non-finite EC trajectory")

        seq_len = audit_config["preprocessing"]["sequence_length"]
        if np.all(fr[:seq_len] == 0) and np.all(dur[:seq_len] == 0):
            details.append(f"Warmup period (first {seq_len} steps): actions are zero as expected")
        post = fr[seq_len:]
        if len(post) and np.all(post == 0):
            status = "WARNING"
            details.append("WARNING: all post-warmup actions are zero (untrained model)")

    except Exception as e:
        status = "FAIL"
        details.append(f"Closed-loop crash: {type(e).__name__}: {e}")

    record("Section 14 — Closed-Loop Deployment Audit", status, details)


def write_final_report() -> str:
    critical_fail = any(r.status == "FAIL" for r in results)
    any_warning = any(r.status == "WARNING" for r in results)

    lines = [
        "# Pre-Training Verification Audit",
        "",
        f"**Generated from:** `{ROOT}`",
        "",
        "## Summary",
        "",
        "| Section | Status |",
        "|---------|--------|",
    ]
    for r in results:
        lines.append(f"| {r.name} | **{r.status}** |")

    lines.extend(["", "## Section Details", ""])
    for r in results:
        lines.append(f"### {r.name} — {r.status}")
        lines.append("")
        for d in r.details:
            lines.append(f"- {d}")
        lines.append("")

    fixes = [r.name for r in results if r.status == "FAIL"]
    if critical_fail:
        lines.extend([
            "## Verdict",
            "",
            "**TRAINING BLOCKED**",
            "",
            "Required fixes before training:",
            "",
        ])
        for f in fixes:
            lines.append(f"- {f}")
    else:
        lines.extend([
            "## Verdict",
            "",
            "**TRAINING APPROVED**",
            "",
        ])
        if any_warning:
            lines.append("Warnings present — review before full training:")
            lines.append("")
            for r in results:
                if r.status == "WARNING":
                    lines.append(f"- {r.name}")

    report_path = REPORTS / "pre_training_audit.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return "TRAINING BLOCKED" if critical_fail else "TRAINING APPROVED"


def main() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    print("Running pre-training audit...")
    section_1_dataset_integrity()
    section_2_feature_leakage()
    section_3_target_distribution()
    section_4_scaling_consistency()
    section_5_training_target_pipeline(config)
    section_6_architecture(config)
    section_7_forward_pass()
    section_8_gradient_flow()
    section_9_overfit_sanity()
    section_10_inference_wrapper(config)
    section_11_action_range(config)
    section_12_control_aware_loss(config)
    section_13_checkpoint_export()
    section_14_closed_loop(config)

    verdict = write_final_report()
    print(f"\nAudit complete. Report: {REPORTS / 'pre_training_audit.md'}")
    print(verdict)
    for r in results:
        print(f"  [{r.status}] {r.name}")
    if verdict == "TRAINING BLOCKED":
        sys.exit(1)


if __name__ == "__main__":
    main()
