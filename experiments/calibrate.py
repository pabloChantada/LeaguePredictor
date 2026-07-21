"""
Win% calibration.

AUC measures RANKING (does it put the winner above?). Calibration measures that the
number MEANS something: out of all the times we say 70%, does 70% win?
For a "win probability" product this matters more than AUC.

Metrics:
  - ECE  (Expected Calibration Error): mean |predicted - actual| per bin.
  - Brier: mean squared error of the probability (lower = better).

Trains on train, FITS the calibration on valid (never on test, or it would lie)
and measures on test. Split per match, same as the rest.
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from train import load_dataset, FEATURES, TARGET

SEED = 42
MODEL_OUT = "models/calibrated_model.joblib"


def ece(y, p, n_bins=20):
    """Expected Calibration Error: weighted mean of |confidence - actual accuracy|."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return total


def report(name, y, p):
    print(f"{name:24s} auc={roc_auc_score(y, p):.4f}  logloss={log_loss(y, p):.4f}  "
          f"brier={brier_score_loss(y, p):.4f}  ECE={ece(y, p):.4f}")


def reliability_table(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    print(f"\n  {'predicted':>12s} {'actual':>8s} {'n':>8s}  gap")
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 30:
            continue
        pm, ym = p[m].mean(), y[m].mean()
        print(f"  {pm:11.1%} {ym:8.1%} {m.sum():8d}  {ym - pm:+.1%}")


def main():
    df = load_dataset()  # filters to soloQ; do not read the CSV raw (see train.load_dataset)
    X, y, groups = df[FEATURES].values, df[TARGET].values, df["match_id"].values

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tv, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_r, va_r = next(inner.split(X[tv], y[tv], groups[tv]))
    train_idx, valid_idx = tv[tr_r], tv[va_r]

    print(f"Train {len(train_idx)} | Valid {len(valid_idx)} | Test {len(test_idx)} rows\n")

    base = GradientBoostingClassifier(random_state=SEED)
    base.fit(X[train_idx], y[train_idx])

    yte = y[test_idx]
    p_base = base.predict_proba(X[test_idx])[:, 1]
    report("uncalibrated", yte, p_base)

    # the calibration is FITTED on valid (data the model did not see)
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        cal.fit(X[valid_idx], y[valid_idx])
        p = cal.predict_proba(X[test_idx])[:, 1]
        report(f"calibrated ({method})", yte, p)

    print("\n--- Reliability UNCALIBRATED (test) ---")
    reliability_table(yte, p_base)

    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    cal.fit(X[valid_idx], y[valid_idx])
    p_cal = cal.predict_proba(X[test_idx])[:, 1]
    print("\n--- Reliability CALIBRATED isotonic (test) ---")
    reliability_table(yte, p_cal)

    joblib.dump({"model": cal, "features": FEATURES}, MODEL_OUT)
    print(f"\nSaved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
