"""
Win-probability baseline from features.csv.

- Classification with predict_proba (the "win %").
- Train/test split PER MATCH (GroupShuffleSplit on match_id), so rows from the
  same match never land in both splits.
- Compares Logistic Regression and Gradient Boosting; reports accuracy,
  log-loss, ROC-AUC, and accuracy per minute bucket.
- Saves the best model to config.MODEL_OUT.
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import config

FEATURES = config.FEATURES
TARGET = config.TARGET


def load_dataset(csv=config.FEATURES_CSV, queue_id=config.QUEUE_ID, tiers=config.TIERS):
    """features.csv filtered to the target queue and elo band.

    Pass tiers=None to disable the elo filter (e.g. to use
    an older crawl with no recorded elos, which comes out as UNKNOWN).
    """
    df = pd.read_csv(csv)
    for col in ("queue_id", "tier"):
        if col not in df.columns:
            raise SystemExit(
                f"{csv} does not have the {col} column (it is from an older version).\n"
                "Rebuild it:  python build_features.py"
            )

    # Safeguard against accidentally training on a different queue than the one we crawled for
    n_before = df["match_id"].nunique()
    df = df[df["queue_id"] == queue_id]
    n_queue = df["match_id"].nunique()
    print(f"Queue {queue_id}: {n_queue}/{n_before} matches "
          f"({n_before - n_queue} dropped for not being soloQ)")

    # Remove matches that are out of the target band (or have no recorded elo)
    if tiers is not None:
        df = df[df["tier"].isin(tiers)]
        n_tier = df["match_id"].nunique()
        print(f"Band {list(tiers)}: {n_tier}/{n_queue} matches "
              f"({n_queue - n_tier} dropped for elo out of band or unrecorded)")

    df = df.reset_index(drop=True)
    if df.empty:
        band_desc = list(tiers) if tiers is not None else "ALL (no tier filter)"
        raise SystemExit(
            f"No matches left (queue {queue_id}, band {band_desc}).\n"
            "  - crawl the target band:                 python crawler.py\n"
            "  - train with the old dataset anyway:     load_dataset(tiers=None)"
        )
    return df


def main(tiers=config.TIERS):
    df = load_dataset(tiers=tiers)
    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df["match_id"].values

    # Split by match_id, so rows from the same match never land in both splits.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    n_games = len(np.unique(groups))
    print(f"Rows: {len(df)}  Matches: {n_games}")
    print(f"Train: {len(Xtr)} rows / {len(np.unique(groups[train_idx]))} matches")
    print(f"Test:  {len(Xte)} rows / {len(np.unique(groups[test_idx]))} matches")

    models = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "GradBoost": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        # use int since it is a binary classification problem (0 or 1)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(yte, pred)
        ll = log_loss(yte, proba)
        auc = roc_auc_score(yte, proba)
        results[name] = (model, acc, ll, auc, proba)
        print(f"{name:10s}  acc={acc:.3f}  logloss={ll:.3f}  auc={auc:.3f}")

    logreg = results["LogReg"][0].named_steps["logisticregression"]
    print("\nLogReg weights (standardized, + favors the blue team):")
    # Sort by absolute value of the coefficient, so the most important features are at the top.
    # Most likely gold, xp and kills
    for f, c in sorted(zip(FEATURES, logreg.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:16s} {c:+.3f}")

    # Select the best model by AUC 
    best_name = max(results, key=lambda k: results[k][3])
    best_model, _, _, _, best_proba = results[best_name]
    # Show acc to see the difference between early, mid and late game
    print(f"\nAccuracy per minute ({best_name}):")
    dte = df.iloc[test_idx].copy()
    dte["proba"] = best_proba
    dte["pred"] = (dte["proba"] >= 0.5).astype(int)
    dte["bucket"] = pd.cut(dte["minute"], [0, 10, 15, 20, 25, 200],
                           labels=["<10", "10-15", "15-20", "20-25", ">25"])
    for b, g in dte.groupby("bucket", observed=True):
        print(f"  min {b:>6}: acc={accuracy_score(g[TARGET], g['pred']):.3f}  (n={len(g)})")

    joblib.dump({"model": best_model, "features": FEATURES}, config.MODEL_OUT)
    print(f"\nBest model ({best_name}) saved to {config.MODEL_OUT}")

if __name__ == "__main__":
    main(tiers=None)