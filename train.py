"""
Win prediction baseline (probability) from features.csv.

Key points:
- It is CLASSIFICATION with a probability output (predict_proba) = the "win %".
- Train/test split PER MATCH (GroupShuffleSplit over match_id): rows from the same
  match cannot land in train and test at once (otherwise the metrics inflate).
- Compares Logistic Regression (interpretable) and Gradient Boosting (non-linear).
- Reports accuracy, log-loss and ROC-AUC, plus accuracy PER MINUTE (early game is
  harder to predict than late game).
- Saves the best model to baseline_model.joblib.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent  # paths anchored to the file, not the CWD
CSV = ROOT / "riot_dataset" / "features.csv"
MODEL_OUT = ROOT / "models" / "baseline_model.joblib"

# LIVE-COMPAT set: 13 features. ALL obtainable from the Live Client Data API, so
# that the model trained here can be served live with no train/serve skew.
#
# It does NOT use gold or xp on purpose: live, they are not available for the 10
# players. Measured, dropping both costs only -0.0020 AUC (0.8674 -> 0.8654),
# because gold is a SUMMARY of cs+kills+objectives, which we do have: keeping the
# causes, we don't lose the sum.
FEATURES = [
    "minute",
    # team state (live: playerList -> scores.kills / creepScore / level)
    "kills_diff", "cs_diff", "level_diff",
    # objectives / structures (live: events) -> the group that contributes the most
    "tower_diff", "inhib_diff", "dragon_diff", "herald_diff", "baron_diff", "grub_diff",
    # momentum (live: keeping a running count between polls)
    "kills_diff_d5", "cs_diff_d5", "level_diff_d5",
    # TODO(championStats/damageStats): they are in the CSV but do NOT enter: -0.001 AUC.
    # TODO(scaling): scaling_diff dropped for now (see build_features.py)
]
TARGET = "blue_win"

# SoloQ 5v5 only. The crawler downloads ALL queues of each player (match-v5
# by-puuid/ids does not filter by queue), so features.csv carries ~19% of matches
# that are NOT the problem we model:
#   - Arena (1750): 2v2v2v2 with no towers/dragons/baron/inhibitors -> the 6
#     objective features are 0 STRUCTURALLY (measured: 100% of their rows), and
#     blue_win comes from teamId==100, which there does not identify the winner ->
#     an essentially random label. It was 19,175 rows of pure noise.
#   - Co-op vs AI (710/870/890): the human team almost always wins -> false bias.
#   - ARAM (450), Swiftplay (480): different map / different rules.
# SoloQ and nothing else: it is the relevant queue and the only one with serious
# rank-based matchmaking. Flex (440) was considered and dropped: it is 5v5 on the
# Rift and measures almost the same (AUC 0.836 vs 0.839 by queue), but they are
# premades playing differently and would only add +6% of matches. Not worth
# muddying the definition of the problem.
QUEUE_SOLOQ = 420

# Elo band that enters training (`tier` column, from crawler.SEED_TIERS).
# The model is served on YOUR matches and the Live Client Data does not expose the
# rank of the 10 players -> elo cannot be a feature: you have to train in the band
# where it is served. See "Elo" in the README.
# TIERS = None disables the filter (useful to compare against the old Challenger+GM
# crawl, which has no provenance recorded and comes out as UNKNOWN).
TIERS = ("EMERALD", "DIAMOND", "MASTER")


def load_dataset(csv=CSV, queue_id=QUEUE_SOLOQ, tiers=TIERS):
    """features.csv filtered to the target queue and elo band.

    SINGLE load point: train and experiments/ go through here so nobody
    accidentally trains with Arena, matches against bots or two blended elo bands.
    """
    df = pd.read_csv(csv)
    for col in ("queue_id", "tier"):
        if col not in df.columns:
            raise SystemExit(
                f"{csv} does not have the {col} column (it is from an older version).\n"
                "Rebuild it:  python build_features.py"
            )
    n_before = df["match_id"].nunique()

    df = df[df["queue_id"] == queue_id]
    n_queue = df["match_id"].nunique()
    print(f"Queue {queue_id}: {n_queue}/{n_before} matches "
          f"({n_before - n_queue} dropped for not being soloQ)")

    if tiers is not None:
        df = df[df["tier"].isin(tiers)]
        n_tier = df["match_id"].nunique()
        print(f"Band {list(tiers)}: {n_tier}/{n_queue} matches "
              f"({n_queue - n_tier} dropped for elo out of band or unrecorded)")

    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit(
            f"No match left (queue {queue_id}, band {list(tiers)}).\n"
            "This is expected until you crawl the band: the old dataset was seeded\n"
            "in Challenger+GM and has no elo provenance recorded (comes out UNKNOWN).\n"
            "  - crawl the target band:                 python crawler.py\n"
            "  - train with the old dataset anyway:     TIERS = None in train.py"
        )
    return df


def main():
    df = load_dataset()
    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df["match_id"].values

    # --- split per match ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    n_games = len(np.unique(groups))
    print(f"Rows: {len(df)}  Matches: {n_games}")
    print(f"Train: {len(Xtr)} rows / {len(np.unique(groups[train_idx]))} matches")
    print(f"Test:  {len(Xte)} rows / {len(np.unique(groups[test_idx]))} matches")
    print(f"Trivial baseline (always majority): acc={max(yte.mean(), 1-yte.mean()):.3f}\n")

    models = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "GradBoost": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(yte, pred)
        ll = log_loss(yte, proba)
        auc = roc_auc_score(yte, proba)
        results[name] = (model, acc, ll, auc, proba)
        print(f"{name:10s}  acc={acc:.3f}  logloss={ll:.3f}  auc={auc:.3f}")

    # --- logistic regression coefficients (interpretable importance) ---
    logreg = results["LogReg"][0].named_steps["logisticregression"]
    print("\nLogReg weights (standardized, + favors the blue team):")
    for f, c in sorted(zip(FEATURES, logreg.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:16s} {c:+.3f}")

    # --- accuracy per minute bucket (best model by auc) ---
    best_name = max(results, key=lambda k: results[k][3])
    best_model, _, _, _, best_proba = results[best_name]
    print(f"\nAccuracy per minute ({best_name}):")
    dte = df.iloc[test_idx].copy()
    dte["proba"] = best_proba
    dte["pred"] = (dte["proba"] >= 0.5).astype(int)
    dte["bucket"] = pd.cut(dte["minute"], [0, 10, 15, 20, 25, 200],
                           labels=["<10", "10-15", "15-20", "20-25", ">25"])
    for b, g in dte.groupby("bucket", observed=True):
        print(f"  min {b:>6}: acc={accuracy_score(g[TARGET], g['pred']):.3f}  (n={len(g)})")

    joblib.dump({"model": best_model, "features": FEATURES}, MODEL_OUT)
    print(f"\nBest model ({best_name}) saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
