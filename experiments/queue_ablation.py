"""
QUEUE ablation: how much does it cost to train with non-soloQ matches?

The crawler requested match-v5 by-puuid/ids WITHOUT the `queue` parameter, so it
pulled all of each player's queues. Measured over features.csv: 19.1% of the 9,839
matches are not soloQ (420). The worst is Arena (1750): 2v2v2v2 with no towers,
dragons, baron or inhibitors -> the 6 objective features are 0 STRUCTURALLY in
100% of their 19,175 rows, and blue_win comes from teamId==100, which there does
not identify the winner. Those were rows with an essentially random label.

The question is not "are those rows bad?" but "do they make the model worse ON
SOLOQ?". So the test is ALWAYS pure soloQ, and the only thing that changes is what
it is trained on:

    A) all queues     (what used to be done)
    B) only soloQ 420 (what train.load_dataset does now)

Note while reading it: A trains with MORE rows than B. If B wins, it wins despite
having less data -> the result is stronger, not weaker.

    python -m experiments.queue_ablation
"""
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from train import CSV, FEATURES, TARGET, QUEUE_SOLOQ
from experiments.calibrate import ece

SEED = 42
TEST_FRAC = 0.2


def evaluate(name, model, Xte, yte):
    p = model.predict_proba(Xte)[:, 1]
    print(f"  {name:22s} auc={roc_auc_score(yte, p):.4f}  "
          f"logloss={log_loss(yte, p):.4f}  brier={brier_score_loss(yte, p):.4f}  "
          f"ECE={ece(yte, p):.4f}")
    return roc_auc_score(yte, p), ece(yte, p)


def main():
    df = pd.read_csv(CSV)
    if "queue_id" not in df.columns:
        raise SystemExit("features.csv without queue_id -> python build_features.py --cached-only")

    print("Queue breakdown in features.csv (matches):")
    by_queue = df.groupby("queue_id")["match_id"].nunique().sort_values(ascending=False)
    for q, n in by_queue.items():
        mark = "  <- target" if q == QUEUE_SOLOQ else ""
        print(f"  {q:>5}: {n:>5} ({n / df['match_id'].nunique():.1%}){mark}")

    # --- test: ALWAYS pure soloQ, and per match (never rows of the same match on both sides)
    soloq_games = df.loc[df.queue_id == QUEUE_SOLOQ, "match_id"].unique()
    rng = np.random.RandomState(SEED)
    test_games = set(rng.choice(soloq_games, size=int(TEST_FRAC * len(soloq_games)),
                               replace=False))
    is_test = df.match_id.isin(test_games)
    test, rest = df[is_test], df[~is_test]

    Xte, yte = test[FEATURES].values, test[TARGET].values
    print(f"\nTest (pure soloQ): {len(test)} rows / {test.match_id.nunique()} matches")

    trainings = {
        "A) all queues": rest,
        "B) only soloQ 420": rest[rest.queue_id == QUEUE_SOLOQ],
    }
    res = {}
    for name, tr in trainings.items():
        print(f"\n{name}: {len(tr)} rows / {tr.match_id.nunique()} matches")
        m = GradientBoostingClassifier(random_state=SEED)
        m.fit(tr[FEATURES].values, tr[TARGET].values)
        res[name] = evaluate("-> on soloQ test", m, Xte, yte)

    (auc_a, ece_a), (auc_b, ece_b) = res["A) all queues"], res["B) only soloQ 420"]
    # ASCII on purpose: the Windows console is cp1252 and a unicode "Delta" blows up
    # the whole script when printing (UnicodeEncodeError).
    print(f"\nAfter cleaning the queue:  AUC {auc_b - auc_a:+.4f}   ECE {ece_b - ece_a:+.4f}")
    print("(AUC: higher is better. ECE: lower is better.)")


if __name__ == "__main__":
    main()
