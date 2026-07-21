"""
SEQUENTIAL (LSTM) win probability model.

Unlike the tabular model (each minute is an independent point), here the network
sees the SEQUENCE of minutes of the match and emits a probability at each minute,
so it can learn the trajectory (snowball, comebacks) by itself.

KEY CORRECTNESS NOTE: the LSTM is UNIDIRECTIONAL. The output at minute t only
depends on the inputs up to t. A bidirectional LSTM would see the future of the
match -> leakage (unrealistic AUC) and would be useless live.

Uses THE SAME per-match test split (random_state=42) as train.py and
train_lgbm.py, and the same feature set, so the AUC is comparable.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from train import CSV, FEATURES, TARGET

SEED = 42
HIDDEN = 64
LAYERS = 1
EPOCHS = 40
PATIENCE = 5
BATCH = 64
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)


class LSTMWinProb(nn.Module):
    """Unidirectional LSTM -> blue-win logit at EACH minute."""

    def __init__(self, n_features, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):            # x: (B, T, F)
        out, _ = self.lstm(x)        # (B, T, H) — out[:, t] only sees up to t
        return self.head(out).squeeze(-1)   # (B, T) logits


def build_sequences(by_game, game_ids, feat_mean, feat_std, max_len):
    """Turns the given matches into padded tensors (X, y, mask, minutes).

    by_game: dict {match_id: sub-dataframe already sorted by minute}.
    """
    n = len(game_ids)
    X = np.zeros((n, max_len, len(FEATURES)), dtype=np.float32)
    y = np.zeros((n, max_len), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=np.float32)
    minutes = np.zeros((n, max_len), dtype=np.int32)

    for i, gid in enumerate(game_ids):
        g = by_game[gid]
        t = min(len(g), max_len)
        feats = (g[FEATURES].values[:t] - feat_mean) / feat_std
        X[i, :t] = feats
        y[i, :t] = g[TARGET].values[0]      # constant label over the whole match
        mask[i, :t] = 1.0
        minutes[i, :t] = g["minute"].values[:t]

    return (torch.tensor(X), torch.tensor(y), torch.tensor(mask), torch.tensor(minutes))


def masked_auc(logits, y, mask):
    """AUC only over the valid timesteps (padding excluded)."""
    sel = mask.bool()
    p = torch.sigmoid(logits[sel]).cpu().numpy()
    t = y[sel].cpu().numpy()
    return roc_auc_score(t, p), p, t


def evaluate(model, loader):
    model.eval()
    L, Y, M, MIN = [], [], [], []
    with torch.no_grad():
        for xb, yb, mb, minb in loader:
            L.append(model(xb)); Y.append(yb); M.append(mb); MIN.append(minb)
    logits, y, mask, minutes = torch.cat(L), torch.cat(Y), torch.cat(M), torch.cat(MIN)
    auc, p, t = masked_auc(logits, y, mask)
    return auc, p, t, minutes[mask.bool()].cpu().numpy()


def main():
    df = pd.read_csv(CSV)
    groups = df["match_id"].values

    # same test split as the tabular models
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_idx, test_idx = next(gss.split(df[FEATURES].values, df[TARGET].values, groups))
    test_games = np.unique(groups[test_idx])
    trainval_games = np.unique(groups[trainval_idx])

    # per-match validation for early stopping
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(trainval_games))
    n_val = int(0.2 * len(trainval_games))
    valid_games = trainval_games[perm[:n_val]]
    train_games = trainval_games[perm[n_val:]]

    # normalization with statistics from train ONLY
    tr_rows = df[df["match_id"].isin(set(train_games))]
    feat_mean = tr_rows[FEATURES].values.mean(axis=0)
    feat_std = tr_rows[FEATURES].values.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    max_len = int(df.groupby("match_id").size().max())
    print(f"Games: train={len(train_games)} valid={len(valid_games)} test={len(test_games)}")
    print(f"Max sequence length: {max_len} minutes\n")

    print("Building sequences...", flush=True)
    # a single groupby pass instead of filtering the df for each match
    by_game = {gid: g.sort_values("minute") for gid, g in df.groupby("match_id")}
    tr = build_sequences(by_game, train_games, feat_mean, feat_std, max_len)
    va = build_sequences(by_game, valid_games, feat_mean, feat_std, max_len)
    te = build_sequences(by_game, test_games, feat_mean, feat_std, max_len)

    tr_loader = DataLoader(TensorDataset(*tr), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(TensorDataset(*va), batch_size=256)
    te_loader = DataLoader(TensorDataset(*te), batch_size=256)

    model = LSTMWinProb(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    best_auc, best_state, bad = -1, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb, mb, _ in tr_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = (lossf(logits, yb) * mb).sum() / mb.sum()  # masked BCE
            loss.backward()
            opt.step()
            total += loss.item()
        auc_va, *_ = evaluate(model, va_loader)
        print(f"epoch {epoch:2d}  train_loss={total/len(tr_loader):.4f}  valid_auc={auc_va:.4f}", flush=True)
        if auc_va > best_auc:
            best_auc, bad = auc_va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping (no improvement in {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    auc, p, t, minutes = evaluate(model, te_loader)
    pred = (p >= 0.5).astype(int)
    print(f"\nTEST  acc={accuracy_score(t, pred):.3f}  logloss={log_loss(t, p):.3f}  auc={auc:.3f}")

    print("\nAccuracy per minute (LSTM):")
    dte = pd.DataFrame({"minute": minutes, "y": t, "pred": pred})
    dte["bucket"] = pd.cut(dte["minute"], [0, 10, 15, 20, 25, 200],
                           labels=["<10", "10-15", "15-20", "20-25", ">25"])
    for b, g in dte.groupby("bucket", observed=True):
        print(f"  min {b:>6}: acc={accuracy_score(g['y'], g['pred']):.3f}  (n={len(g)})")

    torch.save({"state_dict": model.state_dict(), "features": FEATURES,
                "mean": feat_mean, "std": feat_std}, "models/lstm_model.pt")
    print("\nSaved to models/lstm_model.pt")


if __name__ == "__main__":
    main()
