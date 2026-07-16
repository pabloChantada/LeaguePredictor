"""
Modelo SECUENCIAL (LSTM) de win probability.

A diferencia del modelo tabular (cada minuto es un punto independiente), aqui la
red ve la SECUENCIA de minutos de la partida y emite una probabilidad en cada
minuto, pudiendo aprender la trayectoria (snowball, remontadas) por si misma.

CORRECCION CLAVE: la LSTM es UNIDIRECCIONAL. La salida en el minuto t solo
depende de las entradas hasta t. Una LSTM bidireccional veria el futuro de la
partida -> leakage (AUC irreal) y seria inservible en vivo.

Usa EL MISMO split de test por partida (random_state=42) que train.py y
train_lgbm.py, y el mismo set de features, para que el AUC sea comparable.
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
    """LSTM unidireccional -> logit de victoria azul en CADA minuto."""

    def __init__(self, n_features, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):            # x: (B, T, F)
        out, _ = self.lstm(x)        # (B, T, H) — out[:, t] solo ve hasta t
        return self.head(out).squeeze(-1)   # (B, T) logits


def build_sequences(by_game, game_ids, feat_mean, feat_std, max_len):
    """Convierte las partidas dadas en tensores padded (X, y, mask, minutes).

    by_game: dict {match_id: sub-dataframe ya ordenado por minuto}.
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
        y[i, :t] = g[TARGET].values[0]      # etiqueta constante en toda la partida
        mask[i, :t] = 1.0
        minutes[i, :t] = g["minute"].values[:t]

    return (torch.tensor(X), torch.tensor(y), torch.tensor(mask), torch.tensor(minutes))


def masked_auc(logits, y, mask):
    """AUC solo sobre los timesteps validos (padding excluido)."""
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

    # mismo split de test que los modelos tabulares
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_idx, test_idx = next(gss.split(df[FEATURES].values, df[TARGET].values, groups))
    test_games = np.unique(groups[test_idx])
    trainval_games = np.unique(groups[trainval_idx])

    # validacion por partida para early stopping
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(trainval_games))
    n_val = int(0.2 * len(trainval_games))
    valid_games = trainval_games[perm[:n_val]]
    train_games = trainval_games[perm[n_val:]]

    # normalizacion con estadisticos SOLO de train
    tr_rows = df[df["match_id"].isin(set(train_games))]
    feat_mean = tr_rows[FEATURES].values.mean(axis=0)
    feat_std = tr_rows[FEATURES].values.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    max_len = int(df.groupby("match_id").size().max())
    print(f"Games: train={len(train_games)} valid={len(valid_games)} test={len(test_games)}")
    print(f"Longitud maxima de secuencia: {max_len} minutos\n")

    print("Construyendo secuencias...", flush=True)
    # una sola pasada de groupby en vez de filtrar el df por cada partida
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
            loss = (lossf(logits, yb) * mb).sum() / mb.sum()  # BCE enmascarada
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
                print(f"early stopping (sin mejora en {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    auc, p, t, minutes = evaluate(model, te_loader)
    pred = (p >= 0.5).astype(int)
    print(f"\nTEST  acc={accuracy_score(t, pred):.3f}  logloss={log_loss(t, p):.3f}  auc={auc:.3f}")

    print("\nAccuracy por minuto (LSTM):")
    dte = pd.DataFrame({"minute": minutes, "y": t, "pred": pred})
    dte["bucket"] = pd.cut(dte["minute"], [0, 10, 15, 20, 25, 200],
                           labels=["<10", "10-15", "15-20", "20-25", ">25"])
    for b, g in dte.groupby("bucket", observed=True):
        print(f"  min {b:>6}: acc={accuracy_score(g['y'], g['pred']):.3f}  (n={len(g)})")

    torch.save({"state_dict": model.state_dict(), "features": FEATURES,
                "mean": feat_mean, "std": feat_std}, "models/modelo_lstm.pt")
    print("\nGuardado en models/modelo_lstm.pt")


if __name__ == "__main__":
    main()
