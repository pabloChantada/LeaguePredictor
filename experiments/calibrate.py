"""
Calibracion del win%.

El AUC mide ORDENAR (¿pone al ganador por encima?). La calibracion mide que el
numero SIGNIFIQUE algo: de todas las veces que decimos 70%, ¿gana el 70%?
Para un producto de "probabilidad de ganar" esto importa mas que el AUC.

Metricas:
  - ECE  (Expected Calibration Error): error medio |predicho - real| por bins.
  - Brier: error cuadratico medio de la probabilidad (mas bajo = mejor).

Entrena en train, AJUSTA la calibracion en valid (nunca en test, o mentiria) y
mide en test. Split por partida, igual que el resto.
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
MODEL_OUT = "models/modelo_calibrado.joblib"


def ece(y, p, n_bins=20):
    """Expected Calibration Error: media ponderada de |confianza - acierto real|."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return total


def reporte(nombre, y, p):
    print(f"{nombre:24s} auc={roc_auc_score(y, p):.4f}  logloss={log_loss(y, p):.4f}  "
          f"brier={brier_score_loss(y, p):.4f}  ECE={ece(y, p):.4f}")


def tabla_fiabilidad(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    print(f"\n  {'predicho':>12s} {'real':>8s} {'n':>8s}  desvio")
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 30:
            continue
        pm, ym = p[m].mean(), y[m].mean()
        print(f"  {pm:11.1%} {ym:8.1%} {m.sum():8d}  {ym - pm:+.1%}")


def main():
    df = load_dataset()  # filtra a soloQ; no leer el CSV a pelo (ver train.load_dataset)
    X, y, groups = df[FEATURES].values, df[TARGET].values, df["match_id"].values

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tv, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_r, va_r = next(inner.split(X[tv], y[tv], groups[tv]))
    train_idx, valid_idx = tv[tr_r], tv[va_r]

    print(f"Train {len(train_idx)} | Valid {len(valid_idx)} | Test {len(test_idx)} filas\n")

    base = GradientBoostingClassifier(random_state=SEED)
    base.fit(X[train_idx], y[train_idx])

    yte = y[test_idx]
    p_base = base.predict_proba(X[test_idx])[:, 1]
    reporte("SIN calibrar", yte, p_base)

    # la calibracion se AJUSTA en valid (datos que el modelo no vio)
    for metodo in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=metodo)
        cal.fit(X[valid_idx], y[valid_idx])
        p = cal.predict_proba(X[test_idx])[:, 1]
        reporte(f"calibrado ({metodo})", yte, p)

    print("\n--- Fiabilidad SIN calibrar (test) ---")
    tabla_fiabilidad(yte, p_base)

    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    cal.fit(X[valid_idx], y[valid_idx])
    p_cal = cal.predict_proba(X[test_idx])[:, 1]
    print("\n--- Fiabilidad CALIBRADO isotonic (test) ---")
    tabla_fiabilidad(yte, p_cal)

    joblib.dump({"model": cal, "features": FEATURES}, MODEL_OUT)
    print(f"\nGuardado en {MODEL_OUT}")


if __name__ == "__main__":
    main()
