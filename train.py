"""
Baseline de prediccion de victoria (probabilidad) a partir de features.csv.

Puntos clave:
- Es CLASIFICACION con salida de probabilidad (predict_proba) = el "% de ganar".
- Split train/test POR PARTIDA (GroupShuffleSplit sobre match_id): filas del mismo
  match no pueden caer en train y test a la vez (si no, las metricas se inflan).
- Compara Regresion Logistica (interpretable) y Gradient Boosting (no lineal).
- Reporta accuracy, log-loss y ROC-AUC, ademas de accuracy POR MINUTO (el juego
  temprano es mas dificil de predecir que el tardio).
- Guarda el mejor modelo en modelo_baseline.joblib.
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

ROOT = Path(__file__).resolve().parent  # rutas ancladas al fichero, no al CWD
CSV = ROOT / "riot_dataset" / "features.csv"
MODEL_OUT = ROOT / "models" / "modelo_baseline.joblib"

# Set LIVE-COMPAT: 13 features. TODAS obtenibles del Live Client Data API, para
# que el modelo entrenado aqui se pueda servir en vivo sin train/serve skew.
#
# NO usa oro ni xp a proposito: en vivo no estan disponibles para los 10 jugadores.
# Medido, prescindir de ambos cuesta solo -0.0020 de AUC (0.8674 -> 0.8654), porque
# el oro es un RESUMEN de cs+kills+objetivos, que si tenemos: al conservar las
# causas no se pierde la suma.
FEATURES = [
    "minute",
    # estado de equipo (live: playerList -> scores.kills / creepScore / level)
    "kills_diff", "cs_diff", "level_diff",
    # objetivos / estructuras (live: eventos) -> el grupo que mas aporta
    "tower_diff", "inhib_diff", "dragon_diff", "herald_diff", "baron_diff", "grub_diff",
    # momentum (live: llevando la cuenta entre polls)
    "kills_diff_d5", "cs_diff_d5", "level_diff_d5",
    # TODO(championStats/damageStats): estan en el CSV pero NO entran: -0.001 de AUC.
    # TODO(escalado): scaling_diff descartado de momento (ver build_features.py)
]
TARGET = "blue_win"

# Solo soloQ 5v5. El crawler baja TODAS las colas de cada jugador (match-v5
# by-puuid/ids no filtra por cola), asi que features.csv trae ~19% de partidas
# que NO son el problema que modelamos:
#   - Arena (1750): 2v2v2v2 sin torres/dragones/baron/inhibidores -> las 6
#     features de objetivos son 0 ESTRUCTURAL (medido: el 100% de sus filas), y
#     blue_win sale de teamId==100, que ahi no identifica al ganador -> etiqueta
#     practicamente aleatoria. Eran 19.175 filas de ruido puro.
#   - Co-op vs IA (710/870/890): el equipo humano casi siempre gana -> sesgo falso.
#   - ARAM (450), Swiftplay (480): otro mapa / otras reglas.
# SoloQ y nada mas: es la cola relevante y la unica con matchmaking serio por
# rango. Flex (440) se considero y se descarto: es 5v5 en Grieta y mide casi
# igual (AUC 0.836 vs 0.839 por cola), pero son premades jugando distinto y solo
# aportaria un +6% de partidas. No merece ensuciar la definicion del problema.
QUEUE_SOLOQ = 420

# Banda de elo que entra a entrenar (columna `tier`, de crawler.SEED_TIERS).
# El modelo se sirve en TUS partidas y el Live Client Data no expone el rango de
# los 10 jugadores -> el elo no puede ser feature: hay que entrenar en la banda
# en la que se sirve. Ver "El elo" en el README.
# TIERS = None desactiva el filtro (util para comparar contra el crawl viejo de
# Challenger+GM, que no tiene procedencia registrada y sale como UNKNOWN).
TIERS = ("EMERALD", "DIAMOND", "MASTER")


def load_dataset(csv=CSV, queue_id=QUEUE_SOLOQ, tiers=TIERS):
    """features.csv filtrado a la cola y la banda de elo objetivo.

    Punto UNICO de carga: train y experiments/ pasan por aqui para que nadie
    entrene sin querer con Arena, partidas contra bots o dos bandas de elo
    revueltas.
    """
    df = pd.read_csv(csv)
    for col in ("queue_id", "tier"):
        if col not in df.columns:
            raise SystemExit(
                f"{csv} no tiene la columna {col} (es de una version anterior).\n"
                "Reconstruyelo:  python build_features.py"
            )
    n_antes = df["match_id"].nunique()

    df = df[df["queue_id"] == queue_id]
    n_cola = df["match_id"].nunique()
    print(f"Cola {queue_id}: {n_cola}/{n_antes} partidas "
          f"({n_antes - n_cola} descartadas por no ser soloQ)")

    if tiers is not None:
        df = df[df["tier"].isin(tiers)]
        n_tier = df["match_id"].nunique()
        print(f"Banda {list(tiers)}: {n_tier}/{n_cola} partidas "
              f"({n_cola - n_tier} descartadas por elo fuera de banda o sin registrar)")

    df = df.reset_index(drop=True)
    if df.empty:
        raise SystemExit(
            f"No queda ninguna partida (cola {queue_id}, banda {list(tiers)}).\n"
            "Es lo esperado hasta que crawlees la banda: el dataset viejo se sembro\n"
            "en Challenger+GM y no tiene procedencia de elo registrada (sale UNKNOWN).\n"
            "  - crawlear la banda objetivo:            python crawler.py\n"
            "  - entrenar con el dataset viejo igual:   TIERS = None en train.py"
        )
    return df


def main():
    df = load_dataset()
    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df["match_id"].values

    # --- split por partida ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    n_games = len(np.unique(groups))
    print(f"Filas: {len(df)}  Partidas: {n_games}")
    print(f"Train: {len(Xtr)} filas / {len(np.unique(groups[train_idx]))} partidas")
    print(f"Test:  {len(Xte)} filas / {len(np.unique(groups[test_idx]))} partidas")
    print(f"Baseline trivial (siempre mayoria): acc={max(yte.mean(), 1-yte.mean()):.3f}\n")

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

    # --- coeficientes de la regresion logistica (importancia interpretable) ---
    logreg = results["LogReg"][0].named_steps["logisticregression"]
    print("\nPesos LogReg (estandarizados, + favorece al equipo azul):")
    for f, c in sorted(zip(FEATURES, logreg.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:16s} {c:+.3f}")

    # --- accuracy por tramo de minuto (mejor modelo por auc) ---
    best_name = max(results, key=lambda k: results[k][3])
    best_model, _, _, _, best_proba = results[best_name]
    print(f"\nAccuracy por minuto ({best_name}):")
    dte = df.iloc[test_idx].copy()
    dte["proba"] = best_proba
    dte["pred"] = (dte["proba"] >= 0.5).astype(int)
    dte["bucket"] = pd.cut(dte["minute"], [0, 10, 15, 20, 25, 200],
                           labels=["<10", "10-15", "15-20", "20-25", ">25"])
    for b, g in dte.groupby("bucket", observed=True):
        print(f"  min {b:>6}: acc={accuracy_score(g[TARGET], g['pred']):.3f}  (n={len(g)})")

    joblib.dump({"model": best_model, "features": FEATURES}, MODEL_OUT)
    print(f"\nMejor modelo ({best_name}) guardado en {MODEL_OUT}")


if __name__ == "__main__":
    main()
