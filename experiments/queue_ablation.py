"""
Ablacion de COLA: ¿cuanto cuesta entrenar con partidas que no son soloQ?

El crawler pide match-v5 by-puuid/ids SIN el parametro `queue`, asi que se trajo
todas las colas de cada jugador. Medido sobre features.csv: el 19.1% de las 9.839
partidas no es soloQ (420). Lo peor es Arena (1750): 2v2v2v2 sin torres, dragones,
baron ni inhibidores -> las 6 features de objetivos son 0 ESTRUCTURAL en el 100%
de sus 19.175 filas, y blue_win sale de teamId==100, que ahi no identifica al
ganador. Eran filas con la etiqueta practicamente aleatoria.

La pregunta no es "¿son malas esas filas?" sino "¿empeoran el modelo EN SOLOQ?".
Asi que el test es SIEMPRE soloQ puro, y lo unico que cambia es con que se
entrena:

    A) todas las colas  (lo que se venia haciendo)
    B) solo soloQ 420   (lo que hace train.load_dataset ahora)

Ojo al leerlo: A entrena con MAS filas que B. Si B gana, gana pese a tener menos
datos -> el resultado es mas fuerte, no mas debil.

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


def evalua(nombre, modelo, Xte, yte):
    p = modelo.predict_proba(Xte)[:, 1]
    print(f"  {nombre:22s} auc={roc_auc_score(yte, p):.4f}  "
          f"logloss={log_loss(yte, p):.4f}  brier={brier_score_loss(yte, p):.4f}  "
          f"ECE={ece(yte, p):.4f}")
    return roc_auc_score(yte, p), ece(yte, p)


def main():
    df = pd.read_csv(CSV)
    if "queue_id" not in df.columns:
        raise SystemExit("features.csv sin queue_id -> python build_features.py --cached-only")

    print("Reparto de colas en features.csv (partidas):")
    por_cola = df.groupby("queue_id")["match_id"].nunique().sort_values(ascending=False)
    for q, n in por_cola.items():
        marca = "  <- objetivo" if q == QUEUE_SOLOQ else ""
        print(f"  {q:>5}: {n:>5} ({n / df['match_id'].nunique():.1%}){marca}")

    # --- test: SIEMPRE soloQ puro, y por partida (nunca filas del mismo match a los dos lados)
    soloq_games = df.loc[df.queue_id == QUEUE_SOLOQ, "match_id"].unique()
    rng = np.random.RandomState(SEED)
    test_games = set(rng.choice(soloq_games, size=int(TEST_FRAC * len(soloq_games)),
                               replace=False))
    es_test = df.match_id.isin(test_games)
    test, resto = df[es_test], df[~es_test]

    Xte, yte = test[FEATURES].values, test[TARGET].values
    print(f"\nTest (soloQ puro): {len(test)} filas / {test.match_id.nunique()} partidas")

    entrenos = {
        "A) todas las colas": resto,
        "B) solo soloQ 420": resto[resto.queue_id == QUEUE_SOLOQ],
    }
    res = {}
    for nombre, tr in entrenos.items():
        print(f"\n{nombre}: {len(tr)} filas / {tr.match_id.nunique()} partidas")
        m = GradientBoostingClassifier(random_state=SEED)
        m.fit(tr[FEATURES].values, tr[TARGET].values)
        res[nombre] = evalua("-> en test soloQ", m, Xte, yte)

    (auc_a, ece_a), (auc_b, ece_b) = res["A) todas las colas"], res["B) solo soloQ 420"]
    # ASCII a proposito: la consola de Windows es cp1252 y un "Delta" unicode
    # revienta el script entero al imprimir (UnicodeEncodeError).
    print(f"\nAl limpiar la cola:  AUC {auc_b - auc_a:+.4f}   ECE {ece_b - ece_a:+.4f}")
    print("(AUC: mas alto mejor. ECE: mas bajo mejor.)")


if __name__ == "__main__":
    main()
