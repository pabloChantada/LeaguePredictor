"""
Nucleo de la prediccion EN VIVO.

Lee el estado de la partida en curso del Live Client Data API (corre en tu propio
PC mientras juegas, en cualquier modo: ranked, normal, CUSTOM o practice tool),
construye las MISMAS 13 features con las que se entreno el modelo y devuelve el
win%.

Por que estas 13 y no el oro: el modelo se entreno a proposito sin oro ni xp
porque esta API no los da para los 10 jugadores. Ver train.FEATURES.

Uso directo (modo consola):   python live_predict.py
Dashboard:                    streamlit run live_dashboard.py
"""
import json
import time
import warnings

import joblib
import requests
import urllib3

from train import FEATURES, MODEL_OUT

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# El endpoint es local y usa un certificado autofirmado de Riot -> verify=False.
URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
POLL_SECONDS = 10
DELTA_WINDOW = 5  # minutos, igual que en build_features.DELTA_WINDOW
# Margen para dar el reloj por "retrocedido". No hace falta que sea grande:
# gameTime es monotono dentro de una partida; solo cubre jitter del float.
NEW_GAME_TOLERANCE = 5.0  # segundos

BLUE, RED = "ORDER", "CHAOS"   # ORDER = lado azul (teamId 100) = perspectiva del modelo


class NoGameRunning(Exception):
    """No hay partida en curso (o el cliente no expone el endpoint todavia)."""


def fetch_live_data(timeout=3):
    try:
        r = requests.get(URL, verify=False, timeout=timeout)
    except requests.exceptions.RequestException as ex:
        raise NoGameRunning(f"No se pudo conectar a {URL}: {ex}") from ex
    if r.status_code != 200:
        raise NoGameRunning(f"El cliente respondio {r.status_code}")
    return r.json()


def game_time(data):
    """Segundos transcurridos de la partida en curso."""
    return data.get("gameData", {}).get("gameTime", 0.0)


def is_new_game(prev_time, cur_time):
    """True si `cur_time` es de una partida DISTINTA a la de `prev_time`.

    El Live Client Data API no expone ningun id de partida (gameData solo trae
    gameMode/gameTime/mapName), asi que la unica señal fiable es que el reloj
    retroceda: al empezar otra partida vuelve a ~0.

    Reconectar a la MISMA partida (un corte del cliente, un F5 del dashboard)
    deja el reloj avanzando -> no dispara reset y el historial sobrevive, que es
    justo lo que queremos: resetear por una desconexion perderia la curva.
    """
    return prev_time is not None and cur_time < prev_time - NEW_GAME_TOLERANCE


def _other(team):
    return RED if team == BLUE else BLUE


def _team_by_player(data):
    """Nombre de jugador -> equipo, para atribuir eventos a un bando.

    KillerName en los eventos trae SOLO el game name ("ChantaClown"), sin el tag,
    mientras que riotId/summonerName vienen con el ("ChantaClown#milk"). Sin
    riotIdGameName ningun objetivo se atribuye y dragon/baron/grub salen a 0.
    """
    out = {}
    for p in data.get("allPlayers", []):
        for key in ("riotIdGameName", "riotId", "summonerName"):
            if p.get(key):
                out[p[key]] = p.get("team")
    return out


def _structure_owner(name):
    """Equipo DUEÑO de la estructura, leido de su nombre interno.

    El cliente las nombra "Turret_TOrder_L0_P3_..." / "Inhib_TChaos_L1_P1_...";
    se acepta tambien el formato antiguo "_T1_"/"_T2_". Devuelve None si no se
    reconoce, para no atribuir a ciegas.
    """
    if "TOrder" in name or "_T1_" in name:
        return BLUE
    if "TChaos" in name or "_T2_" in name:
        return RED
    return None


# Contadores que read_counters devuelve, en orden de marcador. La clave es la
# misma que usa read_state para nombrar su diff ("kills" -> "kills_diff").
COUNTERS = ["kills", "cs", "level", "towers", "inhibs",
            "dragons", "heralds", "barons", "grubs"]
# nombre de la feature del modelo para cada contador (no todos son plural regular)
DIFF_NAME = {"kills": "kills_diff", "cs": "cs_diff", "level": "level_diff",
             "towers": "tower_diff", "inhibs": "inhib_diff", "dragons": "dragon_diff",
             "heralds": "herald_diff", "barons": "baron_diff", "grubs": "grub_diff"}


def read_counters(data):
    """Contadores BRUTOS por equipo: {contador: {ORDER: n, CHAOS: n}}.

    Separado de read_state porque el modelo solo quiere diffs, pero el dashboard
    necesita los absolutos de cada bando para pintar el marcador.
    """
    kills = {BLUE: 0, RED: 0}
    cs = {BLUE: 0, RED: 0}
    level = {BLUE: 0, RED: 0}
    for p in data.get("allPlayers", []):
        t = p.get("team")
        if t not in kills:
            continue
        sc = p.get("scores", {})
        kills[t] += sc.get("kills", 0)
        cs[t] += sc.get("creepScore", 0)
        level[t] += p.get("level", 0)

    towers = {BLUE: 0, RED: 0}
    inhibs = {BLUE: 0, RED: 0}
    dragons = {BLUE: 0, RED: 0}
    heralds = {BLUE: 0, RED: 0}
    barons = {BLUE: 0, RED: 0}
    grubs = {BLUE: 0, RED: 0}

    by_player = _team_by_player(data)
    for e in data.get("events", {}).get("Events", []):
        name = e.get("EventName")
        # Estructuras: el NOMBRE dice de quien era (T1 = ORDER, T2 = CHAOS).
        # Es mas fiable que KillerName, porque un esbirro puede tirar la torre.
        if name == "TurretKilled":
            owner = _structure_owner(e.get("TurretKilled", ""))
            if owner:
                towers[_other(owner)] += 1
        elif name == "InhibKilled":
            owner = _structure_owner(e.get("InhibKilled", ""))
            if owner:
                inhibs[_other(owner)] += 1
        # Monstruos epicos: hay que mirar quien lo mato.
        elif name in ("DragonKill", "HeraldKill", "BaronKill", "HordeKill"):
            team = by_player.get(e.get("KillerName"))
            if team not in kills:
                continue  # lo mato algo que no es un jugador -> no atribuible
            if name == "DragonKill":
                dragons[team] += 1
            elif name == "HeraldKill":
                heralds[team] += 1
            elif name == "BaronKill":
                barons[team] += 1
            else:
                grubs[team] += 1

    return {"kills": kills, "cs": cs, "level": level, "towers": towers,
            "inhibs": inhibs, "dragons": dragons, "heralds": heralds,
            "barons": barons, "grubs": grubs}


def read_state(data):
    """Estado del juego en este instante, como diffs (azul − rojo).

    Es la perspectiva del modelo: todas sus features son diferencias.
    """
    c = read_counters(data)
    state = {"minute": int(game_time(data) // 60)}
    for key in COUNTERS:
        state[DIFF_NAME[key]] = c[key][BLUE] - c[key][RED]
    return state


def build_features(state, history):
    """Estado + momentum -> el vector de 13 features en el orden de FEATURES.

    history: lista de estados pasados (dicts de read_state), en orden.
    Los _d5 comparan contra el estado de hace DELTA_WINDOW minutos; si aun no
    existe (partida joven), valen 0 — igual que en build_features.py.
    """
    prev = None
    target = state["minute"] - DELTA_WINDOW
    for h in history:
        if h["minute"] <= target:
            prev = h  # el mas reciente que no supere el objetivo
    feats = dict(state)
    for col in ("kills_diff", "cs_diff", "level_diff"):
        feats[f"{col}_d5"] = (state[col] - prev[col]) if prev else 0
    return feats


def active_team(data):
    """Equipo del jugador que esta en este PC (para mostrar 'tu' win%)."""
    me = data.get("activePlayer", {})
    name = me.get("riotId") or me.get("summonerName")
    return _team_by_player(data).get(name, BLUE)


def load_model(path=MODEL_OUT):
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def predict_blue_winrate(model, features, feature_names=FEATURES):
    X = [[features[f] for f in feature_names]]
    return float(model.predict_proba(X)[0][1])


def main():
    model, names = load_model()
    print(f"Modelo cargado ({len(names)} features). Esperando partida...\n")
    history = []
    last_t = None
    while True:
        try:
            data = fetch_live_data()
        except NoGameRunning as ex:
            print(f"  ... sin partida ({ex})", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        # Partida nueva -> historial limpio, o los _d5 compararian contra la anterior.
        t = game_time(data)
        if is_new_game(last_t, t):
            history.clear()
            print("\n--- Partida nueva: historial reiniciado ---\n", flush=True)
        last_t = t

        state = read_state(data)
        if not history or history[-1]["minute"] != state["minute"]:
            history.append(state)
        feats = build_features(state, history)
        p_blue = predict_blue_winrate(model, feats, names)

        mine = active_team(data)
        p_mine = p_blue if mine == BLUE else 1 - p_blue
        print(f"[min {state['minute']:>2}]  TU EQUIPO ({mine}): {p_mine:6.1%}   "
              f"| azul {p_blue:5.1%} | k{state['kills_diff']:+d} "
              f"cs{state['cs_diff']:+d} torres{state['tower_diff']:+d}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
