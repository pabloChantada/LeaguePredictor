"""
Core of the LIVE prediction.

Reads the state of the ongoing match from the Live Client Data API (runs on your
own PC while you play, in any mode: ranked, normal, CUSTOM or practice tool),
builds the SAME 13 features the model was trained on and returns the win %.

Why these 13 and not gold: the model was trained on purpose without gold or xp
because this API does not give them for the 10 players. See train.FEATURES.

Direct use (console mode):   python live_predict.py
Dashboard:                   streamlit run live_dashboard.py
"""
import json
import time
import warnings

import joblib
import requests
import urllib3

from train import FEATURES, MODEL_OUT

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# The endpoint is local and uses a self-signed Riot certificate -> verify=False.
URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
POLL_SECONDS = 10
DELTA_WINDOW = 5  # minutes, same as build_features.DELTA_WINDOW
# Margin to consider the clock "went backwards". It does not need to be large:
# gameTime is monotonic within a match; it only covers float jitter.
NEW_GAME_TOLERANCE = 5.0  # seconds

BLUE, RED = "ORDER", "CHAOS"   # ORDER = blue side (teamId 100) = model's perspective


class NoGameRunning(Exception):
    """No match in progress (or the client does not expose the endpoint yet)."""


def fetch_live_data(timeout=3):
    try:
        r = requests.get(URL, verify=False, timeout=timeout)
    except requests.exceptions.RequestException as ex:
        raise NoGameRunning(f"Could not connect to {URL}: {ex}") from ex
    if r.status_code != 200:
        raise NoGameRunning(f"The client responded {r.status_code}")
    return r.json()


def game_time(data):
    """Seconds elapsed in the ongoing match."""
    return data.get("gameData", {}).get("gameTime", 0.0)


def is_new_game(prev_time, cur_time):
    """True if `cur_time` is from a DIFFERENT match than `prev_time`.

    The Live Client Data API does not expose any match id (gameData only carries
    gameMode/gameTime/mapName), so the only reliable signal is the clock going
    backwards: when another match starts it returns to ~0.

    Reconnecting to the SAME match (a client drop, an F5 on the dashboard) leaves
    the clock moving forward -> does not trigger a reset and the history survives,
    which is exactly what we want: resetting on a disconnect would lose the curve.
    """
    return prev_time is not None and cur_time < prev_time - NEW_GAME_TOLERANCE


def _other(team):
    return RED if team == BLUE else BLUE


def _team_by_player(data):
    """Player name -> team, to attribute events to a side.

    KillerName in the events carries ONLY the game name ("ChantaClown"), without
    the tag, while riotId/summonerName come with it ("ChantaClown#milk"). Without
    riotIdGameName no objective gets attributed and dragon/baron/grub come out as 0.
    """
    out = {}
    for p in data.get("allPlayers", []):
        for key in ("riotIdGameName", "riotId", "summonerName"):
            if p.get(key):
                out[p[key]] = p.get("team")
    return out


def _structure_owner(name):
    """OWNING team of the structure, read from its internal name.

    The client names them "Turret_TOrder_L0_P3_..." / "Inhib_TChaos_L1_P1_...";
    the old format "_T1_"/"_T2_" is also accepted. Returns None if unrecognized,
    so as not to attribute blindly.
    """
    if "TOrder" in name or "_T1_" in name:
        return BLUE
    if "TChaos" in name or "_T2_" in name:
        return RED
    return None


# Counters that read_counters returns, in scoreboard order. The key is the same
# one read_state uses to name its diff ("kills" -> "kills_diff").
COUNTERS = ["kills", "cs", "level", "towers", "inhibs",
            "dragons", "heralds", "barons", "grubs"]
# model feature name for each counter (not all are regular plurals)
DIFF_NAME = {"kills": "kills_diff", "cs": "cs_diff", "level": "level_diff",
             "towers": "tower_diff", "inhibs": "inhib_diff", "dragons": "dragon_diff",
             "heralds": "herald_diff", "barons": "baron_diff", "grubs": "grub_diff"}


def read_counters(data):
    """RAW per-team counters: {counter: {ORDER: n, CHAOS: n}}.

    Split from read_state because the model only wants diffs, but the dashboard
    needs each side's absolutes to draw the scoreboard.
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
        # Structures: the NAME says whose it was (T1 = ORDER, T2 = CHAOS).
        # It is more reliable than KillerName, because a minion can take the tower.
        if name == "TurretKilled":
            owner = _structure_owner(e.get("TurretKilled", ""))
            if owner:
                towers[_other(owner)] += 1
        elif name == "InhibKilled":
            owner = _structure_owner(e.get("InhibKilled", ""))
            if owner:
                inhibs[_other(owner)] += 1
        # Epic monsters: we have to look at who killed it.
        elif name in ("DragonKill", "HeraldKill", "BaronKill", "HordeKill"):
            team = by_player.get(e.get("KillerName"))
            if team not in kills:
                continue  # killed by something that is not a player -> not attributable
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
    """Game state at this instant, as diffs (blue - red).

    This is the model's perspective: all its features are differences.
    """
    c = read_counters(data)
    state = {"minute": int(game_time(data) // 60)}
    for key in COUNTERS:
        state[DIFF_NAME[key]] = c[key][BLUE] - c[key][RED]
    return state


def build_features(state, history):
    """State + momentum -> the vector of 13 features in FEATURES order.

    history: list of past states (dicts from read_state), in order.
    The _d5 compare against the state from DELTA_WINDOW minutes ago; if it does not
    exist yet (young match), they are 0 — same as in build_features.py.
    """
    prev = None
    target = state["minute"] - DELTA_WINDOW
    for h in history:
        if h["minute"] <= target:
            prev = h  # the most recent one that does not exceed the target
    feats = dict(state)
    for col in ("kills_diff", "cs_diff", "level_diff"):
        feats[f"{col}_d5"] = (state[col] - prev[col]) if prev else 0
    return feats


def active_team(data):
    """Team of the player on THIS PC (to show 'your' win %)."""
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
    print(f"Model loaded ({len(names)} features). Waiting for a match...\n")
    history = []
    last_t = None
    while True:
        try:
            data = fetch_live_data()
        except NoGameRunning as ex:
            print(f"  ... no match ({ex})", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        # New match -> clean history, otherwise the _d5 would compare against the previous one.
        t = game_time(data)
        if is_new_game(last_t, t):
            history.clear()
            print("\n--- New match: history reset ---\n", flush=True)
        last_t = t

        state = read_state(data)
        if not history or history[-1]["minute"] != state["minute"]:
            history.append(state)
        feats = build_features(state, history)
        p_blue = predict_blue_winrate(model, feats, names)

        mine = active_team(data)
        p_mine = p_blue if mine == BLUE else 1 - p_blue
        print(f"[min {state['minute']:>2}]  YOUR TEAM ({mine}): {p_mine:6.1%}   "
              f"| blue {p_blue:5.1%} | k{state['kills_diff']:+d} "
              f"cs{state['cs_diff']:+d} towers{state['tower_diff']:+d}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
