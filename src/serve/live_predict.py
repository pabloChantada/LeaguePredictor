"""
Core of the LIVE prediction.

Reads the state of the ongoing match from the Live Client Data API, 
builds the SAME 13 features the model was trained on, and asks the
deployed FastAPI service for the win %.

This script only talks to two HTTP endpoints: 
- the Live Client Data API (localhost) to read the match
- the public prediction API to score it.

Direct use (console mode):   python live_predict.py
Dashboard:                   streamlit run live_dashboard.py
"""
import os
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LIVE_CLIENT_URL = "https://host.docker.internal:2999/liveclientdata/allgamedata"
PREDICT_API_URL = "http://host.docker.internal:8000/predict"
LIVE_TIMEOUT = 3

# Generous on purpose: Render/Fly free tier can cold-start a sleeping service
# in 20-50s, and that's normal, not a failure.
PREDICT_TIMEOUT = 30

POLL_SECONDS = 10
DELTA_WINDOW = 5  # minutesbuild_features.DELTA_WINDOW
NEW_GAME_TOLERANCE = 5.0  # seconds; covers gameTime float jitter, not real resets

BLUE, RED = "ORDER", "CHAOS"   # ORDER = blue side (teamId 100), model's perspective

# Field names/order the API expects. This client should be able to run
# with zero training-pipeline dependencies installed. Must match api/main.py.
FEATURES = [
    "minute",
    "kills_diff", "cs_diff", "level_diff",
    "tower_diff", "inhib_diff", "dragon_diff", "herald_diff", "baron_diff", "grub_diff",
    "kills_diff_d5", "cs_diff_d5", "level_diff_d5",
]


class NoGameRunning(Exception):
    """No match in progress (or the client does not expose the endpoint yet)."""


class PredictionUnavailable(Exception):
    """The prediction API could not be reached or returned an error."""


def fetch_live_data(timeout=LIVE_TIMEOUT):
    try:
        r = requests.get(LIVE_CLIENT_URL, verify=False, timeout=timeout)
    except requests.exceptions.RequestException as ex:
        raise NoGameRunning(f"Could not connect to {LIVE_CLIENT_URL}: {ex}") from ex
    # != 200 -> not ok
    if r.status_code != 200:
        raise NoGameRunning(f"The client responded {r.status_code}")
    return r.json()


def game_time(data):
    """Seconds elapsed in the ongoing match."""
    return data.get("gameData", {}).get("gameTime", 0.0)


def is_new_game(prev_time, cur_time):
    """True if `cur_time` is from a DIFFERENT match than `prev_time`.

    The Live Client Data API does not expose any match id (gameData only
    carries gameMode/gameTime/mapName). So if we reconnect to the same match after a disconnect, the clock keeps moving forward, not resetting to 0. 
    """
    return prev_time is not None and cur_time < prev_time - NEW_GAME_TOLERANCE


def _other(team):
    return RED if team == BLUE else BLUE


def _team_by_player(data):
    """Player name -> team, to attribute events to a side.

    KillerName in the events carries ONLY the game name ("USERNAME"),
    without the tag, while riotId/summonerName come with it
    ("USERNAMEn#tag"). Without riotIdGameName no objective gets
    attributed and dragon/baron/grub come out as 0.
    """
    out = {}
    for p in data.get("allPlayers", []):
        for key in ("riotIdGameName", "riotId", "summonerName"):
            if p.get(key):
                out[p[key]] = p.get("team")
    return out


def _structure_owner(name):
    """
    The API uses old name convetions for structures, so we have to parse them to know which team owned it.

    The client names them "Turret_TOrder_L0_P3_..." /
    "Inhib_TChaos_L1_P1_..."; the old format "_T1_"/"_T2_" is also accepted.
    Returns None if unrecognized, so as not to attribute blindly.
    """
    if "TOrder" in name or "_T1_" in name:
        return BLUE
    if "TChaos" in name or "_T2_" in name:
        return RED
    return None


# Counters that read_counters returns, in scoreboard order. The key is the
# same one read_state uses to name its diff ("kills" -> "kills_diff").
COUNTERS = ["kills", "cs", "level", "towers", "inhibs",
            "dragons", "heralds", "barons", "grubs"]
# model feature name for each counter (not all are regular plurals)
DIFF_NAME = {"kills": "kills_diff", "cs": "cs_diff", "level": "level_diff",
             "towers": "tower_diff", "inhibs": "inhib_diff", "dragons": "dragon_diff",
             "heralds": "herald_diff", "barons": "baron_diff", "grubs": "grub_diff"}


def read_counters(data):
    """RAW per-team counters: {counter: {BLUE: n, RED: n}}.

    Split from read_state because the model only wants diffs, but the
    dashboard needs each side's absolutes to draw the scoreboard.
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
    # The instance of data we get it's a event log for a given moment
    # saved in the JSON file as "events": {"Events": [ ... ]}. Each event has a name and a payload.
    for e in data.get("events", {}).get("Events", []):
        name = e.get("EventName")
        # Structures: the NAME says whose it was (T1 = ORDER, T2 = CHAOS).
        # More reliable than KillerName, since a minion can take the tower.
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
                continue  # killed by something that is not a player (should never happen, but just in case)
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

    This is the model's perspective.
    """
    c = read_counters(data)
    state = {"minute": int(game_time(data) // 60)}
    for key in COUNTERS:
        # Return the diff (blue - red) for each counter, as the model expects.
        state[DIFF_NAME[key]] = c[key][BLUE] - c[key][RED]
    return state


def build_features(state, history):
    """State + momentum -> the vector of 13 features in FEATURES order.

    history: list of past states (dicts from read_state), in order.
    The _d5 compare against the state from DELTA_WINDOW minutes ago; if it
    does not exist yet (young match), they are 0 — same as in
    build_features.py.
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
    """Team of the player on THIS PC."""
    me = data.get("activePlayer", {})
    name = me.get("riotId") or me.get("summonerName")
    return _team_by_player(data).get(name, BLUE)


def warm_up_api(timeout=PREDICT_TIMEOUT):
    """Ping /health once so the caller can report a cold start explicitly
    instead of the first real prediction silently taking 20-50s."""
    health_url = PREDICT_API_URL.rsplit("/predict", 1)[0] + "/health"
    try:
        requests.get(health_url, timeout=timeout)
    except requests.exceptions.RequestException:
        pass  # best-effort; predict_blue_winrate() will raise properly if it's really down


def predict_blue_winrate(features):
    """POST the feature vector to the deployed API and return p(blue wins)."""
    # data we send to the model in a json format from the features dict, in the same order as FEATURES
    payload = {f: features[f] for f in FEATURES}
    try:
        r = requests.post(PREDICT_API_URL, json=payload, timeout=PREDICT_TIMEOUT)
    except requests.exceptions.RequestException as ex:
        raise PredictionUnavailable(f"Could not reach {PREDICT_API_URL}: {ex}") from ex
    if r.status_code != 200:
        raise PredictionUnavailable(f"API responded {r.status_code}: {r.text}")
    # return the probability of the positive class (blue wins) from the API response
    return r.json()["p_blue"]


def main():
    print(f"Prediction API: {PREDICT_API_URL}")
    warm_up_api()
    print("Waiting for a match...\n")
    history = []
    last_t = None  # seconds elapsed in the previous iteration, to detect new matches
    while True:
        try:
            data = fetch_live_data()
        except NoGameRunning as ex:
            print(f"  ... no match ({ex})", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        # New match -> clean history, otherwise the _d5 would compare against
        # the previous one.
        t = game_time(data)
        if is_new_game(last_t, t):
            history.clear()
            print("\n--- New match: history reset ---\n", flush=True)
        last_t = t

        state = read_state(data)
        if not history or history[-1]["minute"] != state["minute"]:
            history.append(state)
        feats = build_features(state, history)

        try:
            p_blue = predict_blue_winrate(feats)
        except PredictionUnavailable as ex:
            print(f"  ... prediction API unavailable ({ex})", flush=True)
            time.sleep(POLL_SECONDS)
            continue

        # Obtain user team and compute its win probability from the model's perspective
        mine = active_team(data)
        p_mine = p_blue if mine == BLUE else 1 - p_blue
        print(f"[min {state['minute']:>2}]  YOUR TEAM ({mine}): {p_mine:6.1%}   "
              f"| blue {p_blue:5.1%} | k{state['kills_diff']:+d} "
              f"cs{state['cs_diff']:+d} towers{state['tower_diff']:+d}", flush=True)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()