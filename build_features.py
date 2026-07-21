"""
Builds the TEMPORAL feature table (one row per minute) for the baseline.

Reads the match JSON files from riot_dataset/matches/, loads (and caches) the
timeline of each one and, for every minute of the match, writes a row with the
game state at that minute (features as differences, blue perspective), the minute
as a feature, momentum deltas (change over the last DELTA_WINDOW min) and the
target blue_win. The match_id is stored so we can split train/test PER MATCH.

Reuses the key + rate limiter from crawler.py. Resumable: timelines are cached on
disk, so re-running it does not re-download what was already fetched.
"""
import os
import sys
import csv
import json

import crawler  # only for the dataset paths (RAW_DIR / TIMELINE_DIR / RANKS_CSV)

MINUTE_START = 5    # first minute we emit (before that it is too noisy)
MINUTE_STEP = 1     # every how many minutes we emit a row
DELTA_WINDOW = 5    # window for the momentum features (change over N min)

RAW_DIR = crawler.RAW_DIR
OUT_DIR = crawler.OUT_DIR
TIMELINE_DIR = crawler.TIMELINE_DIR   # filled by crawler.py; here it is only read
FEATURES_CSV = os.path.join(OUT_DIR, "features.csv")

# elite monster types (monsterType in ELITE_MONSTER_KILL)
DRAGON = "DRAGON"
HERALD = "RIFTHERALD"
BARON = "BARON_NASHOR"
GRUB = "HORDE"  # void grubs
# building types (buildingType in BUILDING_KILL)
TOWER = "TOWER_BUILDING"
INHIB = "INHIBITOR_BUILDING"

# roles: Riot's teamPosition -> short key (fixed order for stable columns)
ROLE_MAP = {"TOP": "TOP", "JUNGLE": "JGL", "MIDDLE": "MID",
            "BOTTOM": "BOT", "UTILITY": "SUP"}
ROLES = ["TOP", "JGL", "MID", "BOT", "SUP"]

# TODO(scaling): add champion identity via comp scaling (scaling_diff).
# Scaffold already in place in make_scaling_table.py + champion_scaling.csv (to be
# curated by hand). Dropped for now: with placeholder values it did not move the
# AUC. Reconsider only if the model stalls; then try the interaction scaling_diff * minute.

# from these we compute the momentum delta.
# cs/level included because they ARE computable live (Live Client Data gives
# creepScore and level; exact gold and xp are NOT) -> they allow a live-compatible model.
DELTA_BASE = ["gold_diff", "xp_diff", "kills_diff", "cs_diff", "level_diff"]


def get_timeline(match_id):
    """Match timeline from disk. None if it is not there.

    Does NOT download: fetching them is crawler.py's job, which saves match and
    timeline in the same pass. This keeps this script pure disk -> CSV:
    deterministic, re-runnable and with no dependency on the API key or the rate
    limiter.
    """
    path = os.path.join(TIMELINE_DIR, f"{match_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# The elo band is recorded by the crawler (it is the one that knows which tier each
# seed came from); here it is only read. Matches from old crawls have no row and
# come out as UNKNOWN -> train.load_dataset discards them.
load_ranks = crawler.load_ranks


def snapshot(pid_team, role_pairs, frames, minute):
    """Game state (features as differences) at the given minute.

    role_pairs: {role_key: {"blue": pid, "red": pid}} for the per-position diffs.
    """
    pf_by_pid = {int(k): v for k, v in frames[minute]["participantFrames"].items()}

    # --- per-team economy in the minute's frame (gold / xp / cs) ---
    blue_gold = red_gold = blue_xp = red_xp = blue_cs = red_cs = 0
    blue_level = red_level = 0  # level: an xp proxy that DOES exist live
    # championStats/damageStats. NOTE: measured in the model they gave -0.001 AUC
    # (they looked orthogonal to gold, but they are not orthogonal to the full set:
    # they correlate with level/xp/time). They do NOT enter FEATURES in train.py;
    # they are still computed only to have them available in the EDA. See the TODO there.
    team_extra = {100: {}, 200: {}}
    for pid, pf in pf_by_pid.items():
        team = pid_team.get(pid)
        cs = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)
        if team == 100:
            blue_gold += pf.get("totalGold", 0); blue_xp += pf.get("xp", 0); blue_cs += cs
            blue_level += pf.get("level", 0)
        elif team == 200:
            red_gold += pf.get("totalGold", 0); red_xp += pf.get("xp", 0); red_cs += cs
            red_level += pf.get("level", 0)
        if team in team_extra:
            cst, dst = pf.get("championStats", {}), pf.get("damageStats", {})
            for key, val in (("health_max", cst.get("healthMax", 0)),
                             ("armor", cst.get("armor", 0)),
                             ("attack_dmg", cst.get("attackDamage", 0)),
                             ("dmg_taken", dst.get("totalDamageTaken", 0)),
                             ("dmg_champs", dst.get("totalDamageDoneToChampions", 0))):
                team_extra[team][key] = team_extra[team].get(key, 0) + val

    # --- per-role diffs (gold/xp): how much more farmed OUR role is than theirs ---
    role_feats = {}
    for rk in ROLES:
        pair = role_pairs.get(rk, {})
        b, r = pair.get("blue"), pair.get("red")
        if b in pf_by_pid and r in pf_by_pid:
            role_feats[f"gold_diff_{rk}"] = pf_by_pid[b].get("totalGold", 0) - pf_by_pid[r].get("totalGold", 0)
            role_feats[f"xp_diff_{rk}"] = pf_by_pid[b].get("xp", 0) - pf_by_pid[r].get("xp", 0)
        else:  # unassigned role (autofill/remake) -> 0
            role_feats[f"gold_diff_{rk}"] = 0
            role_feats[f"xp_diff_{rk}"] = 0

    # --- events accumulated up to the minute (kills / structures / objectives) ---
    blue_kills = red_kills = blue_towers = red_towers = blue_inhibs = red_inhibs = 0
    blue_drakes = red_drakes = blue_heralds = red_heralds = 0
    blue_barons = red_barons = blue_grubs = red_grubs = 0
    for fr in frames[: minute + 1]:
        for e in fr["events"]:
            etype = e["type"]
            if etype == "CHAMPION_KILL":
                team = pid_team.get(e.get("killerId"))
                if team == 100: blue_kills += 1
                elif team == 200: red_kills += 1
            elif etype == "BUILDING_KILL":
                owner = e.get("teamId")  # owning team -> the other one takes it down
                btype = e.get("buildingType")
                if btype == TOWER:
                    if owner == 200: blue_towers += 1
                    elif owner == 100: red_towers += 1
                elif btype == INHIB:
                    if owner == 200: blue_inhibs += 1
                    elif owner == 100: red_inhibs += 1
            elif etype == "ELITE_MONSTER_KILL":
                team = e.get("killerTeamId")
                monster = e.get("monsterType")
                if monster == DRAGON:
                    if team == 100: blue_drakes += 1
                    elif team == 200: red_drakes += 1
                elif monster == HERALD:
                    if team == 100: blue_heralds += 1
                    elif team == 200: red_heralds += 1
                elif monster == BARON:
                    if team == 100: blue_barons += 1
                    elif team == 200: red_barons += 1
                elif monster == GRUB:
                    if team == 100: blue_grubs += 1
                    elif team == 200: red_grubs += 1

    feats = {
        "gold_diff": blue_gold - red_gold,
        "xp_diff": blue_xp - red_xp,
        "cs_diff": blue_cs - red_cs,
        "level_diff": blue_level - red_level,
        "kills_diff": blue_kills - red_kills,
        "tower_diff": blue_towers - red_towers,
        "inhib_diff": blue_inhibs - red_inhibs,
        "dragon_diff": blue_drakes - red_drakes,
        "herald_diff": blue_heralds - red_heralds,
        "baron_diff": blue_barons - red_barons,
        "grub_diff": blue_grubs - red_grubs,
    }
    for key in ("health_max", "armor", "attack_dmg", "dmg_taken", "dmg_champs"):
        feats[f"{key}_diff"] = team_extra[100].get(key, 0) - team_extra[200].get(key, 0)
    feats.update(role_feats)
    return feats


def rows_for_match(match, timeline, band=("UNKNOWN", "")):
    """A list of rows (one per minute) for a match, with momentum deltas."""
    info = match["info"]
    pid_team = {p["participantId"]: p["teamId"] for p in info["participants"]}
    blue_win = 1 if any(t["teamId"] == 100 and t["win"] for t in info["teams"]) else 0
    match_id = match["metadata"]["matchId"]
    # major patch (e.g. "16.13") so we can slice by version in the EDA
    patch = ".".join(info.get("gameVersion", "").split(".")[:2])
    # Match queue. ESSENTIAL: the crawler downloads ALL queues of each player
    # (match-v5 by-puuid/ids does not filter), so Arena, co-op vs AI, ARAM... which
    # are NOT the problem we model, all get in here. train.load_dataset() filters
    # on this. See QUEUE_SOLOQ in train.py.
    queue_id = info.get("queueId")

    # role -> {"blue": pid, "red": pid} map from teamPosition
    role_pairs = {}
    for p in info["participants"]:
        rk = ROLE_MAP.get(p.get("teamPosition"))
        if not rk:
            continue
        side = "blue" if p["teamId"] == 100 else "red"
        role_pairs.setdefault(rk, {})[side] = p["participantId"]

    frames = timeline["info"]["frames"]
    last = len(frames) - 1
    if last < MINUTE_START:
        return []  # match shorter than the first minute we emit

    # snapshot per minute, indexed by minute so we can compute deltas
    snaps = {m: snapshot(pid_team, role_pairs, frames, m)
             for m in range(MINUTE_START, last + 1, MINUTE_STEP)}

    rows = []
    for m, snap in snaps.items():
        row = {"match_id": match_id, "queue_id": queue_id,
               "tier": band[0], "division": band[1], "patch": patch,
               "minute": m, "blue_win": blue_win}
        row.update(snap)
        prev = snaps.get(m - DELTA_WINDOW)
        for col in DELTA_BASE:
            row[f"{col}_d{DELTA_WINDOW}"] = (snap[col] - prev[col]) if prev else 0
        rows.append(row)
    return rows


def build():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]
    ranks = load_ranks()
    print(f"{len(files)} matches on disk, {len(ranks)} with an elo band recorded")
    all_rows = []
    games_ok = games_short = games_no_timeline = 0
    for i, fn in enumerate(files, 1):
        match_id = fn[:-5]
        try:
            with open(os.path.join(RAW_DIR, fn), encoding="utf-8") as f:
                match = json.load(f)
            timeline = get_timeline(match_id)
            if timeline is None:
                games_no_timeline += 1
                continue  # no timeline on disk -> re-run crawler.py
            rows = rows_for_match(match, timeline, ranks.get(match_id, ("UNKNOWN", "")))
        except Exception as ex:
            print(f"  ! {match_id}: {ex}", flush=True)
            continue
        if not rows:
            games_short += 1
            continue
        all_rows.extend(rows)
        games_ok += 1
        if i % 25 == 0:
            print(f"[{i}/{len(files)}] matches -> {len(all_rows)} rows", flush=True)

    if not all_rows:
        print("No rows. Did you already download matches with crawler.py?")
        return

    fieldnames = list(all_rows[0].keys())
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_blue = sum(r["blue_win"] for r in all_rows)
    print(f"\nWrote {FEATURES_CSV}")
    print(f"  valid matches: {games_ok}  (short: {games_short}, "
          f"no timeline: {games_no_timeline})")
    print(f"  rows (minute x match): {len(all_rows)}")
    print(f"  blue_win balance: {n_blue}/{len(all_rows)} = {n_blue/len(all_rows):.3f}")
    per_tier = {}
    for r in all_rows:
        per_tier[r["tier"]] = per_tier.get(r["tier"], 0) + 1
    print("  rows per elo band:")
    for t, n in sorted(per_tier.items(), key=lambda x: -x[1]):
        print(f"    {t:>10}: {n:>7}")


if __name__ == "__main__":
    build()
