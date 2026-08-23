"""
Builds features.csv from the raw match + timeline JSONs in riot_dataset/.

For every match, write one row per minute with the game state at that minute,
momentum deltas over config.DELTA_WINDOW minutes, and the target blue_win.
match_id is kept so train.py can split train/test per match.
"""
import os
import csv
import json
import sys
import config
import crawler  # for load_ranks()

# constants in the format of the api, the docs are a bit strange and inconsistent
# for example grubs are called "HORDE" and in someplaces i think it's "HORDE_MINION" or "GRUB"
DRAGON, HERALD, BARON, GRUB = "DRAGON", "RIFTHERALD", "BARON_NASHOR", "HORDE"
TOWER, INHIB = "TOWER_BUILDING", "INHIBITOR_BUILDING"

# Same for roles, just standarize the values from the api to a human readble and normal value
ROLE_MAP = {"TOP": "TOP", "JUNGLE": "JGL", "MIDDLE": "MID", "BOTTOM": "BOT", "UTILITY": "SUP"}
ROLES = ["TOP", "JGL", "MID", "BOT", "SUP"]

load_ranks = crawler.load_ranks


def get_timeline(match_id):
    """Load a cached timeline from disk, or None if not downloaded yet."""
    path = os.path.join(config.TIMELINE_DIR, f"{match_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def snapshot(pid_team, role_pairs, frames, minute):
    """Feature dict (diffs) for the game state at the given minute.
    100 = blue, 200 = red. Deltas are computed from the previous snapshot.
    """

    # Obtain the participantFrames for this minute, keyed by participantId
    pf_by_pid = {int(k): v for k, v in frames[minute]["participantFrames"].items()}

    # All values at 0 since we only want the difference not the absolute values
    blue_gold = red_gold = blue_xp = red_xp = blue_cs = red_cs = 0
    blue_level = red_level = 0

    # Compute the gold and level for each participant and sum them up for each team
    for pid, pf in pf_by_pid.items():
        team = pid_team.get(pid)
        cs = pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)
        # Sum up the gold, xp, cs and level for each team
        if team == 100:
            blue_gold += pf.get("totalGold", 0); blue_xp += pf.get("xp", 0); blue_cs += cs
            blue_level += pf.get("level", 0)
        elif team == 200:
            red_gold += pf.get("totalGold", 0); red_xp += pf.get("xp", 0); red_cs += cs
            red_level += pf.get("level", 0)

    # Check for each role; i.e: ADC >>> SUPP in gold and exp value
    role_feats = {}
    for rk in ROLES:
        pair = role_pairs.get(rk, {})
        b, r = pair.get("blue"), pair.get("red")
        # Obtain the gold and xp difference for each role
        if b in pf_by_pid and r in pf_by_pid:
            role_feats[f"gold_diff_{rk}"] = pf_by_pid[b].get("totalGold", 0) - pf_by_pid[r].get("totalGold", 0)
            role_feats[f"xp_diff_{rk}"] = pf_by_pid[b].get("xp", 0) - pf_by_pid[r].get("xp", 0)
        else:  # unassigned role (autofill/remake)
            role_feats[f"gold_diff_{rk}"] = 0
            role_feats[f"xp_diff_{rk}"] = 0

    # Count kills, objectives, towers, inhibitors, etc. up to this minute
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
                owner = e.get("teamId")  # owning team; the OTHER team gets credit
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
    # Add the role-specific gold/xp differences to the main feature dict
    feats.update(role_feats)
    return feats


def rows_for_match(match, timeline, band=("UNKNOWN", "")):
    """One row per minute for a match, including momentum deltas."""
    info = match["info"]
    pid_team = {p["participantId"]: p["teamId"] for p in info["participants"]}
    blue_win = 1 if any(t["teamId"] == 100 and t["win"] for t in info["teams"]) else 0
    match_id = match["metadata"]["matchId"]
    patch = ".".join(info.get("gameVersion", "").split(".")[:2])
    queue_id = info.get("queueId")

    role_pairs = {}
    for p in info["participants"]:
        rk = ROLE_MAP.get(p.get("teamPosition"))
        if not rk:
            continue
        side = "blue" if p["teamId"] == 100 else "red"
        role_pairs.setdefault(rk, {})[side] = p["participantId"]

    frames = timeline["info"]["frames"]
    last = len(frames) - 1
    # Avoid info before min 5
    if last < config.MINUTE_START:
        return []

    # Create a dict of snapshots for each minute, with the game state at that minute
    snaps = {m: snapshot(pid_team, role_pairs, frames, m)
             for m in range(config.MINUTE_START, last + 1, config.MINUTE_STEP)}

    rows = []
    for m, snap in snaps.items():
        row = {"match_id": match_id, "queue_id": queue_id,
               "tier": band[0], "division": band[1], "patch": patch,
               "minute": m, "blue_win": blue_win}
        row.update(snap)
        prev = snaps.get(m - config.DELTA_WINDOW)
        # Add the delta for gold, xp, kills, cs, and level over the DELTA_WINDOW
        for col in config.DELTA_BASE:
            row[f"{col}_d{config.DELTA_WINDOW}"] = (snap[col] - prev[col]) if prev else 0
        rows.append(row)
    return rows


def build():
    files = [f for f in os.listdir(config.RAW_DIR) if f.endswith(".json")]
    ranks = load_ranks()
    print(f"{len(files)} matches on disk, {len(ranks)} with an elo band recorded")

    all_rows = []
    games_ok = games_short = games_no_timeline = 0
    for i, fn in enumerate(files, 1):
        # Get the match_id from the filename 
        match_id = fn[:-5]
        try:
            with open(os.path.join(config.RAW_DIR, fn), encoding="utf-8") as f:
                match = json.load(f)
            timeline = get_timeline(match_id)
            if timeline is None:
                games_no_timeline += 1
                continue
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
        print("No rows. Download the matches using crawler.py.")
        return

    # Write the rows to features.csv with the appropriate fieldnames
    fieldnames = list(all_rows[0].keys())
    with open(config.FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_blue = sum(r["blue_win"] for r in all_rows)
    print(f"\nWrote {config.FEATURES_CSV}")
    print(f"  valid matches: {games_ok}  (short: {games_short}, no timeline: {games_no_timeline})")
    print(f"  rows (minute x match): {len(all_rows)}")
    print(f"  blue_win balance: {n_blue}/{len(all_rows)} = {n_blue/len(all_rows):.3f}")

    # We dont use the elo, but it's useful to see how many rows we have per elo band
    per_tier = {}
    for r in all_rows:
        per_tier[r["tier"]] = per_tier.get(r["tier"], 0) + 1
    print("  rows per elo band:")
    # Sort the tiers by number of rows and print them in descending order
    for t, n in sorted(per_tier.items(), key=lambda x: -x[1]):
        print(f"    {t:>10}: {n:>7}")

if __name__ == "__main__":
    build()