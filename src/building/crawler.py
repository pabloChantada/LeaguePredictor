import os
import csv
import json
import time
import random
import requests
from collections import deque

import config

HEADERS = {"X-Riot-Token": config.API_KEY}


def require_api_key():
    """Raise a clear error if API_KEY isn't set. Only called on network paths,
    so build_features.py can import this module without a .env file."""
    if not config.API_KEY:
        raise RuntimeError(
            "Missing API_KEY. Copy .env.example to .env and put your key from "
            "https://developer.riotgames.com/ (dev keys expire every 24h)."
        )

# since the values are unique we use sets intead of dicts or list
seen_match_ids = set()
seen_puuids = set()
# we have a timer so well use a deque to store the last N request times, and pop from the left when they are too old
player_queue = deque()

_req_times = deque()


def _throttle():
    """Block until another request is safe under the rate limits."""
    while True:
        now = time.time()
        while _req_times and now - _req_times[0] > config.WINDOW_SECONDS:
            _req_times.popleft()
        if len(_req_times) >= config.MAX_PER_WINDOW:
            time.sleep(config.WINDOW_SECONDS - (now - _req_times[0]) + 0.1)
            continue
        in_last_second = sum(1 for t in _req_times if now - t < 1.0)
        if in_last_second >= config.MAX_PER_SECOND:
            time.sleep(0.2)
            continue
        break
    _req_times.append(time.time())


def load_existing():
    """Populate seen_match_ids from disk so we don't re-fetch on resume."""
    for fn in os.listdir(config.RAW_DIR):
        if fn.endswith(".json"):
            seen_match_ids.add(fn[:-5])


def riot_get(url, params=None, retries=5):
    require_api_key()
    for attempt in range(retries):
        _throttle()
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        # 429 -> rate limit
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry_after + 0.2)
            continue
        # 500+ -> server error, retry with backoff
        if r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        # raise http errors (404, 403, etc) so the caller can handle them
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed request: {url}")


def get_league_entries(tier, division, page=1):
    """Players of a tier with divisions (Iron..Diamond). 400 above Diamond."""
    url = f"https://{config.PLATFORM}/lol/league/v4/entries/{config.QUEUE}/{tier}/{division}"
    return riot_get(url, params={"page": page})


def get_apex_league(tier):
    """Players of a tier without divisions (Master/GM/Challenger), unpaginated."""
    url = f"https://{config.PLATFORM}/lol/league/v4/{tier.lower()}leagues/by-queue/{config.QUEUE}"
    return riot_get(url).get("entries", [])


def get_match_ids(puuid, start=0, count=20):
    return riot_get(f"https://{config.REGION}/lol/match/v5/matches/by-puuid/{puuid}/ids",
                    params={"start": start, "count": count, "queue": config.QUEUE_ID})


def get_match(match_id):
    return riot_get(f"https://{config.REGION}/lol/match/v5/matches/{match_id}")


def get_timeline(match_id):
    return riot_get(f"https://{config.REGION}/lol/match/v5/matches/{match_id}/timeline")


def save_match(match):
    match_id = match["metadata"]["matchId"]
    with open(os.path.join(config.RAW_DIR, f"{match_id}.json"), "w", encoding="utf-8") as f:
        json.dump(match, f)


def save_timeline(match_id, timeline):
    with open(os.path.join(config.TIMELINE_DIR, f"{match_id}.json"), "w", encoding="utf-8") as f:
        json.dump(timeline, f)


def _tier_candidates(tier, divisions):
    """[(puuid, tier, division)] for a tier, via whichever endpoint applies."""
    out = []
    # If divisions is None, we are in the high elo (Master/Grandmaster/Challenger) which have no divisions. Otherwise, we are in the lower tiers (Iron..Diamond) which have divisions.
    if divisions is None:
        try:
            # obtain the tiers
            entries = get_apex_league(tier)
        except Exception as ex:
            print(f"  ! {tier}: {ex}", flush=True)
            return out
        # obtain players from the tiers, and add them to the output list
        out += [(e["puuid"], tier, e.get("rank", "I")) for e in entries if e.get("puuid")]
        print(f"  {tier}: {len(entries)} players in the league", flush=True)
        return out

    for division in divisions:
        before = len(out)
        # iterate through the division and pages 
        for page in range(1, config.SEED_PAGES_PER_DIVISION + 1):
            try:
                entries = get_league_entries(tier, division, page=page)
            except Exception as ex:
                print(f"  ! {tier} {division} p{page}: {ex}", flush=True)
                break
            if not entries:
                break
            out += [(e["puuid"], tier, division) for e in entries if e.get("puuid")]
        print(f"  {tier} {division}: {len(out) - before} players", flush=True)
    return out


def seed_from_tiers():
    """Seed players only from config.SEED_TIERS (no random walk across the ladder)."""
    seeds = []
    for tier, divisions in config.SEED_TIERS.items():
        # players in the tier, with divisions if applicable
        cand = _tier_candidates(tier, divisions)
        # limit the number of seeds per tier to avoid over-representing a single tier in the crawl
        if len(cand) > config.SEED_MAX_PER_TIER:
            # randomize the candidates
            cand = random.sample(cand, config.SEED_MAX_PER_TIER)
            print(f"  {tier}: trimmed to {config.SEED_MAX_PER_TIER} seeds", flush=True)
        seeds += cand

    random.shuffle(seeds)

    added = 0
    # track seen puuids to avoid duplicates in the player queue
    for puuid, tier, division in seeds:
        if puuid not in seen_puuids:
            seen_puuids.add(puuid)
            player_queue.append((puuid, tier, division))
            added += 1
    return added


def load_ranks():
    """match_id -> (tier, division) of the seed player, from config.RANKS_CSV."""
    if not os.path.exists(config.RANKS_CSV):
        return {}
    with open(config.RANKS_CSV, encoding="utf-8") as f:
        return {r["match_id"]: (r["tier"], r["division"]) for r in csv.DictReader(f)}


def crawl():
    # Target counts only matches in the target band (not everything on disk,
    # which may include an old crawl from a different band).
    in_band = len(load_ranks())
    is_new = not os.path.exists(config.RANKS_CSV)

    with open(config.RANKS_CSV, "a", newline="", encoding="utf-8") as fr:
        # tracker for the match_id -> (tier, division) of the seed player
        w = csv.writer(fr)
        if is_new:
            w.writerow(["match_id", "tier", "division"])

        while player_queue and in_band < config.TARGET_MATCHES:
            puuid, tier, division = player_queue.popleft()

            try:
                match_ids = get_match_ids(puuid, start=0, count=config.MATCHES_PER_PLAYER)
            except Exception:
                continue

            # Randomize the order of match_ids to avoid biasing towards the most recent matches of a player. 
            random.shuffle(match_ids)

            for match_id in match_ids:
                if match_id in seen_match_ids:
                    continue

                try:
                    match = get_match(match_id)
                    timeline = get_timeline(match_id)
                except Exception:
                    continue

                # Timeline saved first: load_existing() treats "match JSON on
                # disk" as done, so this guarantees a crash mid-save is retried.
                save_timeline(match_id, timeline)
                save_match(match)
                w.writerow([match_id, tier, division])
                fr.flush()

                seen_match_ids.add(match_id)
                in_band += 1
                print(f"[{in_band}/{config.TARGET_MATCHES}] {match_id} ({tier} {division})"
                      f"  (queue: {len(player_queue)} players)", flush=True)

                # Save progress every SAVE_EVERY matches in the target band, so we can resume if interrupted.
                if in_band % config.SAVE_EVERY == 0:
                    with open(os.path.join(config.OUT_DIR, "progress.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "matches_in_band": in_band,
                            "target": config.TARGET_MATCHES,
                            "queued_players": len(player_queue),
                            "seen_players": len(seen_puuids),
                            "matches_on_disk": len(seen_match_ids),
                        }, f)

                if in_band >= config.TARGET_MATCHES:
                    break


if __name__ == "__main__":
    require_api_key()
    load_existing()
    with_rank = load_ranks()
    print(f"Matches already on disk: {len(seen_match_ids)} "
          f"({len(with_rank)} with elo provenance recorded)", flush=True)
    band = ", ".join(f"{t} {'/'.join(d) if d else '(no divisions)'}"
                     for t, d in config.SEED_TIERS.items())
    print(f"Seeding in: {band}  (queue {config.QUEUE}, id {config.QUEUE_ID})", flush=True)
    n = seed_from_tiers()
    print(f"Seeded {n} players from the target band", flush=True)
    crawl()
    print(f"Total unique matches: {len(seen_match_ids)}", flush=True)