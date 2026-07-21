import os
import csv
import json
import time
import random
import requests
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

# Project root: anchored to the file, NOT the CWD -> the scripts work from any
# directory.
ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing API_KEY. Copy .env.example to .env and put your key from "
        "https://developer.riotgames.com/ (dev keys expire every 24h)."
    )

HEADERS = {"X-Riot-Token": API_KEY}

PLATFORM = "euw1.api.riotgames.com"
REGION = "europe.api.riotgames.com"

# NOTE: these are TWO distinct identifiers of the same queue, and each API wants its own.
#   QUEUE     (string) -> league-v4, to SEED players by league.
#   QUEUE_ID  (int)    -> match-v5,  to FILTER which matches we download.
# Using only the first one was the bug: by-puuid/ids without `queue` returns ALL of
# the player's queues (Arena, co-op vs AI, ARAM...) and 19.1% of the dataset ended
# up not being soloQ. See experiments/queue_ablation.py.
QUEUE = "RANKED_SOLO_5x5"
QUEUE_ID = 420
TARGET_MATCHES = 10000          # target for the baseline; raise this once you have a personal key
MATCHES_PER_PLAYER = 20
SAVE_EVERY = 25

# --- target elo band ---------------------------------------------------------
# Elo mainly affects CALIBRATION: in low elo, advantages convert worse, so the
# same state deserves a probability closer to 50%. A high-elo model served in mid
# elo comes out OVERCONFIDENT. Since the Live Client Data does not expose the rank
# of the 10 players, elo cannot be a feature: you have to train in the band where
# it will be served.
#
# tier -> divisions to seed, or None if the tier has no divisions.
# Master/GM/Challenger have none (everyone is "I") and CANNOT be requested via
# entries/{queue}/MASTER/{div}: that returns 400. They go through their own
# endpoint ({tier}leagues/by-queue), which returns the whole league in one shot.
# Grandmaster and Challenger are left out on purpose: the target is the mid-high
# band, not the top of the ladder.
SEED_TIERS = {
    "EMERALD": ("I", "II", "III", "IV"),
    "DIAMOND": ("I", "II", "III", "IV"),
    "MASTER": None,
}
SEED_PAGES_PER_DIVISION = 8     # league-v4 gives ~205 players per page
# Cap on seeds per tier. It exists because the two endpoints have different SHAPE:
# Emerald/Diamond give ~205 per page (~6.5k per tier with 8 pages), but
# masterleagues dumps 10,000 at once. Without a cap, Master would be the majority
# of the queue and the band would drift upward by an API accident, not by a
# decision. Raise or lower it to weight one tier over another.
SEED_MAX_PER_TIER = 3000

# Development key limits: 20 req/s and 100 req every 120 s.
# We stay below them to avoid constant 429s.
MAX_PER_SECOND = 18
MAX_PER_WINDOW = 95
WINDOW_SECONDS = 120

OUT_DIR = ROOT / "riot_dataset"
RAW_DIR = OUT_DIR / "matches"
TIMELINE_DIR = OUT_DIR / "timelines"   # downloaded by THIS script, not build_features
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TIMELINE_DIR, exist_ok=True)

# Provenance of each match: match_id -> (tier, division) of the seed player.
# NOT cosmetic. riot_dataset/matches/ already contains an old crawl seeded in
# Master+; without this, a new crawl in Emerald gets MIXED with it in the same
# directory and features.csv comes out with two elo bands blended together,
# with nothing to flag it. Same lesson as queue_id: record the provenance and
# filter at load time (train.load_dataset). Matches without a row here stay UNKNOWN.
RANKS_CSV = OUT_DIR / "match_rank.csv"

seen_match_ids = set()
seen_puuids = set()
player_queue = deque()

_req_times = deque()

def _throttle():
    """Block until it is safe to make another request under the limits."""
    while True:
        now = time.time()
        while _req_times and now - _req_times[0] > WINDOW_SECONDS:
            _req_times.popleft()
        # 2-min window
        if len(_req_times) >= MAX_PER_WINDOW:
            time.sleep(WINDOW_SECONDS - (now - _req_times[0]) + 0.1)
            continue
        # 1-s window
        in_last_second = sum(1 for t in _req_times if now - t < 1.0)
        if in_last_second >= MAX_PER_SECOND:
            time.sleep(0.2)
            continue
        break
    _req_times.append(time.time())

def load_existing():
    """Count already-downloaded matches so we can resume without re-fetching them."""
    for fn in os.listdir(RAW_DIR):
        if fn.endswith(".json"):
            seen_match_ids.add(fn[:-5])

def riot_get(url, params=None, retries=5):
    for attempt in range(retries):
        _throttle()
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "1"))
            time.sleep(retry_after + 0.2)
            continue
        if r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed request: {url}")

def get_league_entries(tier, division, page=1):
    """Players of a tier WITH divisions (EMERALD I, DIAMOND IV, ...).

    Paginated endpoint, ~205 per page. Only valid from IRON to DIAMOND: with
    MASTER and above it returns 400 (verified).
    """
    url = f"https://{PLATFORM}/lol/league/v4/entries/{QUEUE}/{tier}/{division}"
    return riot_get(url, params={"page": page})


def get_apex_league(tier):
    """Players of a tier WITHOUT divisions (master/grandmaster/challenger).

    Different endpoint and different shape: it returns a LeagueList with the whole
    league in a single response (Master ~10,000 entries), unpaginated.
    """
    url = f"https://{PLATFORM}/lol/league/v4/{tier.lower()}leagues/by-queue/{QUEUE}"
    return riot_get(url).get("entries", [])

def get_match_ids(puuid, start=0, count=20):
    # `queue` is NOT optional: without it, the API returns all of the player's queues.
    return riot_get(f"https://{REGION}/lol/match/v5/matches/by-puuid/{puuid}/ids",
                    params={"start": start, "count": count, "queue": QUEUE_ID})

def get_match(match_id):
    url = f"https://{REGION}/lol/match/v5/matches/{match_id}"
    return riot_get(url)

def get_timeline(match_id):
    url = f"https://{REGION}/lol/match/v5/matches/{match_id}/timeline"
    return riot_get(url)

def save_match(match):
    match_id = match["metadata"]["matchId"]
    path = os.path.join(RAW_DIR, f"{match_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(match, f)

def save_timeline(match_id, timeline):
    path = os.path.join(TIMELINE_DIR, f"{match_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline, f)

def _tier_candidates(tier, divisions):
    """[(puuid, tier, division)] for a tier, via whichever endpoint applies."""
    out = []
    if divisions is None:                     # master/gm/challenger: a single call
        try:
            entries = get_apex_league(tier)
        except Exception as ex:
            print(f"  ! {tier}: {ex}", flush=True)
            return out
        out += [(e["puuid"], tier, e.get("rank", "I")) for e in entries if e.get("puuid")]
        print(f"  {tier}: {len(entries)} players in the league", flush=True)
        return out

    for division in divisions:                # emerald/diamond: paginated
        before = len(out)
        for page in range(1, SEED_PAGES_PER_DIVISION + 1):
            try:
                entries = get_league_entries(tier, division, page=page)
            except Exception as ex:
                print(f"  ! {tier} {division} p{page}: {ex}", flush=True)
                break
            if not entries:
                break                          # ran out of pages
            out += [(e["puuid"], tier, division) for e in entries if e.get("puuid")]
        print(f"  {tier} {division}: {len(out) - before} players", flush=True)
    return out


def seed_from_tiers():
    """Seed players ONLY from the target elo band (SEED_TIERS).

    No random walk on purpose. The old crawler pushed 5 participants of each match
    onto the queue, which makes the elo drift down the ladder with no measure or
    control. Seeding straight from the tier keeps the band fixed by construction,
    and these tiers have plenty of players for TARGET_MATCHES without expanding.
    """
    seeds = []
    for tier, divisions in SEED_TIERS.items():
        cand = _tier_candidates(tier, divisions)
        if len(cand) > SEED_MAX_PER_TIER:     # don't let the endpoint shape decide the mix
            cand = random.sample(cand, SEED_MAX_PER_TIER)
            print(f"  {tier}: trimmed to {SEED_MAX_PER_TIER} seeds", flush=True)
        seeds += cand

    # Shuffle so we don't drain EMERALD before touching MASTER: if the crawl is cut
    # halfway, the band stays spread out and not biased toward one tier.
    random.shuffle(seeds)

    added = 0
    for puuid, tier, division in seeds:
        if puuid not in seen_puuids:
            seen_puuids.add(puuid)
            player_queue.append((puuid, tier, division))
            added += 1
    return added

def load_ranks():
    """match_id -> (tier, division) of the seed player, from RANKS_CSV.

    This is the ELO BAND of the match. Honest approximation: we label it with the
    rank of the player we reached it through, and soloQ matchmaking keeps the other
    9 close. Used by crawl() (to know how many it has) and by build_features (to
    emit the `tier` column).
    """
    if not os.path.exists(RANKS_CSV):
        return {}
    with open(RANKS_CSV, encoding="utf-8") as f:
        return {r["match_id"]: (r["tier"], r["division"]) for r in csv.DictReader(f)}

def crawl():
    # The target counts ONLY matches in the target band, not everything on disk:
    # riot_dataset/matches/ already carries an old crawl (Challenger+GM) and,
    # counting it, the crawl would finish instantly without downloading anything.
    # seen_match_ids is still used to avoid re-downloading, but not to decide when
    # to stop.
    in_band = len(load_ranks())
    is_new = not os.path.exists(RANKS_CSV)

    with open(RANKS_CSV, "a", newline="", encoding="utf-8") as fr:
        w = csv.writer(fr)
        if is_new:
            w.writerow(["match_id", "tier", "division"])

        while player_queue and in_band < TARGET_MATCHES:
            puuid, tier, division = player_queue.popleft()

            try:
                match_ids = get_match_ids(puuid, start=0, count=MATCHES_PER_PLAYER)
            except Exception:
                continue

            random.shuffle(match_ids)

            for match_id in match_ids:
                if match_id in seen_match_ids:
                    continue

                try:
                    match = get_match(match_id)
                    timeline = get_timeline(match_id)
                except Exception:
                    continue

                # Timeline FIRST: load_existing() considers a match done if its
                # match JSON exists, so saving it last makes "match on disk" imply
                # "timeline on disk". If we crash in between, the match isn't there
                # and the whole thing is retried.
                save_timeline(match_id, timeline)
                save_match(match)
                w.writerow([match_id, tier, division])
                fr.flush()

                seen_match_ids.add(match_id)
                in_band += 1
                print(f"[{in_band}/{TARGET_MATCHES}] {match_id} ({tier} {division})"
                      f"  (queue: {len(player_queue)} players)", flush=True)

                if in_band % SAVE_EVERY == 0:
                    with open(os.path.join(OUT_DIR, "progress.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "matches_in_band": in_band,
                            "target": TARGET_MATCHES,
                            "queued_players": len(player_queue),
                            "seen_players": len(seen_puuids),
                            "matches_on_disk": len(seen_match_ids),
                        }, f)

                if in_band >= TARGET_MATCHES:
                    break

if __name__ == "__main__":
    load_existing()
    with_rank = load_ranks()
    print(f"Matches already on disk: {len(seen_match_ids)} "
          f"({len(with_rank)} with elo provenance recorded)", flush=True)
    band = ", ".join(f"{t} {'/'.join(d) if d else '(no divisions)'}"
                     for t, d in SEED_TIERS.items())
    print(f"Seeding in: {band}  (queue {QUEUE}, id {QUEUE_ID})", flush=True)
    n = seed_from_tiers()
    print(f"Seeded {n} players from the target band", flush=True)
    crawl()
    print(f"Total unique matches: {len(seen_match_ids)}", flush=True)
