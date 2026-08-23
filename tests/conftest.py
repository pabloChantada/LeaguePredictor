"""Shared fixtures for the test suite.

Nothing here talks to the network or needs a real Riot API key: crawler.py only
requires one once you actually try to fetch something (see require_api_key()).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# add the src/ and src/building/ directories to sys.path so that tests can import from them
for target_dir in [ROOT, ROOT / "src" / "building", ROOT / "src" / "serve"]:
    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch):
    """Make sure no test accidentally relies on a real .env / API key."""
    monkeypatch.delenv("API_KEY", raising=False)


def make_participant(pid, team_id, position):
    return {"participantId": pid, "teamId": team_id, "teamPosition": position}


def make_match(match_id="MATCH_1", blue_wins=True, queue_id=420,
               game_version="16.13.567.1234", participants=None):
    """A minimal match-v5 payload, just the fields build_features.py reads."""
    if participants is None:
        positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        participants = (
            [make_participant(i + 1, 100, pos) for i, pos in enumerate(positions)]
            + [make_participant(i + 6, 200, pos) for i, pos in enumerate(positions)]
        )
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "participants": participants,
            "teams": [
                {"teamId": 100, "win": blue_wins},
                {"teamId": 200, "win": not blue_wins},
            ],
            "gameVersion": game_version,
            "queueId": queue_id,
        },
    }


def make_participant_frame(gold=0, xp=0, minions=0, jungle_minions=0, level=1):
    return {
        "totalGold": gold, "xp": xp, "level": level,
        "minionsKilled": minions, "jungleMinionsKilled": jungle_minions,
        "championStats": {"healthMax": 0, "armor": 0, "attackDamage": 0},
        "damageStats": {"totalDamageTaken": 0, "totalDamageDoneToChampions": 0},
    }


def make_timeline(n_minutes, participant_frames_by_minute=None, events_by_minute=None):
    """A minimal match-v5 timeline: one frame per minute, indices 0..n_minutes.

    participant_frames_by_minute(minute) -> {pid: participantFrame}, defaults to
    all-zero frames for participants 1..10.
    events_by_minute(minute) -> list of event dicts, defaults to [].
    """
    frames = []
    for m in range(n_minutes + 1):
        if participant_frames_by_minute is not None:
            pf = participant_frames_by_minute(m)
        else:
            pf = {str(pid): make_participant_frame() for pid in range(1, 11)}
        events = events_by_minute(m) if events_by_minute is not None else []
        frames.append({"participantFrames": pf, "events": events})
    return {"info": {"frames": frames}}
