# tests/conftest.py
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

mock_config = MagicMock()
mock_config.API_KEY = "fake_key"
mock_config.PLATFORM = "https://fake.api.riotgames.com"
mock_config.REGION = "https://fake.api.riotgames.com"
mock_config.QUEUE = "RANKED_SOLO_5x5"
mock_config.QUEUE_ID = 420
mock_config.WINDOW_SECONDS = 10
mock_config.MAX_PER_WINDOW = 20
mock_config.MAX_PER_SECOND = 2
mock_config.MINUTE_START = 5
mock_config.MINUTE_STEP = 1
mock_config.DELTA_WINDOW = 5
mock_config.FEATURES = [
    "minute",
    "kills_diff", "cs_diff", "level_diff",
    "tower_diff", "inhib_diff", "dragon_diff", "herald_diff", "baron_diff", "grub_diff",
    "kills_diff_d5", "cs_diff_d5", "level_diff_d5",
]
mock_config.TARGET = "blue_win"
mock_config.DELTA_BASE = ["gold_diff", "xp_diff", "kills_diff", "cs_diff", "level_diff"]
mock_config.RAW_DIR = "./fake_raw"
mock_config.TIMELINE_DIR = "./fake_timeline"
mock_config.FEATURES_CSV = "./fake_features.csv"
mock_config.RANKS_CSV = "./fake_ranks.csv"
mock_config.OUT_DIR = "./fake_out"
mock_config.TARGET_MATCHES = 100
mock_config.MATCHES_PER_PLAYER = 20
mock_config.SEED_PAGES_PER_DIVISION = 1
mock_config.SEED_MAX_PER_TIER = 100
mock_config.SEED_TIERS = {"DIAMOND": ["I", "II"]}

sys.modules['config'] = mock_config

sys.modules['crawler'] = MagicMock()

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
