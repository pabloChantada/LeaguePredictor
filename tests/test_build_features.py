import src.building.build_features as bf
from conftest import make_match, make_participant_frame, make_timeline
from src.building.config import *

def _pid_team():
    return {i: 100 for i in range(1, 6)} | {i: 200 for i in range(6, 11)}


def _role_pairs():
    roles = ["TOP", "JGL", "MID", "BOT", "SUP"]
    return {rk: {"blue": i + 1, "red": i + 6} for i, rk in enumerate(roles)}


class TestSnapshot:
    def test_gold_xp_cs_diffs(self):
        frames = [{
            "participantFrames": {
                **{str(i): make_participant_frame(gold=500, xp=400, minions=10, level=3)
                   for i in range(1, 6)},
                **{str(i): make_participant_frame(gold=300, xp=250, minions=6, level=2)
                   for i in range(6, 11)},
            },
            "events": [],
        }]
        snap = bf.snapshot(_pid_team(), _role_pairs(), frames, minute=0)
        assert snap["gold_diff"] == (500 - 300) * 5
        assert snap["xp_diff"] == (400 - 250) * 5
        assert snap["cs_diff"] == (10 - 6) * 5
        assert snap["level_diff"] == (3 - 2) * 5

    def test_events_accumulate_up_to_minute(self):
        """Kills/objectives should be counted cumulatively across all frames <= minute."""
        base_pf = {str(i): make_participant_frame() for i in range(1, 11)}
        frames = [
            {"participantFrames": base_pf,
             "events": [{"type": "CHAMPION_KILL", "killerId": 1}]},  # blue kill @0
            {"participantFrames": base_pf,
             "events": [{"type": "CHAMPION_KILL", "killerId": 6},   # red kill @1
                        {"type": "ELITE_MONSTER_KILL", "killerTeamId": 100,
                         "monsterType": "DRAGON"}]},
        ]
        snap0 = bf.snapshot(_pid_team(), _role_pairs(), frames, minute=0)
        assert snap0["kills_diff"] == 1  # only the minute-0 event counted

        snap1 = bf.snapshot(_pid_team(), _role_pairs(), frames, minute=1)
        assert snap1["kills_diff"] == 0  # 1 blue kill, 1 red kill by minute 1
        assert snap1["dragon_diff"] == 1

    def test_building_kill_owner_is_the_building_that_fell(self):
        """teamId on BUILDING_KILL is the OWNER; the other team gets credit."""
        base_pf = {str(i): make_participant_frame() for i in range(1, 11)}
        frames = [{
            "participantFrames": base_pf,
            "events": [{"type": "BUILDING_KILL", "teamId": 200, "buildingType": "TOWER_BUILDING"}],
        }]
        snap = bf.snapshot(_pid_team(), _role_pairs(), frames, minute=0)
        assert snap["tower_diff"] == 1  # blue took down a red (owner=200) tower

    def test_unassigned_role_defaults_to_zero(self):
        base_pf = {str(i): make_participant_frame(gold=100) for i in range(1, 11)}
        frames = [{"participantFrames": base_pf, "events": []}]
        role_pairs = {"TOP": {"blue": 1}}  # red side missing (autofill/remake)
        snap = bf.snapshot(_pid_team(), role_pairs, frames, minute=0)
        assert snap["gold_diff_TOP"] == 0
        assert snap["xp_diff_TOP"] == 0


class TestRowsForMatch:
    def test_short_match_returns_no_rows(self):
        match = make_match()
        timeline = make_timeline(n_minutes=bf.config.MINUTE_START - 1)
        assert bf.rows_for_match(match, timeline) == []

    def test_blue_win_and_metadata_columns(self):
        match = make_match(match_id="M42", blue_wins=True, queue_id=420,
                            game_version="16.13.567.1")
        timeline = make_timeline(n_minutes=bf.config.MINUTE_START)
        rows = bf.rows_for_match(match, timeline, band=("DIAMOND", "II"))
        assert len(rows) == 1
        row = rows[0]
        assert row["match_id"] == "M42"
        assert row["blue_win"] == 1
        assert row["queue_id"] == 420
        assert row["patch"] == "16.13"
        assert row["tier"] == "DIAMOND"
        assert row["division"] == "II"

    def test_momentum_delta_zero_before_window_then_real_after(self):
        """_d5 should be 0 until DELTA_WINDOW minutes of history exist, then the
        real difference against the snapshot from DELTA_WINDOW minutes ago."""
        def pf_by_minute(m):
            # blue gold grows by 100/min, red stays flat -> gold_diff grows linearly
            return {
                **{str(i): make_participant_frame(gold=100 * m) for i in range(1, 6)},
                **{str(i): make_participant_frame(gold=0) for i in range(6, 11)},
            }

        match = make_match(blue_wins=False)
        n = bf.config.MINUTE_START + bf.config.DELTA_WINDOW
        timeline = make_timeline(n_minutes=n, participant_frames_by_minute=pf_by_minute)
        rows = bf.rows_for_match(match, timeline)
        by_minute = {r["minute"]: r for r in rows}

        first_row = by_minute[bf.config.MINUTE_START]
        assert first_row["gold_diff_d5"] == 0  # no minute (start - 5) recorded

        later_minute = bf.config.MINUTE_START + bf.config.DELTA_WINDOW
        later_row = by_minute[later_minute]
        expected = later_row["gold_diff"] - by_minute[later_minute - bf.config.DELTA_WINDOW]["gold_diff"]
        assert later_row["gold_diff_d5"] == expected
        assert expected == 5 * 100 * bf.config.DELTA_WINDOW  # 5 blue players * 100 gold/min * 5 min
