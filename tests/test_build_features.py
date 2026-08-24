import csv
import json

import pytest

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

    @pytest.mark.parametrize("monster, key", [
        ("DRAGON", "dragon_diff"), ("RIFTHERALD", "herald_diff"),
        ("BARON_NASHOR", "baron_diff"), ("HORDE", "grub_diff"),
    ])
    def test_elite_monster_kills_attributed_to_the_red_side_too(self, monster, key):
        base_pf = {str(i): make_participant_frame() for i in range(1, 11)}
        frames = [{
            "participantFrames": base_pf,
            "events": [{"type": "ELITE_MONSTER_KILL", "killerTeamId": 200, "monsterType": monster}],
        }]
        snap = bf.snapshot(_pid_team(), _role_pairs(), frames, minute=0)
        assert snap[key] == -1  # red side got the kill -> diff is negative

    def test_unassigned_role_defaults_to_zero(self):
        base_pf = {str(i): make_participant_frame(gold=100) for i in range(1, 11)}
        frames = [{"participantFrames": base_pf, "events": []}]
        role_pairs = {"TOP": {"blue": 1}}  # red side missing (autofill/remake)
        snap = bf.snapshot(_pid_team(), role_pairs, frames, minute=0)
        assert snap["gold_diff_TOP"] == 0
        assert snap["xp_diff_TOP"] == 0


class TestRowsForMatch:
    def test_participant_with_unmapped_team_position_is_skipped(self):
        """teamPosition values outside ROLE_MAP (e.g. "" on a remake) must
        not blow up role_pairs construction -- just skip that participant."""
        positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", ""]  # last one unmapped
        from conftest import make_participant
        participants = (
            [make_participant(i + 1, 100, pos) for i, pos in enumerate(positions)]
            + [make_participant(i + 6, 200, pos) for i, pos in enumerate(positions)]
        )
        match = make_match(participants=participants)
        timeline = make_timeline(n_minutes=bf.config.MINUTE_START)
        rows = bf.rows_for_match(match, timeline)
        assert len(rows) == 1
        # SUP role never got a pair -> defaults to 0, doesn't crash
        assert rows[0]["gold_diff_SUP"] == 0

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


class TestGetTimeline:
    def test_returns_none_when_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bf.config, "TIMELINE_DIR", tmp_path)
        assert bf.get_timeline("MISSING_MATCH") is None

    def test_loads_cached_timeline_from_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bf.config, "TIMELINE_DIR", tmp_path)
        payload = {"info": {"frames": [{"participantFrames": {}, "events": []}]}}
        (tmp_path / "M1.json").write_text(json.dumps(payload), encoding="utf-8")
        assert bf.get_timeline("M1") == payload


class TestBuild:
    def _setup(self, tmp_path, monkeypatch, matches):
        """matches: list of (match_id, blue_wins, n_minutes) to write to disk."""
        raw_dir = tmp_path / "matches"
        timeline_dir = tmp_path / "timelines"
        raw_dir.mkdir()
        timeline_dir.mkdir()
        monkeypatch.setattr(bf.config, "RAW_DIR", raw_dir)
        monkeypatch.setattr(bf.config, "TIMELINE_DIR", timeline_dir)
        monkeypatch.setattr(bf.config, "FEATURES_CSV", tmp_path / "features.csv")
        monkeypatch.setattr(bf.config, "RANKS_CSV", tmp_path / "match_rank.csv")
        monkeypatch.setattr(bf, "load_ranks", lambda: {})

        for match_id, blue_wins, n_minutes in matches:
            match = make_match(match_id=match_id, blue_wins=blue_wins)
            (raw_dir / f"{match_id}.json").write_text(json.dumps(match), encoding="utf-8")
            timeline = make_timeline(n_minutes=n_minutes)
            (timeline_dir / f"{match_id}.json").write_text(json.dumps(timeline), encoding="utf-8")

        return tmp_path / "features.csv"

    def test_writes_one_row_per_minute_per_match(self, tmp_path, monkeypatch, capsys):
        out_csv = self._setup(tmp_path, monkeypatch, [
            ("M1", True, bf.config.MINUTE_START + 2),
            ("M2", False, bf.config.MINUTE_START),
        ])
        bf.build()
        assert out_csv.exists()
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert {r["match_id"] for r in rows} == {"M1", "M2"}
        m1_rows = [r for r in rows if r["match_id"] == "M1"]
        assert len(m1_rows) == 3  # MINUTE_START, +1, +2
        assert all(r["blue_win"] == "1" for r in m1_rows)

    def test_match_with_no_timeline_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "matches"
        timeline_dir = tmp_path / "timelines"
        raw_dir.mkdir()
        timeline_dir.mkdir()
        monkeypatch.setattr(bf.config, "RAW_DIR", raw_dir)
        monkeypatch.setattr(bf.config, "TIMELINE_DIR", timeline_dir)
        monkeypatch.setattr(bf.config, "FEATURES_CSV", tmp_path / "features.csv")
        monkeypatch.setattr(bf, "load_ranks", lambda: {})

        match = make_match(match_id="NO_TIMELINE")
        (raw_dir / "NO_TIMELINE.json").write_text(json.dumps(match), encoding="utf-8")
        # no matching timeline file written on purpose

        bf.build()
        assert not (tmp_path / "features.csv").exists()  # no rows -> nothing written

    def test_no_matches_on_disk_prints_message_and_does_not_crash(self, tmp_path, monkeypatch, capsys):
        raw_dir = tmp_path / "matches"
        raw_dir.mkdir()
        monkeypatch.setattr(bf.config, "RAW_DIR", raw_dir)
        monkeypatch.setattr(bf.config, "FEATURES_CSV", tmp_path / "features.csv")
        monkeypatch.setattr(bf, "load_ranks", lambda: {})

        bf.build()
        assert not (tmp_path / "features.csv").exists()
        assert "No rows" in capsys.readouterr().out

    def test_elo_band_columns_populated_from_ranks_csv(self, tmp_path, monkeypatch):
        out_csv = self._setup(tmp_path, monkeypatch, [("M1", True, bf.config.MINUTE_START)])
        monkeypatch.setattr(bf, "load_ranks", lambda: {"M1": ("EMERALD", "III")})
        bf.build()
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["tier"] == "EMERALD"
        assert rows[0]["division"] == "III"
