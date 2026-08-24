"""
Tests for src/serve/live_dashboard.py.

Only the pure/near-pure rendering and parsing functions are covered here
(clock, palette, draw_curve, draw_scoreboard, load_demo_match). main() and
reset_match() are Streamlit session-state orchestration that needs a real
ScriptRunContext to behave predictably -- consistent with the rest of the
suite, which doesn't drive crawler.crawl() or train.main() end-to-end
either, those are left to manual/integration testing.
"""
import json

import matplotlib
matplotlib.use("Agg")  # headless: no display available in CI/containers
import matplotlib.figure
import pytest

import src.serve.live_dashboard as ld


# ---------------------------------------------------------------------------
# clock
# ---------------------------------------------------------------------------

class TestClock:
    def test_formats_minutes_and_seconds(self):
        assert ld.clock(725) == "12:05"

    def test_pads_single_digit_seconds(self):
        assert ld.clock(63) == "1:03"

    def test_zero(self):
        assert ld.clock(0) == "0:00"

    def test_exact_minute_boundary(self):
        assert ld.clock(600) == "10:00"

    def test_accepts_floats_from_game_time(self):
        assert ld.clock(725.9) == "12:05"


# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------

class FakeTheme:
    def __init__(self, theme_type):
        self.type = theme_type


class FakeContext:
    def __init__(self, theme=None):
        self.theme = theme


class TestPalette:
    def test_defaults_to_light_without_a_theme(self):
        # st.context.theme is unavailable outside a real Streamlit session.
        assert ld.palette() == ld.LIGHT

    def test_dark_theme_selects_dark_palette(self, monkeypatch):
        monkeypatch.setattr(ld.st, "context", FakeContext(theme=FakeTheme("dark")))
        assert ld.palette() == ld.DARK

    def test_light_theme_selects_light_palette(self, monkeypatch):
        monkeypatch.setattr(ld.st, "context", FakeContext(theme=FakeTheme("light")))
        assert ld.palette() == ld.LIGHT


# ---------------------------------------------------------------------------
# draw_curve
# ---------------------------------------------------------------------------

class TestDrawCurve:
    def test_returns_a_matplotlib_figure(self):
        curve = [(0, 0.5), (5, 0.6), (10, 0.75)]
        fig = ld.draw_curve(curve, ld.lp.BLUE, ld.LIGHT)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_y_axis_is_fixed_to_the_0_100_percent_range(self):
        curve = [(0, 0.1), (5, 0.9)]
        fig = ld.draw_curve(curve, ld.lp.BLUE, ld.LIGHT)
        ax = fig.axes[0]
        assert ax.get_ylim() == (0, 100)

    def test_perspective_flips_for_the_red_team(self):
        """draw_curve plots the caller's OWN win%, so blue p=0.8 must render
        as 20% when the caller is on the red side."""
        curve = [(0, 0.8)]
        fig_blue = ld.draw_curve(curve, ld.lp.BLUE, ld.LIGHT)
        fig_red = ld.draw_curve(curve, ld.lp.RED, ld.LIGHT)
        line_blue = fig_blue.axes[0].lines[0].get_ydata()
        line_red = fig_red.axes[0].lines[0].get_ydata()
        assert line_blue[-1] == pytest.approx(80.0)
        assert line_red[-1] == pytest.approx(20.0)

    def test_does_not_crash_on_a_single_point(self):
        fig = ld.draw_curve([(0, 0.5)], ld.lp.BLUE, ld.LIGHT)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_does_not_crash_when_the_curve_crosses_fifty_percent(self):
        curve = [(0, 0.3), (5, 0.5), (10, 0.7), (15, 0.4)]
        fig = ld.draw_curve(curve, ld.lp.BLUE, ld.LIGHT)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# draw_scoreboard
# ---------------------------------------------------------------------------

class TestDrawScoreboard:
    def test_renders_one_markdown_call_with_all_counters(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ld.st, "markdown", lambda html, **kw: calls.append(html))

        counters = {k: {ld.lp.BLUE: 3, ld.lp.RED: 1} for k in ld.lp.COUNTERS}
        ld.draw_scoreboard(counters, ld.LIGHT)

        assert len(calls) == 1
        html = calls[0]
        assert "BLUE" in html and "RED" in html
        for label in ld.LABELS.values():
            assert label in html

    def test_shows_the_correct_per_side_absolute_values(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ld.st, "markdown", lambda html, **kw: calls.append(html))

        counters = {k: {ld.lp.BLUE: 0, ld.lp.RED: 0} for k in ld.lp.COUNTERS}
        counters["kills"] = {ld.lp.BLUE: 7, ld.lp.RED: 2}
        ld.draw_scoreboard(counters, ld.LIGHT)

        html = calls[0]
        assert ">7<" in html
        assert ">2<" in html


# ---------------------------------------------------------------------------
# load_demo_match
# ---------------------------------------------------------------------------

def _demo_frame(events=None, minions=0, level=1):
    pf = {str(pid): {"minionsKilled": minions, "jungleMinionsKilled": 0, "level": level}
          for pid in range(1, 11)}
    return {"events": events or [], "participantFrames": pf}


def _write_demo_file(folder, name, frames):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps({"info": {"frames": frames}}), encoding="utf-8")


class TestLoadDemoMatch:
    def test_returns_empty_list_when_no_demo_files_exist(self, tmp_path):
        empty_dir = tmp_path / "no_demos"
        assert ld.load_demo_match(folder_path=str(empty_dir)) == []

    def test_one_entry_per_frame(self, tmp_path):
        frames = [_demo_frame(), _demo_frame(), _demo_frame()]
        _write_demo_file(tmp_path, "match1.json", frames)
        history = ld.load_demo_match(folder_path=str(tmp_path))
        assert len(history) == 3
        state0, counters0 = history[0]
        assert state0["minute"] == 0

    def test_champion_kill_attributed_by_killer_id_range(self, tmp_path):
        frames = [
            _demo_frame(events=[{"type": "CHAMPION_KILL", "killerId": 1}]),  # blue (1-5)
            _demo_frame(events=[{"type": "CHAMPION_KILL", "killerId": 7}]),  # red (6-10)
        ]
        _write_demo_file(tmp_path, "match1.json", frames)
        history = ld.load_demo_match(folder_path=str(tmp_path))
        state0, counters0 = history[0]
        assert counters0["kills"][ld.lp.BLUE] == 1
        assert counters0["kills"][ld.lp.RED] == 0
        state1, counters1 = history[1]
        assert counters1["kills"][ld.lp.RED] == 1

    def test_building_kill_credits_the_non_owning_team(self, tmp_path):
        """teamId on the event is the structure's OWNER (per the same Riot
        convention build_features.py and live_predict.py both rely on)."""
        frames = [_demo_frame(events=[
            {"type": "BUILDING_KILL", "teamId": 200, "buildingType": "TOWER_BUILDING"},
        ])]
        _write_demo_file(tmp_path, "match1.json", frames)
        state0, counters0 = ld.load_demo_match(folder_path=str(tmp_path))[0]
        assert counters0["towers"][ld.lp.BLUE] == 1  # blue destroyed red's tower
        assert counters0["towers"][ld.lp.RED] == 0

    @pytest.mark.parametrize("monster_type, counter_key", [
        ("DRAGON", "dragons"), ("RIFTHERALD", "heralds"),
        ("BARON_NASHOR", "barons"), ("HORDE", "grubs"),
    ])
    def test_elite_monsters_attributed_by_killer_id_range(self, tmp_path, monster_type, counter_key):
        frames = [_demo_frame(events=[
            {"type": "ELITE_MONSTER_KILL", "killerId": 3, "monsterType": monster_type},
        ])]
        _write_demo_file(tmp_path, "match1.json", frames)
        state0, counters0 = ld.load_demo_match(folder_path=str(tmp_path))[0]
        assert counters0[counter_key][ld.lp.BLUE] == 1
        assert counters0[counter_key][ld.lp.RED] == 0

    def test_cs_and_level_summed_per_side_from_participant_frames(self, tmp_path):
        frames = [_demo_frame(minions=8, level=4)]
        _write_demo_file(tmp_path, "match1.json", frames)
        state0, counters0 = ld.load_demo_match(folder_path=str(tmp_path))[0]
        # 5 players/side * 8 cs, 5 players/side * level 4
        assert counters0["cs"][ld.lp.BLUE] == 40
        assert counters0["cs"][ld.lp.RED] == 40
        assert counters0["level"][ld.lp.BLUE] == 20
        assert counters0["level"][ld.lp.RED] == 20
        assert state0["cs_diff"] == 0
        assert state0["level_diff"] == 0
