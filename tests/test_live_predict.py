"""
Tests for src/serve/live_predict.py.

main() (the infinite polling loop) is intentionally not tested here, in
keeping with the rest of the suite (crawler.crawl() and train.main() aren't
covered end-to-end either) -- it's a thin orchestration wrapper around the
functions below, which carry all the actual logic and are covered directly.
"""
import importlib
import sys

import pytest
import requests

import src.serve.live_predict as lp


# ---------------------------------------------------------------------------
# fetch_live_data / game_time
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class TestFetchLiveData:
    def test_returns_json_on_200(self, monkeypatch):
        payload = {"gameData": {"gameTime": 42.0}}
        monkeypatch.setattr(lp.requests, "get", lambda *a, **k: FakeResponse(200, payload))
        assert lp.fetch_live_data() == payload

    def test_non_200_raises_no_game_running(self, monkeypatch):
        monkeypatch.setattr(lp.requests, "get", lambda *a, **k: FakeResponse(404))
        with pytest.raises(lp.NoGameRunning):
            lp.fetch_live_data()

    def test_connection_error_raises_no_game_running(self, monkeypatch):
        def raise_conn_error(*a, **k):
            raise requests.exceptions.ConnectionError("no client")
        monkeypatch.setattr(lp.requests, "get", raise_conn_error)
        with pytest.raises(lp.NoGameRunning):
            lp.fetch_live_data()

    def test_hits_the_live_client_url_with_ssl_verification_disabled(self, monkeypatch):
        seen = {}
        def fake_get(url, verify=None, timeout=None):
            seen["url"] = url
            seen["verify"] = verify
            return FakeResponse(200, {})
        monkeypatch.setattr(lp.requests, "get", fake_get)
        lp.fetch_live_data()
        assert seen["url"] == lp.LIVE_CLIENT_URL
        assert seen["verify"] is False  # self-signed cert from the LoL client


class TestGameTime:
    def test_reads_nested_game_time(self):
        assert lp.game_time({"gameData": {"gameTime": 123.4}}) == 123.4

    def test_defaults_to_zero_when_missing(self):
        assert lp.game_time({}) == 0.0
        assert lp.game_time({"gameData": {}}) == 0.0


class TestIsNewGame:
    def test_normal_forward_progress_is_not_a_new_game(self):
        assert lp.is_new_game(100.0, 105.0) is False

    def test_small_backward_jitter_is_not_a_new_game(self):
        assert lp.is_new_game(100.0, 96.0) is False  # within NEW_GAME_TOLERANCE

    def test_exactly_at_tolerance_boundary_is_not_a_new_game(self):
        assert lp.is_new_game(100.0, 100.0 - lp.NEW_GAME_TOLERANCE) is False

    def test_large_backward_jump_is_a_new_game(self):
        assert lp.is_new_game(100.0, 10.0) is True

    def test_first_poll_ever_is_not_a_new_game(self):
        assert lp.is_new_game(None, 10.0) is False


# ---------------------------------------------------------------------------
# team helpers
# ---------------------------------------------------------------------------

class TestOther:
    def test_swaps_sides(self):
        assert lp._other(lp.BLUE) == lp.RED
        assert lp._other(lp.RED) == lp.BLUE


class TestTeamByPlayer:
    def test_maps_each_present_name_key_to_team(self):
        data = {"allPlayers": [
            {"riotIdGameName": "Faker", "riotId": "Faker#KR1", "team": lp.BLUE},
        ]}
        out = lp._team_by_player(data)
        assert out["Faker"] == lp.BLUE
        assert out["Faker#KR1"] == lp.BLUE

    def test_falls_back_to_summoner_name_when_riot_ids_absent(self):
        data = {"allPlayers": [{"summonerName": "OldSchool", "team": lp.RED}]}
        assert lp._team_by_player(data) == {"OldSchool": lp.RED}

    def test_no_players_returns_empty_dict(self):
        assert lp._team_by_player({}) == {}


class TestStructureOwner:
    def test_new_naming_convention(self):
        assert lp._structure_owner("Turret_TOrder_L0_P3") == lp.BLUE
        assert lp._structure_owner("Inhib_TChaos_L1_P1") == lp.RED

    def test_legacy_naming_convention(self):
        assert lp._structure_owner("Mid_T1_Tower") == lp.BLUE
        assert lp._structure_owner("Top_T2_Tower") == lp.RED

    def test_unrecognized_name_returns_none(self):
        assert lp._structure_owner("Unknown_Structure") is None


# ---------------------------------------------------------------------------
# read_counters / read_state
# ---------------------------------------------------------------------------

def _player(team, kills=0, cs=0, level=1, name="P"):
    return {"team": team, "level": level,
            "scores": {"kills": kills, "creepScore": cs},
            "riotIdGameName": name}


class TestReadCounters:
    def test_sums_kills_cs_level_per_team_from_scores(self):
        data = {"allPlayers": [
            _player(lp.BLUE, kills=3, cs=50, level=6, name="B1"),
            _player(lp.BLUE, kills=1, cs=40, level=5, name="B2"),
            _player(lp.RED, kills=2, cs=45, level=5, name="R1"),
        ], "events": {"Events": []}}
        c = lp.read_counters(data)
        assert c["kills"] == {lp.BLUE: 4, lp.RED: 2}
        assert c["cs"] == {lp.BLUE: 90, lp.RED: 45}
        assert c["level"] == {lp.BLUE: 11, lp.RED: 5}

    def test_tower_credited_to_the_team_that_did_not_own_it(self):
        data = {"allPlayers": [], "events": {"Events": [
            {"EventName": "TurretKilled", "TurretKilled": "Turret_TOrder_L0_P3"},
        ]}}
        c = lp.read_counters(data)
        # BLUE-owned tower fell -> RED gets credit
        assert c["towers"] == {lp.BLUE: 0, lp.RED: 1}

    def test_inhib_credited_to_the_team_that_did_not_own_it(self):
        data = {"allPlayers": [], "events": {"Events": [
            {"EventName": "InhibKilled", "InhibKilled": "Inhib_TChaos_L1_P1"},
        ]}}
        c = lp.read_counters(data)
        # RED-owned inhib fell -> BLUE gets credit
        assert c["inhibs"] == {lp.BLUE: 1, lp.RED: 0}

    def test_unrecognized_structure_name_is_not_attributed(self):
        data = {"allPlayers": [], "events": {"Events": [
            {"EventName": "TurretKilled", "TurretKilled": "Weird_Unknown_Name"},
        ]}}
        c = lp.read_counters(data)
        assert c["towers"] == {lp.BLUE: 0, lp.RED: 0}

    @pytest.mark.parametrize("event_name, counter_key", [
        ("DragonKill", "dragons"),
        ("HeraldKill", "heralds"),
        ("BaronKill", "barons"),
        ("HordeKill", "grubs"),
    ])
    def test_epic_monsters_credited_to_the_killers_team(self, event_name, counter_key):
        data = {"allPlayers": [_player(lp.BLUE, name="Jungler")],
                "events": {"Events": [{"EventName": event_name, "KillerName": "Jungler"}]}}
        c = lp.read_counters(data)
        assert c[counter_key][lp.BLUE] == 1
        assert c[counter_key][lp.RED] == 0

    def test_monster_kill_by_unknown_killer_is_skipped(self):
        """KillerName not found in allPlayers (e.g. killed by a minion/pet)
        must not attribute the objective to either side."""
        data = {"allPlayers": [_player(lp.BLUE, name="Jungler")],
                "events": {"Events": [{"EventName": "DragonKill", "KillerName": "SomePet"}]}}
        c = lp.read_counters(data)
        assert c["dragons"] == {lp.BLUE: 0, lp.RED: 0}


class TestReadState:
    def test_combines_minute_and_diffs(self):
        data = {"gameData": {"gameTime": 725.0},  # 12:05 -> minute 12
                "allPlayers": [
                    _player(lp.BLUE, kills=5, cs=100, level=8, name="B1"),
                    _player(lp.RED, kills=2, cs=80, level=6, name="R1"),
                ],
                "events": {"Events": []}}
        state = lp.read_state(data)
        assert state["minute"] == 12
        assert state["kills_diff"] == 3
        assert state["cs_diff"] == 20
        assert state["level_diff"] == 2
        assert state["tower_diff"] == 0


# ---------------------------------------------------------------------------
# build_features (momentum window)
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_no_history_five_minutes_back_yields_zero_deltas(self):
        state = {"minute": 3, "kills_diff": 1, "cs_diff": 10, "level_diff": 0,
                  "tower_diff": 0}
        feats = lp.build_features(state, history=[])
        assert feats["kills_diff_d5"] == 0
        assert feats["cs_diff_d5"] == 0
        assert feats["level_diff_d5"] == 0

    def test_uses_the_most_recent_snapshot_at_or_before_the_target_minute(self):
        state = {"minute": 10, "kills_diff": 5, "cs_diff": 20, "level_diff": 2,
                  "tower_diff": 1}
        history = [
            {"minute": 4, "kills_diff": 0, "cs_diff": 0, "level_diff": 0},
            {"minute": 5, "kills_diff": 2, "cs_diff": 10, "level_diff": 1},
            {"minute": 6, "kills_diff": 3, "cs_diff": 15, "level_diff": 1},
        ]
        feats = lp.build_features(state, history)
        assert feats["kills_diff_d5"] == 5 - 2
        assert feats["cs_diff_d5"] == 20 - 10
        assert feats["level_diff_d5"] == 2 - 1
        assert feats["tower_diff"] == 1  # absolute fields pass through untouched


# ---------------------------------------------------------------------------
# active_team
# ---------------------------------------------------------------------------

class TestActiveTeam:
    def test_uses_riot_id_when_present(self):
        data = {"activePlayer": {"riotId": "Me#EUW"},
                "allPlayers": [{"riotId": "Me#EUW", "team": lp.RED}]}
        assert lp.active_team(data) == lp.RED

    def test_falls_back_to_summoner_name(self):
        data = {"activePlayer": {"summonerName": "OldMe"},
                "allPlayers": [{"summonerName": "OldMe", "team": lp.BLUE}]}
        assert lp.active_team(data) == lp.BLUE

    def test_defaults_to_blue_when_player_not_found(self):
        assert lp.active_team({"activePlayer": {}, "allPlayers": []}) == lp.BLUE


# ---------------------------------------------------------------------------
# warm_up_api / predict_blue_winrate (talking to the deployed API)
# ---------------------------------------------------------------------------

class TestWarmUpApi:
    def test_pings_the_health_endpoint_derived_from_predict_url(self, monkeypatch):
        seen = {}
        def fake_get(url, timeout=None):
            seen["url"] = url
            return FakeResponse(200)
        monkeypatch.setattr(lp.requests, "get", fake_get)
        lp.warm_up_api()
        assert seen["url"] == lp.PREDICT_API_URL.rsplit("/predict", 1)[0] + "/health"

    def test_swallows_connection_errors_silently(self, monkeypatch):
        def raise_error(*a, **k):
            raise requests.exceptions.ConnectionError("cold start / down")
        monkeypatch.setattr(lp.requests, "get", raise_error)
        lp.warm_up_api()  # must not raise


class TestPredictBlueWinrate:
    VALID_FEATURES = {
        "minute": 15, "kills_diff": 2, "cs_diff": 15, "level_diff": 1,
        "tower_diff": 1, "inhib_diff": 0, "dragon_diff": 1, "herald_diff": 1,
        "baron_diff": 0, "grub_diff": 3, "kills_diff_d5": 0, "cs_diff_d5": 5,
        "level_diff_d5": 0, "irrelevant_extra_key": 999,
    }

    def test_returns_p_blue_from_the_response(self, monkeypatch):
        monkeypatch.setattr(
            lp.requests, "post",
            lambda url, json=None, timeout=None: FakeResponse(200, {"p_blue": 0.63, "p_red": 0.37}),
        )
        assert lp.predict_blue_winrate(self.VALID_FEATURES) == 0.63

    def test_payload_only_includes_features_in_order_dropping_extras(self, monkeypatch):
        captured = {}
        def fake_post(url, json=None, timeout=None):
            captured["json"] = json
            captured["url"] = url
            return FakeResponse(200, {"p_blue": 0.5, "p_red": 0.5})
        monkeypatch.setattr(lp.requests, "post", fake_post)
        lp.predict_blue_winrate(self.VALID_FEATURES)
        assert list(captured["json"].keys()) == lp.FEATURES
        assert "irrelevant_extra_key" not in captured["json"]
        assert captured["url"] == lp.PREDICT_API_URL

    def test_non_200_response_raises_prediction_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            lp.requests, "post",
            lambda url, json=None, timeout=None: FakeResponse(503, {}),
        )
        with pytest.raises(lp.PredictionUnavailable):
            lp.predict_blue_winrate(self.VALID_FEATURES)

    def test_connection_error_raises_prediction_unavailable(self, monkeypatch):
        def raise_error(*a, **k):
            raise requests.exceptions.ConnectionError("api is down")
        monkeypatch.setattr(lp.requests, "post", raise_error)
        with pytest.raises(lp.PredictionUnavailable):
            lp.predict_blue_winrate(self.VALID_FEATURES)

    def test_timeout_is_generous_for_free_tier_cold_starts(self):
        assert lp.PREDICT_TIMEOUT >= 20


# ---------------------------------------------------------------------------
# API_URL environment override
# ---------------------------------------------------------------------------

class TestApiUrlFromEnv:
    def test_default_points_at_docker_internal_host(self):
        # module-level default when API_URL isn't set (the conftest env
        # fixture doesn't touch API_URL, and this module was first imported
        # without it set in the test session).
        assert lp.PREDICT_API_URL.endswith("/predict")

    def test_api_url_env_var_overrides_the_default(self, tmp_path, monkeypatch):
        """Reimport a private copy so we don't mutate the shared lp module
        (other tests in this file rely on its current PREDICT_API_URL)."""
        monkeypatch.setenv("API_URL", "https://leaguepredictor.onrender.com")

        fake_pkg_dir = tmp_path / "src" / "serve"
        fake_pkg_dir.mkdir(parents=True)
        source = open(lp.__file__, encoding="utf-8").read()
        (fake_pkg_dir / "live_predict_isolated.py").write_text(source, encoding="utf-8")

        sys.path.insert(0, str(fake_pkg_dir))
        sys.modules.pop("live_predict_isolated", None)
        try:
            fresh = importlib.import_module("live_predict_isolated")
            assert fresh.PREDICT_API_URL == "https://leaguepredictor.onrender.com/predict"
        finally:
            sys.modules.pop("live_predict_isolated", None)
            sys.path.remove(str(fake_pkg_dir))
