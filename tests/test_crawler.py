import csv
import json

import pytest
import requests

import src.building.crawler as crawler
from src.building.config import *


class FakeResponse:
    """Minimal stand-in for requests.Response covering what crawler.py uses."""

    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")


class TestRequireApiKey:
    def test_raises_without_key(self, monkeypatch):
        # Apuntamos a crawler.config en lugar de crawler
        monkeypatch.setattr(crawler.config, "API_KEY", None)
        with pytest.raises(RuntimeError):
            crawler.require_api_key()

    def test_passes_with_key(self, monkeypatch):
        monkeypatch.setattr(crawler.config, "API_KEY", "fake-key")
        crawler.require_api_key()  # should not raise

    def test_riot_get_checks_the_key_before_any_request(self, monkeypatch):
        monkeypatch.setattr(crawler.config, "API_KEY", None)
        with pytest.raises(RuntimeError):
            crawler.riot_get("https://example.invalid")


class TestLoadRanks:
    def test_missing_file_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crawler.config, "RANKS_CSV", tmp_path / "nope.csv")
        assert crawler.load_ranks() == {}

    def test_reads_tier_division_by_match_id(self, tmp_path, monkeypatch):
        path = tmp_path / "match_rank.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["match_id", "tier", "division"])
            w.writerow(["M1", "DIAMOND", "II"])
            w.writerow(["M2", "MASTER", "I"])
        monkeypatch.setattr(crawler.config, "RANKS_CSV", path)
        ranks = crawler.load_ranks()
        assert ranks == {"M1": ("DIAMOND", "II"), "M2": ("MASTER", "I")}


class TestTierCandidates:
    def test_apex_tier_uses_single_call_endpoint(self, monkeypatch):
        # Las funciones sí están en crawler, esto se queda igual
        monkeypatch.setattr(crawler, "get_apex_league",
                             lambda tier: [{"puuid": "p1", "rank": "I"},
                                           {"puuid": "p2", "rank": "I"}])
        out = crawler._tier_candidates("MASTER", None)
        assert out == [("p1", "MASTER", "I"), ("p2", "MASTER", "I")]

    def test_divisions_paginate_until_empty_page(self, monkeypatch):
        pages = {
            ("EMERALD", "IV", 1): [{"puuid": "a"}, {"puuid": "b"}],
            ("EMERALD", "IV", 2): [],  # stop here
        }
        monkeypatch.setattr(
            crawler, "get_league_entries",
            lambda tier, division, page=1: pages.get((tier, division, page), []),
        )
        out = crawler._tier_candidates("EMERALD", ("IV",))
        assert out == [("a", "EMERALD", "IV"), ("b", "EMERALD", "IV")]

    def test_entries_without_puuid_are_skipped(self, monkeypatch):
        monkeypatch.setattr(crawler, "get_apex_league",
                             lambda tier: [{"rank": "I"}])  # no puuid
        out = crawler._tier_candidates("MASTER", None)
        assert out == []


class TestSeedFromTiers:
    def test_dedups_and_populates_the_queue(self, monkeypatch):
        monkeypatch.setattr(crawler.config, "SEED_TIERS", {"EMERALD": ("IV",)})
        monkeypatch.setattr(crawler.config, "SEED_MAX_PER_TIER", 100)
        monkeypatch.setattr(
            crawler, "_tier_candidates",
            lambda tier, divisions: [("p1", "EMERALD", "IV"),
                                      ("p1", "EMERALD", "IV"),  # duplicate puuid
                                      ("p2", "EMERALD", "IV")],
        )
        crawler.seen_puuids.clear()
        crawler.player_queue.clear()

        added = crawler.seed_from_tiers()
        assert added == 2
        assert crawler.seen_puuids == {"p1", "p2"}
        assert len(crawler.player_queue) == 2

    def test_trims_to_seed_max_per_tier(self, monkeypatch):
        candidates = [(f"p{i}", "EMERALD", "IV") for i in range(10)]
        monkeypatch.setattr(crawler.config, "SEED_TIERS", {"EMERALD": ("IV",)})
        monkeypatch.setattr(crawler.config, "SEED_MAX_PER_TIER", 3)
        monkeypatch.setattr(crawler, "_tier_candidates",
                             lambda tier, divisions: candidates)
        crawler.seen_puuids.clear()
        crawler.player_queue.clear()

        added = crawler.seed_from_tiers()
        assert added == 3


class TestThrottle:
    def test_allows_requests_under_the_limits(self, monkeypatch):
        """With generous limits, _throttle should never sleep."""
        monkeypatch.setattr(crawler.config, "MAX_PER_SECOND", 1000)
        monkeypatch.setattr(crawler.config, "MAX_PER_WINDOW", 1000)
        crawler._req_times.clear()
        sleeps = []
        monkeypatch.setattr(crawler.time, "sleep", lambda s: sleeps.append(s))
        for _ in range(5):
            crawler._throttle()
        assert sleeps == []
        assert len(crawler._req_times) == 5


@pytest.fixture
def api_key(monkeypatch):
    """riot_get requires a key before it will even try a request."""
    monkeypatch.setattr(crawler.config, "API_KEY", "fake-key")
    monkeypatch.setattr(crawler, "HEADERS", {"X-Riot-Token": "fake-key"})


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """No test in this module should actually block on time.sleep."""
    monkeypatch.setattr(crawler.time, "sleep", lambda s: None)


class TestRiotGet:
    def test_returns_json_on_success(self, monkeypatch, api_key):
        monkeypatch.setattr(crawler.requests, "get",
                             lambda *a, **k: FakeResponse(200, {"ok": True}))
        assert crawler.riot_get("https://example.invalid") == {"ok": True}

    def test_429_retries_after_the_retry_after_header(self, monkeypatch, api_key):
        calls = {"n": 0}
        slept = []
        monkeypatch.setattr(crawler.time, "sleep", lambda s: slept.append(s))

        def fake_get(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse(429, headers={"Retry-After": "2"})
            return FakeResponse(200, {"ok": True})

        monkeypatch.setattr(crawler.requests, "get", fake_get)
        result = crawler.riot_get("https://example.invalid")
        assert result == {"ok": True}
        assert calls["n"] == 2
        assert slept[0] == pytest.approx(2.2)  # Retry-After + 0.2 buffer

    def test_429_without_retry_after_header_defaults_to_one_second(self, monkeypatch, api_key):
        calls = {"n": 0}

        def fake_get(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse(429)  # no Retry-After header
            return FakeResponse(200, {"ok": True})

        monkeypatch.setattr(crawler.requests, "get", fake_get)
        assert crawler.riot_get("https://example.invalid") == {"ok": True}

    def test_5xx_is_retried_with_backoff(self, monkeypatch, api_key):
        calls = {"n": 0}

        def fake_get(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                return FakeResponse(503)
            return FakeResponse(200, {"ok": True})

        monkeypatch.setattr(crawler.requests, "get", fake_get)
        assert crawler.riot_get("https://example.invalid") == {"ok": True}
        assert calls["n"] == 3

    def test_4xx_other_than_429_raises_immediately_without_retrying(self, monkeypatch, api_key):
        calls = {"n": 0}

        def fake_get(*a, **k):
            calls["n"] += 1
            return FakeResponse(404)

        monkeypatch.setattr(crawler.requests, "get", fake_get)
        with pytest.raises(requests.exceptions.HTTPError):
            crawler.riot_get("https://example.invalid")
        assert calls["n"] == 1  # no retry for a plain client error

    def test_exhausting_retries_on_persistent_5xx_raises_runtime_error(self, monkeypatch, api_key):
        monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: FakeResponse(500))
        with pytest.raises(RuntimeError, match="Failed request"):
            crawler.riot_get("https://example.invalid", retries=3)

    def test_missing_api_key_raises_before_any_request_is_attempted(self, monkeypatch):
        monkeypatch.setattr(crawler.config, "API_KEY", None)
        called = {"n": 0}
        monkeypatch.setattr(crawler.requests, "get", lambda *a, **k: called.__setitem__("n", 1))
        with pytest.raises(RuntimeError):
            crawler.riot_get("https://example.invalid")
        assert called["n"] == 0


class TestEndpointWrappers:
    """Each wrapper just builds the right URL/params for riot_get."""

    def test_get_league_entries_url_and_params(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(crawler, "riot_get",
                             lambda url, params=None: seen.update(url=url, params=params))
        crawler.get_league_entries("DIAMOND", "II", page=3)
        assert seen["url"].endswith(f"/lol/league/v4/entries/{crawler.config.QUEUE}/DIAMOND/II")
        assert seen["params"] == {"page": 3}

    def test_get_apex_league_returns_entries_list(self, monkeypatch):
        monkeypatch.setattr(crawler, "riot_get",
                             lambda url: {"entries": [{"puuid": "p1"}]})
        assert crawler.get_apex_league("MASTER") == [{"puuid": "p1"}]

    def test_get_apex_league_missing_entries_key_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(crawler, "riot_get", lambda url: {})
        assert crawler.get_apex_league("MASTER") == []

    def test_get_match_ids_url_and_params(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(crawler, "riot_get",
                             lambda url, params=None: seen.update(url=url, params=params))
        crawler.get_match_ids("puuid-123", start=10, count=5)
        assert "puuid-123/ids" in seen["url"]
        assert seen["params"] == {"start": 10, "count": 5, "queue": crawler.config.QUEUE_ID}

    def test_get_match_url(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(crawler, "riot_get", lambda url: seen.setdefault("url", url))
        crawler.get_match("EUW1_123")
        assert seen["url"].endswith("/lol/match/v5/matches/EUW1_123")

    def test_get_timeline_url(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(crawler, "riot_get", lambda url: seen.setdefault("url", url))
        crawler.get_timeline("EUW1_123")
        assert seen["url"].endswith("/lol/match/v5/matches/EUW1_123/timeline")


class TestSaveMatchAndTimeline:
    def test_save_match_writes_json_keyed_by_match_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crawler.config, "RAW_DIR", tmp_path)
        match = {"metadata": {"matchId": "EUW1_999"}, "info": {}}
        crawler.save_match(match)
        written = json.loads((tmp_path / "EUW1_999.json").read_text())
        assert written == match

    def test_save_timeline_writes_json_keyed_by_match_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crawler.config, "TIMELINE_DIR", tmp_path)
        timeline = {"info": {"frames": []}}
        crawler.save_timeline("EUW1_999", timeline)
        written = json.loads((tmp_path / "EUW1_999.json").read_text())
        assert written == timeline


class TestLoadExisting:
    def test_populates_seen_match_ids_from_json_files_on_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crawler.config, "RAW_DIR", tmp_path)
        (tmp_path / "M1.json").write_text("{}")
        (tmp_path / "M2.json").write_text("{}")
        (tmp_path / "not_a_match.txt").write_text("ignore me")
        crawler.seen_match_ids.clear()
        crawler.load_existing()
        assert crawler.seen_match_ids == {"M1", "M2"}


class TestCrawl:
    def _one_player_two_matches(self, monkeypatch, tmp_path):
        monkeypatch.setattr(crawler.config, "RANKS_CSV", tmp_path / "match_rank.csv")
        monkeypatch.setattr(crawler.config, "RAW_DIR", tmp_path / "matches")
        monkeypatch.setattr(crawler.config, "TIMELINE_DIR", tmp_path / "timelines")
        monkeypatch.setattr(crawler.config, "OUT_DIR", tmp_path)
        (tmp_path / "matches").mkdir()
        (tmp_path / "timelines").mkdir()
        monkeypatch.setattr(crawler.config, "TARGET_MATCHES", 2)
        monkeypatch.setattr(crawler.config, "SAVE_EVERY", 1)

        monkeypatch.setattr(crawler, "get_match_ids", lambda puuid, start=0, count=20: ["M1", "M2"])
        monkeypatch.setattr(crawler, "get_match",
                             lambda mid: {"metadata": {"matchId": mid}, "info": {}})
        monkeypatch.setattr(crawler, "get_timeline", lambda mid: {"info": {"frames": []}})

        crawler.seen_match_ids.clear()
        crawler.player_queue.clear()
        crawler.player_queue.append(("puuid-1", "DIAMOND", "II"))

    def test_crawls_until_target_matches_reached(self, tmp_path, monkeypatch):
        self._one_player_two_matches(monkeypatch, tmp_path)
        crawler.crawl()

        assert crawler.seen_match_ids == {"M1", "M2"}
        assert (tmp_path / "matches" / "M1.json").exists()
        assert (tmp_path / "matches" / "M2.json").exists()
        assert (tmp_path / "timelines" / "M1.json").exists()

    def test_writes_ranks_csv_with_header_and_rows(self, tmp_path, monkeypatch):
        self._one_player_two_matches(monkeypatch, tmp_path)
        crawler.crawl()

        with open(tmp_path / "match_rank.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["tier"] == "DIAMOND"
        assert rows[0]["division"] == "II"

    def test_already_seen_match_ids_are_skipped(self, tmp_path, monkeypatch):
        self._one_player_two_matches(monkeypatch, tmp_path)
        crawler.seen_match_ids.add("M1")  # pretend M1 was already crawled
        crawler.crawl()
        assert not (tmp_path / "matches" / "M1.json").exists()
        assert (tmp_path / "matches" / "M2.json").exists()

    def test_a_player_whose_match_ids_call_fails_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crawler.config, "RANKS_CSV", tmp_path / "match_rank.csv")
        monkeypatch.setattr(crawler.config, "RAW_DIR", tmp_path / "matches")
        monkeypatch.setattr(crawler.config, "TIMELINE_DIR", tmp_path / "timelines")
        monkeypatch.setattr(crawler.config, "OUT_DIR", tmp_path)
        (tmp_path / "matches").mkdir()
        (tmp_path / "timelines").mkdir()
        monkeypatch.setattr(crawler.config, "TARGET_MATCHES", 100)
        monkeypatch.setattr(crawler.config, "SAVE_EVERY", 1)

        def flaky_get_match_ids(puuid, start=0, count=20):
            if puuid == "bad-puuid":
                raise RuntimeError("network error")
            return []

        monkeypatch.setattr(crawler, "get_match_ids", flaky_get_match_ids)
        crawler.seen_match_ids.clear()
        crawler.player_queue.clear()
        crawler.player_queue.append(("bad-puuid", "DIAMOND", "II"))
        crawler.player_queue.append(("good-puuid", "DIAMOND", "II"))

        crawler.crawl()  # must not raise, queue should drain
        assert len(crawler.player_queue) == 0