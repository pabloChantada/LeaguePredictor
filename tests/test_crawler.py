import csv

import pytest

import src.building.crawler as crawler
from src.building.config import *


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