"""
Tests for src/building/config.py.

config.py runs its module-level code (load_dotenv, os.makedirs) at import
time, so most of these tests reload the module under a controlled directory
layout instead of asserting against the already-imported singleton.

Note: PROJECT_ROOT = Path(__file__).resolve().parent.parent resolves to the
`src/` directory, not the repo root, because config.py itself lives two
levels down at src/building/config.py. That's intentional and matches the
Docker image (`COPY src ./src`) and MODEL_PATH ("src/models/...") in app.py
-- riot_dataset/ and models/ both end up as siblings of `building/` inside
`src/`, not at the repo root. These tests pin that behaviour down so a
future refactor that moves config.py doesn't silently relocate it.
"""
import importlib
import sys

import pytest


def _reload_config_under(tmp_path, monkeypatch, env=None):
    """Copy config.py's source into tmp_path/src/building/config.py and
    import it fresh, so PROJECT_ROOT resolves under tmp_path instead of the
    real repo.
    """
    import src.building.config as real_config
    src_text = open(real_config.__file__, encoding="utf-8").read()

    fake_pkg_dir = tmp_path / "src" / "building"
    fake_pkg_dir.mkdir(parents=True)
    (fake_pkg_dir / "config.py").write_text(src_text, encoding="utf-8")

    monkeypatch.delenv("API_KEY", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)

    sys.path.insert(0, str(fake_pkg_dir))
    sys.modules.pop("config", None)
    try:
        return importlib.import_module("config")
    finally:
        sys.modules.pop("config", None)
        sys.path.remove(str(fake_pkg_dir))


class TestProjectRoot:
    def test_project_root_resolves_to_the_src_directory(self, tmp_path, monkeypatch):
        """Two levels up from src/building/config.py is src/, not the repo root."""
        fresh = _reload_config_under(tmp_path, monkeypatch)
        assert fresh.PROJECT_ROOT == tmp_path / "src"

    def test_output_dirs_are_created_as_siblings_of_building(self, tmp_path, monkeypatch):
        fresh = _reload_config_under(tmp_path, monkeypatch)
        assert fresh.RAW_DIR.exists()
        assert fresh.TIMELINE_DIR.exists()
        assert fresh.MODEL_DIR.exists()
        assert fresh.RAW_DIR == tmp_path / "src" / "riot_dataset" / "matches"
        assert fresh.MODEL_OUT == tmp_path / "src" / "models" / "baseline_model.joblib"

    def test_no_env_file_and_no_env_var_leaves_api_key_none(self, tmp_path, monkeypatch):
        fresh = _reload_config_under(tmp_path, monkeypatch)
        assert fresh.API_KEY is None

    def test_api_key_is_read_from_env_var(self, tmp_path, monkeypatch):
        fresh = _reload_config_under(tmp_path, monkeypatch, env={"API_KEY": "test-key-123"})
        assert fresh.API_KEY == "test-key-123"

    def test_env_file_at_project_root_is_loaded(self, tmp_path, monkeypatch):
        """load_dotenv(PROJECT_ROOT / '.env') should pick up a real .env file
        placed at PROJECT_ROOT (i.e. under src/, per the resolution above),
        even with no API_KEY already in the environment."""
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / ".env").write_text("API_KEY=from-dotenv-file\n", encoding="utf-8")
        fresh = _reload_config_under(tmp_path, monkeypatch)
        assert fresh.API_KEY == "from-dotenv-file"


class TestFeatureAndTierConstants:
    def test_features_exclude_gold_and_xp(self):
        """Gold/xp aren't available from the Live Client Data API, so they
        must never leak into the live-serving feature set."""
        import src.building.config as config
        assert "gold_diff" not in config.FEATURES
        assert "xp_diff" not in config.FEATURES
        assert config.FEATURES[0] == "minute"
        assert len(config.FEATURES) == 13
        assert config.TARGET == "blue_win"

    def test_features_has_no_duplicates(self):
        import src.building.config as config
        assert len(config.FEATURES) == len(set(config.FEATURES))

    def test_tiers_matches_seed_tiers_keys(self):
        import src.building.config as config
        assert config.TIERS == tuple(config.SEED_TIERS.keys())

    def test_delta_base_columns_are_all_valid_snapshot_keys(self):
        import src.building.config as config
        expected_snapshot_keys = {
            "gold_diff", "xp_diff", "cs_diff", "level_diff", "kills_diff",
            "tower_diff", "inhib_diff", "dragon_diff", "herald_diff",
            "baron_diff", "grub_diff",
        }
        assert set(config.DELTA_BASE).issubset(expected_snapshot_keys)

    def test_rate_limits_stay_under_riot_dev_key_caps(self):
        """Riot dev keys allow 20 req/s and 100 req/120s; config must stay
        strictly under both or crawler.py will get 429s constantly."""
        import src.building.config as config
        assert config.MAX_PER_SECOND < 20
        assert config.MAX_PER_WINDOW < 100
