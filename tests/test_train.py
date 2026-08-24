"""Tests for src/building/train.py."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

import src.building.train as train


FEATURES = train.FEATURES  # config.FEATURES, the live-compatible 13 columns


def _synthetic_dataset(tmp_path, n_matches=24, rows_per_match=4, queue_id=420,
                        tier="DIAMOND", noise_seed=0, filename="features.csv"):
    """A features.csv-shaped file where blue_win is a clean (learnable)
    function of the features, so both models can actually fit something
    and AUC comparisons aren't just noise."""
    rng = np.random.RandomState(noise_seed)
    rows = []
    for m in range(n_matches):
        match_id = f"M{m}"
        # a fixed "match strength" so every minute-row of a match tends the
        # same way -- keeps the label learnable per-match, like real games.
        strength = rng.choice([-1, 1])
        for r in range(rows_per_match):
            minute = 5 + r
            kills_diff = strength * rng.randint(1, 5)
            row = {
                "match_id": match_id, "queue_id": queue_id, "tier": tier,
                "division": "II", "patch": "16.13", "minute": minute,
                "blue_win": 1 if strength > 0 else 0,
                "kills_diff": kills_diff,
                "cs_diff": strength * rng.randint(0, 20),
                "level_diff": strength * rng.randint(0, 3),
                "tower_diff": strength * rng.randint(0, 2),
                "inhib_diff": 0, "dragon_diff": strength * rng.randint(0, 2),
                "herald_diff": 0, "baron_diff": 0,
                "grub_diff": strength * rng.randint(0, 2),
                "kills_diff_d5": kills_diff, "cs_diff_d5": 0, "level_diff_d5": 0,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


class TestLoadDataset:
    def test_happy_path_keeps_matching_queue_and_tier(self, tmp_path):
        path = _synthetic_dataset(tmp_path, queue_id=420, tier="DIAMOND")
        df = train.load_dataset(csv=path, queue_id=420, tiers=("DIAMOND",))
        assert set(df["queue_id"]) == {420}
        assert set(df["tier"]) == {"DIAMOND"}
        assert len(df) > 0

    def test_filters_out_rows_from_a_different_queue(self, tmp_path):
        rng = np.random.RandomState(1)
        rows = []
        for m in range(4):
            for feat_row in range(2):
                rows.append({
                    "match_id": f"S{m}", "queue_id": 440,  # flex, not soloQ
                    "tier": "DIAMOND", "division": "II", "patch": "16.13",
                    "minute": 5, "blue_win": 1,
                    **{f: 0 for f in FEATURES},
                })
        solo_path = _synthetic_dataset(tmp_path, n_matches=4, filename="solo.csv")
        flex_df = pd.DataFrame(rows)
        combined = pd.concat([pd.read_csv(solo_path), flex_df], ignore_index=True)
        combined_path = tmp_path / "combined.csv"
        combined.to_csv(combined_path, index=False)

        df = train.load_dataset(csv=combined_path, queue_id=420, tiers=None)
        assert set(df["queue_id"]) == {420}

    def test_filters_out_tiers_outside_the_target_band(self, tmp_path):
        diamond_path = _synthetic_dataset(tmp_path, n_matches=4, tier="DIAMOND", filename="d.csv")
        iron_path = _synthetic_dataset(tmp_path, n_matches=4, tier="IRON", filename="i.csv")
        combined = pd.concat([pd.read_csv(diamond_path), pd.read_csv(iron_path)], ignore_index=True)
        combined_path = tmp_path / "combined.csv"
        combined.to_csv(combined_path, index=False)

        df = train.load_dataset(csv=combined_path, queue_id=420, tiers=("DIAMOND",))
        assert set(df["tier"]) == {"DIAMOND"}

    def test_tiers_none_disables_the_elo_filter(self, tmp_path):
        diamond_path = _synthetic_dataset(tmp_path, n_matches=4, tier="DIAMOND", filename="d.csv")
        iron_path = _synthetic_dataset(tmp_path, n_matches=4, tier="IRON", filename="i.csv")
        combined = pd.concat([pd.read_csv(diamond_path), pd.read_csv(iron_path)], ignore_index=True)
        combined_path = tmp_path / "combined.csv"
        combined.to_csv(combined_path, index=False)

        df = train.load_dataset(csv=combined_path, queue_id=420, tiers=None)
        assert set(df["tier"]) == {"DIAMOND", "IRON"}

    def test_missing_queue_id_column_raises_system_exit(self, tmp_path):
        df = pd.DataFrame({"match_id": ["M1"], "tier": ["DIAMOND"], "minute": [5],
                            "blue_win": [1], **{f: [0] for f in FEATURES}})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(SystemExit):
            train.load_dataset(csv=path)

    def test_missing_tier_column_raises_system_exit(self, tmp_path):
        df = pd.DataFrame({"match_id": ["M1"], "queue_id": [420], "minute": [5],
                            "blue_win": [1], **{f: [0] for f in FEATURES}})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(SystemExit):
            train.load_dataset(csv=path)

    def test_empty_result_after_filtering_raises_system_exit(self, tmp_path):
        path = _synthetic_dataset(tmp_path, tier="DIAMOND")
        with pytest.raises(SystemExit):
            train.load_dataset(csv=path, queue_id=420, tiers=("CHALLENGER",))

    def test_wrong_queue_id_with_no_matches_raises_system_exit(self, tmp_path):
        path = _synthetic_dataset(tmp_path, queue_id=420)
        with pytest.raises(SystemExit):
            train.load_dataset(csv=path, queue_id=999, tiers=None)


class TestMainTrainsAndSavesTheBestModel:
    def _patch_default_csv(self, monkeypatch, path):
        """main() calls load_dataset(tiers=tiers) with no csv= override, so
        the csv default (bound at import time) must be patched directly."""
        defaults = list(train.load_dataset.__defaults__)
        defaults[0] = path
        monkeypatch.setattr(train.load_dataset, "__defaults__", tuple(defaults))

    def test_saves_a_joblib_bundle_with_model_and_features(self, tmp_path, monkeypatch):
        csv_path = _synthetic_dataset(tmp_path, n_matches=30, rows_per_match=5)
        model_out = tmp_path / "model.joblib"
        self._patch_default_csv(monkeypatch, csv_path)
        monkeypatch.setattr(train.config, "MODEL_OUT", model_out)

        train.main(tiers=None)

        assert model_out.exists()
        bundle = joblib.load(model_out)
        assert set(bundle.keys()) == {"model", "features"}
        assert bundle["features"] == FEATURES
        assert hasattr(bundle["model"], "predict_proba")

    def test_saved_model_predicts_the_learnable_signal_reasonably_well(self, tmp_path, monkeypatch):
        """Sanity check on the whole pipeline: with a clean, learnable
        signal the saved model should do much better than a coin flip."""
        csv_path = _synthetic_dataset(tmp_path, n_matches=40, rows_per_match=5, noise_seed=7)
        model_out = tmp_path / "model.joblib"
        self._patch_default_csv(monkeypatch, csv_path)
        monkeypatch.setattr(train.config, "MODEL_OUT", model_out)

        train.main(tiers=None)

        bundle = joblib.load(model_out)
        df = pd.read_csv(csv_path)
        X = df[FEATURES].values
        y = df["blue_win"].values
        proba = bundle["model"].predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        accuracy = (preds == y).mean()
        assert accuracy > 0.8  # kills_diff sign alone should make this easy


def test_group_shuffle_split_never_mixes_a_match_between_train_and_test():
    """train.py's core safeguard against leakage: GroupShuffleSplit on
    match_id, so rows from the same match never land in both splits."""
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array(["match_1", "match_1", "match_2", "match_2", "match_3", "match_3"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    train_matches = set(groups[train_idx])
    test_matches = set(groups[test_idx])
    assert train_matches.isdisjoint(test_matches)
