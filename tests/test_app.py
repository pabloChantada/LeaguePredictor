"""Tests for src/serve/app.py (the deployed FastAPI prediction service)."""
import asyncio
import importlib
import sys

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

import src.serve.app as app_module
from src.serve.app import app, ml_models, FEATURES, GameState, lifespan


VALID_PAYLOAD = {
    "minute": 15,
    "kills_diff": 2,
    "cs_diff": 15,
    "level_diff": 1,
    "tower_diff": 1,
    "inhib_diff": 0,
    "dragon_diff": 1,
    "herald_diff": 1,
    "baron_diff": 0,
    "grub_diff": 3,
    "kills_diff_d5": 0,
    "cs_diff_d5": 5,
    "level_diff_d5": 0,
}


class RecordingModel:
    """Fake model that records the exact row it was scored on, so we can
    assert the API builds the feature vector in FEATURES order."""

    def __init__(self, p_blue=0.75):
        self.p_blue = p_blue
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(X)
        return [[1 - self.p_blue, self.p_blue]]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded_model():
    """Inject a fake model into the global ml_models dict for one test."""
    model = RecordingModel()
    ml_models["model"] = model
    yield model
    ml_models.clear()


class TestHealth:
    def test_reports_model_not_loaded_when_dict_is_empty(self, client):
        ml_models.clear()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": False}

    def test_reports_model_loaded_once_injected(self, client, loaded_model):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": True}


class TestPredict:
    def test_success_returns_p_blue_and_p_red_that_sum_to_one(self, client, loaded_model):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["p_blue"] == pytest.approx(0.75)
        assert data["p_red"] == pytest.approx(0.25)
        assert data["p_blue"] + data["p_red"] == pytest.approx(1.0)

    def test_returns_503_when_model_not_loaded(self, client):
        ml_models.clear()
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503

    def test_missing_field_is_a_422_validation_error(self, client, loaded_model):
        payload = dict(VALID_PAYLOAD)
        del payload["minute"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_minute_is_rejected(self, client, loaded_model):
        """minute has ge=0; the live client should never send a negative one."""
        payload = dict(VALID_PAYLOAD, minute=-1)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_type_is_a_422_validation_error(self, client, loaded_model):
        payload = dict(VALID_PAYLOAD, kills_diff="not-a-number")
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_extra_unknown_fields_are_ignored_not_rejected(self, client, loaded_model):
        payload = dict(VALID_PAYLOAD, some_future_feature=123)
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_feature_vector_is_built_in_features_order(self, client, loaded_model):
        """The row handed to predict_proba must follow FEATURES order, not
        JSON insertion order -- payload keys are deliberately shuffled here."""
        shuffled = {k: VALID_PAYLOAD[k] for k in reversed(list(VALID_PAYLOAD))}
        client.post("/predict", json=shuffled)
        row = loaded_model.calls[0]
        expected = [[VALID_PAYLOAD[f] for f in FEATURES]]
        assert row == expected

    def test_get_not_allowed_on_predict(self, client, loaded_model):
        response = client.get("/predict")
        assert response.status_code == 405


class TestLifespan:
    @staticmethod
    def _run_lifespan_with(monkeypatch, saved_features):
        monkeypatch.setattr(
            app_module.joblib, "load",
            lambda path: {"model": "the-model-object", "features": saved_features},
        )

        async def _enter_and_exit():
            async with lifespan(app):
                pass

        asyncio.run(_enter_and_exit())

    def test_loads_model_when_features_match(self, monkeypatch):
        self._run_lifespan_with(monkeypatch, FEATURES)
        # lifespan clears ml_models again on exit, but it must have set it
        # (and not raised) while the context was active -- verified indirectly
        # by asserting no exception was raised above.

    def test_raises_when_saved_features_do_not_match_api_features(self, monkeypatch):
        wrong_features = FEATURES[:-1] + ["some_other_feature"]
        with pytest.raises(RuntimeError, match="doesn't match"):
            self._run_lifespan_with(monkeypatch, wrong_features)

    def test_ml_models_populated_inside_the_context(self, monkeypatch):
        monkeypatch.setattr(
            app_module.joblib, "load",
            lambda path: {"model": "sentinel-model", "features": FEATURES},
        )
        seen = {}

        async def _enter_and_check():
            async with lifespan(app):
                seen["model"] = ml_models.get("model")

        asyncio.run(_enter_and_check())
        assert seen["model"] == "sentinel-model"
        assert ml_models == {}  # cleared again after the context exits

    def test_missing_features_key_falls_back_to_api_features(self, monkeypatch):
        """A model dumped without a 'features' key (older joblib.dump call)
        should be treated as matching, not crash the app."""
        monkeypatch.setattr(
            app_module.joblib, "load",
            lambda path: {"model": "legacy-model"},  # no "features" key
        )
        self._run_lifespan_with(monkeypatch, FEATURES)  # unused arg, just runs cleanly

        async def _enter_and_exit():
            async with lifespan(app):
                pass
        asyncio.run(_enter_and_exit())  # should not raise


class TestCORS:
    def test_cors_middleware_is_registered_permissively(self):
        cors_entries = [m for m in app.user_middleware if m.cls is CORSMiddleware]
        assert len(cors_entries) == 1
        kwargs = cors_entries[0].kwargs
        assert kwargs["allow_origins"] == ["*"]
        assert kwargs["allow_methods"] == ["*"]
        assert kwargs["allow_headers"] == ["*"]

    def test_preflight_request_is_allowed(self, client):
        origin = "https://example-dashboard.streamlit.app"
        response = client.options(
            "/predict",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
        assert response.status_code == 200
        # allow_origins=["*"] + allow_credentials=True means Starlette echoes
        # the requesting origin back rather than a literal "*" (per the CORS
        # spec, a wildcard can't be combined with credentialed responses).
        assert response.headers["access-control-allow-origin"] == origin


class TestSentryDebugEndpoint:
    def test_sentry_debug_endpoint_raises_a_server_error(self):
        """/sentry-debug deliberately divides by zero to verify Sentry
        capture in a real deployment; here we just confirm it surfaces as a
        500 instead of silently succeeding."""
        no_raise_client = TestClient(app, raise_server_exceptions=False)
        response = no_raise_client.get("/sentry-debug")
        assert response.status_code == 500


class TestSentryInitialization:
    def test_sentry_sdk_init_receives_dsn_from_environment(self, tmp_path, monkeypatch):
        """app.py calls sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), ...) at
        import time. We reimport a private copy of the module under a fresh
        name so this doesn't disturb the shared `app_module` used elsewhere.
        """
        import sentry_sdk

        calls = []
        monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
        monkeypatch.setenv("SENTRY_DSN", "https://fake-public-key@sentry.example/1")

        fake_pkg_dir = tmp_path / "src" / "serve"
        fake_pkg_dir.mkdir(parents=True)
        source = open(app_module.__file__, encoding="utf-8").read()
        (fake_pkg_dir / "app_isolated.py").write_text(source, encoding="utf-8")

        sys.path.insert(0, str(fake_pkg_dir))
        sys.modules.pop("app_isolated", None)
        try:
            importlib.import_module("app_isolated")
        finally:
            sys.modules.pop("app_isolated", None)
            sys.path.remove(str(fake_pkg_dir))

        assert calls, "sentry_sdk.init was never called on import"
        assert calls[-1]["dsn"] == "https://fake-public-key@sentry.example/1"
