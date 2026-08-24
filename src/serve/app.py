"""
FastAPI service for the LeaguePredictor gradient boosting model.

Deployed standalone, decoupled from src/serve/ and src/building/: this
process only ever sees a feature vector already computed by the client and
POSTed to it. It never calls the Riot Live Client Data API itself. 
"""
from contextlib import asynccontextmanager
import sentry_sdk
import joblib
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Plain constants here, not importing src/building/config.py
FEATURES = [
    "minute",
    "kills_diff", "cs_diff", "level_diff",
    "tower_diff", "inhib_diff", "dragon_diff", "herald_diff", "baron_diff", "grub_diff",
    "kills_diff_d5", "cs_diff_d5", "level_diff_d5",
]
MODEL_PATH = "src/models/baseline_model.joblib"

ml_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load(MODEL_PATH)
    saved_features = model.get("features", FEATURES)
    if saved_features != FEATURES:
        raise RuntimeError(
            "FEATURES in main.py doesn't match what the model was trained on.\n"
            f"  model: {saved_features}\n  api:   {FEATURES}"
        )
    ml_models["model"] = model["model"]
    yield
    # Cleanup after the app shuts down
    ml_models.clear()


sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    send_default_pii=True,
    traces_sample_rate=1.0,
)


app = FastAPI(
    title="League of Legends Win Predictor",
    description="Predicts blue-side win probability from live in-game features.",
    version="1.0.0",
    lifespan=lifespan, # Use the lifespan context manager to load the model at startup and clean up at shutdown
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambia "*" por la URL de tu Streamlit Cloud
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request and response validation
class GameState(BaseModel):
    """One feature vector, matching train.FEATURES exactly."""
    minute: int = Field(..., ge=0)
    kills_diff: int
    cs_diff: int
    level_diff: int
    tower_diff: int
    inhib_diff: int
    dragon_diff: int
    herald_diff: int
    baron_diff: int
    grub_diff: int
    kills_diff_d5: int
    cs_diff_d5: int
    level_diff_d5: int


class PredictionResponse(BaseModel):
    p_blue: float
    p_red: float

@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0
    
# Health endpoint for live deployment
@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "model_loaded": "model" in ml_models}

@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
async def get_prediction(state: GameState):
    model = ml_models.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Build the feature vector in the same order as FEATURES
    row = [[getattr(state, f) for f in FEATURES]]
    # Give the generated vector to the model for prediction [0] is the first row, [1] is the probability of the positive class (blue win)
    p_blue = float(model.predict_proba(row)[0][1])
    return PredictionResponse(p_blue=p_blue, p_red=1 - p_blue)