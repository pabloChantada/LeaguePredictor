# League of Legends Live Win Predictor

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Codecov](https://codecov.io/gh/pabloChantada/LeaguePredictor/branch/main/graph/badge.svg)
[![Tests](https://github.com/pabloChantada/LeaguePredictor/actions/workflows/ci.yaml/badge.svg)](https://github.com/pabloChantada/LeaguePredictor/actions/workflows/ci.yaml)
![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![Render](<https://img.shields.io/badge/Deployed%20on-Render-000000?style=flat&logo=render>)

Real-time League of Legends win probability, updated every minute of the game.

Reads the currently running match through Riot's Live Client Data API and estimates the team's probability of winning. Nothing is injected into the game and no memory is modified; the application only reads the local API already exposed by the League of Legends client.

https://github.com/user-attachments/assets/c748e293-6b4a-4716-9889-78b9d1e2cd54

## Live Demos and Monitoring

| Service        | Link                                                                 | Description                                                      |
| :------------- | :------------------------------------------------------------------- | :--------------------------------------------------------------- |
| Live Dashboard | [Streamlit Cloud](https://chantaclown-leaguepredictor.streamlit.app/) | Interactive web UI with live probability curves and scoreboards. |
| Prediction API | [Render](https://league-of-legends-win-predictor.onrender.com)        | FastAPI service serving the Gradient Boosting model.             |
| Test Coverage  | [Codecov](https://app.codecov.io/gh/pabloChantada/LeaguePredictor)    | CI/CD coverage reports for the core pipeline.                    |
| Error Tracking | [Sentry](https://chantaclown.sentry.io/projects/league_predictor/)    | Real-time crash reporting and performance monitoring.            |

## Overview

A machine learning model trained on 7,956 ranked Solo Queue (from Emerald to Master) matches that estimates the probability of either team winning at every minute of the game using 13 features describing the current game state:

- Kills, CS, Champion levels
- Towers, Inhibitors
- Dragons, Rift Heralds, Baron Nashors, Void Grubs
- Five-minute momentum (deltas in kills, CS, and levels)

The goal is to produce well-calibrated probabilities. Whenever the model predicts a 70% chance of winning, the corresponding team should actually win roughly 70% of the time.

| Metric                           | Value                                                     |
| :------------------------------- | :-------------------------------------------------------- |
| ECE (Expected Calibration Error) | 1.1%                                                      |
| ROC-AUC                          | 0.836 (not a 0.95+ due to game randomness)                |
| Dataset                          | 7,956 Solo Queue matches / 186,635 minute snapshots (EUW) |
| Model                            | Gradient Boosting Classifier                              |

## Pipeline

1. Data Collection: `crawler.py` downloads match metadata and timelines from the Riot API.
2. Feature Engineering: `build_features.py` converts timelines into `features.csv` (one row per minute).
3. Training: `train.py` trains, evaluates, and exports the model to `src/models/baseline_model.joblib`.
4. Live Prediction: `live_predict.py` reads the local client API and queries the FastAPI service every 10 seconds.

Training is performed exclusively from the match timeline to prevent target leakage. The model intentionally uses only features available through the Live Client API.

## Quick Start

### Option 1: Docker (Recommended)

The project includes a fully configured Docker environment. No local Python setup is required.

```bash
git clone https://github.com/pabloChantada/LeaguePredictor.git
cd LeaguePredictor
docker compose up -d --build
```

Once running, open `http://localhost:8501` in your browser to see the live dashboard.

### Option 2: Local Development (with uv)

This project uses `uv` for dependency management.

```bash
git clone https://github.com/pabloChantada/LeaguePredictor.git
cd LeaguePredictor

# Install dependencies and create virtual environment
uv sync

# Run the live dashboard
uv run streamlit run src/serve/live_dashboard.py
```

For a lightweight terminal-only version, run `uv run python src/serve/live_predict.py`.

## Training Your Own Model

If you want to retrain the model on a different region or rank, you will need a Riot Games API key.

1. Configure API Key:
   ```bash
   cp .env.example .env
   # Edit .env and add your API_KEY from https://developer.riotgames.com/
   ```
2. Crawl Data (Note: Dev keys expire every 24h. The crawler supports resuming):
   ```bash
   uv run python src/building/crawler.py
   ```
3. Build Features:
   ```bash
   uv run python src/building/build_features.py
   ```
4. Train and Evaluate:
   ```bash
   uv run python src/building/train.py
   ```

The target region and ranked tiers can be modified via constants in `src/building/crawler.py` (`SEED_TIERS`, `PLATFORM`, `REGION`).

## Testing

The pipeline's core logic (feature extraction, dataset filtering, live-state parsing, and API routing) has a comprehensive unit test suite. It runs with no API key, no dataset, and no running game using extensive mocking.

```bash
# Run tests with coverage report
uv run pytest -q --cov=src.building --cov=src.serve --cov-report=term-missing
```

Coverage reports are automatically uploaded to Codecov via GitHub Actions on every push.

## Limitations

- Designed for Ranked Solo Queue (5v5) only. ARAM, Arena, Co-op vs AI, and other game modes follow different dynamics, making probabilities a bit unreliable.
- Rank matters. The current model was trained on mid-to-high elo matches. Lower-ranked games tend to convert advantages less consistently, meaning probabilities can become overconfident outside the training distribution.
- Upper performance limit. A ROC-AUC around 0.84 reflects the inherent uncertainty of League of Legends. Extensive experiments with LSTMs, LightGBM, and hyperparameter tuning did not yield significant improvements over Gradient Boosting.
- Development API keys expire every 24 hours. This only affects data collection. Live prediction relies exclusively on Riot's local client API.

## Acknowledgements and Disclaimer

Match data is provided by the Riot Games API. This project is not endorsed by Riot Games and does not reflect the views or opinions of Riot Games.
