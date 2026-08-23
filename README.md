# League of Legends Live Win Prediction

**Real-time League of Legends win probability, updated every minute of the game.**

Reads your currently running match through Riot's **Live Client Data API** and estimates your team's probability of winning, updating every 10 seconds. Nothing is injected into the game and no memory is modified—the application only uses the local API already exposed by the League of Legends client.

![Live Dashboard](docs/dashboard.png)

## Overview

A machine learning model trained on **7,956 ranked Solo Queue matches** that estimates the probability of either team winning at every minute of the game using 13 features describing the current game state:

- Kills
- CS
- Champion levels
- Towers
- Inhibitors
- Dragons
- Rift Heralds
- Baron Nashors
- Void Grubs
- Five-minute momentum

The goal is not simply to predict the winner—it is to produce **well-calibrated probabilities**. Whenever the model predicts a 70% chance of winning, the corresponding team should actually win roughly 70% of the time.

| Metric | Value |
|---|---:|
| **ECE** (Expected Calibration Error) | **1.1%** |
| ROC-AUC | 0.836 |
| Dataset | 7,956 Solo Queue matches / 186,635 minute snapshots (EUW) |
| Model | Gradient Boosting (13 features) |

## Requirements

- Python 3.10+
- **For live predictions:** League of Legends installed on the same computer.
- **For training:** a Riot Games API key from https://developer.riotgames.com/.

## Installation

```bash
git clone <repository>
cd league_model
pip install -r requirements.txt
```

If you only want live predictions, you're done—the trained model is already included in `models/`.

To collect data and train your own model:

```bash
cp .env.example .env
```

Then add your Riot API key to the `.env` file.

## Usage

### Live Win Probability

Launch a League of Legends match and start the dashboard:

```bash
streamlit run live_dashboard.py
```

The application waits until it detects an active game and then automatically starts displaying:

- Current win probability
- Probability change over the last five minutes
- Full probability timeline
- Objective scoreboard for both teams

The graph is always shown **from your team's perspective**. The shaded area changes color depending on which team is currently favored, regardless of whether you are playing Blue or Red side.

For a lightweight terminal version:

```bash
python live_predict.py
```

> The live pipeline works in **any game mode** (Ranked, Normal, Custom, Practice Tool, etc.). However, the model was trained exclusively on Ranked Solo Queue matches, so probabilities outside that distribution should not be interpreted literally.

## Training Your Own Model

```bash
python crawler.py
python build_features.py
python train.py
```

The crawler requires **several hours** to collect around 10,000 matches because every game requires downloading both the match metadata and its timeline, while Riot development keys have strict rate limits.

Development keys also **expire every 24 hours**. If you receive a 401 error, generate a new key and restart the crawler—it automatically resumes where it left off.

The target region and ranked tiers can be modified through the constants defined at the beginning of `crawler.py` (`SEED_TIERS`, `PLATFORM`, and `REGION`).

## Tests

The pipeline's core logic (feature extraction, dataset filtering, live-state
parsing, calibration metrics) has a unit test suite that runs with no API key,
no dataset and no running game:

```bash
pip install -r requirements-dev.txt
pytest
```

## Experiments

```bash
python -m experiments.calibrate
python -m experiments.queue_ablation
python -m experiments.train_lstm
```

Included experiments evaluate:

- Whether probability calibration improves performance (it does not)
- The effect of training with game modes other than Solo Queue
- Whether sequential models (LSTM) outperform gradient boosting (they do not)

## Pipeline

```text
Riot API  ──crawler.py──>  matches/ + timelines/
                                 │
                                 v
                        build_features.py
                                 │
                                 v
                           features.csv
                                 │
                                 v
                            train.py
                                 │
                                 v
                    models/model_baseline.joblib

League Client
Live Client Data API
        │
        v
 live_predict.py
        │
        v
 Live win probability every 10 seconds
```

Training is performed exclusively from the **match timeline**, not from Riot's final match summary.

Using the final match statistics would introduce target leakage, since those statistics already contain information about the game's outcome.

Each timeline is converted into one training sample per game minute.

The model intentionally uses **only features available through the Live Client API**. Although the timeline contains additional information such as total gold and experience, these values are unavailable for every player during live matches.

Using them would produce a model that cannot be deployed in real time.

Interestingly, excluding gold only reduces ROC-AUC by approximately **0.002**, since gold is already highly correlated with CS, kills, and objective control.

| File | Purpose |
|---|---|
| `crawler.py` | Downloads match metadata and timelines (resume supported). |
| `build_features.py` | Converts timelines into `features.csv` (one row per minute). |
| `train.py` | Trains, evaluates, and exports the model. |
| `live_predict.py` | Core prediction engine and terminal interface. |
| `live_dashboard.py` | Streamlit dashboard. |
| `experiments/` | Experimental models and ablation studies. |
| `notebooks/eda.ipynb` | Exploratory data analysis. |

All commands should be executed from the repository root.

## Calibration

Evaluation was performed on **37,320 unseen samples**, using a **match-level split** to prevent information leakage between training and testing.

| Predicted | Observed | Samples |
|---:|---:|---:|
| 5.1% | 4.4% | 5,057 |
| 14.8% | 14.7% | 3,666 |
| 25.1% | 28.5% | 3,701 |
| 35.0% | 35.7% | 4,234 |
| 44.9% | 46.4% | 3,880 |
| 55.0% | 54.3% | 3,634 |
| 64.9% | 64.9% | 3,408 |
| 74.9% | 74.3% | 3,111 |
| 85.1% | 84.1% | 3,085 |
| 94.8% | 95.8% | 3,544 |

The model is naturally well calibrated.

Applying **Isotonic Regression** or **Platt Scaling** actually degrades calibration (ECE increases from **1.1%** to approximately **2.2%**), so no post-processing calibration is applied.

## Limitations

- **Designed for Ranked Solo Queue (5v5) only.** ARAM, Arena, Co-op vs AI, and other game modes follow different dynamics, objectives, and maps, making the probabilities unreliable.
- **Rank matters.** The current model was trained on high-elo matches. Lower-ranked games tend to convert advantages less consistently, meaning probabilities become overconfident when deployed outside the training distribution. Retraining with your target rank (`SEED_TIERS`) is recommended.
- **There is an upper performance limit.** A ROC-AUC around **0.84** reflects the inherent uncertainty of League of Legends rather than a limitation of the algorithm. Additional data, hyperparameter tuning, LightGBM, and LSTM models did not significantly improve performance.
- **Development API keys expire every 24 hours.** This only affects data collection. Live prediction relies exclusively on Riot's local client API.

## Acknowledgements

Match data is provided by the **Riot Games API**.

This project is **not endorsed by Riot Games** and does not reflect the views or opinions of Riot Games.
