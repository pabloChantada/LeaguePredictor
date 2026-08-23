import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("API_KEY")


# --- Riot API ------
PLATFORM = "euw1.api.riotgames.com"
REGION = "europe.api.riotgames.com"
QUEUE = "RANKED_SOLO_5x5"   # league-v4 wants the string form
QUEUE_ID = 420              # match-v5 wants the int form

# Dev key limits: 20 req/s, 100 req/120s. Stay under both to avoid errors.
MAX_PER_SECOND = 18
MAX_PER_WINDOW = 95
WINDOW_SECONDS = 120

# --- Crawl target ----------
TARGET_MATCHES = 10000
MATCHES_PER_PLAYER = 20
SAVE_EVERY = 25

# --- Elo band -----------------------------------------------------------------
# Crawled AND trained on the same band: elo can't be a model feature since Live
# Client Data doesn't expose the 10 players' ranks. GM/Challenger are skipped on
# purpose (target is mid-high ladder).
SEED_TIERS = {
    "EMERALD": ("I", "II", "III", "IV"),
    "DIAMOND": ("I", "II", "III", "IV"),
    "MASTER": None,   # no divisions
}
TIERS = tuple(SEED_TIERS.keys())    # used by train.load_dataset; set to None there to disable
SEED_PAGES_PER_DIVISION = 8         # league-v4 gives ~205 players/page
SEED_MAX_PER_TIER = 3000            # caps Master (~10k in one call) from dominating

# --- Paths ---------------
OUT_DIR = PROJECT_ROOT / "riot_dataset"
RAW_DIR = OUT_DIR / "matches"
TIMELINE_DIR = OUT_DIR / "timelines"
RANKS_CSV = OUT_DIR / "match_rank.csv"
FEATURES_CSV = OUT_DIR / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_OUT = MODEL_DIR / "baseline_model.joblib"

for _dir in (RAW_DIR, TIMELINE_DIR, MODEL_DIR):
    os.makedirs(_dir, exist_ok=True)

# --- Feature engineering  ---------------
MINUTE_START = 5   # no sense using the first few minutes
MINUTE_STEP = 1
DELTA_WINDOW = 5    # window for momentum features, enough for a tf, respawn, objective cycle
DELTA_BASE = ["gold_diff", "xp_diff", "kills_diff", "cs_diff", "level_diff"]

# --- Model features (train.py) --------------
# Live-compatible set: everything here is obtainable from the Live Client Data
# API. Gold/xp are excluded on purpose (not available live); dropping them
# cost only -0.002 AUC since kills/cs/objectives already capture that signal.
FEATURES = [
    "minute",
    "kills_diff", "cs_diff", "level_diff",
    "tower_diff", "inhib_diff", "dragon_diff", "herald_diff", "baron_diff", "grub_diff",
    "kills_diff_d5", "cs_diff_d5", "level_diff_d5",
]
TARGET = "blue_win"