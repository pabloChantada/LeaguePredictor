"""
Live win% dashboard.

    streamlit run live_dashboard.py

Reads the ongoing match (Live Client Data API), asks the deployed prediction
API every POLL_SECONDS, and draws the probability curve as the match
progresses. live_predict.predict_blue_winrate does an HTTP call to the FastAPI service.

Works in any mode (ranked, normal, CUSTOM, practice tool) as long as the
match runs ON THIS PC. Note: the model was trained on high-elo soloQ 5v5.
"""
import copy
import os
from pathlib import Path
import sys
import time
import json
import glob
import random
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# i know it's a bit shit but it's only here and not really important so
# for Streamlit Cloud
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
try:
    # Try relative import first (works in tests)
    from . import live_predict as lp
except ImportError:
    # Fall back to absolute import (works in Streamlit)
    import live_predict as lp

st.set_page_config(page_title="LoL Win Probability", layout="wide")

# Readable label for each lp.COUNTERS counter, in scoreboard order.
LABELS = {
    "kills": "Kills", "cs": "CS", "level": "Level (sum)",
    "towers": "Towers", "inhibs": "Inhibitors", "dragons": "Dragons",
    "heralds": "Heralds", "barons": "Barons", "grubs": "Void Grubs",
}

LIGHT = {"blue": "#2a78d6", "red": "#e34948", "ink": "#0b0b0b",
         "muted": "#898781", "grid": "#e1e0d9", "base": "#c3c2b7"}
DARK = {"blue": "#3987e5", "red": "#e66767", "ink": "#ffffff",
        "muted": "#898781", "grid": "#2c2c2a", "base": "#383835"}


def palette():
    theme = getattr(st.context, "theme", None)
    return DARK if theme is not None and theme.type == "dark" else LIGHT


@st.cache_resource
def warm_up_once():
    """Ping the prediction API once per session so a cold start on Render/Fly
    happens here, with a visible spinner."""
    lp.warm_up_api()
    return True


def reset_match():
    """Leave the session state as at startup: each match starts clean."""
    st.session_state.hist = []                  # per-minute states (for the momentum)
    st.session_state.curve = []                 # (minute_float, p_blue) for the chart
    st.session_state.last_t = None              # gameTime of the last poll (detects a new match)
    st.session_state.last_mode = "none"         # Track current mode to detect transitions
    st.session_state.current_demo_minute = 0    # Reset demo counter


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def draw_curve(curve, my_team, col):
    """Win% of YOUR team over the course of the match. The fill uses the TEAM colors."""
    df = pd.DataFrame(curve, columns=["t", "p_blue"])
    # win prob
    p = (df.p_blue if my_team == lp.BLUE else 1 - df.p_blue) * 100

    mine = col["blue"] if my_team == lp.BLUE else col["red"]
    theirs = col["red"] if my_team == lp.BLUE else col["blue"]

    # plot the curve
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_alpha(0)              # adapts to Streamlit's background
    ax.set_facecolor("none")

    # divergent fill about 50%: the color says whose the advantage is
    ax.fill_between(df.t, 50, p, where=(p >= 50), interpolate=True, alpha=.18, color=mine, lw=0)
    ax.fill_between(df.t, 50, p, where=(p < 50), interpolate=True, alpha=.18, color=theirs, lw=0)
    ax.plot(df.t, p, lw=2, color=col["ink"], solid_capstyle="round")
    ax.axhline(50, color=col["base"], lw=1)   # solid hairline, never dashed

    # direct label only on the final point (never a number on every point)
    t_end, p_end = df.t.iloc[-1], p.iloc[-1]
    ax.scatter([t_end], [p_end], s=90, color=mine, zorder=3,
               edgecolor=col.get("ring", "none"), linewidth=2)
    ax.annotate(f"{p_end:.0f}%", (t_end, p_end), textcoords="offset points",
                xytext=(10, 0), va="center", fontsize=12, fontweight="bold",
                color=col["ink"])

    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.margins(x=.08)
    ax.set_xlabel("minute", color=col["muted"], fontsize=9)
    ax.set_ylabel("your win prob. (%)", color=col["muted"], fontsize=9)
    ax.grid(axis="y", color=col["grid"], lw=1, alpha=.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=col["muted"], labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(col["base"])
    fig.tight_layout()
    return fig


def draw_scoreboard(counters, col):
    """Blue vs red scoreboard with each side's absolutes."""
    rows = []
    # obtained as absolutes not diffs, and rendered using html
    for k in lp.COUNTERS:
        b, r = counters[k][lp.BLUE], counters[k][lp.RED]
        rows.append(f"""<tr>
            <td style="text-align:right;font-weight:600;width:38%">{b}</td>
            <td style="text-align:center;color:{col['muted']};font-size:.85em">{LABELS[k]}</td>
            <td style="text-align:left;font-weight:600;width:38%">{r}</td></tr>""")
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse">
      <tr>
        <th style="text-align:right;color:{col['blue']}">BLUE</th>
        <th></th>
        <th style="text-align:left;color:{col['red']}">RED</th>
      </tr>
      {''.join(rows)}
    </table>""", unsafe_allow_html=True)


def load_demo_match(folder_path="src/demo_data"):
    """Reads a RANDOM Riot Timeline JSON and precalculates the state minute by minute."""
    files = glob.glob(f"{folder_path}/*.json")
    if not files:
        return []
    
    # random game for demo
    random_file = random.choice(files)
    
    with open(random_file, 'r') as f:
        data = json.load(f)
    
    frames = data.get("info", {}).get("frames", [])
    
    # Team accumulators using lp constants
    c = {k: {lp.BLUE: 0, lp.RED: 0} for k in lp.COUNTERS}
    history = []
    
    for minute, frame in enumerate(frames):
        # Process events (Towers, Kills, Objectives)
        for e in frame.get("events", []):
            evt_type = e.get("type")
            killer_id = e.get("killerId", 0)
            team = lp.BLUE if 1 <= killer_id <= 5 else lp.RED
            
            if evt_type == "CHAMPION_KILL" and killer_id > 0:
                c["kills"][team] += 1
            elif evt_type == "BUILDING_KILL":
                # If affected teamId is 200 (Red), Blue gets the point
                b_team = lp.BLUE if e.get("teamId") == 200 else lp.RED
                b_type = e.get("buildingType")
                if b_type == "TOWER_BUILDING": c["towers"][b_team] += 1
                elif b_type == "INHIBITOR_BUILDING": c["inhibs"][b_team] += 1
            elif evt_type == "ELITE_MONSTER_KILL":
                m_type = e.get("monsterType")
                if m_type == "DRAGON": c["dragons"][team] += 1
                elif m_type == "RIFTHERALD": c["heralds"][team] += 1
                elif m_type == "BARON_NASHOR": c["barons"][team] += 1
                elif m_type == "HORDE": c["grubs"][team] += 1

        # Process participants (CS and Level)
        c["cs"] = {lp.BLUE: 0, lp.RED: 0}
        c["level"] = {lp.BLUE: 0, lp.RED: 0}
        
        for pid_str, pf in frame.get("participantFrames", {}).items():
            pid = int(pid_str)
            team = lp.BLUE if 1 <= pid <= 5 else lp.RED
            c["cs"][team] += pf.get("minionsKilled", 0) + pf.get("jungleMinionsKilled", 0)
            c["level"][team] += pf.get("level", 0)
        
        # Create the 'state' with diffs (Blue - Red) just like read_state
        state = {"minute": minute}
        for key in lp.COUNTERS:
            state[lp.DIFF_NAME[key]] = c[key][lp.BLUE] - c[key][lp.RED]

        history.append((state, copy.deepcopy(c)))  # Store both state and absolute counters for rendering
        
    return history


def main():
    st.title("LoL Live Win Probability")

    st.caption(f"Model served via API | 13 features, AUC 0.836, ECE 1.1% (soloQ) | "
               f"Refresh every {lp.POLL_SECONDS}s | Download the repo at [github.com/pabloChantada/LeaguePredictor](https://github.com/pabloChantada/LeaguePredictor)")

    with st.spinner("Waking up the prediction service (free-tier cold start)..."):
        warm_up_once()

    if "hist" not in st.session_state:
        reset_match()

    slot = st.empty()
    col = palette()

    DEMO_MODE = os.getenv("DEMO_MODE", "False").lower() == "true"
    
    # Initialize demo history and minute counter in session state for robust persistence
    if "demo_history" not in st.session_state and DEMO_MODE:
        st.session_state.demo_history = load_demo_match(folder_path="src/demo_data")
    if "current_demo_minute" not in st.session_state:
        st.session_state.current_demo_minute = 0

    while True:
        is_demo_active = False
        try:
            # Try to fetch real live data
            data = lp.fetch_live_data()
            t = lp.game_time(data)
            
            # clean dashboard
            if st.session_state.get("last_mode") in ("demo", "none"):
                reset_match()
            
            if lp.is_new_game(st.session_state.last_t, t):
                reset_match()
            st.session_state.last_t = t
            
            state = lp.read_state(data)
            my_team = lp.active_team(data)
            ui_data = lp.read_counters(data)
            st.session_state.last_mode = "real"
            
        except lp.NoGameRunning as ex:
            # If no game is running, check if DEMO_MODE is enabled
            if DEMO_MODE and st.session_state.get("demo_history"):
                is_demo_active = True
                
                # clean dashboard
                if st.session_state.get("last_mode") in ("real", "none"):
                    reset_match()
                
                # Load a new demo when the current one is finished
                if st.session_state.current_demo_minute >= len(st.session_state.demo_history):
                    st.session_state.current_demo_minute = 0
                    reset_match()
                    st.session_state.demo_history = load_demo_match(folder_path="src/demo_data")

                # In case the demo history is empty, we cannot proceed
                if st.session_state.demo_history:
                    state, ui_data = st.session_state.demo_history[st.session_state.current_demo_minute]
                    t = state["minute"] * 60
                    
                    my_team = lp.BLUE 
                    
                    st.session_state.last_mode = "demo"
                    st.session_state.current_demo_minute += 1
            else:
                with slot.container():
                    st.info("**No match in progress.**")
                    st.caption(f"Detail: {ex}")
                st.session_state.last_mode = "none"
                time.sleep(lp.POLL_SECONDS)
                continue

        # Momentum computation
        if not st.session_state.hist or st.session_state.hist[-1]["minute"] != state["minute"]:
            st.session_state.hist.append(state)

        feats = lp.build_features(state, st.session_state.hist)

        # API Request
        try:
            p_blue = lp.predict_blue_winrate(feats)
        except lp.PredictionUnavailable as ex:
            with slot.container():
                st.warning("**Prediction service unavailable right now.**")
                st.caption(f"Detail: {ex}")
            time.sleep(lp.POLL_SECONDS)
            continue

        p_mine = p_blue if my_team == lp.BLUE else 1 - p_blue
        st.session_state.curve.append((t / 60, p_blue))

        # UI Rendering
        with slot.container():
            if is_demo_active:
                st.warning("**DEMO MODE ACTIVE**: Replaying a saved match.")
                
            side = "blue" if my_team == lp.BLUE else "red"
            c1, c2 = st.columns([2, 1])
            
            ago5 = [q for tt, q in st.session_state.curve if tt <= t / 60 - 5]
            if ago5:
                before = ago5[-1] if my_team == lp.BLUE else 1 - ago5[-1]
                delta = f"{(p_mine - before) * 100:+.1f} pts in 5 min"
            else:
                delta = None
                
            c1.metric(f"Your team ({side})", f"{p_mine:.1%}", delta=delta)
            c2.metric("Time", clock(t))

            g, m = st.columns([3, 1])
            with g:
                if len(st.session_state.curve) > 1:
                    fig = draw_curve(st.session_state.curve, my_team, col)
                    st.pyplot(fig)
                    plt.close(fig)
            with m:
                draw_scoreboard(ui_data, col)

            with st.expander("Feature vector (what the model sees)"):
                # DEBUG INFO: Confirms visually that the state has reset properly
                st.caption(f"Mode: **{st.session_state.get('last_mode', 'none').upper()}** | Minute: **{state.get('minute', 'N/A')}**")
                st.dataframe(pd.DataFrame([{k: feats[k] for k in lp.FEATURES}]),
                            hide_index=True, width="stretch")

        # In demo mode we can speed up the poll 
        time.sleep(3 if is_demo_active else lp.POLL_SECONDS)

if __name__ == "__main__":
    main()