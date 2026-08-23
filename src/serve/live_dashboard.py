"""
Live win% dashboard.

    streamlit run live_dashboard.py

Reads the ongoing match (Live Client Data API), asks the deployed prediction
API every POLL_SECONDS, and draws the probability curve as the match
progresses. live_predict.predict_blue_winrate does an HTTP call to the FastAPI service.

Works in any mode (ranked, normal, CUSTOM, practice tool) as long as the
match runs ON THIS PC. Note: the model was trained on high-elo soloQ 5v5.
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import src.serve.live_predict as lp

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
    st.session_state.hist = []      # per-minute states (for the momentum)
    st.session_state.curve = []     # (minute_float, p_blue) for the chart
    st.session_state.last_t = None  # gameTime of the last poll (detects a new match)


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


def main():
    st.title("LoL Live Win Probability")
    st.caption(f"Model served via API | 13 features, AUC 0.836, ECE 1.1% (soloQ)· "
               f"refresh every {lp.POLL_SECONDS}s")

    with st.spinner("Waking up the prediction service (free-tier cold start)..."):
        warm_up_once()

    # Initialize session state on first run, or when a new match is detected.
    if "hist" not in st.session_state:
        reset_match()

    slot = st.empty()
    col = palette()

    while True:
        try:
            # obtain the match data
            data = lp.fetch_live_data()
        except lp.NoGameRunning as ex:
            with slot.container():
                st.info("**No match in progress.**")
                st.caption(f"Detail: {ex}")
            time.sleep(lp.POLL_SECONDS)
            continue

        # The match has changed (new game): reset the session state to start a new curve.
        t = lp.game_time(data)
        if lp.is_new_game(st.session_state.last_t, t):
            reset_match()

        # Update the last_t in session state to the current game time
        st.session_state.last_t = t

        # Obtain the state (diffs) and the feature vector for the model, and store them in session state.
        state = lp.read_state(data)
        # one state per minute: that is what the momentum computation expects
        if not st.session_state.hist or st.session_state.hist[-1]["minute"] != state["minute"]:
            st.session_state.hist.append(state)

        feats = lp.build_features(state, st.session_state.hist)

        try:
            p_blue = lp.predict_blue_winrate(feats)
        except lp.PredictionUnavailable as ex:
            with slot.container():
                st.warning("**Prediction service unavailable right now.** "
                           "Retrying automatically, this can happen on a "
                           "free-tier cold start or a brief network failure.")
                st.caption(f"Detail: {ex}")
            time.sleep(lp.POLL_SECONDS)
            continue

        my_team = lp.active_team(data)
        p_mine = p_blue if my_team == lp.BLUE else 1 - p_blue

        # x = real gameTime, not the whole minute: with a poll every 10s the whole
        # minute stacked ~6 points on the same x and the curve came out stepped.
        st.session_state.curve.append((t / 60, p_blue))

        with slot.container():
            # A single big number
            side = "blue" if my_team == lp.BLUE else "red"
            c1, c2 = st.columns([2, 1])
            # trend: how much YOUR win% has moved in the last 5 min
            ago5 = [q for tt, q in st.session_state.curve if tt <= t / 60 - 5]
            if ago5:
                before = ago5[-1] if my_team == lp.BLUE else 1 - ago5[-1]
                # change in wr in the last 5 min, as a signed percentage point (not relative %)
                delta = f"{(p_mine - before) * 100:+.1f} pts in 5 min"
            else:
                delta = None
            c1.metric(f"Your team ({side})", f"{p_mine:.1%}", delta=delta)
            c2.metric("Time", clock(t))

            g, m = st.columns([3, 1])
            with g:
                if len(st.session_state.curve) > 1:
                    st.pyplot(draw_curve(st.session_state.curve, my_team, col))
                    # A discreet note, not an alarm box: it is a warning about how to
                    # read the number, not a problem to attend to.
                    if state["minute"] > 25:
                        st.caption("Minute >25: long matches are close *precisely* "
                                   "because they last trust the number less.")
            with m:
                draw_scoreboard(lp.read_counters(data), col)

            # table twin: every value on the chart is readable without color
            with st.expander("Feature vector (what the model sees)"):
                st.dataframe(pd.DataFrame([{k: feats[k] for k in lp.FEATURES}]),
                             hide_index=True, use_container_width=True)
                st.dataframe(pd.DataFrame(st.session_state.curve,
                                          columns=["minute", "p_blue"]),
                             hide_index=True, use_container_width=True, height=200)

        time.sleep(lp.POLL_SECONDS)

if __name__ == "__main__":
    main()