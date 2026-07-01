"""
WFC Analytics Platform
Professional women's football performance intelligence system.
Built on StatsBomb open data: WSL 2023/24, Frauen Bundesliga 2023/24, Liga F 2023/24.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from mplsoccer import Pitch, VerticalPitch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
import tempfile, os
import io, base64

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WFC Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0a0e1a",
    "surface":  "#111827",
    "surface2": "#1f2937",
    "surface3": "#374151",
    "accent":   "#00d4aa",
    "accent2":  "#f59e0b",
    "danger":   "#ef4444",
    "success":  "#22c55e",
    "text":     "#f9fafb",
    "muted":    "#9ca3af",
    "home":     "#00d4aa",
    "away":     "#f59e0b",
}
PITCH_BG   = "#0d1117"
PITCH_LINE = "#2d4a3e"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body {{ font-size: 17px !important; }}
  .stApp, .stApp * {{ font-family: 'Inter', sans-serif !important; }}
  .stApp {{ background-color: {COLORS['bg']} !important; }}

  /* Force font size on ALL streamlit elements */
  .stMarkdown, .stMarkdown p, .stMarkdown li,
  .stText, .element-container,
  div[data-testid="stText"], div[data-testid="stMarkdown"] {{
    font-size: 16px !important;
    color: {COLORS['text']} !important;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {COLORS['surface']} !important;
    border-right: 1px solid {COLORS['surface3']};
  }}
  section[data-testid="stSidebar"] *,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] p {{
    color: {COLORS['text']} !important;
    font-size: 15px !important;
  }}

  /* Form labels — the main issue */
  label, .stSelectbox label, .stMultiSelect label,
  .stSlider label, .stRadio label, .stToggle label,
  .stCheckbox label, [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] p {{
    font-size: 16px !important;
    font-weight: 500 !important;
    color: {COLORS['text']} !important;
  }}

  /* Selectbox / multiselect text */
  .stSelectbox > div[data-baseweb] *,
  .stMultiSelect > div[data-baseweb] *,
  [data-baseweb="select"] *, [data-baseweb="popover"] * {{
    font-size: 15px !important;
    color: {COLORS['text']} !important;
    background-color: {COLORS['surface2']} !important;
  }}

  /* Slider ticks and values */
  .stSlider [data-testid="stTickBar"] span,
  .stSlider [data-testid="stSlider"] span {{
    font-size: 14px !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['surface']};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"],
  .stTabs [data-baseweb="tab"] p,
  .stTabs [data-baseweb="tab"] span {{
    font-size: 15px !important;
    font-weight: 500;
    color: {COLORS['muted']} !important;
    border-radius: 8px;
    padding: 10px 24px;
  }}
  .stTabs [aria-selected="true"],
  .stTabs [aria-selected="true"] p,
  .stTabs [aria-selected="true"] span {{
    background: {COLORS['accent']} !important;
    color: {COLORS['bg']} !important;
    font-weight: 700 !important;
  }}

  /* Buttons */
  .stButton > button {{
    background: {COLORS['accent']} !important;
    color: {COLORS['bg']} !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 15px !important;
  }}
  .stButton > button:hover {{ background: #00b894 !important; }}

  /* Home button override */
  .home-btn .stButton > button {{
    background: {COLORS['surface2']} !important;
    color: {COLORS['text']} !important;
    border: 1px solid {COLORS['surface3']} !important;
  }}
  .home-btn .stButton > button:hover {{
    border-color: {COLORS['accent']} !important;
  }}

  /* DataFrames */
  div[data-testid="stDataFrame"] td,
  div[data-testid="stDataFrame"] th,
  div[data-testid="stDataFrame"] span {{
    font-size: 15px !important;
  }}

  /* Alert/info boxes */
  .stAlert, .stAlert p, .stAlert span {{ font-size: 15px !important; }}

  /* Caption */
  .stCaption, .stCaption p {{ font-size: 14px !important; color: {COLORS['muted']} !important; }}

  /* Radio */
  .stRadio > div label, .stRadio > div label p {{
    font-size: 15px !important;
  }}

  /* Toggle */
  .stToggle label, .stToggle label p {{ font-size: 15px !important; }}

  /* Cards */
  .card {{ background:{COLORS['surface']}; border:1px solid {COLORS['surface2']}; border-radius:12px; padding:20px 24px; margin-bottom:16px; }}
  .card-accent {{ border-left:4px solid {COLORS['accent']}; }}
  .metric-card {{ background:{COLORS['surface2']}; border-radius:10px; padding:16px 20px; text-align:center; }}
  .metric-value {{ font-size:32px !important; font-weight:700; color:{COLORS['accent']}; line-height:1.1; }}
  .metric-label {{ font-size:13px !important; font-weight:500; color:{COLORS['muted']}; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }}
  .section-header {{ font-size:14px !important; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:{COLORS['accent']}; border-bottom:1px solid {COLORS['surface3']}; padding-bottom:8px; margin-bottom:16px; }}
  .page-title {{ font-size:28px !important; font-weight:700; color:{COLORS['text']}; }}
  .page-subtitle {{ font-size:16px !important; color:{COLORS['muted']}; margin-top:2px; }}

  .badge {{ display:inline-block; padding:3px 12px; border-radius:20px; font-size:12px !important; font-weight:600; text-transform:uppercase; }}
  .badge-wsl {{ background:#1d4ed8; color:white; }}
  .badge-bl  {{ background:#7c3aed; color:white; }}
  .badge-lf  {{ background:#dc2626; color:white; }}

  hr {{ border-color:{COLORS['surface3']}; margin:20px 0; }}

  .landing-card {{ background:{COLORS['surface']}; border:2px solid {COLORS['surface3']}; border-radius:16px; padding:48px 36px; text-align:center; }}
  .landing-icon {{ font-size:56px; margin-bottom:16px; }}
  .landing-title {{ font-size:24px !important; font-weight:700; color:{COLORS['text']}; margin-bottom:10px; }}
  .landing-desc {{ font-size:16px !important; color:{COLORS['muted']}; line-height:1.7; }}
</style>
""", unsafe_allow_html=True)

# ── Competition config ────────────────────────────────────────────────────────

COMP_SLUGS = {
    "WSL":               "WSL",
    "Frauen Bundesliga": "Frauen_Bundesliga",
    "Liga F":            "Liga_F",
}

ALL_COMPETITIONS = list(COMP_SLUGS.keys())

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_matches():
    try:
        return pd.read_csv("./data/matches.csv", parse_dates=["match_date"])
    except FileNotFoundError:
        st.error("**matches.csv not found.** Run `python data_pipeline.py` first.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_competition_data(comp_name: str):
    """Load events and player data for one competition. Cached per competition."""
    slug = COMP_SLUGS.get(comp_name, comp_name.replace(" ", "_"))
    ev_path = f"./data/events_{slug}.csv"
    pl_path = f"./data/players_{slug}.csv"
    try:
        events  = pd.read_csv(ev_path)
        players = pd.read_csv(pl_path, parse_dates=["match_date"])
        return events, players
    except FileNotFoundError as e:
        st.error(f"**Data file missing: {e}** — run `python data_pipeline.py` first.")
        st.stop()


def load_selected_competitions(selected_comps: list):
    """Load and concatenate events + players for a list of competitions."""
    all_events, all_players = [], []
    for comp in selected_comps:
        ev, pl = load_competition_data(comp)
        all_events.append(ev)
        all_players.append(pl)
    events  = pd.concat(all_events,  ignore_index=True) if all_events  else pd.DataFrame()
    players = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame()
    return events, players


def safe_load():
    """Load matches (always) and return; competitions loaded on demand."""
    return load_matches()


# ── Position profiles ─────────────────────────────────────────────────────────

POSITION_GROUPS = {
    "Goalkeeper": ["Goalkeeper"],
    "Defender":   ["Center Back","Left Back","Right Back","Left Center Back",
                   "Right Center Back","Left Wing Back","Right Wing Back"],
    "Midfielder": ["Center Defensive Midfield","Left Defensive Midfield",
                   "Right Defensive Midfield","Center Midfield",
                   "Left Center Midfield","Right Center Midfield",
                   "Left Midfield","Right Midfield"],
    "Forward":    ["Center Attacking Midfield","Left Attacking Midfield",
                   "Right Attacking Midfield","Left Wing","Right Wing",
                   "Center Forward","Left Center Forward","Right Center Forward"],
}

RADAR_AXES = {
    "Goalkeeper": [
        ("save_pct","Save %"),("shots_faced_p90","Shots Faced p90"),
        ("sweeper_actions_p90","Sweeper p90"),("pass_acc","Pass Acc %"),
        ("carries_p90","Carries p90"),("gk_punches_p90","Punches p90"),
    ],
    "Defender": [
        ("pass_acc","Pass Acc %"),("pressures_p90","Pressures p90"),
        ("tackles_p90","Tackles p90"),("duel_win_pct","Duel Win %"),
        ("interceptions_p90","Interceptions p90"),("progressive_passes_p90","Prog Passes p90"),
    ],
    "Midfielder": [
        ("pass_acc","Pass Acc %"),("key_passes_p90","Key Passes p90"),
        ("pressures_p90","Pressures p90"),("duel_win_pct","Duel Win %"),
        ("progressive_carries_p90","Prog Carries p90"),("total_actions_p90","Actions p90"),
    ],
    "Forward": [
        ("xg_p90","xG p90"),("shots_p90","Shots p90"),
        ("shot_acc","Shot Acc %"),("dribble_pct","Dribble Succ %"),
        ("progressive_carries_p90","Prog Carries p90"),("pressures_p90","Pressures p90"),
    ],
}

def get_position_group(position):
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return "Midfielder"


# ── Helpers ───────────────────────────────────────────────────────────────────

COMP_BADGE = {"WSL":"badge-wsl","Frauen Bundesliga":"badge-bl","Liga F":"badge-lf"}

def comp_badge(name):
    cls   = COMP_BADGE.get(name, "badge-wsl")
    short = {"WSL":"WSL","Frauen Bundesliga":"BL","Liga F":"LF"}.get(name, name)
    return f'<span class="badge {cls}">{short}</span>'

def metric_card(value, label):
    return f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'

def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def plotly_dark_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["bg"],
        font=dict(family="Inter", color=COLORS["text"]),
        height=height, margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        legend=dict(bgcolor=COLORS["surface2"], bordercolor=COLORS["surface3"], borderwidth=1),
        xaxis=dict(gridcolor=COLORS["surface2"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["surface2"], zeroline=False),
    )
    return fig

METRIC_LABELS = {
    "minutes":"Minutes","passes":"Passes","pass_completions":"Pass Completions",
    "pass_acc":"Pass Accuracy %","progressive_passes":"Progressive Passes",
    "final_third_passes":"Final Third Passes","pen_area_passes":"Penalty Area Passes",
    "crosses":"Crosses","through_balls":"Through Balls","key_passes":"Key Passes",
    "avg_pass_length":"Avg Pass Length","shots":"Shots","shots_on_target":"Shots on Target",
    "shot_acc":"Shot Accuracy %","goals":"Goals","xg":"xG","xg_per_shot":"xG per Shot",
    "headed_shots":"Headed Shots","first_time_shots":"First Time Shots",
    "duels":"Duels","tackles":"Tackles","aerials":"Aerials","duels_won":"Duels Won",
    "duel_win_pct":"Duel Win %","tackle_win_pct":"Tackle Win %",
    "pressures":"Pressures","interceptions":"Interceptions","clearances":"Clearances",
    "blocks":"Blocks","ball_recoveries":"Ball Recoveries","fouls_committed":"Fouls Committed",
    "fouls_won":"Fouls Won","carries":"Carries","carry_distance":"Carry Distance",
    "progressive_carries":"Progressive Carries","dribbles":"Dribbles",
    "dribbles_success":"Dribbles Completed","dribble_pct":"Dribble Success %",
    "dispossessed":"Dispossessed","total_actions":"Total Actions",
    "under_pressure":"Actions Under Pressure",
    "save_pct":"Save %","saves":"Saves","shots_faced":"Shots Faced",
}
P90_LABELS = {f"{k}_p90": f"{v} p90" for k, v in METRIC_LABELS.items()}
ALL_LABELS  = {**METRIC_LABELS, **P90_LABELS}

def fmt_col(col):
    return ALL_LABELS.get(col, col.replace("_"," ").title())


# ── Pitch visualisation ───────────────────────────────────────────────────────

EVENT_COLORS = {
    "Pass":          {"Made":COLORS["accent"], "Not Made":COLORS["danger"]},
    "Shot":          {"Goal":COLORS["success"],"Saved":COLORS["accent2"],
                      "Off T":COLORS["muted"], "Blocked":COLORS["danger"],"Wayward":COLORS["danger"]},
    "Carry":         {"Made":COLORS["accent"], "default":COLORS["accent"]},
    "Pressure":      {"Made":COLORS["accent2"],"default":COLORS["accent2"]},
    "Duel":          {"Made":COLORS["accent"], "Not Made":COLORS["danger"]},
    "Interception":  {"Made":COLORS["success"],"default":COLORS["success"]},
    "Ball Recovery": {"Made":COLORS["accent"], "default":COLORS["accent"]},
    "Clearance":     {"Made":COLORS["accent2"],"default":COLORS["accent2"]},
    "Block":         {"Made":COLORS["success"],"default":COLORS["success"]},
    "Dribble":       {"Made":COLORS["accent"], "Not Made":COLORS["danger"]},
    "Foul Committed":{"Made":COLORS["danger"], "default":COLORS["danger"]},
    "Foul Won":      {"Made":COLORS["success"],"default":COLORS["success"]},
}
ARROW_EVENTS = {"Pass","Carry"}


def draw_pitch_figure(ev_df, event_type, selected_outcomes, title="",
                      home_team="", away_team="", split_teams=False):
    """
    Dark-theme pitch with events. Returns base64 PNG.
    StatsBomb data: BOTH teams attack left → right (toward x=120). No flip needed.
    When split_teams=True: colour by team (home=teal, away=amber).
    When split_teams=False: colour by outcome.
    Attack direction arrow added top-right.
    """
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG,
                  line_color=PITCH_LINE, linewidth=1.5,
                  goal_type="box", corner_arcs=True)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor(PITCH_BG)
    ax.set_facecolor(PITCH_BG)

    # Attack direction arrow — both teams attack left → right
    ax.annotate("", xy=(118, 5), xytext=(98, 5),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"],
                                lw=1.5, mutation_scale=14))
    ax.text(108, 2.5, "Attack →", ha="center", va="top",
            fontsize=7, color=COLORS["muted"], style="italic")

    color_map = EVENT_COLORS.get(event_type, {})
    legend_patches = []

    if split_teams and "team" in ev_df.columns:
        # Colour by team — home=teal, away=amber
        team_color_map = {}
        if home_team: team_color_map[home_team] = COLORS["home"]
        if away_team: team_color_map[away_team] = COLORS["away"]
        teams_in_data = ev_df["team"].dropna().unique()
        colors_cycle  = [COLORS["home"], COLORS["away"], COLORS["success"], COLORS["accent2"]]
        for i, t in enumerate(teams_in_data):
            if t not in team_color_map:
                team_color_map[t] = colors_cycle[i % len(colors_cycle)]

        for team, color in team_color_map.items():
            subset = ev_df[ev_df["team"] == team]
            if subset.empty: continue
            if event_type in ARROW_EVENTS and "end_x" in subset.columns:
                valid = subset.dropna(subset=["end_x","end_y"])
                if not valid.empty:
                    pitch.arrows(valid["x"], valid["y"], valid["end_x"], valid["end_y"],
                                 ax=ax, color=color, alpha=0.65,
                                 width=1.5, headwidth=6, headlength=4)
            elif event_type == "Shot":
                sizes = (subset["shot_statsbomb_xg"].fillna(0.05) * 1500 + 80).clip(80, 800)
                ax.scatter(subset["x"], subset["y"], c=color, s=sizes,
                           alpha=0.85, zorder=5, linewidths=1, edgecolors="white")
            else:
                ax.scatter(subset["x"], subset["y"], c=color, s=60,
                           alpha=0.85, zorder=5, linewidths=0.8, edgecolors="white")
            legend_patches.append(mpatches.Patch(color=color, label=team))
    else:
        # Colour by outcome
        for outcome in selected_outcomes:
            subset = ev_df[ev_df["outcome"] == outcome]
            if subset.empty: continue
            color = color_map.get(outcome, color_map.get("Made", COLORS["accent"]))
            if event_type in ARROW_EVENTS and "end_x" in subset.columns:
                valid = subset.dropna(subset=["end_x","end_y"])
                if not valid.empty:
                    pitch.arrows(valid["x"], valid["y"], valid["end_x"], valid["end_y"],
                                 ax=ax, color=color, alpha=0.65,
                                 width=1.5, headwidth=6, headlength=4)
            elif event_type == "Shot":
                sizes = (subset["shot_statsbomb_xg"].fillna(0.05) * 1500 + 80).clip(80, 800)
                ax.scatter(subset["x"], subset["y"], c=color, s=sizes,
                           alpha=0.85, zorder=5, linewidths=1, edgecolors="white")
            else:
                ax.scatter(subset["x"], subset["y"], c=color, s=60,
                           alpha=0.85, zorder=5, linewidths=0.8, edgecolors="white")
            legend_patches.append(mpatches.Patch(color=color, label=outcome))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper left", framealpha=0.3,
                  facecolor=COLORS["surface"], edgecolor=COLORS["surface3"],
                  labelcolor=COLORS["text"], fontsize=9)
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=13, fontweight="bold", pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=PITCH_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def draw_shot_map_half_pitch(shots_df, selected_team):
    """
    Vertical half-pitch shot map for ONE team.
    StatsBomb data: all shots already near x=100-120 (attacking end).
    Both teams attack toward x=120 — no flip needed.
    Returns base64 PNG.
    """
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        pitch_color=PITCH_BG,
        line_color=PITCH_LINE,
        linewidth=1.5,
        goal_type="box",
        half=True,
    )
    fig, ax = pitch.draw(figsize=(8, 10))
    fig.patch.set_facecolor(PITCH_BG)
    ax.set_facecolor(PITCH_BG)

    outcome_color = {
        "Goal":          COLORS["success"],
        "Saved":         COLORS["accent2"],
        "Saved to Post": COLORS["accent2"],
        "Blocked":       COLORS["danger"],
        "Wayward":       COLORS["muted"],
        "Off T":         COLORS["muted"],
        "Post":          "#fb7185",
    }

    legend_patches = []
    seen_outcomes  = set()
    for _, row in shots_df.iterrows():
        outcome = row.get("outcome","Off T")
        color   = outcome_color.get(outcome, COLORS["muted"])
        xg      = float(row.get("shot_statsbomb_xg", 0.05) or 0.05)
        size    = np.clip(xg * 1500 + 80, 80, 800)
        marker  = "*" if outcome == "Goal" else "o"
        # VerticalPitch: x=pitch_x, y=pitch_y (pitch.scatter swaps for vertical)
        pitch.scatter(row["x"], row["y"], ax=ax,
                      s=size, c=color, marker=marker,
                      alpha=0.88, zorder=5,
                      linewidths=1.2, edgecolors="white")
        if outcome not in seen_outcomes:
            legend_patches.append(mpatches.Patch(color=color, label=outcome))
            seen_outcomes.add(outcome)

    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower left", framealpha=0.3,
                  facecolor=COLORS["surface"], edgecolor=COLORS["surface3"],
                  labelcolor=COLORS["text"], fontsize=9)

    ax.set_title(f"{selected_team} — Shot Map",
                 color=COLORS["text"], fontsize=12, fontweight="bold", pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=PITCH_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=PITCH_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Passing network ───────────────────────────────────────────────────────────

def draw_passing_network(ev_df, player_df, team_name, home_team, min_passes=2):
    """
    Passing network for starting XI only.
    Home = teal, Away = amber — consistent with all other tabs.
    """
    node_color = COLORS["home"] if team_name == home_team else COLORS["away"]

    passes = ev_df[(ev_df["type"] == "Pass") & (ev_df["team"] == team_name)].copy()
    passes = passes.dropna(subset=["player","x","y"])
    if passes.empty:
        return None

    # Starting XI: top 11 players by involvement in first 45 mins
    early     = passes[passes["minute"] <= 45]
    starters  = early.groupby("player").size().nlargest(11).index.tolist()
    passes    = passes[passes["player"].isin(starters)]
    if passes.empty:
        return None

    avg_pos     = passes.groupby("player")[["x","y"]].mean()
    pass_counts = passes.groupby("player").size().rename("n_passes")
    avg_pos     = avg_pos.join(pass_counts)

    # Build edges from consecutive passes
    made = passes[passes["outcome"] == "Made"].copy().sort_values("minute")
    made["recipient"] = made["player"].shift(-1)
    made  = made[(made["player"] != made["recipient"]) &
                 made["recipient"].isin(starters)].dropna(subset=["recipient"])
    edges = made.groupby(["player","recipient"]).size().reset_index(name="count")
    edges = edges[edges["count"] >= min_passes]

    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG,
                  line_color=PITCH_LINE, linewidth=1.5,
                  goal_type="box", corner_arcs=True)
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor(PITCH_BG)
    ax.set_facecolor(PITCH_BG)

    max_count = edges["count"].max() if not edges.empty else 1
    for _, row in edges.iterrows():
        if row["player"] in avg_pos.index and row["recipient"] in avg_pos.index:
            x1, y1 = avg_pos.loc[row["player"],   ["x","y"]]
            x2, y2 = avg_pos.loc[row["recipient"], ["x","y"]]
            lw = np.clip(row["count"] / max_count * 5, 0.5, 5)
            ax.plot([x1,x2],[y1,y2], color=node_color, alpha=0.35,
                    linewidth=lw, zorder=2)

    max_passes = avg_pos["n_passes"].max() if not avg_pos.empty else 1
    for player, row in avg_pos.iterrows():
        size = np.clip(row.get("n_passes",5) / max_passes * 700 + 120, 120, 820)
        ax.scatter(row["x"], row["y"], s=size, color=node_color,
                   edgecolors="white", linewidths=1.5, zorder=5, alpha=0.92)
        ax.text(row["x"], row["y"] - 4.5, player.split()[-1][:10],
                ha="center", va="top", fontsize=7,
                color=COLORS["text"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["surface"],
                          ec="none", alpha=0.75))

    ax.set_title(f"{team_name} — Starting XI Network",
                 color=node_color, fontsize=12, fontweight="bold", pad=10)
    return fig_to_b64(fig)


# ── Radar chart ───────────────────────────────────────────────────────────────

def build_radar(players_df, player_names, pos_group):
    axes_info = RADAR_AXES.get(pos_group, RADAR_AXES["Midfielder"])
    cols      = [a[0] for a in axes_info]
    labels    = [a[1] for a in axes_info]

    pos_positions = POSITION_GROUPS.get(pos_group, [])
    benchmark = (
        players_df[players_df["position"].isin(pos_positions)]
        .groupby("player")[cols].mean().reset_index()
    )

    # Palette as (hex, rgba_fill) tuples — plotly doesn't accept 8-digit hex
    palette = [
        (COLORS["accent"],  "rgba(0,212,170,0.15)"),
        (COLORS["accent2"], "rgba(245,158,11,0.15)"),
        (COLORS["danger"],  "rgba(239,68,68,0.15)"),
        (COLORS["success"], "rgba(34,197,94,0.15)"),
    ]
    fig = go.Figure()

    for i, pname in enumerate(player_names):
        raw_row = players_df[players_df["player"] == pname][cols].mean()
        vals = []
        for col in cols:
            v     = raw_row.get(col, 0) if not pd.isna(raw_row.get(col, np.nan)) else 0
            bvals = benchmark[col].dropna() if col in benchmark.columns else pd.Series([0])
            pct   = (bvals < v).sum() / len(bvals) * 100 if len(bvals) else 50
            vals.append(round(pct, 1))

        vals_c = vals + [vals[0]]
        labs_c = labels + [labels[0]]
        line_color, fill_color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=labs_c, name=pname,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["surface"],
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(size=9, color=COLORS["muted"]),
                            gridcolor=COLORS["surface3"], linecolor=COLORS["surface3"]),
            angularaxis=dict(tickfont=dict(size=10, color=COLORS["text"]),
                             gridcolor=COLORS["surface3"], linecolor=COLORS["surface3"]),
        ),
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]), showlegend=True,
        legend=dict(bgcolor=COLORS["surface2"], bordercolor=COLORS["surface3"], x=1.05),
        height=450, margin=dict(l=60, r=120, t=30, b=30),
    )
    return fig


# ── Shot motion replay ────────────────────────────────────────────────────────

# ── Squad aggregation helper ──────────────────────────────────────────────────

PCT_METRICS = {"pass_acc","shot_acc","duel_win_pct","tackle_win_pct",
               "dribble_pct","save_pct",}

def aggregate_squad(pdata):
    """Safely aggregate player stats across matches — avoids pandas named-agg conflicts."""
    base_metrics = [
        "minutes","passes","pass_acc","progressive_passes","key_passes","crosses",
        "shots","shots_on_target","shot_acc","goals","xg","xg_per_shot",
        "duels","tackles","duels_won","duel_win_pct","tackle_win_pct",
        "pressures","interceptions","clearances","blocks","ball_recoveries",
        "fouls_committed","fouls_won","carries","carry_distance",
        "progressive_carries","dribbles","dribbles_success","dribble_pct",
        "dispossessed","miscontrols","total_actions","under_pressure",
    ]
    available = [m for m in base_metrics if m in pdata.columns]

    grp = pdata.groupby(["team","player","position","nationality"])
    apps = grp["match_id"].nunique().rename("matches_played")

    agg_parts = []
    for m in available:
        func = "mean" if m in PCT_METRICS else "sum"
        agg_parts.append(grp[m].agg(func).rename(m))

    result = pd.concat([apps] + agg_parts, axis=1).reset_index()
    return result, available



# ── Player pitch maps ─────────────────────────────────────────────────────────

def draw_player_heatmap(ev_df, player_name, title=""):
    from scipy.stats import gaussian_kde
    pe = ev_df[ev_df["player"] == player_name].dropna(subset=["x","y"])
    if len(pe) < 5:
        return None
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG,
                  line_color=PITCH_LINE, linewidth=1.5, goal_type="box", corner_arcs=True)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor(PITCH_BG)
    ax.set_facecolor(PITCH_BG)
    pitch.kdeplot(pe["x"], pe["y"], ax=ax, fill=True,
                  cmap="YlOrRd", alpha=0.65, levels=100, thresh=0.02)
    ax.annotate("", xy=(118,4), xytext=(100,4),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=1.2, mutation_scale=12))
    ax.text(109, 1.8, "Attack →", ha="center", va="top", fontsize=9,
            color=COLORS["muted"], style="italic")
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=13, fontweight="bold", pad=10)
    return fig_to_b64(fig)


def draw_player_actions_pitch(ev_df, player_name, event_types, title=""):
    pe = ev_df[ev_df["player"] == player_name].copy()
    if event_types:
        pe = pe[pe["type"].isin(event_types)]
    pe = pe.dropna(subset=["x","y"])
    if pe.empty:
        return None
    pitch = Pitch(pitch_type="statsbomb", pitch_color=PITCH_BG,
                  line_color=PITCH_LINE, linewidth=1.5, goal_type="box", corner_arcs=True)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor(PITCH_BG)
    ax.set_facecolor(PITCH_BG)
    color_map = {
        "Pass": COLORS["accent"], "Carry": COLORS["accent2"],
        "Pressure": COLORS["danger"], "Shot": COLORS["success"],
        "Duel": "#a78bfa", "Interception": COLORS["success"],
        "Ball Recovery": COLORS["accent"], "Clearance": COLORS["accent2"],
    }
    legend_patches = []
    for etype in (event_types or pe["type"].unique()):
        sub = pe[pe["type"] == etype]
        if sub.empty: continue
        col = color_map.get(etype, COLORS["muted"])
        if etype in {"Pass","Carry"} and "end_x" in sub.columns:
            valid = sub.dropna(subset=["end_x","end_y"])
            if not valid.empty:
                pitch.arrows(valid["x"], valid["y"], valid["end_x"], valid["end_y"],
                             ax=ax, color=col, alpha=0.5, width=1.2, headwidth=5, headlength=3)
        else:
            ax.scatter(sub["x"], sub["y"], c=col, s=50, alpha=0.7,
                       zorder=5, edgecolors="white", linewidths=0.5)
        legend_patches.append(mpatches.Patch(color=col, label=etype))
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper left", framealpha=0.3,
                  facecolor=COLORS["surface"], edgecolor=COLORS["surface3"],
                  labelcolor=COLORS["text"], fontsize=10)
    ax.annotate("", xy=(118,4), xytext=(100,4),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=1.2, mutation_scale=12))
    ax.text(109, 1.8, "Attack →", ha="center", va="top", fontsize=9,
            color=COLORS["muted"], style="italic")
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=13, fontweight="bold", pad=10)
    return fig_to_b64(fig)


# ── Landing ───────────────────────────────────────────────────────────────────

def page_landing():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;padding:40px 0 20px">
      <div style="font-size:46px;font-weight:800;color:{COLORS['text']}">⚽ WFC Analytics</div>
      <div style="font-size:18px;color:{COLORS['muted']};margin-top:10px">
        Women's Football Performance Intelligence · 2023/24
      </div>
      <div style="margin-top:14px">
        {comp_badge('WSL')} &nbsp; {comp_badge('Frauen Bundesliga')} &nbsp; {comp_badge('Liga F')}
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, _, col2, _, col3 = st.columns([4,1,4,1,4])
    with col1:
        st.markdown(f"""
        <div class="landing-card">
          <div class="landing-icon">🔍</div>
          <div class="landing-title">Open Match</div>
          <div class="landing-desc">Deep-dive into a single match.<br>
          Match overview · Player stats · Pitch analysis · Shot map</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Open Match", use_container_width=True, key="btn_match"):
            st.session_state["view"] = "match_selector"
            st.rerun()
    with col2:
        st.markdown(f"""
        <div class="landing-card">
          <div class="landing-icon">📊</div>
          <div class="landing-title">Season Analysis</div>
          <div class="landing-desc">Analyse a team across a season or range.<br>
          Results · Squad · Involvement · Player Intelligence · Pitch</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊  Season Analysis", use_container_width=True, key="btn_season"):
            st.session_state["view"] = "season_selector"
            st.rerun()
    with col3:
        st.markdown(f"""
        <div class="landing-card">
          <div class="landing-icon">👤</div>
          <div class="landing-title">About</div>
          <div class="landing-desc">About the analyst behind this platform.<br>
          Background · Projects · Contact</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("👤  About", use_container_width=True, key="btn_about"):
            st.session_state["view"] = "about"
            st.rerun()


# ── Match selector ────────────────────────────────────────────────────────────

def page_match_selector(matches, players, events):
    with st.sidebar:
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
        comp_opts = ["All"] + sorted(matches["competition"].unique().tolist())
        sel_comp  = st.selectbox("Competition", comp_opts, key="ms_comp")
        filtered  = matches.copy()
        if sel_comp != "All":
            filtered = filtered[filtered["competition"] == sel_comp]
        team_opts = ["All"] + sorted(
            pd.concat([filtered["home_team"], filtered["away_team"]]).unique().tolist())
        sel_team = st.selectbox("Team", team_opts, key="ms_team")
        if sel_team != "All":
            filtered = filtered[(filtered["home_team"]==sel_team)|(filtered["away_team"]==sel_team)]
        if filtered["match_week"].nunique() > 1:
            mw_min, mw_max = int(filtered["match_week"].min()), int(filtered["match_week"].max())
            mw_range = st.slider("Match Week", mw_min, mw_max, (mw_min, mw_max), key="ms_mw")
            filtered = filtered[filtered["match_week"].between(*mw_range)]

    st.markdown('<div class="page-title">🔍 Open Match</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select a match to analyse in detail</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    display_df = filtered.sort_values(["competition","match_date","match_week"]).copy()
    display_df["Score"] = display_df["home_score"].astype(str)+" – "+display_df["away_score"].astype(str)
    display_df["Match"] = display_df["home_team"]+" vs "+display_df["away_team"]
    display_df["Date"]  = pd.to_datetime(display_df["match_date"]).dt.strftime("%d %b %Y")
    display_df["MW"]    = display_df["match_week"].astype(int)
    show_df = display_df[["competition","Date","MW","Match","Score"]].rename(columns={"competition":"Competition"})

    st.info("💡 Select exactly one match then press Open Match.")
    sel = st.dataframe(show_df, use_container_width=True, hide_index=True,
                       on_select="rerun", selection_mode="single-row", height=500, key="ms_table")
    idx = sel.selection.rows if sel.selection else []
    sel_ids = display_df.iloc[idx]["match_id"].tolist() if idx else []

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Open Match →", disabled=len(sel_ids)!=1, key="ms_open"):
        st.session_state["view"]          = "match_detail"
        st.session_state["sel_match_ids"] = sel_ids
        st.rerun()


# ── Season selector ───────────────────────────────────────────────────────────

def page_season_selector(matches, players, events):
    # ── KEY FIX: use separate widget keys that don't conflict with session_state ──
    # Never write to session_state keys that are also used as widget keys

    with st.sidebar:
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

    st.markdown('<div class="page-title">📊 Season Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select a team and match range to analyse</div>',
                unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([1,1,1])
    with col_f1:
        comp_opts = sorted(matches["competition"].unique().tolist())
        # Use unique widget keys with "ss_" prefix — never stored in session_state manually
        sel_comp = st.selectbox("Competition", comp_opts, key="ss_comp")
    with col_f2:
        comp_matches = matches[matches["competition"] == sel_comp]
        all_teams = sorted(pd.concat([comp_matches["home_team"],
                                       comp_matches["away_team"]]).unique().tolist())
        sel_team = st.selectbox("Team", all_teams, key="ss_team")
    with col_f3:
        team_matches = comp_matches[
            (comp_matches["home_team"]==sel_team)|(comp_matches["away_team"]==sel_team)
        ].sort_values("match_week")
        if team_matches.empty:
            st.warning("No matches found for this team.")
            return
        mw_min = int(team_matches["match_week"].min())
        mw_max = int(team_matches["match_week"].max())
        if mw_min == mw_max:
            mw_range = (mw_min, mw_max)
            st.info(f"Only match week {mw_min} available.")
        else:
            mw_range = st.slider("Match Week Range", mw_min, mw_max,
                                 (mw_min, mw_max), key="ss_mw")

    filtered = team_matches[team_matches["match_week"].between(*mw_range)]
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(f"{sel_team} — {sel_comp} · Weeks {mw_range[0]}–{mw_range[1]}")

    # Build preview table
    disp = filtered.copy()
    disp["Opponent"] = disp.apply(lambda r: r["away_team"] if r["home_team"]==sel_team else r["home_team"], axis=1)
    disp["H/A"]   = disp.apply(lambda r: "H" if r["home_team"]==sel_team else "A", axis=1)
    disp["Score"] = disp["home_score"].astype(str)+" – "+disp["away_score"].astype(str)
    disp["Result"]= disp.apply(lambda r:
        "W" if (r["home_team"]==sel_team and r["home_score"]>r["away_score"]) or
               (r["away_team"]==sel_team and r["away_score"]>r["home_score"]) else
        "L" if (r["home_team"]==sel_team and r["home_score"]<r["away_score"]) or
               (r["away_team"]==sel_team and r["away_score"]<r["home_score"]) else "D", axis=1)
    disp["Date"] = pd.to_datetime(disp["match_date"]).dt.strftime("%d %b %Y")

    st.dataframe(disp[["match_week","Date","H/A","Opponent","Score","Result"]].rename(
        columns={"match_week":"MW"}), use_container_width=True, hide_index=True, height=350)

    # KPI row
    wins  = (disp["Result"]=="W").sum()
    draws = (disp["Result"]=="D").sum()
    losses= (disp["Result"]=="L").sum()
    gf = disp.apply(lambda r: r["home_score"] if r["home_team"]==sel_team else r["away_score"], axis=1).sum()
    ga = disp.apply(lambda r: r["away_score"] if r["home_team"]==sel_team else r["home_score"], axis=1).sum()
    kc = st.columns(4)
    kc[0].markdown(metric_card(len(filtered),"Matches"), unsafe_allow_html=True)
    kc[1].markdown(metric_card(f"{wins}W {draws}D {losses}L","Record"), unsafe_allow_html=True)
    kc[2].markdown(metric_card(f"{int(gf)}–{int(ga)}","Goals"), unsafe_allow_html=True)
    kc[3].markdown(metric_card(f"W{mw_range[0]}–W{mw_range[1]}","Range"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📊 Analyse →", use_container_width=False, key="ss_go"):
        # Store values using storage keys (different from widget keys)
        st.session_state["sv_team"]   = sel_team
        st.session_state["sv_comp"]   = sel_comp
        st.session_state["sv_mw"]     = mw_range
        st.session_state["sv_mids"]   = filtered["match_id"].tolist()
        st.session_state["view"]      = "season_view"
        st.rerun()


# ── Season view ───────────────────────────────────────────────────────────────

def page_season_view(matches, players, events):
    # Read from storage keys (sv_*), not widget keys
    sel_team = st.session_state.get("sv_team", "")
    sel_comp = st.session_state.get("sv_comp", "")
    mw_range = st.session_state.get("sv_mw", (1,1))
    sel_ids  = st.session_state.get("sv_mids", [])

    if not sel_ids:
        st.error("No matches selected. Go back and select a team.")
        if st.button("← Back"):
            st.session_state["view"] = "season_selector"
            st.rerun()
        return

    pdata = players[players["match_id"].isin(sel_ids)].copy()
    mdata = matches[matches["match_id"].isin(sel_ids)].copy()
    edata = events[events["match_id"].isin(sel_ids)].copy()

    with st.sidebar:
        st.markdown('<div class="section-header">Season View</div>', unsafe_allow_html=True)
        min_mins = st.slider("Min minutes per match", 10, 90, 30, key="sv_minmins")

    pdata_f = pdata[pdata["minutes"] >= min_mins].copy()

    st.markdown(f"""
    <div class="page-title">📊 {sel_team}</div>
    <div class="page-subtitle">{comp_badge(sel_comp)} · Weeks {mw_range[0]}–{mw_range[1]} · {len(sel_ids)} matches</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Match Results", "👥 Squad Overview",
        "📈 Player Involvement", "🧠 Player Intelligence", "🏟 Team Pitch"
    ])

    # ── Tab 1: Match Results ──────────────────────────────────────────────────
    with tab1:
        section_header(f"{sel_team} — MATCH BY MATCH")
        match_rows = []
        for _, match in mdata.iterrows():
            mid      = match["match_id"]
            is_home  = (match["home_team"] == sel_team)
            opponent = match["away_team"] if is_home else match["home_team"]
            gf = match["home_score"] if is_home else match["away_score"]
            ga = match["away_score"] if is_home else match["home_score"]
            result = "W" if gf>ga else ("L" if gf<ga else "D")
            tp = pdata[(pdata["match_id"]==mid)&(pdata["team"]==sel_team)]
            op = pdata[(pdata["match_id"]==mid)&(pdata["team"]==opponent)]
            def ts(df,col,pct=False):
                if col not in df.columns or df.empty: return 0
                return round(df[col].mean(),1) if pct else round(df[col].sum(),1)
            match_rows.append({
                "MW":int(match["match_week"]),
                "Date":pd.to_datetime(match["match_date"]).strftime("%d %b"),
                "H/A":"H" if is_home else "A","Opponent":opponent,
                "Score":f"{int(gf)}–{int(ga)}","Result":result,
                "xG":ts(tp,"xg"),"Shots":ts(tp,"shots"),"Shots OT":ts(tp,"shots_on_target"),
                "Passes":ts(tp,"passes"),"Pass%":ts(tp,"pass_acc",True),
                "Prog Pass":ts(tp,"progressive_passes"),"Key Pass":ts(tp,"key_passes"),
                "Pressures":ts(tp,"pressures"),"Duels Won":ts(tp,"duels_won"),
                "Prog Carry":ts(tp,"progressive_carries"),
                "opp_xG":ts(op,"xg"),"opp_Shots":ts(op,"shots"),
                "opp_Passes":ts(op,"passes"),"opp_Pass%":ts(op,"pass_acc",True),
                "opp_Pressures":ts(op,"pressures"),
            })
        if not match_rows:
            st.info("No match data available.")
        else:
            mr = pd.DataFrame(match_rows)
            num_cols = [c for c in mr.columns if c not in ["MW","Date","H/A","Opponent","Score","Result"]]
            avg_row  = {c: round(mr[c].mean(),1) for c in num_cols}
            avg_row.update({"MW":"—","Date":"AVG","H/A":"—","Opponent":"All","Score":"—","Result":"—"})
            show_mr  = pd.concat([mr, pd.DataFrame([avg_row])], ignore_index=True)
            tcols    = ["MW","Date","H/A","Opponent","Score","Result","xG","Shots","Shots OT",
                        "Passes","Pass%","Prog Pass","Key Pass","Pressures","Duels Won","Prog Carry"]
            st.dataframe(show_mr[tcols], use_container_width=True, hide_index=True, height=420)

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("TEAM AVG vs OPPONENTS AVG")
            cmp_list = ["xG","Shots","Passes","Pass%","Pressures"]
            avg_t = mr[cmp_list].mean()
            avg_o = mr[["opp_xG","opp_Shots","opp_Passes","opp_Pass%","opp_Pressures"]].mean()
            avg_o.index = cmp_list
            kc = st.columns(len(cmp_list))
            for col_w, metric in zip(kc, cmp_list):
                tv = round(avg_t[metric],1); ov = round(avg_o[metric],1)
                diff = tv-ov
                dc = COLORS["success"] if diff>=0 else COLORS["danger"]
                col_w.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:13px;color:{COLORS['muted']};margin-bottom:8px;font-weight:600">{metric}</div>
                  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
                    <div style="text-align:center">
                      <div style="font-size:24px;font-weight:700;color:{COLORS['home']}">{tv}</div>
                      <div style="font-size:12px;color:{COLORS['muted']}">Team</div>
                    </div>
                    <div style="font-size:18px;color:{dc};font-weight:700">{"+" if diff>=0 else ""}{round(diff,1)}</div>
                    <div style="text-align:center">
                      <div style="font-size:24px;font-weight:700;color:{COLORS['away']}">{ov}</div>
                      <div style="font-size:12px;color:{COLORS['muted']}">Opp</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            section_header("PERFORMANCE BY 15-MINUTE PERIOD")
            p_metric = st.selectbox("Metric",
                ["Total Actions","Passes","Pressures","Shots","Duels","Interceptions"],
                key="sv_period_metric")
            type_map = {"Total Actions":["Pass","Carry","Pressure","Duel"],"Passes":["Pass"],
                        "Pressures":["Pressure"],"Shots":["Shot"],"Duels":["Duel"],
                        "Interceptions":["Interception"]}
            plabels = ["0–15","15–30","30–45","45–60","60–75","75–90+"]
            pbins   = [-1,15,30,45,60,75,200]
            fig_p   = go.Figure()
            for grp, lbl, col in [
                (edata[edata["team"]==sel_team], sel_team, COLORS["home"]),
                (edata[edata["team"]!=sel_team], "Opponents", COLORS["away"])
            ]:
                sub = grp[grp["type"].isin(type_map[p_metric])].copy()
                if not sub.empty and "minute" in sub.columns:
                    sub["pb"] = pd.cut(sub["minute"],bins=pbins,labels=plabels)
                    cnt = sub.groupby(["match_id","pb"],observed=True).size().reset_index(name="n")
                    avg = cnt.groupby("pb",observed=True)["n"].mean().reset_index()
                    fig_p.add_trace(go.Bar(x=avg["pb"],y=avg["n"].round(1),name=lbl,marker_color=col))
            plotly_dark_layout(fig_p, height=320)
            fig_p.update_layout(barmode="group")
            st.plotly_chart(fig_p, use_container_width=True)

    # ── Tab 2: Squad Overview ─────────────────────────────────────────────────
    with tab2:
        section_header("SQUAD PERFORMANCE TABLE")
        p90 = st.toggle("Per 90 minutes", value=True, key="sv_p90")
        spd = pdata_f[pdata_f["team"]==sel_team].copy()
        agg, avail = aggregate_squad(spd)
        if p90:
            for m in avail:
                if m in agg.columns and m not in PCT_METRICS and m!="minutes":
                    agg[m] = (agg[m]/agg["minutes"].replace(0,np.nan)*90).round(2)
        rename_m = {m:fmt_col(m) for m in avail if m in agg.columns}
        st.dataframe(agg.rename(columns=rename_m).round(2),
                     use_container_width=True, hide_index=True, height=520,
                     column_config={"player":st.column_config.TextColumn("Player"),
                                    "position":st.column_config.TextColumn("Position"),
                                    "matches_played":st.column_config.NumberColumn("Apps")})
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("TOP PERFORMERS")
        raw_top, _ = aggregate_squad(spd)
        top_list = [("xg","xG","🎯"),("pressures","Pressures","⚡"),
                    ("key_passes","Key Passes","🎁"),("duels_won","Duels Won","💪"),
                    ("progressive_carries","Prog Carries","🏃"),("interceptions","Interceptions","🛡️")]
        tc = st.columns(len(top_list))
        for cw,(m,lbl,icon) in zip(tc,top_list):
            if m in raw_top.columns and not raw_top[m].isna().all():
                top = raw_top.nlargest(1,m).iloc[0]
                cw.markdown(metric_card(f"{icon} {round(top[m],1)}",
                                        f"{lbl} — {top['player'].split()[-1]}"),
                            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("SQUAD PERFORMANCE MAP")
        st.caption(
            "X = Total successful actions (completed passes + duels won + dribbles completed + shots on target + interceptions + recoveries). "
            "Y = Action success rate %. Bubble size = minutes played. "
            "Top-right = high volume AND high quality."
        )

        # Compute per-player totals
        ff = spd.groupby("player").agg(
            minutes           = ("minutes",           "sum"),
            passes_done       = ("pass_completions",  "sum"),
            passes_total      = ("passes",            "sum"),
            duels_won         = ("duels_won",         "sum"),
            duels_total       = ("duels",             "sum"),
            dribbles_done     = ("dribbles_success",  "sum"),
            dribbles_total    = ("dribbles",          "sum"),
            shots_on_target   = ("shots_on_target",   "sum"),
            shots_total       = ("shots",             "sum"),
            interceptions     = ("interceptions",     "sum"),
            recoveries        = ("ball_recoveries",   "sum"),
        ).reset_index()

        ff["successful_actions"] = (
            ff["passes_done"] + ff["duels_won"] + ff["dribbles_done"] +
            ff["shots_on_target"] + ff["interceptions"] + ff["recoveries"]
        )
        ff["total_actions"] = (
            ff["passes_total"] + ff["duels_total"] + ff["dribbles_total"] +
            ff["shots_total"] + ff["interceptions"] + ff["recoveries"]
        )
        ff["success_rate"] = (
            ff["successful_actions"] / ff["total_actions"].replace(0, np.nan) * 100
        ).round(1)
        ff = ff.dropna(subset=["success_rate"])
        ff = ff[ff["minutes"] >= min_mins]

        if not ff.empty:
            fig_ff = go.Figure()
            fig_ff.add_trace(go.Scatter(
                x=ff["successful_actions"],
                y=ff["success_rate"],
                mode="markers+text",
                text=ff["player"].apply(lambda p: p.split()[-1]),
                textposition="top center",
                textfont=dict(size=10, color=COLORS["muted"]),
                marker=dict(
                    size=ff["minutes"].apply(lambda m: np.clip(m/15, 8, 28)),
                    color=COLORS["accent"],
                    opacity=0.8,
                    line=dict(color="white", width=1),
                ),
                customdata=np.stack([
                    ff["player"], ff["minutes"].astype(int),
                    ff["successful_actions"].astype(int),
                    ff["success_rate"],
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Minutes: %{customdata[1]}<br>"
                    "Successful actions: %{customdata[2]}<br>"
                    "Success rate: %{customdata[3]:.1f}%"
                    "<extra></extra>"
                ),
            ))
            # Quadrant lines at median
            med_x = ff["successful_actions"].median()
            med_y = ff["success_rate"].median()
            fig_ff.add_vline(x=med_x, line_dash="dash", line_color=COLORS["surface3"], line_width=1)
            fig_ff.add_hline(y=med_y, line_dash="dash", line_color=COLORS["surface3"], line_width=1)
            xm, ym = ff["successful_actions"].max(), ff["success_rate"].max()
            for lbl2, xp, yp in [
                ("⭐ Elite",       xm*0.88, ym*0.97),
                ("💎 Efficient",   med_x*0.05, ym*0.97),
                ("📦 High Volume", xm*0.88, med_y*0.6),
                ("👁️ Assess",     med_x*0.05, med_y*0.6),
            ]:
                fig_ff.add_annotation(x=xp, y=yp, text=lbl2, showarrow=False,
                                      font=dict(color=COLORS["muted"], size=12))

            fig_ff.update_layout(
                xaxis_title="Successful Actions (volume)",
                yaxis_title="Action Success Rate (%)",
            )
            plotly_dark_layout(fig_ff, height=450)
            st.plotly_chart(fig_ff, use_container_width=True)
            st.caption("Bubble size = minutes played. Hover for full details.")

    # ── Tab 3: Load & Workload ────────────────────────────────────────────────
    with tab3:
        spd3 = pdata_f[pdata_f["team"]==sel_team].copy()
        c_l,c_r = st.columns([1,2])
        with c_l:
            section_header("SELECTION")
            all_p = sorted(spd3["player"].unique().tolist())
            sel_p_load = st.multiselect("Players",all_p,
                default=all_p[:4] if len(all_p)>=4 else all_p, key="sv_load_players")
            load_opts = {"Total Actions p90":"total_actions_p90","Pressures p90":"pressures_p90",
                         "Pass Accuracy %":"pass_acc","Minutes Played":"minutes",
                         "Actions Under Pressure p90":"under_pressure_p90",
                         "xG per match":"xg","Progressive Carries p90":"progressive_carries_p90"}
            load_lbl    = st.selectbox("Metric",list(load_opts.keys()),key="sv_load_metric")
            load_metric = load_opts[load_lbl]
        with c_r:
            section_header("TREND ACROSS MATCHWEEKS")
            if sel_p_load:
                ld = spd3[spd3["player"].isin(sel_p_load)].copy()
                cu = load_metric if load_metric in ld.columns else load_metric.replace("_p90","")
                if cu in ld.columns:
                    fig_t = go.Figure()
                    for pn in sel_p_load:
                        pf = ld[ld["player"]==pn].sort_values("match_week")
                        if pf.empty: continue
                        fig_t.add_trace(go.Scatter(x=pf["match_week"],y=pf[cu],
                            mode="lines+markers",name=pn,line=dict(width=2.5),marker=dict(size=8),
                            hovertemplate=f"<b>{pn}</b><br>MW %{{x}}<br>{load_lbl}: %{{y:.1f}}<extra></extra>"))
                    plotly_dark_layout(fig_t,height=360)
                    fig_t.update_xaxes(title="Match Week",tickmode="linear",dtick=1)
                    fig_t.update_yaxes(title=load_lbl)
                    st.plotly_chart(fig_t, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("ACTIONS HEATMAP")
        if sel_p_load and "total_actions_p90" in spd3.columns:
            piv = (spd3[spd3["player"].isin(sel_p_load)]
                   .groupby(["player","match_week"])["total_actions_p90"].mean()
                   .reset_index().pivot(index="player",columns="match_week",values="total_actions_p90"))
            if not piv.empty:
                piv.columns = [int(c) for c in piv.columns]
                fig_hm = px.imshow(piv,
                    color_continuous_scale=[[0,COLORS["danger"]],[0.5,COLORS["surface2"]],[1,COLORS["success"]]],
                    aspect="auto",labels=dict(x="Match Week",y="Player",color="Actions p90"))
                fig_hm.update_layout(paper_bgcolor=COLORS["bg"],font=dict(color=COLORS["text"]),
                    height=300,margin=dict(l=20,r=20,t=20,b=20),
                    xaxis=dict(tickmode="linear",dtick=1))
                st.plotly_chart(fig_hm, use_container_width=True)

    # ── Tab 4: Player Intelligence ────────────────────────────────────────────
    with tab4:
        spd4 = pdata_f[pdata_f["team"]==sel_team].copy()
        mode = st.radio("View mode",["Single Player","Player Comparison"],horizontal=True,key="sv_intel_mode")
        pp = (spd4.groupby("player")["position"]
              .agg(lambda x: x.mode()[0] if len(x)>0 else "—").reset_index())
        p_opts = {f"{r['player']} ({r['position']})":r["player"]
                  for _,r in pp.sort_values("player").iterrows()}

        if mode == "Single Player":
            sel_pl = st.selectbox("Select player",list(p_opts.keys()),key="sv_single_p")
            sel_p  = p_opts[sel_pl]
            if sel_p:
                pdf  = spd4[spd4["player"]==sel_p]
                pg   = get_position_group(pdf["position"].mode()[0] if not pdf.empty else "Unknown")
                nat  = pdf["nationality"].mode()[0] if not pdf.empty else ""
                apps = pdf["match_id"].nunique()
                mins = int(pdf["minutes"].sum())
                xgt  = round(pdf["xg"].sum(),2)
                pat  = round(pdf["pass_acc"].mean(),1)

                st.markdown(f"""
                <div class="card card-accent">
                  <div style="font-size:20px;font-weight:700;color:{COLORS['text']}">{sel_p}</div>
                  <div style="color:{COLORS['muted']};margin-top:4px;font-size:15px">{sel_team} · {nat} · {pg}</div>
                  <div style="margin-top:14px;display:flex;gap:36px">
                    <div><span style="font-size:26px;font-weight:700;color:{COLORS['accent']}">{apps}</span>
                         <span style="font-size:13px;color:{COLORS['muted']}"> APPS</span></div>
                    <div><span style="font-size:26px;font-weight:700;color:{COLORS['accent']}">{mins}</span>
                         <span style="font-size:13px;color:{COLORS['muted']}"> MINS</span></div>
                    <div><span style="font-size:26px;font-weight:700;color:{COLORS['accent']}">{xgt}</span>
                         <span style="font-size:13px;color:{COLORS['muted']}"> xG</span></div>
                    <div><span style="font-size:26px;font-weight:700;color:{COLORS['accent']}">{pat}%</span>
                         <span style="font-size:13px;color:{COLORS['muted']}"> PASS ACC</span></div>
                  </div>
                </div>""", unsafe_allow_html=True)

                cr,ct = st.columns([1,1])
                with cr:
                    section_header(f"RADAR — vs {pg.upper()} AVERAGE")
                    fig_r = build_radar(players,[sel_p],pg)
                    st.plotly_chart(fig_r, use_container_width=True)
                with ct:
                    section_header("METRIC TREND")
                    t_opts = {"Actions p90":"total_actions_p90","Pass Accuracy %":"pass_acc",
                              "xG":"xg","Pressures p90":"pressures_p90","Minutes":"minutes"}
                    t_lbl    = st.selectbox("Metric",list(t_opts.keys()),key="sv_trend_metric")
                    t_metric = t_opts[t_lbl]
                    cu = t_metric if t_metric in pdf.columns else t_metric.replace("_p90","")
                    if cu in pdf.columns:
                        tdf = pdf.sort_values("match_week")
                        fig_tr = go.Figure()
                        fig_tr.add_trace(go.Scatter(x=tdf["match_week"],y=tdf[cu],
                            mode="lines+markers",
                            line=dict(color=COLORS["accent"],width=2.5),
                            marker=dict(size=8,color=COLORS["accent"])))
                        if len(tdf)>=3:
                            roll = tdf[cu].rolling(3,center=True).mean()
                            fig_tr.add_trace(go.Scatter(x=tdf["match_week"],y=roll,
                                mode="lines",name="3-match avg",
                                line=dict(color=COLORS["accent2"],width=1.5,dash="dot")))
                        plotly_dark_layout(fig_tr,height=340)
                        fig_tr.update_xaxes(title="Match Week",tickmode="linear",dtick=1)
                        fig_tr.update_yaxes(title=t_lbl)
                        st.plotly_chart(fig_tr, use_container_width=True)

                # Pitch maps
                st.markdown("<br>", unsafe_allow_html=True)
                section_header(f"PITCH MAPS — {sel_p}")
                pev = edata[edata["player"]==sel_p].copy()
                pt1,pt2 = st.tabs(["Action Markers","Density Heatmap"])
                with pt1:
                    aet = sorted(pev["type"].dropna().unique().tolist())
                    set_ = st.multiselect("Event types",aet,
                        default=[t for t in ["Pass","Carry","Shot","Pressure"] if t in aet],
                        key="sv_player_events")
                    if set_:
                        b64 = draw_player_actions_pitch(pev,sel_p,set_,
                            title=f"{sel_p} — {', '.join(set_)} ({len(sel_ids)} matches)")
                        if b64:
                            st.markdown(f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:10px">',
                                        unsafe_allow_html=True)
                        else: st.info("Not enough data.")
                with pt2:
                    b64h = draw_player_heatmap(pev,sel_p,
                        title=f"{sel_p} — Activity Heatmap ({len(sel_ids)} matches)")
                    if b64h:
                        st.markdown(f'<img src="data:image/png;base64,{b64h}" style="width:100%;border-radius:10px">',
                                    unsafe_allow_html=True)
                    else: st.info("Not enough location data.")
        else:
            sel_psl = st.multiselect("Select 2–3 players",list(p_opts.keys()),
                default=list(p_opts.keys())[:2] if len(p_opts)>=2 else list(p_opts.keys()),
                max_selections=3,key="sv_cmp_players")
            if len(sel_psl)>=2:
                sel_ps = [p_opts[l] for l in sel_psl]
                pg = get_position_group(spd4[spd4["player"]==sel_ps[0]]["position"].mode()[0])
                cr2,ct2 = st.columns([1,1])
                with cr2:
                    section_header(f"RADAR — {pg.upper()} PROFILE")
                    fig_r2 = build_radar(players,sel_ps,pg)
                    st.plotly_chart(fig_r2, use_container_width=True)
                with ct2:
                    section_header("HEAD-TO-HEAD")
                    cmp_m = ["minutes","pass_acc","progressive_passes_p90","key_passes_p90",
                             "xg_p90","shots_p90","shot_acc","pressures_p90","duels_won_p90",
                             "duel_win_pct","tackles_p90","interceptions_p90","clearances_p90",
                             "progressive_carries_p90","dribble_pct","carry_distance_p90","total_actions_p90"]
                    rows = []
                    for m in cmp_m:
                        base = m.replace("_p90","")
                        cu = m if m in spd4.columns else (base if base in spd4.columns else None)
                        if cu is None: continue
                        row = {"Metric":fmt_col(m)}; vals = {}
                        for p in sel_ps:
                            v = spd4[spd4["player"]==p][cu].mean()
                            vals[p] = round(v,2) if not pd.isna(v) else 0.0
                            row[p]  = vals[p]
                        row["Best"] = f"✅ {max(vals,key=vals.get).split()[-1]}"
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=540)

    # ── Tab 5: Team Pitch ─────────────────────────────────────────────────────
    with tab5:
        cf1,cf2 = st.columns([1,3])
        with cf1:
            section_header("FILTERS")
            pt_sel = st.radio("Team",[sel_team,"Opponents"],key="sv_pitch_team")
            aet2   = sorted(edata["type"].dropna().unique().tolist())
            set2   = st.selectbox("Event Type",aet2,key="sv_pitch_event")
            aout2  = sorted(edata[edata["type"]==set2]["outcome"].dropna().unique().tolist())
            sout2  = st.multiselect("Outcome",aout2,default=aout2,key="sv_pitch_outcome")
            ev_f2  = edata[edata["team"]==sel_team] if pt_sel==sel_team else edata[edata["team"]!=sel_team]
            tplist = ["All"]+sorted(ev_f2["player"].dropna().unique().tolist())
            spl2   = st.selectbox("Player",tplist,key="sv_pitch_player")
            amws   = sorted(mdata["match_week"].unique().tolist())
            smws   = st.multiselect("Match Weeks",amws,default=amws,key="sv_pitch_mw")
        with cf2:
            section_header(f"{set2.upper()} — {pt_sel} · {len(smws)} matchweeks")
            mids_mw = mdata[mdata["match_week"].isin(smws)]["match_id"].tolist()
            evf = edata[(edata["match_id"].isin(mids_mw))&(edata["type"]==set2)].copy()
            if pt_sel==sel_team: evf = evf[evf["team"]==sel_team]
            else:                evf = evf[evf["team"]!=sel_team]
            if spl2!="All":      evf = evf[evf["player"]==spl2]
            if sout2:            evf = evf[evf["outcome"].isin(sout2)]
            st.caption(f"{len(evf)} events across {len(smws)} matchweeks")
            if not evf.empty:
                b64tp = draw_pitch_figure(evf,set2,sout2 if sout2 else aout2,
                    title=f"{set2} · {pt_sel}",
                    home_team=sel_team,away_team="Opponents",split_teams=False)
                st.markdown(f'<img src="data:image/png;base64,{b64tp}" style="width:100%;border-radius:10px">',
                            unsafe_allow_html=True)
            else:
                st.info("No events match current filters.")


# ── About page ────────────────────────────────────────────────────────────────

def page_about():
    st.markdown('<div class="page-title">👤 About</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col_bio, col_links = st.columns([2, 1])

    with col_bio:
        st.markdown(f"""
        <div class="card card-accent">
          <div style="font-size:26px;font-weight:800;color:{COLORS['text']}">Thanos Paidoulias</div>
          <div style="font-size:16px;color:{COLORS['accent']};margin-top:4px;font-weight:600">
            Data Scientist · Football Analytics
          </div>
          <div style="margin-top:16px;font-size:15px;color:{COLORS['muted']};line-height:1.8">
            Data scientist with expertise in big data analysis and machine learning,
            focused on applying analytical methods to sports performance, player evaluation,
            and decision-making.
          </div>

          <div style="margin-top:20px">
            <div style="font-size:13px;font-weight:600;color:{COLORS['accent']};
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
              THIS PLATFORM
            </div>
            <div style="font-size:15px;color:{COLORS['muted']};line-height:1.8">
              Built as a performance intelligence tool for women's football, powered by
              StatsBomb open event data across the WSL, Frauen Bundesliga, and Liga F (2023/24).
              Demonstrates what an AMS analytics layer could look like in a professional club —
              match intelligence, squad profiling, player involvement trends, and pitch analysis.
            </div>
          </div>

          <div style="margin-top:20px">
            <div style="font-size:13px;font-weight:600;color:{COLORS['accent']};
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
              EDUCATION & CERTIFICATIONS
            </div>
            <div style="font-size:15px;color:{COLORS['muted']};line-height:2.0">
              🎓 MSc Applied Statistics · Athens University of Economics and Business<br>
              🎓 BSc Economics · Athens University of Economics and Business<br>
              📋 Certificate in Football Tactical Analysis · Barça Innovation Hub<br>
              📋 Game Analysis & Fitness Reports · InStat Scout System<br>
              📋 FA Level 1 in Talent · The Football Association
            </div>
          </div>

          <div style="margin-top:20px">
            <div style="font-size:13px;font-weight:600;color:{COLORS['accent']};
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">
              TECHNICAL STACK
            </div>
            <div style="font-size:15px;color:{COLORS['muted']};line-height:1.8">
              Python · R · SQL · Streamlit · Shiny · Plotly · mplsoccer · StatsBomb<br>
              Azure · Databricks · Tableau · Power BI · Machine Learning · Git
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_links:
        st.markdown(f"""
        <div class="card" style="margin-bottom:16px">
          <div style="font-size:13px;font-weight:600;color:{COLORS['accent']};
                      text-transform:uppercase;letter-spacing:0.08em;margin-bottom:16px">
            CONTACT & LINKS
          </div>
          <div style="margin-bottom:16px">
            <div style="font-size:12px;color:{COLORS['muted']};margin-bottom:4px;font-weight:600">EMAIL</div>
            <a href="mailto:thanospaidoulias@gmail.com"
               style="color:{COLORS['text']};font-size:15px;text-decoration:none;">
              thanospaidoulias@gmail.com
            </a>
          </div>
          <div style="margin-bottom:16px">
            <div style="font-size:12px;color:{COLORS['muted']};margin-bottom:4px;font-weight:600">LINKEDIN</div>
            <a href="https://www.linkedin.com/in/thanos-paidoulias/" target="_blank"
               style="color:{COLORS['accent']};font-size:15px;text-decoration:none;">
              linkedin.com/in/thanos-paidoulias ↗
            </a>
          </div>
          <div style="margin-bottom:16px">
            <div style="font-size:12px;color:{COLORS['muted']};margin-bottom:4px;font-weight:600">GITHUB</div>
            <a href="https://github.com/ThanosPaidoulias" target="_blank"
               style="color:{COLORS['accent']};font-size:15px;text-decoration:none;">
              github.com/ThanosPaidoulias ↗
            </a>
          </div>
          <div>
            <div style="font-size:12px;color:{COLORS['muted']};margin-bottom:4px;font-weight:600">SOCCERAPP</div>
            <a href="https://thanospaidoulias.shinyapps.io/soccerapp/" target="_blank"
               style="color:{COLORS['accent']};font-size:15px;text-decoration:none;">
              Live Scouting App ↗
            </a>
          </div>
        </div>

        <div class="card">
          <div style="font-size:13px;font-weight:600;color:{COLORS['accent']};
                      text-transform:uppercase;letter-spacing:0.08em;margin-bottom:14px">
            SELECTED PROJECTS
          </div>
          <div style="font-size:14px;color:{COLORS['muted']};line-height:2.2">
            ⚽ MSc Thesis: Event data to predict player performance<br>
            📊 EPL Transfer Dashboard (Tableau · 2010–2020)<br>
            🎲 Serie A Bayesian Simulation<br>
            🥅 Football Penalty Analysis
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────


def page_match_detail(matches, players, events):
    sel_ids  = st.session_state.get("sel_match_ids", [])
    match_id = sel_ids[0] if sel_ids else None
    if match_id is None:
        st.error("No match selected.")
        return

    match_row  = matches[matches["match_id"] == match_id].iloc[0]
    home_team  = match_row["home_team"]
    away_team  = match_row["away_team"]
    home_score = int(match_row["home_score"])
    away_score = int(match_row["away_score"])
    comp       = match_row["competition"]
    mdate      = pd.to_datetime(match_row["match_date"]).strftime("%d %b %Y")
    mweek      = int(match_row["match_week"])

    pdata = players[players["match_id"] == match_id].copy()
    edata = events[events["match_id"] == match_id].copy()

    with st.sidebar:
        st.markdown('<div class="section-header">Match Detail</div>', unsafe_allow_html=True)
        if st.button("← Back to Match Selector"):
            st.session_state["view"] = "match_selector"
            st.rerun()

    result_color = (COLORS["accent"] if home_score > away_score
                    else COLORS["danger"] if home_score < away_score else COLORS["muted"])
    st.markdown(f"""
    <div class="card" style="text-align:center">
      <div style="font-size:11px;color:{COLORS['muted']};text-transform:uppercase;letter-spacing:0.1em">
        {comp_badge(comp)} · {mdate} · Match Week {mweek}
      </div>
      <div style="margin-top:12px;display:flex;align-items:center;justify-content:center;gap:24px">
        <div style="font-size:18px;font-weight:700;color:{COLORS['text']};text-align:right;width:220px">{home_team}</div>
        <div style="font-size:36px;font-weight:800;color:{result_color};min-width:90px;text-align:center">
          {home_score} – {away_score}
        </div>
        <div style="font-size:18px;font-weight:700;color:{COLORS['text']};text-align:left;width:220px">{away_team}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Match Overview","Player Stats","Pitch Analysis","Shot Map"
    ])

    # ── Tab 1: Match Overview ─────────────────────────────────────────────────
    with tab1:
        home_p = pdata[pdata["team"] == home_team]
        away_p = pdata[pdata["team"] == away_team]

        def team_sum(df, col, is_pct=False):
            if col not in df.columns: return 0
            return round(df[col].mean(), 1) if is_pct else int(df[col].sum())

        overview_metrics = [
            ("Goals",              "goals",            False),
            ("xG",                 "xg",               False),
            ("Shots",              "shots",            False),
            ("Shots on Target",    "shots_on_target",  False),
            ("Shot Accuracy %",    "shot_acc",         True),
            ("Passes",             "passes",           False),
            ("Pass Accuracy %",    "pass_acc",         True),
            ("Key Passes",         "key_passes",       False),
            ("Progressive Passes", "progressive_passes",False),
            ("Crosses",            "crosses",          False),
            ("Pressures",          "pressures",        False),
            ("Duels Won",          "duels_won",        False),
            ("Duel Win %",         "duel_win_pct",     True),
            ("Tackles",            "tackles",          False),
            ("Interceptions",      "interceptions",    False),
            ("Clearances",         "clearances",       False),
            ("Progressive Carries","progressive_carries",False),
            ("Dribbles Completed", "dribbles_success", False),
            ("Fouls Committed",    "fouls_committed",  False),
        ]

        # ── FIX: aligned 3-column table with HTML ────────────────────────────
        rows_html = ""
        for label, col, is_pct in overview_metrics:
            hv = team_sum(home_p, col, is_pct)
            av = team_sum(away_p, col, is_pct)
            # Bold the winner
            hbold = "font-weight:700;" if hv > av else ""
            abold = "font-weight:700;" if av > hv else ""
            rows_html += f"""
            <tr>
              <td style="text-align:right;padding:6px 12px;{hbold}color:{COLORS['text']}">{hv}</td>
              <td style="text-align:center;padding:6px 12px;color:{COLORS['muted']};font-size:12px">{label}</td>
              <td style="text-align:left;padding:6px 12px;{abold}color:{COLORS['text']}">{av}</td>
            </tr>"""

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;background:{COLORS['surface']};
                      border-radius:10px;overflow:hidden">
          <thead>
            <tr>
              <th style="text-align:right;padding:10px 12px;color:{COLORS['accent']};
                         font-size:13px;font-weight:700;width:38%">{home_team}</th>
              <th style="text-align:center;padding:10px 12px;color:{COLORS['muted']};
                         font-size:11px;font-weight:600;text-transform:uppercase;width:24%">Stat</th>
              <th style="text-align:left;padding:10px 12px;color:{COLORS['accent2']};
                         font-size:13px;font-weight:700;width:38%">{away_team}</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Passing networks ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("PASSING NETWORKS")
        net_col1, net_col2 = st.columns(2)
        with net_col1:
            b64 = draw_passing_network(edata, pdata, home_team, home_team)
            if b64:
                st.markdown(
                    f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:10px">',
                    unsafe_allow_html=True
                )
        with net_col2:
            b64 = draw_passing_network(edata, pdata, away_team, home_team)
            if b64:
                st.markdown(
                    f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:10px">',
                    unsafe_allow_html=True
                )

        # ── 15-min period chart ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("PERFORMANCE BY 15-MINUTE PERIOD")

        # ── FIX: expanded metric list including shots and final third ─────────
        period_metric_opts = {
            "Total Actions":       ["Pass","Carry","Pressure","Duel"],
            "Passes":              ["Pass"],
            "Pressures":           ["Pressure"],
            "Duels":               ["Duel"],
            "Shots":               ["Shot"],
            "Interceptions":       ["Interception"],
            "Ball Recoveries":     ["Ball Recovery"],
            "Fouls":               ["Foul Committed"],
        }
        period_col = st.selectbox("Metric", list(period_metric_opts.keys()), key="period_metric")

        period_labels = ["0–15","15–30","30–45","45–60","60–75","75–90+"]
        period_bins   = [-1,15,30,45,60,75,200]

        ev_sub = edata[edata["type"].isin(period_metric_opts[period_col])].copy()

        # Final third filter for passes
        if period_col == "Passes":
            ev_sub = ev_sub[ev_sub["end_x"] >= 80] if "end_x" in ev_sub.columns else ev_sub
            st.caption("Showing passes into the final third only")

        if not ev_sub.empty and "minute" in ev_sub.columns:
            ev_sub["period_bin"] = pd.cut(ev_sub["minute"], bins=period_bins, labels=period_labels)
            period_counts = (
                ev_sub.groupby(["team","period_bin"], observed=True)
                .size().reset_index(name="count")
            )
            fig_p = px.bar(
                period_counts, x="period_bin", y="count", color="team", barmode="group",
                color_discrete_map={home_team: COLORS["home"], away_team: COLORS["away"]},
                labels={"period_bin":"Period","count":period_col,"team":"Team"},
            )
            plotly_dark_layout(fig_p, height=300)
            st.plotly_chart(fig_p, use_container_width=True)

    # ── Tab 2: Player Stats ───────────────────────────────────────────────────
    with tab2:
        p90_toggle2  = st.toggle("Per 90 minutes", value=False, key="match_p90")
        sel_team_tab2= st.radio("Team", [home_team, away_team, "Both"],
                                horizontal=True, key="match_team")

        show_pdata = pdata.copy()
        if sel_team_tab2 != "Both":
            show_pdata = show_pdata[show_pdata["team"] == sel_team_tab2]

        display_metrics = [
            "player","team","position","nationality","minutes",
            "passes","pass_acc","key_passes","progressive_passes","crosses",
            "shots","shots_on_target","shot_acc","goals","xg",
            "duels","tackles","duels_won","duel_win_pct",
            "pressures","interceptions","clearances","blocks",
            "carries","carry_distance","progressive_carries",
            "dribbles","dribbles_success","dribble_pct",
            "total_actions",
        ]
        if p90_toggle2:
            no_norm = {"player","team","position","nationality","minutes","pass_acc",
                       "shot_acc","duel_win_pct","dribble_pct",}
            p90_cols  = [m+"_p90" for m in display_metrics if m not in no_norm]
            base_keep = [m for m in display_metrics if m in no_norm]
            all_show  = [m for m in base_keep + p90_cols if m in show_pdata.columns]
        else:
            all_show = [m for m in display_metrics if m in show_pdata.columns]

        table_df = show_pdata[all_show].rename(
            columns={m: fmt_col(m) for m in all_show}
        ).round(2)

        st.dataframe(
            table_df, use_container_width=True, hide_index=True, height=520,
            column_config={
                "Player":   st.column_config.TextColumn("Player"),
                "Nat":      st.column_config.TextColumn("Nationality"),
                "Position": st.column_config.TextColumn("Position (this match)"),
            }
        )

    # ── Tab 3: Pitch Analysis ─────────────────────────────────────────────────
    with tab3:
        col_f1, col_f2 = st.columns([1,3])

        with col_f1:
            section_header("FILTERS")

            # Team — default Both so pitch is never empty on first load
            team_opts      = ["Both", home_team, away_team]
            sel_pitch_team = st.selectbox("Team", team_opts, key="pitch_team")

            # Player
            if sel_pitch_team == "Both":
                avail_players = ["All"] + sorted(edata["player"].dropna().unique().tolist())
            else:
                avail_players = ["All"] + sorted(
                    edata[edata["team"] == sel_pitch_team]["player"].dropna().unique().tolist()
                )
            sel_pitch_player = st.selectbox("Player", avail_players, key="pitch_player")

            # Event type
            avail_events = sorted(edata["type"].unique().tolist())
            sel_event    = st.selectbox("Event Type", avail_events, key="pitch_event")

            # ── FIX: default ALL outcomes so pitch is never empty ─────────────
            avail_outcomes = sorted(
                edata[edata["type"] == sel_event]["outcome"].dropna().unique().tolist()
            )
            sel_outcomes = st.multiselect(
                "Outcome", avail_outcomes,
                default=avail_outcomes,   # all selected by default
                key="pitch_outcome"
            )

            # Coordinate sliders
            x_range = st.slider("X coordinate", 0, 120, (0, 120), key="pitch_x")
            y_range = st.slider("Y coordinate", 0, 80,  (0, 80),  key="pitch_y")

            # ── NEW: minute range filter ──────────────────────────────────────
            max_min = int(edata["minute"].max()) if not edata.empty else 90
            min_range = st.slider("Minutes", 0, max_min, (0, max_min), key="pitch_min")

        with col_f2:
            section_header(f"{sel_event.upper()} — {home_team} vs {away_team}")

            ev_filtered = edata[edata["type"] == sel_event].copy()

            if sel_pitch_team != "Both":
                ev_filtered = ev_filtered[ev_filtered["team"] == sel_pitch_team]
            if sel_pitch_player != "All":
                ev_filtered = ev_filtered[ev_filtered["player"] == sel_pitch_player]
            if sel_outcomes:
                ev_filtered = ev_filtered[ev_filtered["outcome"].isin(sel_outcomes)]

            ev_filtered = ev_filtered[
                (ev_filtered["x"] >= x_range[0]) & (ev_filtered["x"] <= x_range[1]) &
                (ev_filtered["y"] >= y_range[0]) & (ev_filtered["y"] <= y_range[1]) &
                (ev_filtered["minute"] >= min_range[0]) & (ev_filtered["minute"] <= min_range[1])
            ]

            st.caption(f"{len(ev_filtered)} events plotted")

            if not ev_filtered.empty:
                # Always colour by team — home=teal, away=amber
                # When one team: use that team's fixed colour
                # When both: split by team with respective colours
                split = True  # always team-coloured
                pitch_b64 = draw_pitch_figure(
                    ev_filtered, sel_event,
                    sel_outcomes if sel_outcomes else avail_outcomes,
                    title=f"{sel_event} · {sel_pitch_team}",
                    home_team=home_team, away_team=away_team,
                    split_teams=split,
                )
                st.markdown(
                    f'<img src="data:image/png;base64,{pitch_b64}" '
                    f'style="width:100%;border-radius:10px">',
                    unsafe_allow_html=True
                )
            else:
                st.info("No events match the current filters.")

    # ── Tab 4: Shot Map ───────────────────────────────────────────────────────
    with tab4:
        shots_all = edata[edata["type"] == "Shot"].copy()

        if shots_all.empty:
            st.info("No shot data available for this match.")
        else:
            shots_all["xG"] = shots_all["shot_statsbomb_xg"].fillna(0.05)

            # Filters — one team at a time, no Both
            filter_col1, filter_col2, filter_col3 = st.columns([1, 2, 1])
            with filter_col1:
                sel_shot_team = st.radio(
                    "Team", [home_team, away_team],
                    horizontal=False, key="shot_team_filter"
                )
            with filter_col2:
                all_outcomes  = sorted(shots_all["outcome"].dropna().unique().tolist())
                sel_shot_outs = st.multiselect("Outcome", all_outcomes,
                                               default=all_outcomes, key="shot_outcome_filter")
            with filter_col3:
                max_min_shot   = int(shots_all["minute"].max())
                shot_min_range = st.slider("Minutes", 0, max_min_shot,
                                           (0, max_min_shot), key="shot_min_range")

            shots_df = shots_all[
                (shots_all["team"] == sel_shot_team) &
                shots_all["outcome"].isin(sel_shot_outs) &
                shots_all["minute"].between(*shot_min_range)
            ].copy()

            col_sm1, col_sm2 = st.columns([2, 1])

            with col_sm1:
                section_header(f"SHOT MAP — {sel_shot_team} · bubble size = xG · ★ = Goal")
                if not shots_df.empty:
                    shot_b64 = draw_shot_map_half_pitch(shots_df, sel_shot_team)
                    st.markdown(
                        f'<img src="data:image/png;base64,{shot_b64}" '
                        f'style="width:100%;border-radius:10px;max-width:700px">',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No shots match current filters.")

            with col_sm2:
                section_header("SHOT LOG")
                shot_log = shots_df[["player","minute","outcome","xG"]]\
                    .sort_values("minute")\
                    .rename(columns={"player":"Player","minute":"Min","outcome":"Outcome"})
                shot_log["xG"] = shot_log["xG"].round(3)
                st.dataframe(shot_log, hide_index=True, height=460,
                             use_container_width=True)


def main():
    # Initialise session state
    if "view" not in st.session_state:
        st.session_state["view"] = "landing"
    if "loaded_comps" not in st.session_state:
        st.session_state["loaded_comps"] = ALL_COMPETITIONS.copy()

    matches = safe_load()

    # Sidebar — always visible
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:14px 0 6px">
          <span style="font-size:22px;font-weight:800;color:{COLORS['accent']}">⚽ WFC</span>
          <span style="font-size:14px;color:{COLORS['muted']};display:block;margin-top:2px">Analytics Platform</span>
        </div>
        """, unsafe_allow_html=True)

        # Home button — always works
        if st.button("⌂  Home", key="global_home", use_container_width=True):
            st.session_state["view"] = "landing"
            st.rerun()

        st.markdown("---")
        st.markdown('<div class="section-header">Data</div>', unsafe_allow_html=True)
        sel_comps = st.multiselect("Competitions loaded", ALL_COMPETITIONS,
                                   default=st.session_state["loaded_comps"],
                                   key="comp_selector")
        if not sel_comps:
            st.warning("Select at least one competition.")
            st.stop()
        st.session_state["loaded_comps"] = sel_comps
        with st.spinner("Loading data..."):
            events, players = load_selected_competitions(sel_comps)
        n = len(matches[matches["competition"].isin(sel_comps)])
        st.caption(f"✅ {n} matches · {players['player'].nunique()} players")
        st.markdown("---")

    matches_f = matches[matches["competition"].isin(sel_comps)].copy()
    view      = st.session_state["view"]

    if   view == "landing":          page_landing()
    elif view == "match_selector":   page_match_selector(matches_f, players, events)
    elif view == "match_detail":     page_match_detail(matches_f, players, events)
    elif view == "season_selector":  page_season_selector(matches_f, players, events)
    elif view == "season_view":      page_season_view(matches_f, players, events)
    elif view == "about":            page_about()
    else:
        st.session_state["view"] = "landing"
        st.rerun()


if __name__ == "__main__":
    main()
