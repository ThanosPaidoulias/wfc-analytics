"""
WFC Analytics Platform — Data Pipeline
Pulls StatsBomb open data for WSL 2023/24, Frauen Bundesliga 2023/24, Liga F 2023/24
Computes all per-player per-match metrics and saves cached CSVs.

Run once before launching the app:
    python data_pipeline.py

Output files (saved to ./data/):
    matches.csv          — all matches across 3 leagues
    player_match.csv     — per player per match metrics (~55 metrics)
    events_raw.csv       — coordinate-flipped raw events for pitch viz
"""

import os
import numpy as np
import pandas as pd
from statsbombpy import sb

# ── Config ──────────────────────────────────────────────────────────────────

COMPETITIONS = [
    {"competition_id": 37,  "season_id": 281, "name": "WSL",               "short": "WSL"},
    {"competition_id": 135, "season_id": 281, "name": "Frauen Bundesliga", "short": "BL"},
    {"competition_id": 182, "season_id": 281, "name": "Liga F",            "short": "LF"},
]

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

PITCH_LENGTH = 120
PITCH_WIDTH  = 80

# ── Helpers ──────────────────────────────────────────────────────────────────

def _xy(val):
    """Safely extract [x, y] from a list-like location field."""
    return val if isinstance(val, list) and len(val) >= 2 else [None, None]


def flip_coords(df, home_team):
    """
    StatsBomb open data already normalises ALL teams to attack left → right (toward x=120).
    No coordinate flip needed. This function is a no-op kept for compatibility.
    NOTE: The Barnsley FC proprietary data required flipping because it stored
    away team coordinates from their own attacking perspective. StatsBomb open
    data does not — both teams always attack toward x=120.
    """
    return df


def parse_minutes(positions, max_minute=90):
    """
    Calculate minutes played and primary position from StatsBomb lineup positions list.
    Returns (minutes_played, primary_position).
    """
    total = 0
    primary = "Unknown"
    for pos in positions:
        start_str = pos.get("from") or "0:00"
        end_str   = pos.get("to")   or f"{min(max_minute, 90)}:00"
        try:
            s = int(start_str.split(":")[0])
            e = int(end_str.split(":")[0])
            total   += max(0, e - s)
            primary  = pos.get("position", primary)
        except (ValueError, AttributeError):
            pass
    return total, primary


def carry_distance(carries_df):
    """Euclidean carry distance from XY start/end coords."""
    if carries_df.empty:
        return 0.0
    dx = carries_df["end_x"] - carries_df["x"]
    dy = carries_df["end_y"] - carries_df["y"]
    return float(np.sqrt(dx**2 + dy**2).sum())


def progressive(start_x, end_x):
    """True if ball moved toward opponent goal (end_x > start_x)."""
    return end_x > start_x


# ── Per-player metrics extractor ─────────────────────────────────────────────

def extract_player_metrics(events, player_name, minutes, position, max_minute):
    """
    Extract all ~55 metrics for one player from a single match events DataFrame.
    """
    pe = events[events["player"] == player_name].copy()
    if pe.empty:
        return None

    m90 = max(minutes, 1) / 90  # divisor for per-90 normalisation

    # ── Passing ──────────────────────────────────────────────────────────────
    passes       = pe[pe["type"] == "Pass"]
    pass_total   = len(passes)
    pass_complete= len(passes[passes["pass_outcome"].isna()])
    pass_acc     = round(pass_complete / pass_total * 100, 1) if pass_total else 0.0

    passes_fwd   = passes[passes["end_x"] > passes["x"]]
    fwd_total    = len(passes_fwd)

    prog_passes  = len(passes_fwd)  # forward = progressive proxy
    final_third  = len(passes[passes["end_x"] >= 80])
    pen_area     = len(passes[
        (passes["end_x"] >= 102) &
        passes["pass_end_location"].apply(lambda v: 18 <= _xy(v)[1] <= 62 if isinstance(v, list) else False)
    ])
    crosses      = int(passes["pass_cross"].sum()) if "pass_cross" in passes.columns else 0
    through_balls= int(passes["pass_through_ball"].sum()) if "pass_through_ball" in passes.columns else 0
    switches     = int(passes["pass_switch"].sum()) if "pass_switch" in passes.columns else 0
    key_passes   = int(passes["pass_shot_assist"].sum()) if "pass_shot_assist" in passes.columns else 0
    avg_pass_len = round(passes["pass_length"].mean(), 1) if pass_total else 0.0

    # forward pass distance bands (using pass_length as proxy)
    fp_0_5   = len(passes_fwd[passes_fwd["pass_length"] < 5])   if "pass_length" in passes_fwd.columns else 0
    fp_5_10  = len(passes_fwd[(passes_fwd["pass_length"] >= 5)  & (passes_fwd["pass_length"] < 10)])  if "pass_length" in passes_fwd.columns else 0
    fp_10_15 = len(passes_fwd[(passes_fwd["pass_length"] >= 10) & (passes_fwd["pass_length"] < 15)])  if "pass_length" in passes_fwd.columns else 0
    fp_15    = len(passes_fwd[passes_fwd["pass_length"] >= 15]) if "pass_length" in passes_fwd.columns else 0

    # ── Shooting ─────────────────────────────────────────────────────────────
    shots       = pe[pe["type"] == "Shot"]
    shot_total  = len(shots)
    shot_on_tgt = len(shots[shots["shot_outcome"].isin(["Saved", "Goal",
                                                         "Saved to Post", "Saved Off Target"])])
    goals       = len(shots[shots["shot_outcome"] == "Goal"])
    xg_total    = round(float(shots["shot_statsbomb_xg"].fillna(0).sum()), 3)
    xg_per_shot = round(xg_total / shot_total, 3) if shot_total else 0.0
    shot_acc    = round(shot_on_tgt / shot_total * 100, 1) if shot_total else 0.0
    headed_shots= len(shots[shots["shot_body_part"] == "Head"]) if "shot_body_part" in shots.columns else 0
    first_time  = int(shots["shot_first_time"].sum()) if "shot_first_time" in shots.columns else 0

    # ── Duels ────────────────────────────────────────────────────────────────
    duels       = pe[pe["type"] == "Duel"]
    duel_total  = len(duels)
    tackles     = len(duels[duels["duel_type"] == "Tackle"]) if "duel_type" in duels.columns else 0
    aerials     = len(duels[duels["duel_type"] == "Aerial Lost"]) if "duel_type" in duels.columns else 0
    duel_won    = len(duels[duels["duel_outcome"].isin(["Won", "Success In Play", "Success Out"])]) \
                  if "duel_outcome" in duels.columns else 0
    duel_win_pct= round(duel_won / duel_total * 100, 1) if duel_total else 0.0
    tackle_won  = len(duels[(duels["duel_type"] == "Tackle") &
                             duels["duel_outcome"].isin(["Won", "Success In Play", "Success Out"])]) \
                  if "duel_type" in duels.columns and "duel_outcome" in duels.columns else 0
    tackle_win_pct = round(tackle_won / tackles * 100, 1) if tackles else 0.0

    # ── Defending ────────────────────────────────────────────────────────────
    pressures    = len(pe[pe["type"] == "Pressure"])
    interceptions= len(pe[pe["type"] == "Interception"])
    clearances   = len(pe[pe["type"] == "Clearance"])
    blocks       = len(pe[pe["type"] == "Block"])
    recoveries   = len(pe[pe["type"] == "Ball Recovery"])
    dribbled_past= len(pe[pe["type"] == "Dribbled Past"])
    fouls_comm   = len(pe[pe["type"] == "Foul Committed"])
    fouls_won    = len(pe[pe["type"] == "Foul Won"])
    yellow_cards = int(pe[pe["type"] == "Foul Committed"]["foul_committed_card"]
                       .eq("Yellow Card").sum()) if "foul_committed_card" in pe.columns else 0

    # ── Carrying / Progression ───────────────────────────────────────────────
    carries_df   = pe[pe["type"] == "Carry"].copy()
    carry_total  = len(carries_df)
    carry_dist   = carry_distance(carries_df)
    prog_carries = len(carries_df[carries_df["end_x"] > carries_df["x"]]) if carry_total else 0
    ft_carries   = len(carries_df[carries_df["end_x"] >= 80]) if carry_total else 0

    dribbles     = pe[pe["type"] == "Dribble"]
    drib_total   = len(dribbles)
    drib_success = len(dribbles[dribbles["dribble_outcome"] == "Complete"]) \
                   if "dribble_outcome" in dribbles.columns else 0
    drib_pct     = round(drib_success / drib_total * 100, 1) if drib_total else 0.0
    dispossessed = len(pe[pe["type"] == "Dispossessed"])
    miscontrols  = len(pe[pe["type"] == "Miscontrol"])

    # ── Physical / workload proxies ──────────────────────────────────────────
    act_types    = ["Pass","Carry","Pressure","Duel","Dribble","Shot",
                    "Ball Recovery","Interception","Clearance","Block"]
    core_types   = ["Pass","Carry","Pressure","Duel"]

    total_actions= len(pe[pe["type"].isin(act_types)])
    under_press  = len(pe[pe["under_pressure"] == True]) if "under_pressure" in pe.columns else 0
    fifty_fifty  = len(pe[pe["type"] == "50/50"])



    # ── OBV (if available) ────────────────────────────────────────────────────
    obv = round(float(pe["obv_total_net"].fillna(0).sum()), 3) \
          if "obv_total_net" in pe.columns else 0.0

    # ── Goalkeeper ────────────────────────────────────────────────────────────
    gk_events   = pe[pe["type"] == "Goal Keeper"] if "type" in pe.columns else pd.DataFrame()
    shots_faced = len(gk_events[gk_events["goalkeeper_type"] == "Shot Faced"]) \
                  if "goalkeeper_type" in gk_events.columns else 0
    saves       = len(gk_events[gk_events["goalkeeper_type"].isin(
                      ["Shot Saved", "Shot Saved to Post", "Shot Saved Off Target"])]) \
                  if "goalkeeper_type" in gk_events.columns else 0
    gk_conceded = len(gk_events[gk_events["goalkeeper_type"] == "Goal Conceded"]) \
                  if "goalkeeper_type" in gk_events.columns else 0
    save_pct    = round(saves / shots_faced * 100, 1) if shots_faced else 0.0
    sweeper     = len(gk_events[gk_events["goalkeeper_type"] == "Keeper Sweeper"]) \
                  if "goalkeeper_type" in gk_events.columns else 0
    punches     = len(gk_events[gk_events["goalkeeper_type"] == "Punch"]) \
                  if "goalkeeper_type" in gk_events.columns else 0

    return {
        # Identity
        "player":            player_name,
        "position":          position,
        "minutes":           minutes,
        # Passing
        "passes":            pass_total,
        "pass_completions":  pass_complete,
        "pass_acc":          pass_acc,
        "progressive_passes":prog_passes,
        "final_third_passes":final_third,
        "pen_area_passes":   pen_area,
        "crosses":           crosses,
        "through_balls":     through_balls,
        "switch_passes":     switches,
        "key_passes":        key_passes,
        "avg_pass_length":   avg_pass_len,
        "fwd_passes_0_5m":   fp_0_5,
        "fwd_passes_5_10m":  fp_5_10,
        "fwd_passes_10_15m": fp_10_15,
        "fwd_passes_15m":    fp_15,
        # Shooting
        "shots":             shot_total,
        "shots_on_target":   shot_on_tgt,
        "shot_acc":          shot_acc,
        "goals":             goals,
        "xg":                xg_total,
        "xg_per_shot":       xg_per_shot,
        "headed_shots":      headed_shots,
        "first_time_shots":  first_time,
        # Duels
        "duels":             duel_total,
        "tackles":           tackles,
        "aerials":           aerials,
        "duels_won":         duel_won,
        "duel_win_pct":      duel_win_pct,
        "tackle_win_pct":    tackle_win_pct,
        # Defending
        "pressures":         pressures,
        "interceptions":     interceptions,
        "clearances":        clearances,
        "blocks":            blocks,
        "ball_recoveries":   recoveries,
        "dribbled_past":     dribbled_past,
        "fouls_committed":   fouls_comm,
        "fouls_won":         fouls_won,
        "yellow_cards":      yellow_cards,
        # Carrying
        "carries":           carry_total,
        "carry_distance":    round(carry_dist, 1),
        "progressive_carries":prog_carries,
        "final_third_carries":ft_carries,
        "dribbles":          drib_total,
        "dribbles_success":  drib_success,
        "dribble_pct":       drib_pct,
        "dispossessed":      dispossessed,
        "miscontrols":       miscontrols,
        # Physical proxies
        "total_actions":     total_actions,
        "under_pressure":    under_press,
        "fifty_fifty":       fifty_fifty,
        "obv":               obv,
        # Goalkeeper
        "shots_faced":       shots_faced,
        "saves":             saves,
        "save_pct":          save_pct,
        "gk_goals_conceded": gk_conceded,
        "sweeper_actions":   sweeper,
        "gk_punches":        punches,
    }


# ── Per-90 normalisation ──────────────────────────────────────────────────────

# Metrics that should NOT be per-90 normalised (rates, averages, identities)
NO_NORM = {
    "player","position","team","competition","match_id","match_date",
    "match_week","home_team","away_team","home_score","away_score","minutes",
    "nationality",
    # rate metrics — already percentages
    "pass_acc","shot_acc","duel_win_pct","tackle_win_pct","dribble_pct","save_pct",
    "xg_per_shot","avg_pass_length",
    # 15-min period counts kept as raw for period charts
    
    
}

def add_per90_cols(df):
    """Add _p90 column for every count metric."""
    count_cols = [c for c in df.columns if c not in NO_NORM]
    for col in count_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[f"{col}_p90"] = (df[col] / df["minutes"].replace(0, np.nan) * 90).round(2)
    return df


# ── Raw events for pitch visualisation ───────────────────────────────────────

PITCH_EVENT_TYPES = [
    "Pass","Shot","Carry","Duel","Pressure","Interception",
    "Ball Recovery","Clearance","Block","Dribble","Foul Committed","Foul Won"
]

def extract_pitch_events(events, home_team, match_id, competition, home_t, away_t):
    """
    Extract and coordinate-flip pitch-level events for a single match.
    Returns a tidy DataFrame ready for visualisation.
    """
    ev = events[events["type"].isin(PITCH_EVENT_TYPES)].copy()
    if ev.empty:
        return pd.DataFrame()

    ev["x"]     = ev["location"].apply(lambda v: _xy(v)[0])
    ev["y"]     = ev["location"].apply(lambda v: _xy(v)[1])
    ev["end_x"] = ev.get("pass_end_location",   pd.Series(dtype=object)).apply(lambda v: _xy(v)[0])
    ev["end_y"] = ev.get("pass_end_location",   pd.Series(dtype=object)).apply(lambda v: _xy(v)[1])

    # carries end location
    carry_mask = ev["type"] == "Carry"
    if carry_mask.any() and "carry_end_location" in ev.columns:
        ev.loc[carry_mask, "end_x"] = ev.loc[carry_mask, "carry_end_location"].apply(lambda v: _xy(v)[0])
        ev.loc[carry_mask, "end_y"] = ev.loc[carry_mask, "carry_end_location"].apply(lambda v: _xy(v)[1])

    ev = flip_coords(ev, home_team)

    # Outcome normalisation (same logic as Barnsley dataCleansing.R)
    shot_gk = ["Shot", "Goal Keeper"]
    fail_outcomes = ["Lost","Incomplete","Lost In Play","Lost Out","Out","Pass Offside"]
    ev["outcome"] = "Made"
    ev.loc[ev["type"].isin(shot_gk), "outcome"] = \
        ev.loc[ev["type"].isin(shot_gk), "shot_outcome"].fillna(
        ev.loc[ev["type"].isin(shot_gk), "goalkeeper_outcome"].fillna("Unknown"))
    ev.loc[~ev["type"].isin(shot_gk) &
           ev["pass_outcome"].isin(fail_outcomes), "outcome"] = "Not Made"
    ev.loc[~ev["type"].isin(shot_gk) &
           ev["duel_outcome"].isin(["Lost In Play","Lost Out"]), "outcome"] = "Not Made"

    keep = ["match_id","competition","home_team","away_team",
            "team","player","type","minute","period",
            "x","y","end_x","end_y","outcome",
            "shot_statsbomb_xg","under_pressure"]
    keep = [c for c in keep if c in ev.columns]
    ev["match_id"]    = match_id
    ev["competition"] = competition
    ev["home_team"]   = home_t
    ev["away_team"]   = away_t
    return ev[keep].dropna(subset=["x","y"])


# ── File naming helpers ────────────────────────────────────────────────────────

def comp_slug(comp_name):
    """Convert competition name to safe filename slug."""
    return comp_name.replace(" ", "_").replace("/", "_")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline():
    all_matches = []

    for comp in COMPETITIONS:
        cid, sid, comp_name = comp["competition_id"], comp["season_id"], comp["name"]
        slug = comp_slug(comp_name)

        print(f"\n{'='*60}")
        print(f"  Processing: {comp_name}")
        print(f"{'='*60}")

        matches = sb.matches(competition_id=cid, season_id=sid)
        matches["competition"] = comp_name
        all_matches.append(matches)

        comp_player_stats = []
        comp_pitch_events = []

        total = len(matches)
        for idx, (_, match) in enumerate(matches.iterrows(), 1):
            mid       = int(match["match_id"])
            home_team = match["home_team"]
            away_team = match["away_team"]
            mweek     = int(match.get("match_week", 0))
            mdate     = str(match["match_date"])
            hscore    = int(match.get("home_score", 0))
            ascore    = int(match.get("away_score", 0))

            print(f"  [{idx:>3}/{total}] {home_team} vs {away_team} (MW {mweek})", end=" ... ")

            try:
                events  = sb.events(match_id=mid)
                lineups = sb.lineups(match_id=mid)
            except Exception as e:
                print(f"SKIP ({e})")
                continue

            max_minute = int(events["minute"].max()) if not events.empty else 90

            # Extract XY coords from nested location fields
            if "location" in events.columns:
                events["x"] = events["location"].apply(lambda v: _xy(v)[0])
                events["y"] = events["location"].apply(lambda v: _xy(v)[1])
            if "pass_end_location" in events.columns:
                events["end_x"] = events["pass_end_location"].apply(lambda v: _xy(v)[0])
                events["end_y"] = events["pass_end_location"].apply(lambda v: _xy(v)[1])
            if "carry_end_location" in events.columns:
                carry_mask = events["type"] == "Carry"
                events.loc[carry_mask, "end_x"] = \
                    events.loc[carry_mask, "carry_end_location"].apply(lambda v: _xy(v)[0])
                events.loc[carry_mask, "end_y"] = \
                    events.loc[carry_mask, "carry_end_location"].apply(lambda v: _xy(v)[1])

            # StatsBomb open data: no coordinate flip needed —
            # both teams already attack toward x=120.
            events = flip_coords(events, home_team)  # no-op

            # Pitch events
            pitch_ev = extract_pitch_events(
                events, home_team, mid, comp_name, home_team, away_team
            )
            if not pitch_ev.empty:
                comp_pitch_events.append(pitch_ev)

            # Player metrics
            for team_name, lineup_df in lineups.items():
                for _, player_row in lineup_df.iterrows():
                    pname       = player_row["player_name"]
                    nickname    = player_row.get("player_nickname", None)
                    nationality = player_row.get("country", "Unknown")
                    jersey      = player_row.get("jersey_number", None)
                    positions   = player_row.get("positions", [])

                    minutes, position = parse_minutes(positions, max_minute)
                    if minutes < 5:
                        continue

                    metrics = extract_player_metrics(
                        events, pname, minutes, position, max_minute
                    )
                    if metrics is None:
                        continue

                    metrics.update({
                        "match_id":    mid,
                        "competition": comp_name,
                        "match_date":  mdate,
                        "match_week":  mweek,
                        "team":        team_name,
                        "home_team":   home_team,
                        "away_team":   away_team,
                        "home_score":  hscore,
                        "away_score":  ascore,
                        "nationality": nationality,
                        "jersey_number": jersey,
                        "nickname": nickname if nickname else pname,
                    })
                    comp_player_stats.append(metrics)

            print("OK")

        # ── Save per-competition files ─────────────────────────────────────────
        print(f"\n  Saving {comp_name} files...")

        # events_<slug>.csv
        if comp_pitch_events:
            ev_df = pd.concat(comp_pitch_events, ignore_index=True)
            ev_path = f"{DATA_DIR}/events_{slug}.csv"
            ev_df.to_csv(ev_path, index=False)
            import os
            size_mb = os.path.getsize(ev_path) / 1024 / 1024
            print(f"  events_{slug}.csv  → {len(ev_df):,} events  ({size_mb:.1f} MB)")

        # players_<slug>.csv
        if comp_player_stats:
            pl_df = pd.DataFrame(comp_player_stats)
            pl_df = add_per90_cols(pl_df)
            id_cols    = ["match_id","competition","match_date","match_week",
                          "home_team","away_team","home_score","away_score",
                          "team","player","nationality","position","minutes"]
            other_cols = [c for c in pl_df.columns if c not in id_cols]
            pl_df      = pl_df[id_cols + other_cols]
            pl_path    = f"{DATA_DIR}/players_{slug}.csv"
            pl_df.to_csv(pl_path, index=False)
            size_mb = os.path.getsize(pl_path) / 1024 / 1024
            print(f"  players_{slug}.csv → {len(pl_df):,} records  ({size_mb:.1f} MB)")

    # ── matches.csv — single file covering all competitions ───────────────────
    print(f"\n{'='*60}")
    print("  Saving matches.csv (all competitions)...")
    matches_df = pd.concat(all_matches, ignore_index=True)
    match_cols = ["match_id","competition","match_date","match_week",
                  "home_team","away_team","home_score","away_score"]
    match_cols = [c for c in match_cols if c in matches_df.columns]
    matches_df[match_cols].to_csv(f"{DATA_DIR}/matches.csv", index=False)
    print(f"  matches.csv        → {len(matches_df)} matches")
    print("\n  Pipeline complete. Ready to launch app.")
    return matches_df


if __name__ == "__main__":
    run_pipeline()

