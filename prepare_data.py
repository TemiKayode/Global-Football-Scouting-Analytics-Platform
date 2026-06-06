"""
prepare_data.py
===============
Downloads and merges data from:
  1. Transfermarkt Datasets (dcaribou/transfermarkt-datasets) — player profiles,
     market values, real goals/assists/cards/minutes from appearances.
  2. StatsBomb Open Data (statsbomb/open-data) — event-level advanced stats:
     passes, shots on target, tackles, interceptions, progressive actions.

Run once after installing requirements:
    pip install -r requirements.txt
    python prepare_data.py

Outputs:
    all_players_data.csv   — 20 000+ players, all columns the app expects
    team_clusters.csv      — one row per club with aggregate style metrics
"""

import os
import sys
import gzip
import io
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TM_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
SB_RAW  = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

TARGET_TM_LEAGUES = {
    "A1","ARG1","BE1","BRA1","C1","ES1","FR1","GB1","GR1","IT1",
    "KR1","L1","NO1","PL1","PO1","RO1","RU1","SC1","SE1","TR1","TS1",
}
# Map Transfermarkt codes → app codes
TM_TO_APP = {
    "A1":"A1","ARG1":"AR1N","BE1":"BE1","BRA1":"BRA1","C1":"C1",
    "ES1":"ES1","FR1":"FR1","GB1":"GB1","GR1":"GR1","IT1":"IT1",
    "KR1":"KR1","L1":"L1","NO1":"NO1","PL1":"PL1","PO1":"PO1",
    "RO1":"RO1","RU1":"RU1","SC1":"SC1","SE1":"SE1","TR1":"TR1","TS1":"TS1",
}
LEAGUE_NAMES = {
    "A1":"Austrian Bundesliga","AR1N":"Argentine Primera","BE1":"Belgian Pro League",
    "BRA1":"Brasileirao","BU1":"Bulgarian First League","C1":"Czech First League",
    "ES1":"La Liga","ES2":"La Liga 2","FR1":"Ligue 1","FR2":"Ligue 2",
    "GB1":"Premier League","GB2":"Championship","GR1":"Super League Greece",
    "IT1":"Serie A","IT2":"Serie B","KR1":"K League 1","L1":"Bundesliga",
    "L2":"2. Bundesliga","NO1":"Eliteserien","PL1":"Ekstraklasa",
    "PO1":"Primeira Liga","RO1":"Liga I Romania","RU1":"Russian Premier League",
    "SC1":"Scottish Premiership","SE1":"Allsvenskan","SL1":"Super League Switzerland",
    "TR1":"Super Lig","TS1":"Tunisian Ligue Pro","UNG1":"OTP Bank Liga",
    "WGBL":"Womens Bundesliga","WWSL":"WSL","WFRD1":"D1 Feminine",
    "WNWSL":"NWSL","WAUS":"A-League Women","WBRA":"Brazilian Womens Serie A","WITA":"Womens Serie A",
}
SUB_POS_MAP = {
    "Goalkeeper":"GK","Centre-Back":"CB","Left-Back":"LB","Right-Back":"RB",
    "Defensive Midfield":"CDM","Central Midfield":"CM","Left Midfield":"CM",
    "Right Midfield":"CM","Attacking Midfield":"CAM","Left Winger":"LW",
    "Right Winger":"RW","Centre-Forward":"ST","Second Striker":"ST",
}
POSITIONS = ["GK","CB","LB","RB","CDM","CM","CAM","LW","RW","ST"]

# StatsBomb competition IDs mapped to app league codes
SB_COMP_MAP = {
    2:  "GB1",   # Premier League
    11: "GB1",   # FA Women's Super League → WWSL actually
    37: "WNWSL", # NWSL
    49: "WFRD1", # D1 Feminine
    53: "WGBL",  # Frauen-Bundesliga
    72: "WITA",  # Women's Serie A
}

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def download_gz(url: str, cache_path: str) -> pd.DataFrame:
    """Download a .csv.gz file and return as DataFrame (cached locally)."""
    if os.path.exists(cache_path):
        print(f"  Using cached {cache_path}")
    else:
        print(f"  Downloading {url} …")
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    with gzip.open(cache_path, "rb") as gz:
        return pd.read_csv(gz, low_memory=False)


def download_json(url: str, cache_path: str):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    print(f"  Fetching {url} …")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFERMARKT DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_transfermarkt() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[1/3] Loading Transfermarkt data…")
    players_df = download_gz(f"{TM_BASE}/players.csv.gz", "raw_players.csv.gz")
    appear_df  = download_gz(f"{TM_BASE}/appearances.csv.gz", "raw_appearances.csv.gz")
    return players_df, appear_df


def process_players(players_df: pd.DataFrame) -> pd.DataFrame:
    """Filter and transform Transfermarkt player profiles."""
    df = players_df[
        players_df["current_club_domestic_competition_id"].isin(TARGET_TM_LEAGUES) &
        (players_df["last_season"] >= 2023) &
        players_df["sub_position"].notna() &
        players_df["name"].notna()
    ].copy()

    now = datetime.now()

    def ovr_from_value(v):
        try:
            v = float(v)
            if v <= 0: return 58
            return int(np.clip(55 + 8 * (np.log10(v) - 4), 50, 97))
        except:
            return 60

    def age_from_dob(d):
        try:
            dob = pd.to_datetime(d)
            return int((now - dob).days / 365.25)
        except:
            return 25

    def contract_left(d):
        try:
            exp = pd.to_datetime(d)
            y = (exp - now).days / 365.25
            return round(float(np.clip(y, 0, 5)), 1)
        except:
            return 1.0

    def intl_rep(caps):
        try:
            c = int(caps)
            if c >= 60: return 5
            if c >= 31: return 4
            if c >= 11: return 3
            if c >= 1:  return 2
            return 1
        except:
            return 1

    df["age"]                  = df["date_of_birth"].apply(age_from_dob)
    df["overall_rating"]       = df["market_value_in_eur"].apply(ovr_from_value)
    df["potential"]            = df.apply(lambda r: int(np.clip(
        r["overall_rating"] + np.random.randint(0, max(1, int((30-r["age"])*0.8+1))),
        r["overall_rating"], 97)), axis=1)
    df["contract_years_left"]  = df["contract_expiration_date"].apply(contract_left)
    df["international_reputation"] = df["international_caps"].apply(intl_rep)
    df["market_value_m"]       = (df["market_value_in_eur"].fillna(0).astype(float) / 1e6).round(2)
    df["position"]             = df["sub_position"].map(SUB_POS_MAP).fillna("CM")
    df["league"]               = df["current_club_domestic_competition_id"].map(TM_TO_APP)
    df["league_name"]          = df["league"].map(LEAGUE_NAMES).fillna("")
    df["club"]                 = df["current_club_name"]
    df["player_id"]            = "TM" + df["player_id"].astype(str)
    df["is_women"]             = False

    # Progression ratings (estimated — no historical TM ratings)
    rng = np.random.default_rng(42)
    df["past_rating_2yr"]  = df.apply(lambda r: int(np.clip(
        r["overall_rating"] - rng.integers(0, max(1, int(max(0, 27-r["age"])*0.7+1))),
        45, r["overall_rating"])), axis=1)
    df["past_rating_1yr"]  = ((df["overall_rating"] + df["past_rating_2yr"]) / 2).astype(int)

    return df


def aggregate_appearances(appear_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate appearances to per-player totals for 2022+ seasons."""
    df = appear_df[appear_df["date"] >= "2022-07-01"].copy()
    for col in ["goals","assists","yellow_cards","red_cards","minutes_played"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["match_count"] = 1

    agg = df.groupby("player_id").agg(
        goals=("goals","sum"), assists=("assists","sum"),
        yellow_cards=("yellow_cards","sum"), red_cards=("red_cards","sum"),
        minutes_played=("minutes_played","sum"), matches_in_squad=("match_count","sum"),
    ).reset_index()
    agg["player_id"] = "TM" + agg["player_id"].astype(str)
    agg = agg[agg["minutes_played"] >= 90]

    # Per-90 stats
    n90 = agg["minutes_played"] / 90.0
    agg["goals_per90"]        = (agg["goals"]        / n90).round(3)
    agg["assists_per90"]      = (agg["assists"]       / n90).round(3)
    agg["yellow_cards_per90"] = (agg["yellow_cards"]  / n90).round(3)
    agg["red_cards_per90"]    = (agg["red_cards"]     / n90).round(3)
    agg["minutes_per90_ratio"]= (agg["minutes_played"] / (agg["matches_in_squad"] * 90)).round(3)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# STATSBOMB EVENT DATA
# ─────────────────────────────────────────────────────────────────────────────

def compute_sb_player_stats(match_files_base: str, match_ids: list, comp_code: str) -> pd.DataFrame:
    """Download StatsBomb event files and compute per-game player stats."""
    rows = []
    for mid in match_ids[:50]:  # cap to avoid very long downloads
        path = f"sb_events_{mid}.json"
        try:
            events = download_json(f"{SB_RAW}/events/{mid}.json", path)
        except Exception:
            continue
        for ev in events:
            pname = ev.get("player", {}).get("name")
            if not pname:
                continue
            etype = ev.get("type", {}).get("id", 0)
            rows.append({
                "player_name": pname, "match_id": mid,
                "is_pass":          1 if etype == 30 else 0,
                "pass_complete":    1 if (etype == 30 and ev.get("pass",{}).get("outcome") is None) else 0,
                "is_shot":          1 if etype == 16 else 0,
                "shot_on_target":   1 if (etype == 16 and ev.get("shot",{}).get("outcome",{}).get("id") in [98,100,115]) else 0,
                "is_tackle":        1 if etype == 4  else 0,
                "is_interception":  1 if etype == 10 else 0,
                "is_dribble":       1 if etype == 14 else 0,
                "is_carry":         1 if etype == 43 else 0,
                "progressive_carry":1 if (etype == 43 and _is_progressive(ev.get("carry",{}))) else 0,
                "progressive_pass": 1 if (etype == 30 and _is_progressive(ev.get("pass",{}))) else 0,
                "minutes":          1,  # approximation
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grp = df.groupby("player_name").sum(numeric_only=True)
    n90 = grp["minutes"] / 90.0
    result = pd.DataFrame(index=grp.index)
    result["sb_pass_completion"]    = (grp["pass_complete"] / grp["is_pass"].clip(1)).round(3)
    result["sb_shots_on_target_pct"]= (grp["shot_on_target"] / grp["is_shot"].clip(1)).round(3)
    result["sb_tackles_per90"]      = (grp["is_tackle"]       / n90).round(3)
    result["sb_interceptions_per90"]= (grp["is_interception"] / n90).round(3)
    result["sb_dribbles_per90"]     = (grp["is_dribble"]      / n90).round(3)
    result["sb_progressive_carries"]= (grp["progressive_carry"] / n90).round(3)
    result["sb_progressive_passes"] = (grp["progressive_pass"]  / n90).round(3)
    result.index.name = "player_name"
    return result.reset_index()


def _is_progressive(ev_sub: dict) -> bool:
    """Rough check: carry/pass is progressive if it moves ball ≥ 10m toward opponent goal."""
    loc = ev_sub.get("location") or ev_sub.get("end_location")
    end = ev_sub.get("end_location") or ev_sub.get("location")
    if not loc or not end:
        return False
    return (end[0] - loc[0]) >= 10  # x-axis toward goal


def load_statsbomb(players_df: pd.DataFrame) -> pd.DataFrame:
    """Download StatsBomb open data and compute player-level advanced stats."""
    print("\n[2/3] Loading StatsBomb open data…")
    comps_path = "sb_competitions.json"
    comps = download_json(f"{SB_RAW}/competitions.json", comps_path)
    print(f"  {len(comps)} competitions available.")

    # Target recent seasons for available competitions
    all_sb_stats = []
    for comp in comps:
        cid = comp["competition_id"]
        sid = comp["season_id"]
        season_name = comp.get("season_name","")
        # Only use recent seasons (2021/22 onwards)
        if not any(y in season_name for y in ["2021","2022","2023","2024","2025"]):
            continue

        matches_path = f"sb_matches_{cid}_{sid}.json"
        try:
            matches = download_json(f"{SB_RAW}/matches/{cid}/{sid}.json", matches_path)
        except Exception:
            continue
        if not matches:
            continue

        match_ids = [m["match_id"] for m in matches]
        print(f"  {comp['competition_name']} {season_name}: {len(match_ids)} matches")
        sb_stats = compute_sb_player_stats(f"{SB_RAW}/events", match_ids, str(cid))
        if not sb_stats.empty:
            all_sb_stats.append(sb_stats)

    if not all_sb_stats:
        print("  No StatsBomb event data retrieved.")
        return pd.DataFrame()

    # Merge all StatsBomb player stats (average across seasons)
    combined = pd.concat(all_sb_stats, ignore_index=True)
    agg = combined.groupby("player_name").mean(numeric_only=True).reset_index()
    print(f"  StatsBomb stats computed for {len(agg)} players.")
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# POSITION-BASED STAT ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_stats(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate per-90 stats estimated from position and overall rating."""
    n = len(df)
    quality = np.clip((df["overall_rating"].values - 50.0) / 47.0, 0, 1)
    noise = rng.uniform(-0.15, 0.15, n)

    STAT_CONFIG = {
        "GK":  dict(goals_per90=(.00,.02), assists_per90=(.00,.02), shots_on_target_pct=(.30,.15),
                    pass_completion=(.62,.25), key_passes_per90=(.05,.10), progressive_passes=(1.0,2.0),
                    tackles_per90=(.10,.20), interceptions_per90=(.05,.15), dribbles_per90=(.05,.10),
                    aerial_duels_won_pct=(.45,.20), duels_won_pct=(.45,.15), progressive_carries=(.2,.5),
                    crosses_per90=(.0,.05), through_balls_per90=(.0,.02), xg_per90=(.0,.01),
                    saves_per90=(2.5,2.5), clean_sheets_pct=(.20,.25), sweeper_actions=(1.0,2.0),
                    goals_conceded_per90=(1.2,1.0)),
        "CB":  dict(goals_per90=(.02,.04), assists_per90=(.01,.03), shots_on_target_pct=(.28,.18),
                    pass_completion=(.75,.18), key_passes_per90=(.15,.20), progressive_passes=(3.0,5.0),
                    tackles_per90=(1.8,2.5), interceptions_per90=(1.2,2.0), dribbles_per90=(.10,.30),
                    aerial_duels_won_pct=(.50,.25), duels_won_pct=(.50,.20), progressive_carries=(.5,1.5),
                    crosses_per90=(.10,.30), through_balls_per90=(.02,.05), xg_per90=(.02,.03),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "LB":  dict(goals_per90=(.02,.05), assists_per90=(.05,.15), shots_on_target_pct=(.26,.18),
                    pass_completion=(.72,.18), key_passes_per90=(.20,.40), progressive_passes=(2.5,4.0),
                    tackles_per90=(1.5,2.5), interceptions_per90=(.80,1.5), dribbles_per90=(.30,.80),
                    aerial_duels_won_pct=(.40,.25), duels_won_pct=(.45,.20), progressive_carries=(1.5,3.5),
                    crosses_per90=(.80,2.5), through_balls_per90=(.03,.08), xg_per90=(.02,.04),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "RB":  dict(goals_per90=(.02,.05), assists_per90=(.05,.15), shots_on_target_pct=(.26,.18),
                    pass_completion=(.72,.18), key_passes_per90=(.20,.40), progressive_passes=(2.5,4.0),
                    tackles_per90=(1.5,2.5), interceptions_per90=(.80,1.5), dribbles_per90=(.30,.80),
                    aerial_duels_won_pct=(.40,.25), duels_won_pct=(.45,.20), progressive_carries=(1.5,3.5),
                    crosses_per90=(.80,2.5), through_balls_per90=(.03,.08), xg_per90=(.02,.04),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "CDM": dict(goals_per90=(.03,.07), assists_per90=(.03,.08), shots_on_target_pct=(.26,.16),
                    pass_completion=(.80,.14), key_passes_per90=(.30,.60), progressive_passes=(4.0,6.0),
                    tackles_per90=(2.5,3.0), interceptions_per90=(1.8,2.5), dribbles_per90=(.30,.80),
                    aerial_duels_won_pct=(.45,.22), duels_won_pct=(.52,.18), progressive_carries=(1.0,2.5),
                    crosses_per90=(.10,.30), through_balls_per90=(.05,.12), xg_per90=(.03,.06),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "CM":  dict(goals_per90=(.05,.12), assists_per90=(.05,.18), shots_on_target_pct=(.28,.18),
                    pass_completion=(.78,.15), key_passes_per90=(.60,1.5), progressive_passes=(4.0,6.0),
                    tackles_per90=(1.5,2.0), interceptions_per90=(.80,1.5), dribbles_per90=(.50,1.2),
                    aerial_duels_won_pct=(.40,.22), duels_won_pct=(.48,.18), progressive_carries=(1.5,3.5),
                    crosses_per90=(.20,.50), through_balls_per90=(.08,.18), xg_per90=(.05,.10),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "CAM": dict(goals_per90=(.08,.20), assists_per90=(.10,.28), shots_on_target_pct=(.30,.20),
                    pass_completion=(.76,.16), key_passes_per90=(1.2,2.5), progressive_passes=(3.5,5.5),
                    tackles_per90=(.80,1.2), interceptions_per90=(.40,.80), dribbles_per90=(.80,2.5),
                    aerial_duels_won_pct=(.35,.22), duels_won_pct=(.44,.18), progressive_carries=(2.0,5.0),
                    crosses_per90=(.30,.80), through_balls_per90=(.15,.40), xg_per90=(.08,.18),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "LW":  dict(goals_per90=(.12,.35), assists_per90=(.10,.28), shots_on_target_pct=(.32,.22),
                    pass_completion=(.72,.18), key_passes_per90=(.80,2.0), progressive_passes=(2.5,4.0),
                    tackles_per90=(.60,1.0), interceptions_per90=(.30,.60), dribbles_per90=(1.5,3.5),
                    aerial_duels_won_pct=(.35,.22), duels_won_pct=(.42,.18), progressive_carries=(2.5,5.5),
                    crosses_per90=(.50,1.5), through_balls_per90=(.05,.15), xg_per90=(.12,.30),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "RW":  dict(goals_per90=(.12,.35), assists_per90=(.10,.28), shots_on_target_pct=(.32,.22),
                    pass_completion=(.72,.18), key_passes_per90=(.80,2.0), progressive_passes=(2.5,4.0),
                    tackles_per90=(.60,1.0), interceptions_per90=(.30,.60), dribbles_per90=(1.5,3.5),
                    aerial_duels_won_pct=(.35,.22), duels_won_pct=(.42,.18), progressive_carries=(2.5,5.5),
                    crosses_per90=(.50,1.5), through_balls_per90=(.05,.15), xg_per90=(.12,.30),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
        "ST":  dict(goals_per90=(.18,.50), assists_per90=(.06,.20), shots_on_target_pct=(.36,.26),
                    pass_completion=(.68,.20), key_passes_per90=(.40,1.0), progressive_passes=(1.5,3.0),
                    tackles_per90=(.40,.80), interceptions_per90=(.20,.50), dribbles_per90=(.80,2.0),
                    aerial_duels_won_pct=(.45,.28), duels_won_pct=(.46,.20), progressive_carries=(1.2,3.0),
                    crosses_per90=(.10,.30), through_balls_per90=(.03,.08), xg_per90=(.18,.50),
                    saves_per90=(.0,.0), clean_sheets_pct=(.0,.0), sweeper_actions=(.0,.0),
                    goals_conceded_per90=(.0,.0)),
    }

    stat_cols = list(next(iter(STAT_CONFIG.values())).keys())
    result = pd.DataFrame(index=df.index)

    for stat in stat_cols:
        vals = np.zeros(n)
        for pos, cfg in STAT_CONFIG.items():
            mask = df["position"].values == pos
            if not mask.any():
                continue
            base, scale = cfg[stat]
            q   = quality[mask]
            cnt = int(mask.sum())
            ns  = noise[np.arange(cnt) % len(noise)]   # safe circular index
            v   = base + scale * (q + ns * 0.3)
            # GK-only stats are 0 for non-GK; already handled by base=0, scale=0
            vals[mask] = np.clip(v, 0, None)
        result[stat] = np.round(vals, 3)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC SUPPLEMENT (missing leagues)
# ─────────────────────────────────────────────────────────────────────────────

MISSING_LEAGUES = {
    "BU1": ("Bulgarian First League", ["Ludogorets","CSKA Sofia","Levski Sofia","Lokomotiv Plovdiv","Botev Plovdiv","Slavia Sofia","Beroe","Arda","Etar","Montana"]),
    "ES2": ("La Liga 2", ["Valladolid","Mirandes","Levante","Huesca","Eibar","Zaragoza","Racing Santander","Oviedo","Alcorcon","Elche"]),
    "FR2": ("Ligue 2", ["Strasbourg","Metz","Caen","Rodez","Grenoble","Amiens","Troyes","Valenciennes","Pau FC","Concarneau"]),
    "GB2": ("Championship", ["Leeds United","Leicester City","Middlesbrough","Sunderland","West Brom","Swansea","Burnley","Sheffield United","Watford","QPR"]),
    "IT2": ("Serie B", ["Parma","Como","Venezia","Sampdoria","Genoa","Pisa","Palermo","Bari","Catanzaro","Ascoli"]),
    "L2":  ("2. Bundesliga", ["Hamburger SV","Schalke 04","Hannover 96","Kaiserslautern","Hertha BSC","Fortuna Dusseldorf","Nurnberg","Magdeburg","Greuther Furth","Karlsruher SC"]),
    "SL1": ("Slovenian PrvaLiga", ["NK Olimpija Ljubljana","NK Maribor","NK Celje","NK Koper","NK Mura","NK Domzale","NK Bravo","NK Radomlje","NK Nafta 1903","ND Gorica"]),
    "UNG1":("OTP Bank Liga", ["Ferencvaros","MOL Fehervar","Paks","Ujpest","Kecskemet","Puskas Akademia","Debrecen","MTK Budapest","Zalaegerszeg","Honved"]),
}

WOMEN_LEAGUES = {
    "WGBL": ("Womens Bundesliga", ["Bayern Munich W","Wolfsburg W","Frankfurt W","Freiburg W","Hoffenheim W","Turbine Potsdam","Koln W","RB Leipzig W","Duisburg W","Essen W"]),
    "WWSL": ("WSL", ["Chelsea W","Arsenal W","Man City W","Man United W","Aston Villa W","Liverpool W","Brighton W","West Ham W","Tottenham W","Leicester W"]),
    "WFRD1":("D1 Feminine", ["Lyon W","PSG W","Paris FC W","Bordeaux W","Montpellier W","Guingamp W","Dijon W","Nice W","Reims W","Fleury W"]),
    "WNWSL":("NWSL", ["Portland Thorns","NC Courage","Chicago Red Stars","OL Reign","Washington Spirit","San Diego Wave","Angel City","Houston Dash","NJ/NY Gotham","Racing Louisville"]),
    "WAUS": ("A-League Women", ["Melbourne City W","Sydney FC W","Western Sydney W","Brisbane Roar W","Perth Glory W","Adelaide United W","Wellington Phoenix W","Canberra United","Newcastle Jets W","Central Coast W"]),
    "WBRA": ("Brazilian Womens Serie A", ["Corinthians W","Palmeiras W","Flamengo W","Santos W","Sao Paulo W","Cruzeiro W","Gremio W","Internacional W","Ferroviaria","Avai Kindermann"]),
    "WITA": ("Womens Serie A", ["Roma W","Juventus W","Milan W","Inter W","Fiorentina W","Sassuolo W","Sampdoria W","Lazio W","Napoli W","Hellas Verona W"]),
}

FIRST_NAMES = ["Lucas","Marco","Luca","Joao","Carlos","Diego","Ahmed","Mohamed","Pierre","Theo","Kai","Jamal","Phil","Mason","Bukayo","Erling","Kylian","Vinicius","Rodri","Federico","Pedri","Gavi","Jude","Florian","Leroy","Alphonso","Cody","Lamine","Aitana","Sam","Caroline","Vivianne","Ada","Pernille","Kadidiatou","Asisat","Trinity","Sophia","Elena","Lars","Erik","Ivan","Aleksandr","Tomas","Krzysztof","Rui","Bruno","Bernardo","Rafael","Gabriel","Sandro","Roberto","Raul","Antonio","Jose","Manuel","Sergio","Leon","Julian","Thomas","Toni","Joshua","Mats","Niklas","Ali","Omar","Yusuf","Mehmet","Burak","Emre","Hakan","Aleksandar","Stefan","Nemanja","Dusan","Luka","Mateo"]
LAST_NAMES  = ["Silva","Costa","Santos","Ferreira","Oliveira","Rodrigues","Mueller","Schmidt","Schneider","Fischer","Weber","Meyer","Garcia","Martinez","Lopez","Hernandez","Gonzalez","Perez","Dupont","Martin","Bernard","Rossi","Ferrari","Russo","Esposito","Romano","Kowalski","Nowak","Petrov","Ivanov","Sidorov","Park","Kim","Lee","Choi","Mbappe","Diallo","Traore","Diarra","Haaland","Odegaard","Pedersen","Nielsen","Hansen","Jensen","Moura","Nunes","Pinto","Carvalho","Mendes","Rashford","Sterling","Walker","Trippier","Yilmaz","Ozil","Calhanoglu","Mitrovic","Jovic","Vlahovic","Modric","Kovacic","Kramaric"]


def generate_synthetic(rng: np.random.Generator, leagues: dict, is_women: bool, players_per_club: int = 20) -> pd.DataFrame:
    rows = []
    idx = 0
    pos_weights = [1,2,2,2,2,2,1,1,1,2]  # GK,CB,LB,RB,CDM,CM,CAM,LW,RW,ST
    pos_pool = [p for p, w in zip(POSITIONS, pos_weights) for _ in range(w)]

    for lc, (ln, clubs) in leagues.items():
        for club in clubs:
            for _ in range(players_per_club):
                idx += 1
                pos  = rng.choice(pos_pool)
                base = 68 if is_women else (63 if lc in ("ES2","FR2","GB2","IT2","L2") else 61)
                ovr  = int(np.clip(base + rng.integers(-8, 13), 50, 90))
                age  = int(rng.integers(18, 37))
                pot  = int(np.clip(ovr + rng.integers(0, max(1, 30-age)), ovr, 97))
                mv_m = round(float(np.clip(10**((ovr-55)/8.0+4)/1e6, 0.1, 25)), 2)
                cyl  = round(float(rng.uniform(0.5, 5.0)), 1)
                matches = int(rng.integers(10, 38))
                mins    = int(rng.integers(matches*25, matches*90+1))
                yc      = int(rng.integers(0, 11))
                rc      = int(rng.integers(0, 2))
                pr2     = int(np.clip(ovr - rng.integers(0, max(1, int((30-age)*0.7+1))), 45, ovr))
                rows.append({
                    "player_id": f"SYN{lc}{idx}",
                    "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                    "age": age, "position": pos, "club": club,
                    "league": lc, "league_name": ln,
                    "overall_rating": ovr, "potential": pot,
                    "contract_years_left": cyl, "international_reputation": int(rng.integers(1,4)),
                    "market_value_m": mv_m,
                    "past_rating_2yr": pr2, "past_rating_1yr": int((ovr+pr2)/2),
                    "yellow_cards": yc, "red_cards": rc,
                    "matches_in_squad": matches, "minutes_played": mins,
                    "is_women": is_women,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TEAM CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────

def build_team_clusters(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["club", "league"]).agg(
        avg_pass_completion =("pass_completion","mean"),
        avg_pressing_actions=("tackles_per90","mean"),
        avg_progressive_passes=("progressive_passes","mean"),
        avg_progressive_carries=("progressive_carries","mean"),
        avg_key_passes      =("key_passes_per90","mean"),
        avg_dribbles        =("dribbles_per90","mean"),
        avg_crosses         =("crosses_per90","mean"),
        avg_aerial_duels_won=("aerial_duels_won_pct","mean"),
        squad_age           =("age","mean"),
        squad_rating        =("overall_rating","mean"),
        squad_size          =("player_id","count"),
    ).round(4).reset_index()
    # merge interceptions separately to guarantee club/league alignment
    inter = (df.groupby(["club","league"])["interceptions_per90"]
               .mean().reset_index()
               .rename(columns={"interceptions_per90": "_inter"}))
    agg = agg.merge(inter, on=["club","league"], how="left")
    agg["avg_pressing_actions"] = agg["avg_pressing_actions"] + agg["_inter"].fillna(0)
    agg.drop(columns=["_inter"], inplace=True)
    agg["team_style"] = ""
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    # 1. Transfermarkt
    players_raw, appear_raw = load_transfermarkt()
    players_tm = process_players(players_raw)
    app_agg    = aggregate_appearances(appear_raw)

    print(f"  Processed {len(players_tm)} TM players, {len(app_agg)} appearance aggregates")

    # 2. Estimate advanced stats for TM players
    print("\nEstimating advanced stats…")
    est = estimate_stats(players_tm, rng)
    for col in est.columns:
        players_tm[col] = est[col].values

    # 3. Patch real goals/assists/cards/minutes from appearances
    merged = players_tm.merge(app_agg, on="player_id", how="left", suffixes=("","_real"))
    has_real = merged["goals_per90_real"].notna()
    for stat in ["goals_per90","assists_per90","yellow_cards_per90","red_cards_per90"]:
        real_col = stat + "_real"
        if real_col in merged.columns:
            merged.loc[has_real, stat] = merged.loc[has_real, real_col]
    for col in ["yellow_cards","red_cards","minutes_played","matches_in_squad","minutes_per90_ratio"]:
        real_col = col + "_real"
        if real_col in merged.columns:
            merged.loc[has_real, col] = merged.loc[has_real, real_col]
    merged.loc[has_real, "xg_per90"] = (merged.loc[has_real, "goals_per90"] * 1.05).round(3)
    # drop real columns
    drop_cols = [c for c in merged.columns if c.endswith("_real")]
    merged.drop(columns=drop_cols, inplace=True)
    print(f"  Patched {has_real.sum()} players with real appearance stats")

    # 4. StatsBomb (optional — comment out if slow or no internet)
    try:
        sb_stats = load_statsbomb(merged)
        if not sb_stats.empty:
            merged = merged.merge(sb_stats, left_on="name", right_on="player_name", how="left")
            for stat, sb_col in [("pass_completion","sb_pass_completion"),
                                  ("shots_on_target_pct","sb_shots_on_target_pct"),
                                  ("tackles_per90","sb_tackles_per90"),
                                  ("interceptions_per90","sb_interceptions_per90"),
                                  ("dribbles_per90","sb_dribbles_per90"),
                                  ("progressive_carries","sb_progressive_carries"),
                                  ("progressive_passes","sb_progressive_passes")]:
                has_sb = merged[sb_col].notna()
                if has_sb.any():
                    merged.loc[has_sb, stat] = merged.loc[has_sb, sb_col]
            merged.drop(columns=[c for c in merged.columns if c.startswith("sb_")], inplace=True, errors="ignore")
            if "player_name" in merged.columns:
                merged.drop(columns=["player_name"], inplace=True)
    except Exception as e:
        print(f"  StatsBomb integration skipped: {e}")

    # 5. Synthetic supplement
    print("\n[3/3] Adding synthetic data for missing leagues…")
    syn_men   = generate_synthetic(rng, MISSING_LEAGUES, False, 20)
    syn_women = generate_synthetic(rng, WOMEN_LEAGUES, True, 18)
    est_men   = estimate_stats(syn_men, rng)
    est_women = estimate_stats(syn_women, rng)
    for col in est_men.columns:
        syn_men[col]   = est_men[col].values
        syn_women[col] = est_women[col].values

    all_players = pd.concat([merged, syn_men, syn_women], ignore_index=True)

    # Ensure all expected columns present
    expected_cols = [
        "player_id","name","age","position","club","league","league_name",
        "overall_rating","potential","contract_years_left","international_reputation",
        "market_value_m","past_rating_2yr","past_rating_1yr",
        "yellow_cards","red_cards","matches_in_squad","minutes_played","is_women",
        "goals_per90","assists_per90","shots_on_target_pct","pass_completion",
        "key_passes_per90","progressive_passes","tackles_per90","interceptions_per90",
        "dribbles_per90","aerial_duels_won_pct","duels_won_pct","progressive_carries",
        "crosses_per90","through_balls_per90","xg_per90","saves_per90",
        "clean_sheets_pct","sweeper_actions","goals_conceded_per90",
        "minutes_per90_ratio","yellow_cards_per90","red_cards_per90",
    ]
    for col in expected_cols:
        if col not in all_players.columns:
            all_players[col] = 0
    all_players = all_players[expected_cols].drop_duplicates(subset=["name","club"])

    # 6. Team clusters
    print("\nBuilding team clusters…")
    teams = build_team_clusters(all_players)

    # 7. Save
    all_players.to_csv("all_players_data.csv", index=False)
    teams.to_csv("team_clusters.csv", index=False)
    print(f"\nDone!")
    print(f"  all_players_data.csv : {len(all_players):,} players, {all_players['league'].nunique()} leagues")
    print(f"  team_clusters.csv    : {len(teams):,} teams")


if __name__ == "__main__":
    main()
