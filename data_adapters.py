"""
Data Adapters — Provider-Agnostic Data Layer
============================================
Standardises player data from multiple sources into a single DataFrame format.

Each adapter outputs the same core columns so the rest of the pipeline
doesn't care where the data came from. Mix and match freely.

Usage:
    from data_adapters import TransfermarktAdapter, FBrefAdapter, StatsBombAdapter

    # Use Transfermarkt CSV (default)
    df = TransfermarktAdapter("raw_players.csv", "raw_appearances.csv").load()

    # Augment with FBref stats (requires fbref-python: pip install fbref)
    df_fbref = FBrefAdapter(league="Premier League", season="2024-25").load()

    # Merge: TM base + FBref stats for overlapping players
    from data_adapters import merge_sources
    df_merged = merge_sources(df, df_fbref, on="name", prefer="fbref_stats")
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD SCHEMA
# Every adapter must output these columns (fill with NaN if unavailable).
# ─────────────────────────────────────────────────────────────────────────────
STANDARD_COLUMNS = [
    # Identity
    "name", "club", "league", "league_name", "nationality", "position",
    "age", "overall_rating", "potential", "market_value_m", "contract_years_left",
    "international_reputation", "is_women",
    # Standard
    "goals_per90", "assists_per90", "minutes_per90_ratio",
    "shots_per90", "shots_on_target_pct",
    "yellow_cards_per90", "red_cards_per90",
    # Shooting
    "npxg_per90", "xg_per90", "npxg_per_shot",
    # Passing
    "pass_completion", "short_pass_completion", "medium_pass_completion",
    "long_pass_completion", "key_passes_per90", "progressive_passes",
    "xa_per90", "through_balls_per90",
    # Creation
    "sca_per90", "gca_per90", "crosses_per90",
    # Defence
    "tackles_per90", "tackles_won_pct", "interceptions_per90",
    "blocks_per90", "clearances_per90", "pressures_per90",
    "pressure_success_pct", "aerial_duels_won_pct", "duels_won_pct",
    # Possession
    "dribbles_per90", "progressive_carries", "touches_per90",
    "touches_att3rd_per90", "progressive_passes_received_per90",
    # History (for progression model)
    "past_rating_1yr", "past_rating_2yr",
    # Source tracking
    "data_source",
]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing standard columns filled with NaN."""
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_COLUMNS + [c for c in df.columns if c not in STANDARD_COLUMNS]]


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DataAdapter:
    """Abstract base — subclass and implement load()."""

    source_name: str = "generic"

    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def _tag_source(self, df: pd.DataFrame) -> pd.DataFrame:
        df["data_source"] = self.source_name
        return df


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER 1 — Transfermarkt CSV  (existing pipeline output)
# ─────────────────────────────────────────────────────────────────────────────

class TransfermarktAdapter(DataAdapter):
    """
    Reads the app's native all_players_data.csv produced by build_data.ps1.
    This is the default data source — no extra dependencies.
    """
    source_name = "transfermarkt"

    def __init__(self, players_csv: str = "all_players_data.csv",
                 appearances_csv: Optional[str] = "raw_appearances.csv"):
        self.players_csv     = Path(players_csv)
        self.appearances_csv = Path(appearances_csv) if appearances_csv else None

    def load(self) -> pd.DataFrame:
        if not self.players_csv.exists():
            raise FileNotFoundError(f"Players CSV not found: {self.players_csv}")
        df = pd.read_csv(self.players_csv, low_memory=False)
        df = self._tag_source(df)
        return _ensure_columns(df)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER 2 — FBref  (via soccerdata library or direct scraping)
# Install: pip install soccerdata
# Docs: https://soccerdata.readthedocs.io
# ─────────────────────────────────────────────────────────────────────────────

class FBrefAdapter(DataAdapter):
    """
    Fetches per-90 stats from FBref using the soccerdata library.
    Requires: pip install soccerdata

    Returns a DataFrame with the same schema — merge with TransfermarktAdapter
    output to get market values + FBref stats on the same rows.
    """
    source_name = "fbref"

    # FBref league IDs used by soccerdata
    _LEAGUE_MAP = {
        "Premier League":      ("ENG-Premier League", "1"),
        "La Liga":             ("ESP-La Liga",         "1"),
        "Serie A":             ("ITA-Serie A",         "1"),
        "Bundesliga":          ("GER-Bundesliga",      "1"),
        "Ligue 1":             ("FRA-Ligue 1",         "1"),
        "Championship":        ("ENG-Championship",    "2"),
        "Primeira Liga":       ("POR-Primeira Liga",   "1"),
        "Eredivisie":          ("NED-Eredivisie",      "1"),
        "Scottish Premiership":("SCO-Scottish Premiership","1"),
    }

    def __init__(self, league: str = "Premier League", season: str = "2024-25"):
        self.league = league
        self.season = season

    def load(self) -> pd.DataFrame:
        try:
            import soccerdata as sd
        except ImportError:
            raise ImportError(
                "soccerdata not installed. Run: pip install soccerdata\n"
                "Note: requires Selenium + Chrome for FBref scraping."
            )

        league_code = self._LEAGUE_MAP.get(self.league, (self.league, "1"))[0]
        fbref       = sd.FBref(leagues=league_code, seasons=self.season)

        dfs = []
        for stat_type in ["standard", "shooting", "passing", "goal_shot_creation",
                          "defense", "possession", "misc"]:
            try:
                raw = fbref.read_player_season_stats(stat_type=stat_type)
                dfs.append(raw)
            except Exception:
                pass

        if not dfs:
            return _ensure_columns(pd.DataFrame())

        merged = dfs[0]
        for d in dfs[1:]:
            try:
                merged = merged.join(d, how="outer", rsuffix=f"_{stat_type}")
            except Exception:
                pass

        df = self._map_fbref_columns(merged.reset_index())
        df = self._tag_source(df)
        return _ensure_columns(df)

    @staticmethod
    def _map_fbref_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Map FBref column names to the standard schema."""
        rename = {
            "player":             "name",
            "squad":              "club",
            "nation":             "nationality",
            "pos":                "position",
            "age":                "age",
            "Gls":                "goals_per90",
            "Ast":                "assists_per90",
            "npxG":               "npxg_per90",
            "xAG":                "xa_per90",
            "Cmp%":               "pass_completion",
            "SCA":                "sca_per90",
            "GCA":                "gca_per90",
            "Tkl":                "tackles_per90",
            "Int":                "interceptions_per90",
            "Blocks":             "blocks_per90",
            "Press":              "pressures_per90",
            "Succ%_pressures":    "pressure_success_pct",
            "Won%_aerial":        "aerial_duels_won_pct",
            "Att 3rd_touches":    "touches_att3rd_per90",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        return df


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER 3 — StatsBomb Open Data  (free, no API key)
# Install: pip install statsbombpy socceraction
# Docs: https://github.com/statsbomb/statsbombpy
# Coverage: selected competitions only (free tier)
# ─────────────────────────────────────────────────────────────────────────────

class StatsBombAdapter(DataAdapter):
    """
    Pulls event data from StatsBomb's free open dataset via statsbombpy.
    Computes per-90 stats from raw events: xG, pass completion, pressures, etc.

    Free competitions available (as of 2025):
      competition_id=9,  season_id=281  → Bundesliga 2023/24  (default, 34 matches)
      competition_id=11, season_id=90   → La Liga 2020/21     (360° data)
      competition_id=16, season_id=4    → Champions League 2018/19
      competition_id=43, season_id=106  → FIFA World Cup 2022  (360° data)
      competition_id=7,  season_id=235  → Ligue 1 2022/23
      competition_id=2,  season_id=27   → Premier League 2015/16
      competition_id=55, season_id=282  → UEFA Euro 2024

    Use StatsBombAdapter.list_competitions() to see the full catalogue.
    For full professional coverage, a StatsBomb IQ subscription is required.

    Requires: pip install statsbombpy
    """
    source_name = "statsbomb"

    def __init__(self, competition_id: int = 9, season_id: int = 281,
                 max_matches: int = 20):
        # Default: Bundesliga 2023/24 — most recent free men's league data
        self.competition_id = competition_id
        self.season_id      = season_id
        self.max_matches    = max_matches

    @staticmethod
    def list_competitions() -> pd.DataFrame:
        """Return all freely available StatsBomb competitions as a DataFrame."""
        try:
            from statsbombpy import sb
            return sb.competitions()
        except ImportError:
            raise ImportError("statsbombpy not installed. Run: pip install statsbombpy")

    def load(self) -> pd.DataFrame:
        try:
            from statsbombpy import sb
        except ImportError:
            raise ImportError("statsbombpy not installed. Run: pip install statsbombpy")

        matches = sb.matches(competition_id=self.competition_id,
                             season_id=self.season_id)
        if matches.empty:
            return _ensure_columns(pd.DataFrame())

        all_events = []
        for mid in matches["match_id"].unique()[:self.max_matches]:
            try:
                evts = sb.events(match_id=mid)
                all_events.append(evts)
            except Exception:
                pass

        if not all_events:
            return _ensure_columns(pd.DataFrame())

        events = pd.concat(all_events, ignore_index=True)
        df = self._aggregate_to_players(events)
        df = self._tag_source(df)
        return _ensure_columns(df)

    @staticmethod
    def _aggregate_to_players(events: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw StatsBomb events to per-player per-90 stats."""
        # Minutes proxy: use fraction of total match events per player
        total_events = len(events)
        max_events   = events["player"].value_counts().max() if total_events else 1

        rows = []
        for player, grp in events.groupby("player"):
            n_events = len(grp)
            # Estimate minutes via event share (starter ≈ 90 min)
            mins_est = max(float(n_events) / max(float(max_events), 1) * 90.0, 1.0)
            p90      = 90.0 / mins_est

            passes    = grp[grp["type"] == "Pass"]
            shots     = grp[grp["type"] == "Shot"]
            pressures = grp[grp["type"] == "Pressure"]
            carries   = grp[grp["type"] == "Carry"]
            dribbles  = grp[grp["type"] == "Dribble"]
            tackles   = grp[grp["type"] == "Dribbled Past"]
            interc    = grp[grp["type"] == "Interception"]
            blocks    = grp[grp["type"] == "Block"]
            clears    = grp[grp["type"] == "Clearance"]

            # Pass completion: StatsBomb convention — NaN outcome = successful
            pass_comp = float(passes["pass_outcome"].isna().mean()) * 100.0 if len(passes) else 0.0

            # Goals & xG — shot_outcome == 'Goal', shot_statsbomb_xg
            goals_raw = int((shots["shot_outcome"] == "Goal").sum()) if "shot_outcome" in shots.columns else 0
            xg_total  = float(shots["shot_statsbomb_xg"].sum()) if "shot_statsbomb_xg" in shots.columns else 0.0

            # Dribble success rate
            drib_comp = 0.0
            if len(dribbles) and "dribble_outcome" in dribbles.columns:
                drib_comp = float((dribbles["dribble_outcome"] == "Complete").mean()) * 100.0

            rows.append({
                "name":               str(player),
                "club":               str(grp["team"].iloc[0]) if not grp.empty else "",
                "position":           str(grp["position"].mode().iloc[0]) if not grp.empty else "CM",
                "goals_per90":        round(goals_raw * p90, 3),
                "xg_per90":           round(xg_total * p90, 3),
                "npxg_per90":         round(xg_total * p90, 3),  # StatsBomb xG is already non-penalty
                "pass_completion":    round(pass_comp, 1),
                "pressures_per90":    round(len(pressures) * p90, 2),
                "tackles_per90":      round(len(tackles) * p90, 2),
                "interceptions_per90":round(len(interc) * p90, 2),
                "blocks_per90":       round(len(blocks) * p90, 2),
                "clearances_per90":   round(len(clears) * p90, 2),
                "dribbles_per90":     round(len(dribbles) * p90, 2),
                "progressive_carries":round(len(carries) * 0.3 * p90, 2),  # ~30% are progressive
                "progressive_passes": round(len(passes) * 0.2 * p90, 2),   # ~20% are progressive
                "minutes_per90_ratio":round(min(mins_est / 90.0, 1.0), 3),
            })

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER 4 — Wyscout / Opta  (stub — requires paid credentials)
# ─────────────────────────────────────────────────────────────────────────────

class WyscoutAdapter(DataAdapter):
    """
    Stub adapter for Wyscout API (paid subscription required).
    Implement _fetch_players() when credentials are available.
    """
    source_name = "wyscout"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("WYSCOUT_API_KEY", "")

    def load(self) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("Set WYSCOUT_API_KEY environment variable.")
        # Implementation placeholder — add Wyscout HTTP calls here
        return _ensure_columns(pd.DataFrame())


class OptaAdapter(DataAdapter):
    """
    Stub adapter for Opta/StatsPerform (paid subscription required).
    Implement _fetch_players() when credentials are available.
    """
    source_name = "opta"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("OPTA_API_KEY", "")

    def load(self) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("Set OPTA_API_KEY environment variable.")
        return _ensure_columns(pd.DataFrame())


# ─────────────────────────────────────────────────────────────────────────────
# MERGE UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def merge_sources(
    base: pd.DataFrame,
    supplement: pd.DataFrame,
    on: str = "name",
    prefer: str = "supplement_stats",
) -> pd.DataFrame:
    """
    Merge two adapter outputs.

    prefer="supplement_stats" — keep base identity columns (market value,
        contract, OVR), overlay supplement's per-90 stats where available.
    prefer="base" — keep all base columns, supplement fills only NaN gaps.
    """
    stat_cols = [c for c in STANDARD_COLUMNS
                 if c not in ("name","club","league","league_name","nationality",
                              "position","age","overall_rating","potential",
                              "market_value_m","contract_years_left",
                              "international_reputation","is_women","data_source")]

    merged = base.merge(supplement[[on] + stat_cols], on=on, how="left",
                        suffixes=("_base", "_supp"))

    if prefer == "supplement_stats":
        for col in stat_cols:
            if f"{col}_supp" in merged.columns:
                merged[col] = merged[f"{col}_supp"].combine_first(merged.get(f"{col}_base", pd.Series()))
                merged.drop(columns=[f"{col}_base", f"{col}_supp"], errors="ignore", inplace=True)
    else:
        for col in stat_cols:
            if f"{col}_base" in merged.columns:
                merged[col] = merged[f"{col}_base"].combine_first(merged.get(f"{col}_supp", pd.Series()))
                merged.drop(columns=[f"{col}_base", f"{col}_supp"], errors="ignore", inplace=True)

    return _ensure_columns(merged)
