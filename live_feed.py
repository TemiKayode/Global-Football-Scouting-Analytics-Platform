"""
Live Data Feed Module — Global Football Scouting Platform
Fetches recent transfers, market values, and squad updates from free public APIs.

Sources (no API key required):
  1. TheSportsDB API v3  — team/player/event lookups (free public tier)
  2. Transfermarkt unofficial JSON endpoints — transfer news, market values
  3. API-Football (RapidAPI)  — if RAPIDAPI_KEY env var is set
  4. football-data.org       — if FOOTBALL_DATA_KEY env var is set

Cache: live_cache.json (default TTL 6 h, refresh on demand)
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CACHE_FILE     = Path(__file__).parent / "live_cache.json"
CACHE_TTL      = int(os.getenv("LIVE_CACHE_TTL_HOURS", "1")) * 3600    # transfers: 1h default
FORMATION_TTL  = int(os.getenv("FORMATION_CACHE_TTL_HOURS", "2")) * 3600  # formations: 2h
TIMEOUT        = 10   # request timeout

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

RAPIDAPI_KEY       = os.getenv("RAPIDAPI_KEY", "")
FOOTBALL_DATA_KEY  = os.getenv("FOOTBALL_DATA_KEY", "")

_TSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

# Football-data.org league IDs (free tier covers these)
_FDORG_LEAGUES = {
    "GB1":  2021,   # Premier League
    "ES1":  2014,   # La Liga
    "IT1":  2019,   # Serie A
    "L1":   2002,   # Bundesliga
    "FR1":  2015,   # Ligue 1
    "GB2":  2016,   # Championship
    "PO1":  2017,   # Primeira Liga
    "SC1":  2003,   # Scottish Premiership
}

# Transfermarkt competition IDs
_TM_COMP_IDS = {
    "GB1": "GB1", "ES1": "ES1", "IT1": "IT1", "L1": "L1", "FR1": "FR1",
    "GB2": "GB2", "PO1": "PO1", "TR1": "TR1", "BE1": "BE1", "SC1": "SC1",
    "AR1N": "AR1N", "BRA1": "BRA1",
}

# TheSportsDB sport/league IDs
_TSDB_LEAGUE_IDS = {
    "GB1":  4328,   # English Premier League
    "ES1":  4335,   # La Liga
    "IT1":  4332,   # Serie A
    "L1":   4331,   # Bundesliga
    "FR1":  4334,   # Ligue 1
    "GB2":  4329,   # Championship
    "PO1":  4344,   # Primeira Liga
    "TR1":  4340,   # Süper Lig
    "SC1":  4330,   # Scottish Premiership
    "BE1":  4397,   # Belgian Pro League
    "BRA1": 4351,   # Brasileirão
    "AR1N": 4406,   # Argentine Primera
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "timestamp":   0,
        "transfers":   [],
        "values":      {},
        "squads":      {},
        "team_info":   {},   # {club_name: {manager, formation, ts}}
        "source":      "none",
    }


def _save_cache(data: dict):
    data["timestamp"] = time.time()
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    try:
        CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )
    except Exception:
        pass


def cache_age_minutes() -> float:
    cache = _load_cache()
    return (time.time() - cache.get("timestamp", 0)) / 60


def is_cache_fresh() -> bool:
    return (time.time() - _load_cache().get("timestamp", 0)) < CACHE_TTL


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: TheSportsDB  (no key, always available)
# ─────────────────────────────────────────────────────────────────────────────

def _tsdb_get(endpoint: str, params: dict = None) -> dict:
    url = f"{_TSDB_BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_tsdb_squad(league_id: int) -> list:
    """Return list of {name, team, position, nationality} from TheSportsDB."""
    data = _tsdb_get("search_all_teams.php", {"l": str(league_id)})
    teams = data.get("teams") or []
    players = []
    for team in teams[:20]:   # rate-limit: max 20 teams per call
        tid = team.get("idTeam")
        if not tid:
            continue
        pdata = _tsdb_get("lookup_all_players.php", {"id": tid})
        for p in (pdata.get("player") or []):
            players.append({
                "name":        p.get("strPlayer", ""),
                "club":        team.get("strTeam", ""),
                "position":    _map_tsdb_pos(p.get("strPosition", "")),
                "nationality": p.get("strNationality", ""),
                "age":         _tsdb_age(p.get("dateBorn", "")),
                "source":      "TheSportsDB",
            })
        time.sleep(0.2)   # polite rate limiting
    return players


def _tsdb_age(dob: str) -> int:
    try:
        born = datetime.strptime(dob[:10], "%Y-%m-%d")
        return int((datetime.now() - born).days / 365.25)
    except Exception:
        return 25


def _map_tsdb_pos(pos: str) -> str:
    pos = pos.upper().strip()
    m = {
        "GOALKEEPER": "GK", "GOALIE": "GK",
        "DEFENDER": "CB", "CENTRE BACK": "CB", "CENTER BACK": "CB",
        "LEFT BACK": "LB", "RIGHT BACK": "RB",
        "MIDFIELDER": "CM", "CENTRAL MIDFIELDER": "CM",
        "DEFENSIVE MIDFIELDER": "CDM", "ATTACKING MIDFIELDER": "CAM",
        "FORWARD": "ST", "STRIKER": "ST", "CENTRE FORWARD": "ST",
        "LEFT WINGER": "LW", "RIGHT WINGER": "RW", "WINGER": "LW",
    }
    for k, v in m.items():
        if k in pos:
            return v
    return "CM"


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: Transfermarkt recent transfers
# ─────────────────────────────────────────────────────────────────────────────

_TM_TRANSFER_URL = "https://www.transfermarkt.com/transfers/transfertagedetail/statistik"
_TM_SEARCH_URL   = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"

def _fetch_tm_transfers(competition_code: str, season: str = "2025") -> list:
    """
    Scrape Transfermarkt's public transfer list for a given competition.
    Returns list of transfer dicts.
    """
    url = f"https://www.transfermarkt.com/transfers/neuestetransfers/statistik"
    params = {"land_id": "", "wettbewerb_id": competition_code,
              "transfer_typ": "alle", "saison_id": season,
              "leihe": "", "plus": "1"}
    transfers = []
    try:
        resp = requests.get(url, params=params, headers={
            **_HEADERS, "Referer": "https://www.transfermarkt.com/"
        }, timeout=TIMEOUT)
        if resp.status_code != 200:
            return transfers

        from html.parser import HTMLParser

        class _TParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self._rows = []
                self._cur  = {}
                self._in_td = False
                self._td_class = ""
                self._col = 0
                self._buf = ""

            def handle_starttag(self, tag, attrs):
                d = dict(attrs)
                if tag == "tr" and "odd" in d.get("class","") + d.get("bgcolor",""):
                    self._cur = {}
                    self._col = 0
                elif tag == "td":
                    self._in_td = True
                    self._td_class = d.get("class","")
                    self._buf = ""

            def handle_endtag(self, tag):
                if tag == "td" and self._in_td:
                    self._in_td = False
                    txt = self._buf.strip()
                    if self._col == 0:   self._cur["name"] = txt
                    elif self._col == 1: self._cur["age"]  = _safe_int(txt, 0)
                    elif self._col == 2: self._cur["position"] = txt
                    elif self._col == 3: self._cur["from_club"] = txt
                    elif self._col == 4: self._cur["to_club"]   = txt
                    elif self._col == 5: self._cur["fee"]       = txt
                    self._col += 1
                elif tag == "tr" and self._cur.get("name"):
                    self._rows.append(dict(self._cur))
                    self._cur = {}
                    self._col = 0

            def handle_data(self, data):
                if self._in_td:
                    self._buf += data

        p = _TParser()
        p.feed(resp.text)
        for row in p._rows:
            if row.get("name") and row.get("to_club"):
                transfers.append({
                    "name":      row["name"],
                    "age":       row.get("age", 0),
                    "position":  row.get("position", ""),
                    "from_club": row.get("from_club", ""),
                    "to_club":   row.get("to_club", ""),
                    "fee":       row.get("fee", "?"),
                    "date":      datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "source":    "Transfermarkt",
                })
    except Exception:
        pass
    return transfers


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3: football-data.org (optional, free API key)
# ─────────────────────────────────────────────────────────────────────────────

_FDORG_BASE = "https://api.football-data.org/v4"


def _fdorg_get(endpoint: str) -> dict:
    if not FOOTBALL_DATA_KEY:
        return {}
    headers = {**_HEADERS, "X-Auth-Token": FOOTBALL_DATA_KEY}
    try:
        r = requests.get(f"{_FDORG_BASE}/{endpoint}", headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_fdorg_squad(league_id: int) -> list:
    data = _fdorg_get(f"competitions/{league_id}/teams")
    players = []
    for team in (data.get("teams") or []):
        for p in (team.get("squad") or []):
            players.append({
                "name":        p.get("name", ""),
                "club":        team.get("name", ""),
                "position":    _map_fdorg_pos(p.get("position", "")),
                "nationality": p.get("nationality", ""),
                "age":         _tsdb_age(p.get("dateOfBirth", "")),
                "source":      "football-data.org",
            })
    return players


def _map_fdorg_pos(pos: str) -> str:
    m = {
        "Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "LB",
        "Right-Back": "RB", "Defensive Midfield": "CDM",
        "Central Midfield": "CM", "Attacking Midfield": "CAM",
        "Left Winger": "LW", "Right Winger": "RW",
        "Centre-Forward": "ST", "Offence": "ST",
    }
    return m.get(pos, "CM")


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: API-Football via RapidAPI (optional)
# ─────────────────────────────────────────────────────────────────────────────

_RAPIDAPI_BASE = "https://api-football-v1.p.rapidapi.com/v3"
_RAPIDAPI_LEAGUE_IDS = {
    "GB1": 39, "ES1": 140, "IT1": 135, "L1": 78, "FR1": 61,
    "GB2": 40, "PO1": 94, "TR1": 203, "BE1": 144, "SC1": 179,
    "BRA1": 71, "AR1N": 128,
}


def _rapid_get(endpoint: str, params: dict) -> dict:
    if not RAPIDAPI_KEY:
        return {}
    headers = {
        **_HEADERS,
        "X-RapidAPI-Key":  RAPIDAPI_KEY,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }
    try:
        r = requests.get(f"{_RAPIDAPI_BASE}/{endpoint}",
                         params=params, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_rapidapi_transfers(league_id: int, season: int = 2025) -> list:
    data = _rapid_get("transfers", {"league": league_id, "season": season})
    transfers = []
    for item in (data.get("response") or []):
        player = item.get("player", {})
        for t in (item.get("transfers") or []):
            transfers.append({
                "name":      player.get("name", ""),
                "age":       0,
                "position":  "",
                "from_club": t.get("teams", {}).get("out", {}).get("name", ""),
                "to_club":   t.get("teams", {}).get("in",  {}).get("name", ""),
                "fee":       t.get("type", "?"),
                "date":      t.get("date", ""),
                "source":    "API-Football",
            })
    return transfers


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_feed(league_codes: list = None, force_refresh: bool = False) -> dict:
    """
    Fetch and return the live data feed.
    Returns a dict with keys:
      transfers   : list of recent transfer dicts
      squad_news  : list of player arrival/departure dicts
      last_updated: ISO timestamp string
      source      : which data source was used
      fresh       : bool — True if data was just fetched

    Uses cached data if fresh (< TTL hours old) unless force_refresh=True.
    """
    cache = _load_cache()
    if not force_refresh and is_cache_fresh():
        cache["fresh"] = False
        return cache

    if league_codes is None:
        league_codes = list(_TSDB_LEAGUE_IDS.keys())

    transfers = []
    source_used = []

    # ── SOURCE: API-Football (best structured data, needs key) ─────────────
    if RAPIDAPI_KEY:
        for lc in league_codes:
            lid = _RAPIDAPI_LEAGUE_IDS.get(lc)
            if lid:
                t = _fetch_rapidapi_transfers(lid)
                transfers.extend(t)
                if t:
                    source_used.append("API-Football")
                time.sleep(0.3)

    # ── SOURCE: football-data.org (optional key) ───────────────────────────
    if FOOTBALL_DATA_KEY and not transfers:
        for lc in league_codes:
            lid = _FDORG_LEAGUES.get(lc)
            if lid:
                players = _fetch_fdorg_squad(lid)
                # Mark all as "signed" if new vs last cache
                cached_names = {t.get("name","") for t in cache.get("transfers",[])}
                for p in players:
                    if p["name"] not in cached_names:
                        transfers.append({
                            "name":      p["name"],
                            "age":       p.get("age", 0),
                            "position":  p.get("position", ""),
                            "from_club": "",
                            "to_club":   p.get("club", ""),
                            "fee":       "?",
                            "date":      datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                            "source":    "football-data.org",
                        })
                if players:
                    source_used.append("football-data.org")
                time.sleep(0.2)

    # ── SOURCE: Transfermarkt scraping (always available) ──────────────────
    if not transfers:
        for lc in league_codes[:5]:   # limit to 5 leagues to be polite
            tm_code = _TM_COMP_IDS.get(lc)
            if tm_code:
                t = _fetch_tm_transfers(tm_code)
                transfers.extend(t)
                if t:
                    source_used.append("Transfermarkt")
                time.sleep(1.0)

    # ── SOURCE: TheSportsDB (always available, squad-level) ────────────────
    squad_news = []
    if not transfers:
        for lc in league_codes[:3]:
            lid = _TSDB_LEAGUE_IDS.get(lc)
            if lid:
                players = _fetch_tsdb_squad(lid)
                squad_news.extend(players)
                if players:
                    source_used.append("TheSportsDB")

    # Deduplicate transfers by name+to_club
    seen = set()
    unique_transfers = []
    for t in transfers:
        key = (t.get("name",""), t.get("to_club",""))
        if key not in seen:
            seen.add(key)
            unique_transfers.append(t)

    result = {
        "transfers":    unique_transfers,
        "squad_news":   squad_news,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "source":       ", ".join(set(source_used)) or "cached",
        "fresh":        True,
    }
    _save_cache(result)
    return result


def transfers_to_df(feed: dict) -> pd.DataFrame:
    """Convert the transfers list from fetch_live_feed() to a clean DataFrame."""
    rows = feed.get("transfers") or []
    if not rows:
        return pd.DataFrame(columns=["name","age","position","from_club","to_club","fee","date","source"])
    df = pd.DataFrame(rows)
    for col in ["name","position","from_club","to_club","fee","date","source"]:
        if col not in df.columns:
            df[col] = ""
    if "age" not in df.columns:
        df["age"] = 0
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%d %b %Y")
    return df[["name","age","position","from_club","to_club","fee","date","source"]].reset_index(drop=True)


def merge_transfers_into_players(players_df: pd.DataFrame,
                                 feed: dict) -> tuple:
    """
    Update players_df with club changes from the live feed.
    Players found in the transfer list with a confirmed new club
    get their club column updated.
    Returns a copy with changes applied and a new 'data_source' column.
    """
    df = players_df.copy()
    if "data_source" not in df.columns:
        df["data_source"] = "CSV"

    transfers = feed.get("transfers") or []
    updated = 0
    for t in transfers:
        pname   = str(t.get("name", "")).strip()
        to_club = str(t.get("to_club", "")).strip()
        if not pname or not to_club:
            continue
        # Fuzzy match on name (exact first, then partial)
        mask_exact = df["name"].str.lower() == pname.lower()
        if mask_exact.any():
            df.loc[mask_exact, "club"]        = to_club
            df.loc[mask_exact, "data_source"] = t.get("source", "live")
            updated += 1
        else:
            # Partial: first + last name
            parts = pname.lower().split()
            if len(parts) >= 2:
                mask_partial = (df["name"].str.lower().str.contains(parts[0], regex=False) &
                                df["name"].str.lower().str.contains(parts[-1], regex=False))
                if mask_partial.sum() == 1:
                    df.loc[mask_partial, "club"]        = to_club
                    df.loc[mask_partial, "data_source"] = t.get("source", "live")
                    updated += 1

    return df, updated


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TEAM INFO  (formations + coaches, cached with FORMATION_TTL)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_tsdb_team_info(club_name: str) -> dict:
    """Query TheSportsDB for team coach/manager (no API key required)."""
    data = _tsdb_get("searchteams.php", {"t": club_name})
    teams = data.get("teams") or []
    if not teams:
        return {}
    t = teams[0]
    return {
        "manager":   str(t.get("strManager", "") or "").strip(),
        "formation": str(t.get("strFormation", "") or "").strip(),
        "team_id":   str(t.get("idTeam", "") or ""),
        "country":   str(t.get("strCountry", "") or ""),
    }


def _fetch_rapidapi_team_formation(club_name: str) -> str:
    """
    Query API-Football: search for team → get last 5 fixtures → parse formation
    from lineup data. Returns the most-used formation string, or "" on failure.
    """
    if not RAPIDAPI_KEY:
        return ""
    try:
        team_data = _rapid_get("teams", {"search": club_name})
        team_id = 0
        for tr in (team_data.get("response") or []):
            t_name = tr.get("team", {}).get("name", "").lower().replace(" fc", "").strip()
            c_name = club_name.lower().replace(" fc", "").strip()
            if t_name == c_name or c_name in t_name or t_name in c_name:
                team_id = int(tr["team"]["id"])
                break
        if not team_id and (team_data.get("response") or []):
            team_id = int(team_data["response"][0]["team"]["id"])
        if not team_id:
            return ""

        fix_data = _rapid_get("fixtures", {"team": team_id, "last": 10, "season": 2025})
        fixture_ids = [
            f["fixture"]["id"]
            for f in (fix_data.get("response") or [])
            if f.get("fixture", {}).get("id")
        ]

        formation_counts: dict = {}
        for fid in fixture_ids[:4]:
            lu_data = _rapid_get("fixtures/lineups", {"fixture": fid})
            for lu in (lu_data.get("response") or []):
                t_info = lu.get("team", {})
                lu_name = t_info.get("name", "").lower()
                if club_name.lower() in lu_name or lu_name in club_name.lower():
                    form = lu.get("formation", "")
                    if form:
                        formation_counts[form] = formation_counts.get(form, 0) + 1
            time.sleep(0.35)

        return max(formation_counts, key=formation_counts.get) if formation_counts else ""
    except Exception:
        return ""


def get_live_team_info(club_name: str, force: bool = False) -> dict:
    """
    Return current manager and most-used formation for a club.
    Cached in live_cache.json with FORMATION_TTL (default 2 h).
    Returns dict with keys: manager, formation, source, fresh.
    """
    cache   = _load_cache()
    t_cache = cache.setdefault("team_info", {})
    now     = time.time()
    entry   = t_cache.get(club_name, {})

    if entry and (now - entry.get("ts", 0)) < FORMATION_TTL and not force:
        return {**entry, "fresh": False}

    result = {"manager": "", "formation": "", "ts": now, "source": ""}

    # TheSportsDB (no key, always available)
    try:
        tsdb = _fetch_tsdb_team_info(club_name)
        if tsdb.get("manager"):
            result["manager"]  = tsdb["manager"]
            result["source"]   = "TheSportsDB"
        if tsdb.get("formation"):
            result["formation"] = tsdb["formation"]
    except Exception:
        pass

    # API-Football for accurate recent formation (needs key)
    if RAPIDAPI_KEY and not result["formation"]:
        try:
            form = _fetch_rapidapi_team_formation(club_name)
            if form:
                result["formation"] = form
                result["source"]    = (result["source"] + "+API-Football").lstrip("+")
        except Exception:
            pass

    t_cache[club_name] = result
    cache["team_info"] = t_cache
    _save_cache(cache)
    return {**result, "fresh": True}


def get_cached_formations() -> dict:
    """Return {club_name: formation_string} from cache (no API calls)."""
    t_info = _load_cache().get("team_info", {})
    return {k: v["formation"] for k, v in t_info.items() if v.get("formation")}


def get_cached_coaches() -> dict:
    """Return {club_name: manager_name} from cache (no API calls)."""
    t_info = _load_cache().get("team_info", {})
    return {k: v["manager"] for k, v in t_info.items() if v.get("manager")}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_int(s: str, default: int = 0) -> int:
    try:
        return int(re.sub(r"[^\d]", "", s))
    except Exception:
        return default


def status_summary(feed: dict) -> str:
    """Return a one-line status string for the sidebar."""
    age_min = round(cache_age_minutes(), 1)
    n = len(feed.get("transfers") or [])
    src = feed.get("source", "unknown")
    ts  = feed.get("last_updated", "")[:16].replace("T", " ")
    return f"{n} transfers | {src} | updated {ts} UTC ({age_min} min ago)"
