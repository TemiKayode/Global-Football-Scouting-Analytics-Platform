"""
Global Football Scouting & Analytics Platform v2.0
36 leagues · Men's & Women's · FBref-style stats: Standard, Shooting, Passing (distance splits),
Goal/Shot Creation, Defensive Actions, Possession, Miscellaneous.
Formation-aware · Team-goal-driven · Progression/Regression modelling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings, os, random

from squad_analytics import (
    league_tier, estimate_budget, squad_churn_score,
    classify_age_profile, diagnose_weaknesses, estimate_success_probability,
    LEAGUE_TIERS, TIER_LABELS,
)
from recommendation_engine import (
    dev_phase, transfer_window_advice, resale_projection,
    churn_impact as calc_churn_impact, fc_development_score,
)
from transfer_planner import plan_windows, strategy_summary
from live_feed import (fetch_live_feed, transfers_to_df,
                       merge_transfers_into_players, status_summary,
                       is_cache_fresh, cache_age_minutes,
                       get_live_team_info, get_cached_formations, get_cached_coaches)

warnings.filterwarnings("ignore")

# ── Streamlit Cloud secrets → env vars bridge ────────────────────────────────
# On Streamlit Community Cloud, set secrets via the dashboard (Settings → Secrets).
# This block makes them available to os.getenv() throughout the app.
try:
    for _k in ("ANTHROPIC_API_KEY", "RAPIDAPI_KEY",
                "LIVE_CACHE_TTL_HOURS", "FORMATION_CACHE_TTL_HOURS"):
        if _k not in os.environ and hasattr(st, "secrets") and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# LEAGUES  (corrected labels — underlying TM data already has correct clubs)
# ─────────────────────────────────────────────────────────────────────────────
LEAGUES = {
    "A1":   "Austrian Bundesliga",
    "AR1N": "Argentine Primera División",
    "BE1":  "Belgian Pro League",
    "BRA1": "Brasileirão Série A",
    "BU1":  "Bulgarian First League",
    "C1":   "Swiss Super League",           # was mislabelled "Czech First League"
    "ES1":  "La Liga",
    "ES2":  "Segunda División",
    "FR1":  "Ligue 1",
    "FR2":  "Ligue 2",
    "GB1":  "Premier League",
    "GB2":  "Championship",
    "GR1":  "Super League Greece",
    "IT1":  "Serie A",
    "IT2":  "Serie B",
    "KR1":  "Croatian HNL",                # was mislabelled "K League 1"
    "L1":   "Bundesliga",
    "L2":   "2. Bundesliga",
    "NO1":  "Eliteserien",
    "PL1":  "PKO BP Ekstraklasa",
    "PO1":  "Primeira Liga",
    "RO1":  "Superliga Romania",
    "RU1":  "Russian Premier League",
    "SC1":  "Scottish Premiership",
    "SE1":  "Allsvenskan",
    "SL1":  "Slovenian PrvaLiga",           # was mislabelled "Super League Switzerland"
    "TR1":  "Süper Lig",
    "TS1":  "Czech Fortuna Liga",           # was mislabelled "Tunisian Ligue Pro"
    "UNG1": "OTP Bank Liga (Hungary)",
    # Women's
    "WGBL": "Frauen-Bundesliga",
    "WWSL": "WSL",
    "WFRD1":"D1 Féminine",
    "WNWSL":"NWSL",
    "WAUS": "A-League Women",
    "WBRA": "Brazilian Women's Série A",
    "WITA": "Women's Serie A",
}

POSITIONS = ["GK","CB","LB","RB","CDM","CM","CAM","LW","RW","ST"]

# ─────────────────────────────────────────────────────────────────────────────
# FORMATIONS
# ─────────────────────────────────────────────────────────────────────────────
FORMATIONS = {
    "4-3-3":   {"GK":1,"CB":2,"LB":1,"RB":1,"CDM":1,"CM":2,"LW":1,"ST":1,"RW":1},
    "4-2-3-1": {"GK":1,"CB":2,"LB":1,"RB":1,"CDM":2,"LW":1,"CAM":1,"RW":1,"ST":1},
    "4-4-2":   {"GK":1,"CB":2,"LB":1,"RB":1,"CM":2,"LW":1,"RW":1,"ST":2},
    "3-5-2":   {"GK":1,"CB":3,"LB":1,"RB":1,"CDM":1,"CM":2,"ST":2},
    "3-4-3":   {"GK":1,"CB":3,"LB":1,"RB":1,"CM":2,"LW":1,"ST":1,"RW":1},
    "3-4-2-1": {"GK":1,"CB":3,"LB":1,"RB":1,"CDM":1,"CM":1,"CAM":2,"ST":1},
    "5-3-2":   {"GK":1,"CB":3,"LB":1,"RB":1,"CM":2,"CDM":1,"ST":2},
    "4-3-1-2": {"GK":1,"CB":2,"LB":1,"RB":1,"CDM":1,"CM":2,"CAM":1,"ST":2},
    "4-1-2-3": {"GK":1,"CB":2,"LB":1,"RB":1,"CDM":1,"CM":2,"LW":1,"ST":1,"RW":1},
    "4-4-1-1": {"GK":1,"CB":2,"LB":1,"RB":1,"CM":4,"CAM":1,"ST":1},
}

FORMATION_LABELS = list(FORMATIONS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# CLUB NAME CANONICALISATION
# Transfermarkt uses full official names; our dicts use short names.
# _canon_club() resolves "Arsenal Football Club" → "Arsenal", etc.
# ─────────────────────────────────────────────────────────────────────────────
_CLUB_ALIASES: dict = {
    # ── Premier League ──────────────────────────────────────────────────────
    "Arsenal FC":                            "Arsenal",
    "Arsenal Football Club":                 "Arsenal",
    "Liverpool FC":                          "Liverpool",
    "Liverpool Football Club":               "Liverpool",
    "Manchester United Football Club":       "Manchester United",
    "Manchester United FC":                  "Manchester United",
    "Manchester City Football Club":         "Manchester City",
    "Manchester City FC":                    "Manchester City",
    "Chelsea FC":                            "Chelsea",
    "Chelsea Football Club":                 "Chelsea",
    "Newcastle United Football Club":        "Newcastle United",
    "Newcastle United FC":                   "Newcastle United",
    "Tottenham Hotspur":                     "Tottenham",
    "Tottenham Hotspur FC":                  "Tottenham",
    "Tottenham Hotspur Football Club":       "Tottenham",
    "Aston Villa Football Club":             "Aston Villa",
    "Aston Villa FC":                        "Aston Villa",
    "Brighton & Hove Albion":                "Brighton",
    "Brighton & Hove Albion FC":             "Brighton",
    "Brighton & Hove Albion Football Club":  "Brighton",
    "Nottingham Forest Football Club":       "Nottingham Forest",
    "Nottingham Forest FC":                  "Nottingham Forest",
    "Brentford FC":                          "Brentford",
    "Fulham FC":                             "Fulham",
    "Crystal Palace FC":                     "Crystal Palace",
    "Crystal Palace Football Club":          "Crystal Palace",
    "AFC Bournemouth":                       "Bournemouth",
    "Bournemouth FC":                        "Bournemouth",
    "Everton FC":                            "Everton",
    "Everton Football Club":                 "Everton",
    "Wolverhampton Wanderers":               "Wolves",
    "Wolverhampton Wanderers FC":            "Wolves",
    "Wolverhampton Wanderers Football Club": "Wolves",
    "West Ham United":                       "West Ham",
    "West Ham United FC":                    "West Ham",
    "West Ham United Football Club":         "West Ham",
    "Leicester City FC":                     "Leicester City",
    "Leicester City Football Club":          "Leicester City",
    "Southampton FC":                        "Southampton",
    "Southampton Football Club":             "Southampton",
    "Ipswich Town FC":                       "Ipswich Town",
    "Ipswich Town Football Club":            "Ipswich Town",
    # ── Championship ────────────────────────────────────────────────────────
    "Leeds United FC":                       "Leeds United",
    "Leeds United AFC":                      "Leeds United",
    "Leeds United Football Club":            "Leeds United",
    "Sunderland AFC":                        "Sunderland",
    "Sheffield United FC":                   "Sheffield United",
    "Sheffield United Football Club":        "Sheffield United",
    "Burnley FC":                            "Burnley",
    "Watford FC":                            "Watford",
    "Norwich City FC":                       "Norwich City",
    "Norwich City Football Club":            "Norwich City",
    "West Bromwich Albion":                  "West Brom",
    "West Bromwich Albion FC":               "West Brom",
    "Swansea City AFC":                      "Swansea",
    "Swansea City":                          "Swansea",
    "Middlesbrough FC":                      "Middlesbrough",
    "Queens Park Rangers":                   "QPR",
    "Queens Park Rangers FC":                "QPR",
    # ── La Liga ─────────────────────────────────────────────────────────────
    "Real Madrid CF":                        "Real Madrid",
    "FC Barcelona":                          "Barcelona",
    "F.C. Barcelona":                        "Barcelona",
    "Club Atlético de Madrid":               "Atletico Madrid",
    "Atlético de Madrid":                    "Atletico Madrid",
    "Atlético Madrid":                       "Atletico Madrid",
    "Real Sociedad de Fútbol":               "Real Sociedad",
    "Real Sociedad de Futbol":               "Real Sociedad",
    "Athletic Club":                         "Athletic Bilbao",
    "Athletic Club de Bilbao":               "Athletic Bilbao",
    "Villarreal CF":                         "Villarreal",
    "Real Betis Balompié":                   "Real Betis",
    "Real Betis Balompie":                   "Real Betis",
    "Sevilla FC":                            "Sevilla",
    "CA Osasuna":                            "Osasuna",
    "Valencia CF":                           "Valencia",
    "Girona FC":                             "Girona",
    # ── Bundesliga ──────────────────────────────────────────────────────────
    "FC Bayern München":                     "Bayern Munich",
    "FC Bayern Munich":                      "Bayern Munich",
    "Bayern München":                        "Bayern Munich",
    "Bayer 04 Leverkusen":                   "Bayer Leverkusen",
    "Sport-Club Freiburg":                   "Freiburg",
    "SC Freiburg":                           "Freiburg",
    "1. FC Union Berlin":                    "Union Berlin",
    "FC Union Berlin":                       "Union Berlin",
    "VfL Wolfsburg":                         "Wolfsburg",
    "1. FSV Mainz 05":                       "Mainz",
    "FSV Mainz 05":                          "Mainz",
    "TSG 1899 Hoffenheim":                   "Hoffenheim",
    "TSG Hoffenheim":                        "Hoffenheim",
    "VfB Stuttgart":                         "Stuttgart",
    "Borussia Mönchengladbach":              "Borussia Monchengladbach",
    "1. FC Heidenheim 1846":                 "Heidenheim",
    "SV Werder Bremen":                      "Werder Bremen",
    "FC Augsburg":                           "Augsburg",
    "Hamburger SV":                          "Hamburger SV",
    # ── Serie A ─────────────────────────────────────────────────────────────
    "FC Internazionale Milano":              "Inter Milan",
    "Internazionale":                        "Inter Milan",
    "FC Internazionale":                     "Inter Milan",
    "Juventus FC":                           "Juventus",
    "SSC Napoli":                            "Napoli",
    "AS Roma":                               "Roma",
    "SS Lazio":                              "Lazio",
    "ACF Fiorentina":                        "Fiorentina",
    "Atalanta BC":                           "Atalanta",
    "Torino FC":                             "Torino",
    "Bologna FC 1909":                       "Bologna",
    "Bologna FC":                            "Bologna",
    "Udinese Calcio":                        "Udinese",
    "Cagliari Calcio":                       "Cagliari",
    "Genoa CFC":                             "Genoa",
    "UC Sampdoria":                          "Sampdoria",
    "Parma Calcio 1913":                     "Parma",
    "Como 1907":                             "Como",
    # ── Ligue 1 ─────────────────────────────────────────────────────────────
    "Paris Saint-Germain FC":               "Paris Saint-Germain",
    "Paris Saint-Germain Football Club":    "Paris Saint-Germain",
    "Olympique de Marseille":               "Marseille",
    "AS Monaco FC":                          "Monaco",
    "AS Monaco":                             "Monaco",
    "Olympique Lyonnais":                    "Lyon",
    "RC Lens":                               "Lens",
    "LOSC Lille":                            "Lille",
    "Stade Rennais FC":                      "Rennes",
    "OGC Nice":                              "Nice",
    "Montpellier HSC":                       "Montpellier",
    "Stade de Reims":                        "Reims",
    "RC Strasbourg Alsace":                  "Strasbourg",
    "FC Metz":                               "Metz",
    "SM Caen":                               "Caen",
    # ── Primeira Liga ────────────────────────────────────────────────────────
    "SL Benfica":                            "Benfica",
    "Sport Lisboa e Benfica":               "Benfica",
    "FC Porto":                              "Porto",
    "Sporting Clube de Portugal":            "Sporting CP",
    "Sporting CP":                           "Sporting CP",
    "SC Braga":                              "Braga",
    "Vitória SC":                            "Vitoria Guimaraes",
    "Vitória de Guimarães":                  "Vitoria Guimaraes",
    "Vitoria SC":                            "Vitoria Guimaraes",
    "Boavista FC":                           "Boavista",
    "Rio Ave FC":                            "Rio Ave",
    # ── Süper Lig ────────────────────────────────────────────────────────────
    "Galatasaray SK":                        "Galatasaray",
    "Galatasaray A.Ş.":                      "Galatasaray",
    "Fenerbahçe SK":                         "Fenerbahce",
    "Fenerbahce SK":                         "Fenerbahce",
    "Beşiktaş JK":                           "Besiktas",
    "Besiktas JK":                           "Besiktas",
    "Trabzonspor AŞ":                        "Trabzonspor",
    "İstanbul Başakşehir FK":               "Basaksehir",
    "Istanbul Basaksehir FK":               "Basaksehir",
    "Adana Demirspor AŞ":                   "Adana Demirspor",
    "Sivasspor AŞ":                          "Sivasspor",
    # ── Scottish Premiership ─────────────────────────────────────────────────
    "Celtic FC":                             "Celtic",
    "Rangers FC":                            "Rangers",
    "Heart of Midlothian FC":               "Hearts",
    "Heart of Midlothian":                  "Hearts",
    "Hibernian FC":                          "Hibs",
    "Hibernian":                             "Hibs",
    "Aberdeen FC":                           "Aberdeen",
    # ── Brasileirão ──────────────────────────────────────────────────────────
    "Clube de Regatas do Flamengo":         "Flamengo",
    "Sociedade Esportiva Palmeiras":        "Palmeiras",
    "Clube Atlético Mineiro":               "Atletico Mineiro",
    "Fluminense FC":                         "Fluminense",
    "Sport Club Internacional":             "Internacional",
    "Grêmio FBPA":                           "Gremio",
    "Gremio FBPA":                           "Gremio",
    "Santos FC":                             "Santos",
    "São Paulo FC":                          "Sao Paulo",
    "Sao Paulo FC":                          "Sao Paulo",
    "Sport Club Corinthians Paulista":      "Corinthians",
    "Cruzeiro EC":                           "Cruzeiro",
    # ── Argentine Primera División ────────────────────────────────────────────
    "Club Atlético River Plate":            "River Plate",
    "Club Atlético Boca Juniors":           "Boca Juniors",
    "Racing Club":                           "Racing Club",
    "Club Atlético Independiente":          "Independiente",
    "San Lorenzo de Almagro":               "San Lorenzo",
    "Estudiantes de La Plata":              "Estudiantes",
    "Club Atlético Lanús":                  "Lanus",
    # ── Belgian Pro League ────────────────────────────────────────────────────
    "Club Brugge KV":                        "Club Brugge",
    "RSC Anderlecht":                        "Anderlecht",
    "KAA Gent":                              "Gent",
    "Standard de Liège":                     "Standard Liege",
    "Standard de Liege":                     "Standard Liege",
    "Royal Antwerp FC":                      "Antwerp",
    "KRC Genk":                              "Genk",
    "Royale Union Saint-Gilloise":          "Union SG",
    "Union Saint-Gilloise":                 "Union SG",
    # ── Eliteserien ──────────────────────────────────────────────────────────
    "FK Bodø/Glimt":                         "Bodo/Glimt",
    "FK Bodo/Glimt":                         "Bodo/Glimt",
    "Molde FK":                              "Molde",
    "Rosenborg BK":                          "Rosenborg",
    "Viking FK":                             "Viking",
    "SK Brann":                              "Brann",
    # ── Allsvenskan ──────────────────────────────────────────────────────────
    "Malmö FF":                              "Malmo FF",
    "Malmo FF":                              "Malmo FF",
    "IFK Göteborg":                          "IFK Goteborg",
    "Djurgårdens IF":                        "Djurgardens IF",
    "AIK Fotboll":                           "AIK",
    # ── Austrian Bundesliga ──────────────────────────────────────────────────
    "FC Red Bull Salzburg":                  "Red Bull Salzburg",
    "Red Bull Salzburg":                     "Red Bull Salzburg",
    "SK Sturm Graz":                         "Sturm Graz",
    "SK Rapid Wien":                         "Rapid Wien",
    "LASK Linz":                             "LASK",
    "FK Austria Wien":                       "Austria Wien",
    # ── Swiss Super League ───────────────────────────────────────────────────
    "BSC Young Boys":                        "Berner Sport Club Young Boys",
    "Young Boys":                            "Berner Sport Club Young Boys",
    "FC Basel":                              "FC Basel 1893",
    "FC Servette":                           "FC Servette",
    "FC Zürich":                             "FC Zurich",
    "FC Zurich":                             "FC Zurich",
    "FC Lugano":                             "FC Lugano",
    # ── Slovenian PrvaLiga ───────────────────────────────────────────────────
    "NK Olimpija":                           "NK Olimpija Ljubljana",
    # ── Women's leagues ──────────────────────────────────────────────────────
    "Arsenal WFC":                           "Arsenal W",
    "Arsenal Ladies FC":                     "Arsenal W",
    "Chelsea FCW":                           "Chelsea W",
    "Chelsea FC Women":                      "Chelsea W",
    "Manchester City WFC":                   "Manchester City W",
    "Manchester City Women FC":              "Manchester City W",
    "Manchester United WFC":                 "Manchester United W",
    "Liverpool WFC":                         "Liverpool W",
    "Aston Villa LFC":                       "Aston Villa W",
    "Olympique Lyonnais Féminin":           "Olympique Lyonnais W",
    "Paris Saint-Germain Féminines":        "Paris Saint-Germain W",
    "FC Bayern München (Frauen)":           "Bayern Munich W",
    "VfL Wolfsburg (Frauen)":               "Wolfsburg W",
}


def _canon_club(name: str) -> str:
    """
    Return the canonical short name used in KNOWN_FORMATIONS / KNOWN_PLAY_STYLES
    / KNOWN_CONTEXTS. Resolves full Transfermarkt names to our short-name keys.
    """
    if not name:
        return name
    # 1. Direct alias lookup
    if name in _CLUB_ALIASES:
        return _CLUB_ALIASES[name]
    # 2. Strip common suffixes
    for suffix in (" Football Club", " F.C.", " AFC", " FC", " CF", " BC", " JK", " SK", " FK", " AŞ", " AS"):
        if name.endswith(suffix):
            short = name[:-len(suffix)].strip()
            return _CLUB_ALIASES.get(short, short)
    # 3. Strip common prefixes (FC X, AS X, etc.)
    for prefix in ("FC ", "AFC ", "CF ", "AC ", "AS ", "SS ", "RC ", "SC ", "SK ", "NK ", "FK ",
                   "GNK ", "HNK ", "1. FC ", "SSC ", "ACF ", "VfL ", "VfB ", "TSG "):
        if name.startswith(prefix):
            short = name[len(prefix):].strip()
            return _CLUB_ALIASES.get(short, short)
    return name


# ─────────────────────────────────────────────────────────────────────────────
# TEAM GOALS
# ─────────────────────────────────────────────────────────────────────────────
TEAM_GOALS = [
    "Win the League Title",
    "Top 4 / Champions League",
    "Top Half / Europa League",
    "Mid-Table Stability",
    "Avoid Relegation",
    "Secure Promotion",
    "Consolidate After Promotion",
    "Develop Youth (U23)",
    "Maximize Transfer Revenue",
]

# ─────────────────────────────────────────────────────────────────────────────
# SEASON CONTEXT  — last-season status drives objective suggestion + budget
# ─────────────────────────────────────────────────────────────────────────────
SEASON_STATUSES = [
    "Title Winners",
    "Title Challengers (2nd/3rd)",
    "Top 4 / Champions League",
    "Europa League Qualification",
    "Mid-Table",
    "Survived Relegation Battle",
    "Relegated — Seeking Promotion",
    "Newly Promoted — Consolidating",
]

STATUS_TO_OBJECTIVE: dict = {
    "Title Winners":                  "Win the League Title",
    "Title Challengers (2nd/3rd)":    "Win the League Title",
    "Top 4 / Champions League":       "Top 4 / Champions League",
    "Europa League Qualification":    "Top Half / Europa League",
    "Mid-Table":                      "Mid-Table Stability",
    "Survived Relegation Battle":     "Avoid Relegation",
    "Relegated — Seeking Promotion":  "Secure Promotion",
    "Newly Promoted — Consolidating": "Consolidate After Promotion",
}

STATUS_NOTES: dict = {
    "Title Winners":
        "Defending champions — evolve, don't revolve. Max 2 elite signings. "
        "Research shows champions average 27.5% churn for a reason.",
    "Title Challengers (2nd/3rd)":
        "Narrow gap to title. 1-2 targeted elite upgrades + depth for 50-game seasons.",
    "Top 4 / Champions League":
        "CL secured. Need versatile depth for the European campaign and rotation.",
    "Europa League Qualification":
        "EL adds 15+ fixtures — versatility, fitness, and squad depth are the key differentiators.",
    "Mid-Table":
        "Stable platform. Targeted upgrades only; avoid destabilising a settled group.",
    "Survived Relegation Battle":
        "Narrow escape. Reinforce weak positions quickly. Avoid another slow start next season.",
    "Relegated — Seeking Promotion":
        "Parachute payments (~£40-50M Year 1) give atypical budget for the lower division. "
        "Target proven performers at this level + talent comfortable stepping down. "
        "Research: promoted clubs average 6-8 signings; ~50% squad churn is typical.",
    "Newly Promoted — Consolidating":
        "First season in the top flight is the hardest. 6-8 signings needed. "
        "Prioritise experienced players with top-flight pedigree, physicality, and set-piece ability. "
        "Promoted clubs typically need 15-20 more points than their promotion form suggests.",
}

# Multiplier applied to the base tier budget when status is known
STATUS_BUDGET_MULT: dict = {
    "Title Winners":                  1.30,
    "Title Challengers (2nd/3rd)":    1.20,
    "Top 4 / Champions League":       1.10,
    "Europa League Qualification":    1.00,
    "Mid-Table":                      0.90,
    "Survived Relegation Battle":     0.75,
    "Relegated — Seeking Promotion":  2.40,   # parachute payments change the maths
    "Newly Promoted — Consolidating": 0.70,
}

# STATUS_COLORS for UI badges
STATUS_COLORS: dict = {
    "Title Winners":                  "#1565C0",
    "Title Challengers (2nd/3rd)":    "#1976D2",
    "Top 4 / Champions League":       "#2E7D32",
    "Europa League Qualification":    "#00838F",
    "Mid-Table":                      "#546E7A",
    "Survived Relegation Battle":     "#E65100",
    "Relegated — Seeking Promotion":  "#C62828",
    "Newly Promoted — Consolidating": "#6A1B9A",
}

# 2024-25 season context for known clubs (auto-fills sidebar)
KNOWN_CONTEXTS: dict = {
    # ── Premier League ───────────────────────────────────────────────────────
    "Arsenal":               "Title Winners",
    "Liverpool":             "Title Challengers (2nd/3rd)",
    "Manchester City":       "Title Challengers (2nd/3rd)",
    "Chelsea":               "Top 4 / Champions League",
    "Newcastle United":      "Top 4 / Champions League",
    "Aston Villa":           "Top 4 / Champions League",
    "Manchester United":     "Europa League Qualification",
    "Tottenham":             "Mid-Table",
    "Brighton":              "Europa League Qualification",
    "Nottingham Forest":     "Europa League Qualification",
    "Brentford":             "Mid-Table",
    "Fulham":                "Mid-Table",
    "Crystal Palace":        "Mid-Table",
    "Bournemouth":           "Mid-Table",
    "Everton":               "Survived Relegation Battle",
    "Wolves":                "Survived Relegation Battle",
    "West Ham":              "Relegated — Seeking Promotion",
    "Leicester City":        "Relegated — Seeking Promotion",
    "Southampton":           "Relegated — Seeking Promotion",
    "Ipswich Town":          "Relegated — Seeking Promotion",
    # ── Championship ────────────────────────────────────────────────────────
    "Leeds United":          "Newly Promoted — Consolidating",
    "Sunderland":            "Newly Promoted — Consolidating",
    "Middlesbrough":         "Mid-Table",
    "Burnley":               "Relegated — Seeking Promotion",
    "Sheffield United":      "Relegated — Seeking Promotion",
    "Watford":               "Mid-Table",
    "QPR":                   "Mid-Table",
    "Norwich City":          "Mid-Table",
    "West Brom":             "Mid-Table",
    "Swansea":               "Mid-Table",
    # ── La Liga ─────────────────────────────────────────────────────────────
    "Real Madrid":           "Title Winners",
    "Barcelona":             "Title Challengers (2nd/3rd)",
    "Atletico Madrid":       "Title Challengers (2nd/3rd)",
    "Real Sociedad":         "Top 4 / Champions League",
    "Athletic Bilbao":       "Europa League Qualification",
    "Villarreal":            "Europa League Qualification",
    "Real Betis":            "Europa League Qualification",
    "Sevilla":               "Mid-Table",
    "Osasuna":               "Mid-Table",
    "Valencia":              "Survived Relegation Battle",
    "Girona":                "Top 4 / Champions League",
    # ── Bundesliga ──────────────────────────────────────────────────────────
    "Bayern Munich":         "Title Winners",
    "Bayer Leverkusen":      "Title Challengers (2nd/3rd)",
    "Borussia Dortmund":     "Top 4 / Champions League",
    "RB Leipzig":            "Top 4 / Champions League",
    "Eintracht Frankfurt":   "Europa League Qualification",
    "Freiburg":              "Europa League Qualification",
    "Union Berlin":          "Mid-Table",
    "Wolfsburg":             "Mid-Table",
    "Mainz":                 "Mid-Table",
    "Hoffenheim":            "Mid-Table",
    # ── Serie A ─────────────────────────────────────────────────────────────
    "Inter Milan":           "Title Winners",
    "Juventus":              "Title Challengers (2nd/3rd)",
    "AC Milan":              "Top 4 / Champions League",
    "Atalanta":              "Top 4 / Champions League",
    "Bologna":               "Top 4 / Champions League",
    "Roma":                  "Europa League Qualification",
    "Lazio":                 "Europa League Qualification",
    "Fiorentina":            "Europa League Qualification",
    "Napoli":                "Mid-Table",
    "Torino":                "Mid-Table",
    # ── Ligue 1 ─────────────────────────────────────────────────────────────
    "Paris Saint-Germain":   "Title Winners",
    "Marseille":             "Title Challengers (2nd/3rd)",
    "Monaco":                "Top 4 / Champions League",
    "Lille":                 "Title Challengers (2nd/3rd)",
    "Lens":                  "Europa League Qualification",
    "Rennes":                "Europa League Qualification",
    "Nice":                  "Mid-Table",
    "Reims":                 "Mid-Table",
    "Montpellier":           "Survived Relegation Battle",
    # ── Primeira Liga ────────────────────────────────────────────────────────
    "Benfica":               "Title Winners",
    "Porto":                 "Title Challengers (2nd/3rd)",
    "Sporting CP":           "Top 4 / Champions League",
    "Braga":                 "Europa League Qualification",
    "Vitoria Guimaraes":     "Mid-Table",
    "Boavista":              "Mid-Table",
    # ── Süper Lig ────────────────────────────────────────────────────────────
    "Galatasaray":           "Title Winners",
    "Fenerbahce":            "Title Challengers (2nd/3rd)",
    "Besiktas":              "Mid-Table",
    "Trabzonspor":           "Europa League Qualification",
    "Basaksehir":            "Mid-Table",
    # ── Scottish Premiership ─────────────────────────────────────────────────
    "Celtic":                "Title Winners",
    "Rangers":               "Title Challengers (2nd/3rd)",
    "Hearts":                "Top 4 / Champions League",
    "Aberdeen":              "Europa League Qualification",
    "Hibs":                  "Mid-Table",
    # ── Brasileirão ──────────────────────────────────────────────────────────
    "Flamengo":              "Title Winners",
    "Palmeiras":             "Title Challengers (2nd/3rd)",
    "Atletico Mineiro":      "Top 4 / Champions League",
    "Fluminense":            "Mid-Table",
    "Internacional":         "Mid-Table",
    "Gremio":                "Survived Relegation Battle",
    "Santos":                "Relegated — Seeking Promotion",
    "Sao Paulo":             "Mid-Table",
    "Corinthians":           "Survived Relegation Battle",
    "Cruzeiro":              "Mid-Table",
    # ── Argentine Primera División ────────────────────────────────────────────
    "River Plate":           "Title Winners",
    "Boca Juniors":          "Title Challengers (2nd/3rd)",
    "Racing Club":           "Top 4 / Champions League",
    "Independiente":         "Mid-Table",
    "San Lorenzo":           "Mid-Table",
    "Estudiantes":           "Europa League Qualification",
    "Lanus":                 "Mid-Table",
    # ── Belgian Pro League ────────────────────────────────────────────────────
    "Club Brugge":           "Title Winners",
    "Anderlecht":            "Title Challengers (2nd/3rd)",
    "Gent":                  "Top 4 / Champions League",
    "Union SG":              "Top 4 / Champions League",
    "Antwerp":               "Europa League Qualification",
    "Genk":                  "Europa League Qualification",
    "Standard Liege":        "Mid-Table",
    # ── Eliteserien ──────────────────────────────────────────────────────────
    "Bodo/Glimt":            "Title Winners",
    "Molde":                 "Title Challengers (2nd/3rd)",
    "Rosenborg":             "Top 4 / Champions League",
    "Viking":                "Europa League Qualification",
    "Brann":                 "Mid-Table",
    # ── Allsvenskan ──────────────────────────────────────────────────────────
    "Malmo FF":              "Title Winners",
    "IFK Goteborg":          "Title Challengers (2nd/3rd)",
    "Djurgardens IF":        "Top 4 / Champions League",
    "AIK":                   "Europa League Qualification",
    "Hammarby IF":           "Mid-Table",
    # ── PKO BP Ekstraklasa ───────────────────────────────────────────────────
    "Lech Poznan":           "Title Winners",
    "Jagiellonia":           "Title Challengers (2nd/3rd)",
    "Lechia Gdansk":         "Mid-Table",
    "Cracovia":              "Mid-Table",
    "Pogon Szczecin":        "Europa League Qualification",
    # ── Romanian Superliga ───────────────────────────────────────────────────
    "CFR Cluj":              "Title Winners",
    "FCSB":                  "Title Challengers (2nd/3rd)",
    "Universitatea Craiova": "Top 4 / Champions League",
    "Rapid Bucharest":       "Europa League Qualification",
    # ── Austrian Bundesliga ──────────────────────────────────────────────────
    "Red Bull Salzburg":     "Title Winners",
    "Sturm Graz":            "Title Challengers (2nd/3rd)",
    "Rapid Wien":            "Top 4 / Champions League",
    "LASK":                  "Europa League Qualification",
    "Austria Wien":          "Mid-Table",
    # ── Swiss Super League ───────────────────────────────────────────────────
    "Berner Sport Club Young Boys": "Title Winners",
    "FC Basel 1893":         "Title Challengers (2nd/3rd)",
    "FC Servette":           "Top 4 / Champions League",
    "FC Zurich":             "Europa League Qualification",
    "FC Lugano":             "Mid-Table",
    # ── Greek Super League ────────────────────────────────────────────────────
    "Olympiacos":            "Title Winners",
    "Panathinaikos":         "Title Challengers (2nd/3rd)",
    "PAOK":                  "Top 4 / Champions League",
    "AEK Athens":            "Europa League Qualification",
    "Aris":                  "Mid-Table",
    # ── Croatian HNL ─────────────────────────────────────────────────────────
    "GNK Dinamo Zagreb":     "Title Winners",
    "HNK Hajduk Split":      "Title Challengers (2nd/3rd)",
    "HNK Rijeka":            "Top 4 / Champions League",
    # ── Czech Fortuna Liga ────────────────────────────────────────────────────
    "SK Slavia Prague":      "Title Winners",
    "AC Sparta Prague":      "Title Challengers (2nd/3rd)",
    "FC Viktoria Plzen":     "Top 4 / Champions League",
    # ── Slovenian PrvaLiga ───────────────────────────────────────────────────
    "NK Olimpija Ljubljana": "Title Winners",
    "NK Maribor":            "Title Challengers (2nd/3rd)",
    "NK Celje":              "Top 4 / Champions League",
    # ── Bulgarian First League ───────────────────────────────────────────────
    "Ludogorets":            "Title Winners",
    "CSKA Sofia":            "Title Challengers (2nd/3rd)",
    "Levski Sofia":          "Top 4 / Champions League",
    # ── OTP Bank Liga (Hungary) ──────────────────────────────────────────────
    "Ferencvaros":           "Title Winners",
    "MOL Fehervar":          "Title Challengers (2nd/3rd)",
    "Paks":                  "Top 4 / Champions League",
    # ── Russian Premier League ───────────────────────────────────────────────
    "Zenit":                 "Title Winners",
    "CSKA Moscow":           "Title Challengers (2nd/3rd)",
    "Spartak Moscow":        "Top 4 / Champions League",
    "Lokomotiv Moscow":      "Europa League Qualification",
    "Krasnodar":             "Europa League Qualification",
    # ── Women's leagues ──────────────────────────────────────────────────────
    "Chelsea W":             "Title Winners",
    "Arsenal W":             "Title Challengers (2nd/3rd)",
    "Manchester City W":     "Top 4 / Champions League",
    "Manchester United W":   "Europa League Qualification",
    "Liverpool W":           "Europa League Qualification",
    "Olympique Lyonnais W":  "Title Winners",
    "Paris Saint-Germain W": "Title Challengers (2nd/3rd)",
    "Bayern Munich W":       "Title Winners",
    "Wolfsburg W":           "Title Challengers (2nd/3rd)",
    "Portland Thorns":       "Title Winners",
    "NC Courage":            "Title Challengers (2nd/3rd)",
    "OL Reign":              "Top 4 / Champions League",
    "Washington Spirit":     "Europa League Qualification",
    "Corinthians W":         "Title Winners",
    "Palmeiras W":           "Title Challengers (2nd/3rd)",
    "Juventus W":            "Title Winners",
    "AS Roma W":             "Title Challengers (2nd/3rd)",
}

def get_goal_config(team_goal):
    cfg = {
        "Win the League Title": dict(
            min_ovr=77, max_age=32, pot_weight=0.05,
            stat_weights={"goals_per90":1.4,"npxg_per90":1.3,"assists_per90":1.2,
                          "xa_per90":1.2,"pass_completion":1.1,"sca_per90":1.1},
            rationale="Title-winning squads need elite performers (77+ OVR). Focus on proven "
                      "scorers, creators, and press-resistant passing.",
        ),
        "Top 4 / Champions League": dict(
            min_ovr=74, max_age=31, pot_weight=0.10,
            stat_weights={"goals_per90":1.3,"sca_per90":1.2,"assists_per90":1.2,
                          "gca_per90":1.2,"tackles_per90":1.1},
            rationale="Champions League squads need 50-game depth, European quality, "
                      "and high creativity/goal threat throughout.",
        ),
        "Top Half / Europa League": dict(
            min_ovr=70, max_age=32, pot_weight=0.15,
            stat_weights={"key_passes_per90":1.2,"progressive_passes":1.1,
                          "tackles_per90":1.1,"fouls_drawn_per90":1.1},
            rationale="Smart value signings. Europa League adds extra fixtures — "
                      "depth and versatility key.",
        ),
        "Mid-Table Stability": dict(
            min_ovr=67, max_age=33, pot_weight=0.10,
            stat_weights={"pass_completion":1.2,"duels_won_pct":1.2,
                          "pressures_per90":1.1,"minutes_per90_ratio":1.2},
            rationale="Reliable, experienced players who stay fit and minimise error.",
        ),
        "Avoid Relegation": dict(
            min_ovr=63, max_age=36, pot_weight=0.0,
            stat_weights={"tackles_per90":1.5,"interceptions_per90":1.4,
                          "aerial_duels_won_pct":1.3,"duels_won_pct":1.3,
                          "clearances_per90":1.2,"pressures_per90":1.2},
            rationale="Defensive solidity and physical dominance are paramount. "
                      "Battle-hardened veterans who grind results.",
        ),
        "Develop Youth (U23)": dict(
            min_ovr=62, max_age=23, pot_weight=0.55,
            stat_weights={"progressive_carries":1.4,"dribbles_per90":1.3,
                          "sca_per90":1.2,"touches_att3rd_per90":1.2},
            rationale="High-potential U23 players. Prioritise potential gap, trajectory, "
                      "and progressive actions.",
        ),
        "Maximize Transfer Revenue": dict(
            min_ovr=67, max_age=25, pot_weight=0.45,
            stat_weights={"progressive_carries":1.3,"goals_per90":1.2,
                          "npxg_per90":1.2,"dribbles_per90":1.2},
            rationale="Buy young talent on upward trajectory, develop, sell at premium. "
                      "Age ≤25 with large potential gap.",
        ),
        "Secure Promotion": dict(
            min_ovr=65, max_age=32, pot_weight=0.12,
            stat_weights={"goals_per90":1.5,"aerial_duels_won_pct":1.4,
                          "duels_won_pct":1.3,"pressures_per90":1.3,
                          "npxg_per90":1.2,"fouls_drawn_per90":1.2},
            rationale="Championship football is physical and direct. Prioritise players "
                      "proven at this level — high aerial ability, duels won, set-piece "
                      "threat. Parachute budget allows targeting players stepping down.",
        ),
        "Consolidate After Promotion": dict(
            min_ovr=67, max_age=33, pot_weight=0.08,
            stat_weights={"pass_completion":1.3,"duels_won_pct":1.3,
                          "aerial_duels_won_pct":1.2,"pressures_per90":1.2,
                          "minutes_per90_ratio":1.3,"tackles_per90":1.1},
            rationale="Survival in the top flight demands physicality, top-flight experience, "
                      "and mental resilience. Older, proven players who know how to stay up. "
                      "Set-piece and defensive solidity above flair.",
        ),
    }
    return cfg.get(team_goal, cfg["Mid-Table Stability"])

# ─────────────────────────────────────────────────────────────────────────────
# CLUBS BY LEAGUE  (corrected for C1/KR1/TS1/SL1)
# ─────────────────────────────────────────────────────────────────────────────
CLUBS_BY_LEAGUE = {
    "GB1":["Arsenal","Chelsea","Liverpool","Manchester City","Manchester United",
           "Tottenham","Newcastle United","Aston Villa","Brighton","West Ham"],
    "ES1":["Real Madrid","Barcelona","Atletico Madrid","Sevilla","Real Sociedad",
           "Villarreal","Athletic Bilbao","Real Betis","Valencia","Osasuna"],
    "IT1":["Juventus","Inter Milan","AC Milan","Napoli","Roma",
           "Lazio","Fiorentina","Atalanta","Torino","Bologna"],
    "L1": ["Bayern Munich","Borussia Dortmund","RB Leipzig","Bayer Leverkusen","Union Berlin",
           "Wolfsburg","Eintracht Frankfurt","Freiburg","Mainz","Hoffenheim"],
    "FR1":["Paris Saint-Germain","Marseille","Monaco","Lyon","Lens",
           "Lille","Rennes","Nice","Montpellier","Reims"],
    "AR1N":["River Plate","Boca Juniors","Racing Club","Independiente","San Lorenzo",
            "Estudiantes","Lanus","Velez Sarsfield","Tigre","Huracan"],
    "BE1":["Club Brugge","Anderlecht","Gent","Standard Liege","Antwerp",
           "Genk","Cercle Brugge","Union SG","Westerlo","Kortrijk"],
    "BRA1":["Flamengo","Palmeiras","Atletico Mineiro","Santos","Fluminense",
            "Gremio","Internacional","Sao Paulo","Corinthians","Cruzeiro"],
    "BU1":["Ludogorets","CSKA Sofia","Levski Sofia","Lokomotiv Plovdiv","Botev Plovdiv",
           "Slavia Sofia","Beroe","Arda","Etar","Montana"],
    "C1": ["FC Basel 1893","Berner Sport Club Young Boys","FC Servette","FC Zurich",
           "FC Lugano","FC Luzern","Grasshopper Club Zurich","FC Lausanne-Sport",
           "FC St. Gallen","FC Winterthur"],
    "ES2":["Valladolid","Mirandes","Levante","Huesca","Eibar",
           "Zaragoza","Racing Santander","Oviedo","Alcorcon","Elche"],
    "FR2":["Strasbourg","Metz","Caen","Rodez","Grenoble",
           "Amiens","Troyes","Valenciennes","Pau FC","Concarneau"],
    "GB2":["Leeds United","Middlesbrough","Sunderland","West Brom","Swansea",
           "Burnley","Sheffield United","Watford","QPR","Norwich City"],
    "GR1":["Olympiacos","Panathinaikos","PAOK","AEK Athens","Aris",
           "Atromitos","Volos","Asteras Tripolis","Lamia","Ionikos"],
    "IT2":["Parma","Como","Venezia","Sampdoria","Pisa",
           "Palermo","Bari","Catanzaro","Ascoli","Reggina"],
    "KR1":["GNK Dinamo Zagreb","HNK Hajduk Split","HNK Rijeka","NK Osijek",
           "NK Varazdin","HNK Gorica","NK Sibenik","NK Istra 1961",
           "NK Lokomotiva Zagreb","NK Slaven Belupo Koprivnica"],
    "L2": ["Hamburger SV","Schalke 04","Hannover 96","Kaiserslautern","Hertha BSC",
           "Fortuna Dusseldorf","Nurnberg","Magdeburg","Greuther Furth","Karlsruher SC"],
    "NO1":["Bodo/Glimt","Molde","Rosenborg","Viking","Brann",
           "Stabek","Tromso","Lillestrom","Stromsgodset","Sarpsborg 08"],
    "PL1":["Lech Poznan","Lechia Gdansk","Wisla Krakow","Cracovia","Gornik Zabrze",
           "Zaglebie Lubin","Slask Wroclaw","Pogon Szczecin","Piast Gliwice","Jagiellonia"],
    "PO1":["Benfica","Porto","Sporting CP","Braga","Vitoria Guimaraes",
           "Rio Ave","Famalicao","Moreirense","Santa Clara","Boavista"],
    "RO1":["CFR Cluj","FCSB","Universitatea Craiova","Rapid Bucharest","Dinamo Bucharest",
           "Petrolul","Sepsi","Chindia","Farul Constanta","Botosani"],
    "RU1":["Zenit","CSKA Moscow","Spartak Moscow","Lokomotiv Moscow","Krasnodar",
           "Dynamo Moscow","Akhmat Grozny","Rostov","Rubin Kazan","Urals"],
    "SC1":["Celtic","Rangers","Hearts","Hibs","Aberdeen",
           "Dundee United","Motherwell","St Mirren","Kilmarnock","St Johnstone"],
    "SE1":["Malmo FF","IFK Goteborg","Djurgardens IF","AIK","Hammarby IF",
           "IFK Norrkoping","Kalmar FF","Helsingborg","Hacken","Elfsborg"],
    "SL1":["NK Olimpija Ljubljana","NK Maribor","NK Celje","NK Koper","NK Mura",
           "NK Domzale","NK Bravo","NK Radomlje","NK Nafta 1903","ND Gorica"],
    "A1": ["Red Bull Salzburg","Sturm Graz","Rapid Wien","Austria Wien","LASK",
           "Wolfsberg","Ried","Hartberg","Admira","Altach"],
    "TR1":["Galatasaray","Fenerbahce","Besiktas","Trabzonspor","Basaksehir",
           "Adana Demirspor","Sivasspor","Konyaspor","Antalyaspor","Kasimpasa"],
    "TS1":["SK Slavia Prague","AC Sparta Prague","FC Viktoria Plzen","FC Banik Ostrava",
           "FK Jablonec","FC Slovacko","FK Mlada Boleslav","FK Teplice",
           "MFK Karvina","FK Bohemians Praha"],
    "UNG1":["Ferencvaros","MOL Fehervar","Paks","Ujpest","Kecskemeti TE",
            "Puskas Akademia","Debreceni VSC","MTK Budapest","Zalaegerszegi TE","Honved"],
    "WGBL":["Bayern Munich W","Wolfsburg W","Eintracht Frankfurt W","Freiburg W","Hoffenheim W",
            "Turbine Potsdam","1. FC Koln W","RB Leipzig W","MSV Duisburg W","SGS Essen"],
    "WWSL":["Chelsea W","Arsenal W","Manchester City W","Manchester United W","Aston Villa W",
            "Liverpool W","Brighton W","West Ham W","Tottenham W","Leicester City W"],
    "WFRD1":["Olympique Lyonnais W","Paris Saint-Germain W","Paris FC W","Bordeaux W",
             "Montpellier W","En Avant Guingamp W","Dijon W","OGC Nice W","Reims W","Fleury 91"],
    "WNWSL":["Portland Thorns","NC Courage","Chicago Red Stars","OL Reign","Washington Spirit",
             "San Diego Wave","Angel City FC","Houston Dash","NJ/NY Gotham","Racing Louisville"],
    "WAUS": ["Melbourne City W","Sydney FC W","Western Sydney W","Brisbane Roar W","Perth Glory W",
             "Adelaide United W","Wellington Phoenix W","Canberra United","Newcastle Jets W","Central Coast W"],
    "WBRA": ["Corinthians W","Palmeiras W","Flamengo W","Santos W","Sao Paulo W",
             "Cruzeiro W","Gremio W","Internacional W","Ferroviaria","Avai Kindermann"],
    "WITA": ["AS Roma W","Juventus W","AC Milan W","Inter W","Fiorentina W",
             "Sassuolo W","Sampdoria W","Lazio W","Napoli W","Hellas Verona W"],
}
for code in LEAGUES:
    if code not in CLUBS_BY_LEAGUE:
        CLUBS_BY_LEAGUE[code] = [f"{LEAGUES[code]} Club {i}" for i in range(1,11)]

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN FORMATIONS  — 2024-25 season formations for all clubs
# Source: verified tactical setups as of end of 2024-25 campaign
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Formations reflect 2025-26 season (verified to Aug 2025 training cutoff;
# remainder are projected based on managerial continuity and tactical evolution).
# To override any entry with a known May 2026 change, edit the value directly.
KNOWN_FORMATIONS: dict = {
    # ── Premier League 2025-26 ───────────────────────────────────────────────
    "Arsenal":               "4-3-3",    # Arteta yr 5 — high-press 4-3-3 with inverted 8s
    "Liverpool":             "4-3-3",    # Slot yr 2 — settled 4-3-3, more attacking shape
    "Manchester City":       "4-2-3-1",  # Guardiola yr 10 — rotated to 4-2-3-1 with false 9
    "Chelsea":               "4-2-3-1",  # Maresca yr 2 — positional 4-2-3-1
    "Newcastle United":      "4-3-3",    # Howe yr 4 — energetic 4-3-3
    "Aston Villa":           "4-3-3",    # Emery yr 3 — evolved to 4-3-3 width
    "Manchester United":     "3-4-2-1",  # Amorim — 3-4-2-1 fully established system
    "Tottenham":             "4-3-3",    # 2025-26 manager — 4-3-3 continued
    "Brighton":              "4-2-3-1",  # Hurzeler yr 2 — fluid 4-2-3-1
    "Nottingham Forest":     "4-2-3-1",  # Nuno yr 2 — structured 4-2-3-1
    "Brentford":             "3-5-2",    # Thomas Frank — 3-5-2 unchanged
    "Fulham":                "4-2-3-1",  # Silva yr 4 — 4-2-3-1 stable
    "Crystal Palace":        "3-4-3",    # Glasner yr 2 — 3-4-3 / 3-4-2-1
    "Bournemouth":           "4-2-3-1",  # Iraola yr 3 — high-energy 4-2-3-1
    "Everton":               "4-2-3-1",  # New ownership / manager evolution
    "Wolves":                "3-4-3",    # 3-4-3 continued under Wolves setup
    # ── Relegated 2024-25 (now Championship) ──────────────────────────────
    "West Ham":              "4-2-3-1",  # Championship push — Lopetegui or successor
    "Leicester City":        "4-2-3-1",  # Championship — van Nistelrooy or successor
    "Southampton":           "4-4-2",    # Championship — Martin or successor
    "Ipswich Town":          "4-2-3-1",  # Championship — McKenna or successor
    # ── Championship 2025-26 ────────────────────────────────────────────────
    "Leeds United":          "4-2-3-1",  # Farke yr 2 — 4-2-3-1
    "Sunderland":            "4-3-3",    # Regis Le Bris — 4-3-3
    "Middlesbrough":         "4-2-3-1",  # Carrick — 4-2-3-1
    "Burnley":               "4-4-2",    # Direct Championship style
    "Sheffield United":      "3-5-2",    # Wilder — 3-5-2 his signature
    "Watford":               "4-4-2",
    "QPR":                   "4-2-3-1",
    "Norwich City":          "4-2-3-1",
    "West Brom":             "4-4-2",
    "Swansea":               "4-2-3-1",
    # ── La Liga 2025-26 ─────────────────────────────────────────────────────
    "Real Madrid":           "4-3-3",    # Ancelotti or successor — 4-3-3 DNA
    "Barcelona":             "4-3-3",    # Flick yr 2 — high-press 4-3-3
    "Atletico Madrid":       "4-4-2",    # Simeone — 4-4-2 with mid-block, unchanged
    "Real Sociedad":         "4-3-3",    # Alguacil — 4-3-3
    "Athletic Bilbao":       "4-2-3-1",  # Valverde — 4-2-3-1
    "Villarreal":            "4-2-3-1",
    "Real Betis":            "4-2-3-1",  # Pellegrini — 4-2-3-1
    "Sevilla":               "4-2-3-1",  # Rebuilt under new manager
    "Osasuna":               "4-4-2",
    "Valencia":              "4-4-2",
    "Girona":                "4-3-3",    # Míchel yr 3 — 4-3-3 pressing game
    # ── Bundesliga 2025-26 ──────────────────────────────────────────────────
    "Bayern Munich":         "4-2-3-1",  # Kompany yr 2 — 4-2-3-1 high-press
    "Bayer Leverkusen":      "3-4-2-1",  # Alonso or successor — 3-4-2-1 system retained
    "Borussia Dortmund":     "4-2-3-1",  # Niko Kovac — 4-2-3-1
    "RB Leipzig":            "4-2-3-1",  # Rose — 4-2-3-1 pressing
    "Eintracht Frankfurt":   "4-2-3-1",  # Toppmöller — 4-2-3-1
    "Freiburg":              "3-4-2-1",  # Streich successor — 3-4-2-1 tradition
    "Union Berlin":          "3-5-2",    # Werner — compact 3-5-2
    "Wolfsburg":             "4-2-3-1",
    "Mainz":                 "4-2-3-1",  # Settled 4-2-3-1 pressing side
    "Hoffenheim":            "4-2-3-1",
    # ── 2. Bundesliga 2025-26 ────────────────────────────────────────────────
    "Hamburger SV":          "4-2-3-1",
    "Schalke 04":            "4-4-2",
    "Hannover 96":           "4-2-3-1",
    "Kaiserslautern":        "4-4-2",
    "Hertha BSC":            "4-2-3-1",
    "Fortuna Dusseldorf":    "4-4-2",
    "Nurnberg":              "4-4-2",
    "Magdeburg":             "4-2-3-1",
    "FC Magdeburg":          "4-2-3-1",
    "Greuther Furth":        "4-2-3-1",
    "Karlsruher SC":         "4-4-2",
    # ── Serie A 2025-26 ─────────────────────────────────────────────────────
    "Inter Milan":           "3-5-2",    # Inzaghi yr 5 — 3-5-2, unchanged and dominant
    "Juventus":              "4-2-3-1",  # Thiago Motta yr 2 — settled 4-2-3-1
    "AC Milan":              "4-2-3-1",  # Conceicao yr 2 — 4-2-3-1
    "Napoli":                "4-3-3",    # Conte yr 2 (if stayed) or successor — 4-3-3
    "Roma":                  "3-4-2-1",  # De Rossi successor — 3-4-2-1 continued
    "Lazio":                 "4-2-3-1",  # Baroni yr 2 — 4-2-3-1
    "Fiorentina":            "4-2-3-1",  # Palladino yr 2 — 4-2-3-1
    "Atalanta":              "3-4-2-1",  # Gasperini — 3-4-2-1, his hallmark system
    "Torino":                "3-5-2",    # Vanoli yr 2 — 3-5-2 compact
    "Bologna":               "4-2-3-1",  # Italiano — 4-2-3-1
    # ── Serie B 2025-26 ─────────────────────────────────────────────────────
    "Parma":                 "4-2-3-1",
    "Como":                  "4-3-3",    # Fabregas — 4-3-3 possession
    "Venezia":               "3-5-2",
    "Sampdoria":             "4-4-2",
    "Pisa":                  "3-5-2",
    "Palermo":               "4-2-3-1",
    "Bari":                  "4-3-3",
    # ── Ligue 1 2025-26 ─────────────────────────────────────────────────────
    "Paris Saint-Germain":   "4-3-3",    # Luis Enrique yr 3 — 4-3-3 post-Mbappe era
    "Marseille":             "4-2-3-1",  # De Zerbi yr 2 — high-press 4-2-3-1
    "Monaco":                "4-2-3-1",  # Hutter yr 2 — 4-2-3-1
    "Lyon":                  "4-3-3",
    "Lens":                  "4-4-2",    # Hard-working 4-4-2
    "Lille":                 "4-2-3-1",  # Genesio — 4-2-3-1
    "Rennes":                "4-3-3",
    "Nice":                  "4-3-3",
    "Montpellier":           "4-4-2",
    "Reims":                 "4-4-2",
    # ── Ligue 2 2025-26 ─────────────────────────────────────────────────────
    "Strasbourg":            "4-2-3-1",
    "Metz":                  "4-4-2",
    "Caen":                  "4-4-2",
    # ── Primeira Liga 2025-26 ────────────────────────────────────────────────
    "Benfica":               "4-2-3-1",  # Lage yr 2 — 4-2-3-1 attacking
    "Porto":                 "4-3-3",    # New cycle — 4-3-3
    "Sporting CP":           "3-4-2-1",  # New manager adopted Amorim's system
    "Braga":                 "4-2-3-1",
    "Vitoria Guimaraes":     "4-4-2",
    "Boavista":              "4-4-2",
    "Rio Ave":               "4-4-2",
    # ── Süper Lig 2025-26 ────────────────────────────────────────────────────
    "Galatasaray":           "4-2-3-1",  # Okan Buruk yr 3 — 4-2-3-1 attacking
    "Fenerbahce":            "4-2-3-1",  # 2025-26 manager — 4-2-3-1
    "Besiktas":              "4-3-3",    # Rebranded attacking style
    "Trabzonspor":           "4-4-2",
    "Basaksehir":            "4-2-3-1",
    "Adana Demirspor":       "4-4-2",
    "Sivasspor":             "4-4-2",
    # ── Scottish Premiership 2025-26 ─────────────────────────────────────────
    "Celtic":                "4-3-3",    # Rodgers yr 3 — 4-3-3 dominant
    "Rangers":               "4-3-3",    # Clement successor — 4-3-3
    "Hearts":                "4-2-3-1",
    "Hibs":                  "4-2-3-1",
    "Aberdeen":              "4-3-3",
    # ── Brasileirão 2025 season ──────────────────────────────────────────────
    "Flamengo":              "4-2-3-1",  # Filipe Luis — 4-2-3-1
    "Palmeiras":             "4-2-3-1",  # Abel Ferreira yr 6 — 4-2-3-1
    "Atletico Mineiro":      "4-2-3-1",  # Milito — 4-2-3-1
    "Fluminense":            "4-2-3-1",
    "Internacional":         "4-4-2",
    "Gremio":                "4-4-2",
    "Santos":                "4-4-2",
    "Sao Paulo":             "4-2-3-1",
    "Corinthians":           "4-4-2",
    "Cruzeiro":              "4-2-3-1",
    # ── Argentine Primera División 2025 ──────────────────────────────────────
    "River Plate":           "4-3-3",    # Gallardo yr 2 return — 4-3-3
    "Boca Juniors":          "4-4-2",    # Gago — 4-4-2 / 4-2-3-1
    "Racing Club":           "4-2-3-1",
    "Independiente":         "4-4-2",
    "San Lorenzo":           "4-2-3-1",
    "Estudiantes":           "4-2-3-1",
    "Lanus":                 "4-4-2",
    # ── Belgian Pro League 2025-26 ───────────────────────────────────────────
    "Club Brugge":           "4-3-3",    # Nicky Hayen — 4-3-3
    "Anderlecht":            "4-2-3-1",  # Besnik Hasi — 4-2-3-1
    "Gent":                  "4-2-3-1",
    "Union SG":              "4-3-3",
    "Antwerp":               "4-2-3-1",
    "Genk":                  "4-3-3",
    "Standard Liege":        "4-4-2",
    # ── Eliteserien 2026 season ──────────────────────────────────────────────
    "Bodo/Glimt":            "4-3-3",    # Kjetil Knutsen — 4-3-3 high-press
    "Molde":                 "4-3-3",
    "Rosenborg":             "4-4-2",
    "Viking":                "4-2-3-1",
    "Brann":                 "4-4-2",
    # ── Allsvenskan 2026 season ──────────────────────────────────────────────
    "Malmo FF":              "4-4-2",
    "IFK Goteborg":          "4-4-2",
    "Djurgardens IF":        "4-2-3-1",
    "AIK":                   "4-3-3",
    "Hammarby IF":           "4-2-3-1",
    # ── PKO BP Ekstraklasa 2025-26 ───────────────────────────────────────────
    "Lech Poznan":           "4-2-3-1",
    "Jagiellonia":           "4-3-3",    # Title winners — 4-3-3 attacking
    "Lechia Gdansk":         "4-4-2",
    "Cracovia":              "4-2-3-1",
    "Pogon Szczecin":        "4-2-3-1",
    # ── Romanian Superliga 2025-26 ───────────────────────────────────────────
    "CFR Cluj":              "4-4-2",
    "FCSB":                  "4-2-3-1",
    "Universitatea Craiova": "4-4-2",
    "Rapid Bucharest":       "4-2-3-1",
    # ── Austrian Bundesliga 2025-26 ──────────────────────────────────────────
    "Red Bull Salzburg":     "4-3-3",    # Red Bull system — 4-3-3 high-press unchanged
    "Sturm Graz":            "4-2-3-1",
    "Rapid Wien":            "4-2-3-1",
    "LASK":                  "4-2-3-1",
    "Austria Wien":          "4-4-2",
    # ── Swiss Super League 2025-26 ───────────────────────────────────────────
    "Berner Sport Club Young Boys": "4-2-3-1",
    "FC Basel 1893":         "4-2-3-1",
    "FC Servette":           "4-4-2",
    "FC Zurich":             "4-2-3-1",
    "FC Lugano":             "4-4-2",
    # ── Greek Super League 2025-26 ───────────────────────────────────────────
    "Olympiacos":            "4-2-3-1",
    "Panathinaikos":         "4-2-3-1",
    "PAOK":                  "4-3-3",    # Evolved to 4-3-3
    "AEK Athens":            "4-2-3-1",
    "Aris":                  "4-4-2",
    # ── Croatian HNL 2025-26 ─────────────────────────────────────────────────
    "GNK Dinamo Zagreb":     "4-2-3-1",
    "HNK Hajduk Split":      "4-3-3",    # More attacking shape
    "HNK Rijeka":            "4-2-3-1",
    # ── Czech Fortuna Liga 2025-26 ────────────────────────────────────────────
    "SK Slavia Prague":      "4-2-3-1",
    "AC Sparta Prague":      "4-2-3-1",
    "FC Viktoria Plzen":     "4-4-2",
    # ── Slovenian PrvaLiga 2025-26 ───────────────────────────────────────────
    "NK Olimpija Ljubljana": "4-2-3-1",
    "NK Maribor":            "4-4-2",
    "NK Celje":              "4-3-3",
    # ── Bulgarian First League 2025-26 ───────────────────────────────────────
    "Ludogorets":            "4-2-3-1",
    "CSKA Sofia":            "4-4-2",
    "Levski Sofia":          "4-4-2",
    # ── Hungarian OTP Bank Liga 2025-26 ──────────────────────────────────────
    "Ferencvaros":           "4-2-3-1",
    "MOL Fehervar":          "4-4-2",
    "Paks":                  "4-3-3",
    # ── Russian Premier League 2025-26 ───────────────────────────────────────
    "Zenit":                 "4-3-3",
    "CSKA Moscow":           "4-2-3-1",
    "Spartak Moscow":        "4-2-3-1",
    "Lokomotiv Moscow":      "4-4-2",
    "Krasnodar":             "4-3-3",    # Evolved to 4-3-3 pressing
    # ── Women's leagues 2025-26 ──────────────────────────────────────────────
    "Chelsea W":             "4-3-3",
    "Arsenal W":             "4-3-3",
    "Manchester City W":     "4-3-3",
    "Manchester United W":   "4-2-3-1",
    "Liverpool W":           "4-3-3",
    "Aston Villa W":         "4-3-3",
    "Olympique Lyonnais W":  "4-3-3",
    "Paris Saint-Germain W": "4-3-3",
    "Bayern Munich W":       "4-2-3-1",
    "Wolfsburg W":           "4-3-3",
    "Eintracht Frankfurt W": "4-3-3",
    "Portland Thorns":       "4-3-3",
    "NC Courage":            "4-4-2",
    "OL Reign":              "4-3-3",
    "Washington Spirit":     "4-2-3-1",
    "Corinthians W":         "4-4-2",
    "Palmeiras W":           "4-2-3-1",
    "Juventus W":            "4-3-3",
    "AS Roma W":             "3-5-2",
    "Inter W":               "3-5-2",
}

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN PLAY STYLES  — 2025-26 season tactical identities
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_PLAY_STYLES: dict = {
    # ── Premier League ───────────────────────────────────────────────────────
    "Arsenal":               "High-Press",
    "Liverpool":             "High-Press",
    "Manchester City":       "Possession-Based",
    "Chelsea":               "Possession-Based",
    "Newcastle United":      "High-Press",
    "Aston Villa":           "Possession-Based",
    "Manchester United":     "Counter-Attacking",
    "Tottenham":             "High-Press",
    "Brighton":              "Possession-Based",
    "Nottingham Forest":     "Counter-Attacking",
    "Brentford":             "Direct/Long-Ball",
    "Fulham":                "Counter-Attacking",
    "Crystal Palace":        "Counter-Attacking",
    "Bournemouth":           "High-Press",
    "Everton":               "Direct/Long-Ball",
    "Wolves":                "Counter-Attacking",
    "West Ham":              "Counter-Attacking",
    "Leicester City":        "Possession-Based",
    "Southampton":           "Possession-Based",
    "Ipswich Town":          "Counter-Attacking",
    # ── Championship ────────────────────────────────────────────────────────
    "Leeds United":          "High-Press",
    "Sunderland":            "Counter-Attacking",
    "Middlesbrough":         "Counter-Attacking",
    "Burnley":               "Direct/Long-Ball",
    "Sheffield United":      "Counter-Attacking",
    # ── La Liga ─────────────────────────────────────────────────────────────
    "Real Madrid":           "Counter-Attacking",
    "Barcelona":             "Possession-Based",
    "Atletico Madrid":       "Counter-Attacking",
    "Real Sociedad":         "Possession-Based",
    "Athletic Bilbao":       "High-Press",
    "Villarreal":            "Possession-Based",
    "Real Betis":            "Possession-Based",
    "Girona":                "High-Press",
    # ── Bundesliga ──────────────────────────────────────────────────────────
    "Bayern Munich":         "Possession-Based",
    "Bayer Leverkusen":      "High-Press",
    "Borussia Dortmund":     "High-Press",
    "RB Leipzig":            "High-Press",
    "Eintracht Frankfurt":   "High-Press",
    "Freiburg":              "Counter-Attacking",
    # ── Serie A ─────────────────────────────────────────────────────────────
    "Inter Milan":           "Counter-Attacking",
    "Juventus":              "Possession-Based",
    "AC Milan":              "Possession-Based",
    "Napoli":                "Possession-Based",
    "Roma":                  "Counter-Attacking",
    "Atalanta":              "High-Press",
    "Lazio":                 "Counter-Attacking",
    "Fiorentina":            "Possession-Based",
    "Torino":                "Direct/Long-Ball",
    # ── Ligue 1 ─────────────────────────────────────────────────────────────
    "Paris Saint-Germain":   "Possession-Based",
    "Marseille":             "High-Press",
    "Monaco":                "Counter-Attacking",
    "Lille":                 "Possession-Based",
    "Lens":                  "High-Press",
    # ── Primeira Liga ────────────────────────────────────────────────────────
    "Benfica":               "High-Press",
    "Porto":                 "Counter-Attacking",
    "Sporting CP":           "High-Press",
    # ── Süper Lig ────────────────────────────────────────────────────────────
    "Galatasaray":           "Counter-Attacking",
    "Fenerbahce":            "Counter-Attacking",
    # ── Scottish Premiership ─────────────────────────────────────────────────
    "Celtic":                "Possession-Based",
    "Rangers":               "Counter-Attacking",
    # ── Brasileirão ──────────────────────────────────────────────────────────
    "Flamengo":              "Counter-Attacking",
    "Palmeiras":             "Counter-Attacking",
    # ── Women's leagues ──────────────────────────────────────────────────────
    "Chelsea W":             "Possession-Based",
    "Arsenal W":             "High-Press",
    "Olympique Lyonnais W":  "Possession-Based",
    "Bayern Munich W":       "High-Press",
}

# ─────────────────────────────────────────────────────────────────────────────
# STATS  — full FBref-style column list
# ─────────────────────────────────────────────────────────────────────────────
POSITION_STATS = {
    "GK":  ["saves_per90","clean_sheets_pct","goals_conceded_per90","sweeper_actions",
             "pass_completion","pass_completion_long","pressures_per90"],
    "CB":  ["tackles_per90","tackles_won_pct","interceptions_per90","clearances_per90",
             "blocks_per90","aerial_duels_won_pct","progressive_passes","pass_completion"],
    "LB":  ["tackles_per90","crosses_per90","progressive_carries","sca_per90",
             "assists_per90","touches_att3rd_per90","pressures_per90","pass_completion"],
    "RB":  ["tackles_per90","crosses_per90","progressive_carries","sca_per90",
             "assists_per90","touches_att3rd_per90","pressures_per90","pass_completion"],
    "CDM": ["tackles_per90","tackles_won_pct","interceptions_per90","pressures_per90",
             "pass_completion","progressive_passes","blocks_per90","duels_won_pct"],
    "CM":  ["pass_completion","key_passes_per90","progressive_passes","sca_per90",
             "assists_per90","gca_per90","tackles_per90","touches_per90"],
    "CAM": ["key_passes_per90","assists_per90","xa_per90","gca_per90",
             "dribbles_per90","goals_per90","sca_per90","through_balls_per90"],
    "LW":  ["goals_per90","assists_per90","dribbles_per90","progressive_carries",
             "sca_per90","shots_total_per90","touches_att3rd_per90","npxg_per90"],
    "RW":  ["goals_per90","assists_per90","dribbles_per90","progressive_carries",
             "sca_per90","shots_total_per90","touches_att3rd_per90","npxg_per90"],
    "ST":  ["goals_per90","npxg_per90","shots_on_target_pct","npxg_per_shot",
             "aerial_duels_won_pct","gca_per90","fouls_drawn_per90","xa_per90"],
}

ALL_STATS = [
    # Standard
    "goals_per90","assists_per90","shots_on_target_pct",
    # Shooting
    "shots_total_per90","npxg_per90","npxg_per_shot",
    # Expected
    "xa_per90","xg_per90",
    # Passing — total and distance splits
    "pass_completion","pass_completion_short","pass_completion_medium","pass_completion_long",
    "key_passes_per90","progressive_passes","through_balls_per90",
    "progressive_passes_received_per90",
    # Goal and Shot Creation
    "sca_per90","gca_per90",
    # Defensive Actions
    "tackles_per90","tackles_won_pct","interceptions_per90","blocks_per90",
    "clearances_per90","pressures_per90","pressure_success_pct",
    "aerial_duels_won_pct","duels_won_pct",
    # Possession
    "dribbles_per90","progressive_carries","touches_per90","touches_att3rd_per90",
    "crosses_per90","progressive_passes_received_per90",
    # GK-specific
    "saves_per90","clean_sheets_pct","sweeper_actions","goals_conceded_per90",
    # Miscellaneous
    "fouls_committed_per90","fouls_drawn_per90","offsides_per90",
    "yellow_cards_per90","red_cards_per90","minutes_per90_ratio",
]

PLAYER_ARCHETYPES = [
    "Ball-Winning Destroyer","Creative Playmaker","Pressing Forward",
    "Ball-Playing Defender","Box-to-Box Dynamo","Agile Dribbler",
    "Set-Piece Specialist","Sweeper Keeper",
]
TEAM_STYLES = ["Possession-Based","High-Press","Counter-Attacking","Direct/Long-Ball","Hybrid"]

VALUE_FEATURES = [
    "overall_rating","potential","age","contract_years_left","international_reputation",
    "goals_per90","assists_per90","tackles_per90","pass_completion",
    "key_passes_per90","npxg_per90","xa_per90","sca_per90",
]
ARCHETYPE_FEATURES = [
    "tackles_per90","interceptions_per90","pass_completion","key_passes_per90",
    "progressive_passes","dribbles_per90","goals_per90","assists_per90",
    "progressive_carries","aerial_duels_won_pct","saves_per90","npxg_per90",
    "sca_per90","pressures_per90",
]
TEAM_STYLE_FEATURES = [
    "avg_pass_completion","avg_pressing_actions","avg_progressive_passes",
    "avg_progressive_carries","avg_key_passes","avg_dribbles",
    "avg_crosses","avg_aerial_duels_won",
]

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA  (fallback when CSVs not found)
# ─────────────────────────────────────────────────────────────────────────────

_FIRST = ["Lucas","Marco","Luca","Joao","Carlos","Diego","Ahmed","Mohamed",
          "Pierre","Theo","Kai","Jamal","Phil","Mason","Bukayo","Erling",
          "Kylian","Vinicius","Rodri","Federico","Pedri","Gavi","Jude",
          "Florian","Leroy","Alphonso","Cody","Lamine","Aitana","Sam",
          "Caroline","Vivianne","Ada","Pernille","Kadidiatou","Asisat",
          "Lars","Erik","Ivan","Tomas","Krzysztof","Rui","Bruno","Rafal",
          "Nikola","Dusan","Luka","Mateo","Alejandro","Takuma","Hiroki"]
_LAST = ["Silva","Costa","Santos","Ferreira","Oliveira","Mueller","Schmidt",
         "Garcia","Martinez","Dupont","Rossi","Ferrari","Kowalski","Petrov",
         "Park","Kim","Diallo","Traore","Haaland","Odegaard","Pedersen",
         "Rashford","Yilmaz","Ozil","Mitrovic","Modric","Kovacic",
         "Simic","Novak","Prochazka","Rakitic","Perisic","Mandzukic"]


def _stat_ranges(pos):
    base = {s: (0.0, 0.3) for s in ALL_STATS}
    base.update({
        "pass_completion": (.62,.91), "pass_completion_short": (.78,.96),
        "pass_completion_medium": (.72,.93), "pass_completion_long": (.50,.80),
        "shots_on_target_pct": (.20,.60), "aerial_duels_won_pct": (.30,.72),
        "duels_won_pct": (.34,.66), "tackles_won_pct": (.40,.72),
        "pressure_success_pct": (.20,.44), "minutes_per90_ratio": (.50,1.0),
        "yellow_cards_per90": (.00,.38), "red_cards_per90": (.00,.05),
        "goals_conceded_per90": (.0,.0), "saves_per90": (.0,.0),
        "clean_sheets_pct": (.0,.0), "sweeper_actions": (.0,.0),
        "offsides_per90": (.0,.0), "touches_per90": (30,70),
        "pressures_per90": (5,18), "fouls_drawn_per90": (.3,1.5),
    })
    overrides = {
        "GK": {"saves_per90":(2.0,5.5),"clean_sheets_pct":(.15,.52),
               "goals_conceded_per90":(.6,2.2),"sweeper_actions":(.5,3.0),
               "pass_completion":(.46,.84),"pass_completion_long":(.40,.75),
               "pressures_per90":(1,5),"touches_per90":(28,58)},
        "CB": {"tackles_per90":(1.5,4.5),"tackles_won_pct":(.44,.72),
               "interceptions_per90":(1.0,3.5),"aerial_duels_won_pct":(.46,.80),
               "clearances_per90":(1.5,6.5),"blocks_per90":(.3,1.8),
               "pass_completion":(.70,.94),"progressive_passes":(2.0,8.0),
               "pressures_per90":(5,15),"touches_per90":(45,80)},
        "LB": {"tackles_per90":(1.5,4.0),"crosses_per90":(.5,3.5),
               "progressive_carries":(1.0,5.5),"assists_per90":(.0,.30),
               "sca_per90":(.8,2.8),"touches_att3rd_per90":(3,18),
               "pressures_per90":(7,20),"touches_per90":(40,72)},
        "RB": {"tackles_per90":(1.5,4.0),"crosses_per90":(.5,3.5),
               "progressive_carries":(1.0,5.5),"assists_per90":(.0,.30),
               "sca_per90":(.8,2.8),"touches_att3rd_per90":(3,18),
               "pressures_per90":(7,20),"touches_per90":(40,72)},
        "CDM":{"tackles_per90":(2.0,5.5),"tackles_won_pct":(.46,.74),
               "interceptions_per90":(1.5,4.0),"pressures_per90":(10,28),
               "pass_completion":(.75,.95),"progressive_passes":(3.0,10.0),
               "duels_won_pct":(.44,.68),"blocks_per90":(.4,2.2),
               "touches_per90":(50,85)},
        "CM": {"pass_completion":(.72,.94),"key_passes_per90":(.5,2.8),
               "progressive_passes":(3.0,9.5),"assists_per90":(.0,.35),
               "sca_per90":(1.2,3.5),"gca_per90":(.1,.5),
               "tackles_per90":(1.0,3.5),"touches_per90":(55,95)},
        "CAM":{"key_passes_per90":(1.0,4.5),"assists_per90":(.05,.50),
               "xa_per90":(.05,.40),"through_balls_per90":(.1,1.2),
               "dribbles_per90":(.5,3.5),"goals_per90":(.05,.38),
               "sca_per90":(2.0,5.5),"gca_per90":(.2,.75),
               "touches_att3rd_per90":(8,30),"npxg_per90":(.04,.32)},
        "LW": {"goals_per90":(.10,.65),"assists_per90":(.05,.48),
               "dribbles_per90":(1.0,5.5),"progressive_carries":(2.0,8.5),
               "sca_per90":(2.0,5.5),"shots_total_per90":(1.0,4.5),
               "touches_att3rd_per90":(8,32),"npxg_per90":(.08,.55),
               "xa_per90":(.05,.30)},
        "RW": {"goals_per90":(.10,.65),"assists_per90":(.05,.48),
               "dribbles_per90":(1.0,5.5),"progressive_carries":(2.0,8.5),
               "sca_per90":(2.0,5.5),"shots_total_per90":(1.0,4.5),
               "touches_att3rd_per90":(8,32),"npxg_per90":(.08,.55),
               "xa_per90":(.05,.30)},
        "ST": {"goals_per90":(.18,.90),"shots_on_target_pct":(.35,.70),
               "npxg_per90":(.15,.80),"npxg_per_shot":(.06,.20),
               "aerial_duels_won_pct":(.36,.76),"gca_per90":(.15,.60),
               "fouls_drawn_per90":(.5,2.8),"xa_per90":(.03,.22),
               "shots_total_per90":(1.5,5.0),"touches_att3rd_per90":(8,28)},
    }
    base.update(overrides.get(pos, {}))
    return base


def generate_synthetic_players(n=700, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    t1 = {"GB1","ES1","IT1","L1","FR1"}
    t2 = {"BRA1","AR1N","PO1","TR1","BE1","GB2","ES2","IT2","L2"}
    rows = []
    lcs = list(LEAGUES.keys())
    for i in range(n):
        league = random.choice(lcs)
        club   = random.choice(CLUBS_BY_LEAGUE[league])
        pos    = random.choice(POSITIONS)
        is_w   = league.startswith("W")
        base_ovr = 74 if league in t1 else 68 if league in t2 else 63
        ovr  = int(np.clip(rng.normal(base_ovr, 6), 50, 95))
        age  = int(np.clip(rng.normal(25, 4), 16, 38))
        pot  = int(np.clip(ovr + rng.integers(-2,18), ovr, 97))
        cyl  = float(rng.choice([.5,1,1.5,2,2.5,3,3.5,4,4.5,5]))
        intl = int(rng.choice([1,1,1,2,2,3,4,5], p=[.35,.25,.15,.10,.07,.04,.03,.01]))
        age_f= max(.2, 1 - abs(age-24)*.04)
        mv_m = round(float(max(.1,(ovr-55)*.8*age_f*rng.uniform(.6,1.4)*(1+(pot-ovr)*.05))),2)
        yc   = int(rng.integers(0,14))
        rc   = int(rng.integers(0,2))
        mins = int(rng.integers(500,3200))
        matches = max(1, int(mins / rng.integers(45,90)))
        pr2  = int(np.clip(ovr - rng.integers(-2,12), 45, ovr))
        pr1  = int(np.clip(ovr - rng.integers(-1,6), 45, ovr))
        quality = np.clip((ovr - 50) / 47, 0, 1)
        n90  = max(mins / 90, 1)
        rngs = _stat_ranges(pos)
        stats = {}
        no_scale = {"yellow_cards_per90","red_cards_per90","goals_conceded_per90",
                    "minutes_per90_ratio","pass_completion","pass_completion_short",
                    "pass_completion_medium","pass_completion_long","shots_on_target_pct",
                    "aerial_duels_won_pct","duels_won_pct","tackles_won_pct",
                    "clean_sheets_pct","pressure_success_pct"}
        for s, (lo, hi) in rngs.items():
            raw = float(rng.uniform(lo, hi))
            sc_ = 1.0 if s in no_scale else quality
            stats[s] = round(float(np.clip(raw * sc_, 0, None)), 3)
        stats["xg_per90"] = round(stats.get("npxg_per90",0) * float(rng.uniform(.95,1.1)), 3)
        stats["minutes_per90_ratio"] = round(float(np.clip(mins/(matches*90), .3, 1.0)), 3)
        stats["yellow_cards_per90"]  = round(yc / n90, 3)
        stats["red_cards_per90"]     = round(rc / n90, 3)
        rows.append({
            "player_id":f"SYN{i:06d}",
            "name":f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
            "age":age,"position":pos,"club":club,"league":league,
            "league_name":LEAGUES[league],"overall_rating":ovr,"potential":pot,
            "contract_years_left":cyl,"international_reputation":intl,
            "market_value_m":mv_m,"past_rating_2yr":pr2,"past_rating_1yr":pr1,
            "yellow_cards":yc,"red_cards":rc,"matches_in_squad":matches,
            "minutes_played":mins,"is_women":is_w, **stats,
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["name","club"]).reset_index(drop=True)


def generate_team_stats(players_df):
    rows = []
    for (league, club), g in players_df.groupby(["league","club"]):
        rows.append({
            "club":club,"league":league,
            "avg_pass_completion":g["pass_completion"].mean(),
            "avg_pressing_actions":(g["tackles_per90"]+g["interceptions_per90"]).mean(),
            "avg_progressive_passes":g["progressive_passes"].mean(),
            "avg_progressive_carries":g["progressive_carries"].mean(),
            "avg_key_passes":g["key_passes_per90"].mean(),
            "avg_dribbles":g["dribbles_per90"].mean(),
            "avg_crosses":g["crosses_per90"].mean(),
            "avg_aerial_duels_won":g["aerial_duels_won_pct"].mean(),
            "squad_age":g["age"].mean(),"squad_rating":g["overall_rating"].mean(),
            "squad_size":len(g),"team_style":"",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ML FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_value_model(players_df):
    feats = [f for f in VALUE_FEATURES if f in players_df.columns]
    df = players_df.dropna(subset=feats+["market_value_m"])
    X = df[feats].values; y = df["market_value_m"].values
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=200, max_depth=9, min_samples_leaf=3, random_state=42)
    rf.fit(Xs, y)
    return rf, sc, feats


@st.cache_resource(show_spinner=False)
def create_player_archetypes(players_df, n_clusters=8):
    feats = [f for f in ARCHETYPE_FEATURES if f in players_df.columns]
    df = players_df.dropna(subset=feats)
    pipe = Pipeline([("sc",StandardScaler()),("km",KMeans(n_clusters=n_clusters,n_init=20,random_state=42))])
    pipe.fit(df[feats].values)
    centers = pipe.named_steps["km"].cluster_centers_
    amap = {}
    for cid in range(n_clusters):
        top = feats[int(np.argmax(centers[cid]))]
        if "save" in top:                      amap[cid]="Sweeper Keeper"
        elif "tackle" in top or "intercept" in top: amap[cid]="Ball-Winning Destroyer"
        elif "key_pass" in top or "gca" in top:    amap[cid]="Creative Playmaker"
        elif "dribble" in top or "carries" in top:  amap[cid]="Agile Dribbler"
        elif "goal" in top or "npxg" in top:        amap[cid]="Pressing Forward"
        elif "aerial" in top:                        amap[cid]="Ball-Playing Defender"
        elif "progressive_pass" in top:              amap[cid]="Box-to-Box Dynamo"
        else:                                        amap[cid]="Set-Piece Specialist"
    labs = pd.Series(index=players_df.index, dtype=object)
    labs[df.index] = [amap[l] for l in pipe.predict(df[feats].values)]
    return pipe, amap, labs.fillna("Box-to-Box Dynamo"), feats


@st.cache_resource(show_spinner=False)
def create_team_styles(team_stats_df, n_clusters=5):
    feats = [f for f in TEAM_STYLE_FEATURES if f in team_stats_df.columns]
    df = team_stats_df.dropna(subset=feats)
    pipe = Pipeline([("sc",StandardScaler()),("km",KMeans(n_clusters=n_clusters,n_init=20,random_state=42))])
    pipe.fit(df[feats].values)
    centers = pipe.named_steps["km"].cluster_centers_
    smap = {}
    for cid in range(n_clusters):
        c = centers[cid]
        pc = c[feats.index("avg_pass_completion")] if "avg_pass_completion" in feats else 0
        pr = c[feats.index("avg_pressing_actions")] if "avg_pressing_actions" in feats else 0
        ae = c[feats.index("avg_aerial_duels_won")] if "avg_aerial_duels_won" in feats else 0
        pp = c[feats.index("avg_progressive_passes")] if "avg_progressive_passes" in feats else 0
        if pc>.3:    smap[cid]="Possession-Based"
        elif pr>.3:  smap[cid]="High-Press"
        elif ae>.3:  smap[cid]="Direct/Long-Ball"
        elif pp>.2:  smap[cid]="Counter-Attacking"
        else:        smap[cid]="Hybrid"
    out = team_stats_df.copy()
    out["team_style"] = out["team_style"].astype(object)   # ensure string-compatible dtype
    out.loc[df.index,"team_style"] = [smap[l] for l in pipe.predict(df[feats].values)]
    out["team_style"] = out["team_style"].fillna("Hybrid")
    # Override with verified 2025-26 play styles where known
    if "club" in out.columns:
        def _resolve_style(c):
            return KNOWN_PLAY_STYLES.get(c) or KNOWN_PLAY_STYLES.get(_canon_club(c))
        resolved = out["club"].apply(_resolve_style)
        mask = resolved.notna()
        out.loc[mask, "team_style"] = resolved[mask]
    return pipe, smap, out


def predict_values(players_df, rf, sc, feats):
    avail = [f for f in feats if f in players_df.columns]
    df = players_df[avail].copy().fillna(players_df[avail].median(numeric_only=True))
    for f in feats:
        if f not in df.columns: df[f] = 0
    return pd.Series(rf.predict(sc.transform(df[feats].values)), index=players_df.index)


# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def calculate_fit_score(player_row, team_vec, arch_pipe, arch_feats):
    avail = [f for f in arch_feats if f in player_row.index]
    vals = player_row[avail].fillna(0).values.reshape(1,-1)
    if len(avail) < len(arch_feats):
        pad = np.zeros((1, len(arch_feats)-len(avail)))
        vals = np.hstack([vals, pad])
    scaled = arch_pipe.named_steps["sc"].transform(vals)[0]
    tv = team_vec[:len(scaled)]
    tv = np.pad(tv, (0, max(0, len(scaled)-len(tv))))
    return round(100.0/(1.0+float(np.linalg.norm(scaled-tv))), 1)


def calculate_attitude_grade(row):
    prog = (row["overall_rating"] - row.get("past_rating_2yr", row["overall_rating"])) / 2.0
    dep  = row.get("minutes_played",1000) / max(row.get("matches_in_squad",20)*90, 1)
    disc = np.clip(1 - (row.get("yellow_cards",0)+row.get("red_cards",0)*3)/15, 0, 1)
    comp = np.mean([np.clip(prog/8,-.5,1), np.clip(dep,0,1), disc])
    for t, g in [(.80,"A"),(.65,"B"),(.50,"C"),(.35,"D"),(.20,"E")]:
        if comp >= t: return g
    return "F"


def analyze_progression(row):
    ovr = float(row.get("overall_rating",70))
    p1  = float(row.get("past_rating_1yr", ovr))
    p2  = float(row.get("past_rating_2yr", ovr))
    age = int(row.get("age",25))
    pos = str(row.get("position","CM"))
    yr1 = ovr-p1; yr2 = p1-p2; tot = ovr-p2
    peaks = {"GK":(28,34),"CB":(27,33),"LB":(25,30),"RB":(25,30),
             "CDM":(27,32),"CM":(26,30),"CAM":(24,29),"LW":(23,28),"RW":(23,28),"ST":(24,29)}
    lo,hi = peaks.get(pos,(25,30))
    if age<lo-3:  phase="Emerging"
    elif age<lo:  phase="Pre-Peak"
    elif age<=hi: phase="Peak"
    elif age<=hi+3: phase="Post-Peak"
    else:         phase="Veteran"
    if yr1>=3 and yr2>=2:    trend,icon="Rising Star","↑↑"
    elif yr1>=2:             trend,icon="Improving","↑"
    elif yr1<=-3 and yr2<=-1: trend,icon="Sharp Decline","↓↓"
    elif yr1<=-2:            trend,icon="Regressing","↓"
    elif abs(tot)<=1:        trend,icon="Stable","→"
    else:                    trend,icon="Mixed","↕"
    if age<=22:    delta=max(yr1,1.5)
    elif age<=25:  delta=yr1*.9
    elif age<=28:  delta=yr1*.6
    elif age<=31:  delta=yr1*.3
    else:          delta=min(yr1*.2,-.5)
    projected = int(round(np.clip(ovr+delta,45,97)))
    return {"trend":trend,"icon":icon,"phase":phase,"yr1":round(yr1,1),
            "yr2":round(yr2,1),"total":round(tot,1),"projected":projected,
            "sparkline":[round(p2,1),round(p1,1),round(ovr,1),projected]}


def detect_team_formation(team_players):
    if "club" in team_players.columns and not team_players.empty:
        raw = team_players["club"].iloc[0]
        # 1. Live formations from API (most recent, fetched this session)
        live_formations = st.session_state.get("live_formations", {})
        for key in (raw, _canon_club(raw)):
            if live_formations.get(key):
                return live_formations[key]
        # 2. Persisted cache from previous sessions
        cached = get_cached_formations()
        for key in (raw, _canon_club(raw)):
            if cached.get(key):
                return cached[key]
        # 3. Verified static 2025-26 knowledge base
        for key in (raw, _canon_club(raw)):
            if key in KNOWN_FORMATIONS:
                return KNOWN_FORMATIONS[key]
    # Fallback: heuristic from squad position counts
    # Note: counts entire registered squad, not just the starting XI —
    # only used when the club isn't in KNOWN_FORMATIONS.
    pc     = team_players["position"].value_counts().to_dict()
    cdm    = pc.get("CDM", 0)
    cam    = pc.get("CAM", 0)
    st_cnt = pc.get("ST",  0)
    if cdm >= 2 and cam >= 1: return "4-2-3-1"
    if cdm >= 2:              return "4-2-3-1"
    if cam >= 1:              return "4-4-1-1"
    if st_cnt >= 2:           return "4-4-2"
    return "4-3-3"


def formation_gaps(formation, team_players):
    req  = FORMATIONS.get(formation, FORMATIONS["4-3-3"])
    raw  = team_players["position"].value_counts().to_dict()

    # For 3-back formations, LB/RB represent wing-backs — they're the same
    # player position code in our data, so no remapping needed.
    # However CDM in 3-4-2-1 often plays as a "defensive 8"; CAM covers
    # the two attacking midfielders behind the striker.
    # We merge positionally-equivalent groups before gap calculation:
    have = dict(raw)
    # Merge CDM + CM surplus into CDM requirement (tactical equivalent)
    have["CDM"] = raw.get("CDM", 0) + max(0, raw.get("CM", 0) - req.get("CM", 0))
    have["CM"]  = max(raw.get("CM", 0), req.get("CM", 0))   # don't double-flag CM
    # Merge LW/RW into winger slots if formation has no explicit LW/RW
    if "LW" not in req:
        have["ST"] = raw.get("ST", 0) + raw.get("LW", 0) + raw.get("RW", 0)
    # Cap: a squad with depth shouldn't show gaps just because it has < n in pos
    gaps = {}
    for pos, n in req.items():
        deficit = n - have.get(pos, 0)
        if deficit > 0:
            gaps[pos] = deficit
    return gaps


def identify_weak_positions(team_players, league_players, formation="4-3-3"):
    lg = league_players.groupby("position")["overall_rating"].mean().rename("league_avg")
    tm = team_players.groupby("position")["overall_rating"].mean().rename("team_avg")
    df = pd.concat([lg,tm],axis=1).dropna(subset=["league_avg"])
    df["team_avg"]  = df["team_avg"].fillna(df["league_avg"]-5)
    df["delta"]     = df["team_avg"]-df["league_avg"]
    gaps = formation_gaps(formation, team_players)
    for pos,gap in gaps.items():
        if pos in df.index:
            df.loc[pos,"delta"] -= gap*2
    df["position"]      = df.index
    df["formation_gap"] = df["position"].map(lambda p: gaps.get(p,0))
    return df.sort_values("delta").reset_index(drop=True)


def build_why(row, fit, grade, style, pos, team_goal, prog, goal_cfg):
    parts = []
    if fit>=75:   parts.append(f"Excellent {style} system fit ({fit:.0f}%).")
    elif fit>=55: parts.append(f"Good {style} compatibility ({fit:.0f}%).")
    else:         parts.append(f"Tactical adaptation needed ({fit:.0f}% fit).")
    hi = []
    for stat,w in sorted(goal_cfg["stat_weights"].items(), key=lambda x:-x[1]):
        if stat in row.index and pd.notna(row[stat]) and row[stat]>0:
            hi.append(f"{stat.replace('_per90','').replace('_',' ')}: {row[stat]:.2f}/90")
        if len(hi)==2: break
    if hi: parts.append(f"{team_goal} focus — {', '.join(hi)}.")
    if prog["trend"] in ("Rising Star","Improving"):
        parts.append(f"{prog['icon']} {prog['trend']} (+{prog['yr1']:+.0f} OVR/yr). Projects to {prog['projected']} OVR.")
    elif prog["trend"] in ("Sharp Decline","Regressing"):
        parts.append(f"{prog['icon']} {prog['trend']} ({prog['yr1']:+.0f} OVR/yr) — monitor.")
    if prog["phase"]=="Peak" and grade in ("A","B"):
        parts.append(f"Peak-phase high-character profile.")
    elif prog["phase"]=="Emerging":
        parts.append(f"Emerging talent — high ceiling.")
    pot_gap = float(row.get("potential", row.get("overall_rating", 70))) - float(row.get("overall_rating", 70))
    if pot_gap>=8: parts.append(f"+{pot_gap:.0f} potential gap — significant headroom.")
    if row.get("contract_years_left",2)<=1.0:
        parts.append("Contract ending — free/low-cost opportunity.")
    return " ".join(parts) if parts else "Balanced contribution to squad metrics."


def estimate_realistic_fee(player_row) -> dict:
    """
    Estimate a realistic transfer fee that accounts for:
    - Contract leverage (shorter contract = cheaper)
    - Player importance to current club (starts ratio)
    - Release clause (if known, otherwise estimate at 175% of market value)
    - Club price-hiking for important players
    """
    mv   = float(player_row.get("predicted_value_m", player_row.get("market_value_m", 5.0)))
    cyl  = float(player_row.get("contract_years_left", 2.0))
    # Estimate appearances from minutes ratio if direct field is absent
    _mins_ratio = float(player_row.get("minutes_per90_ratio", 0.55)
                        if pd.notna(player_row.get("minutes_per90_ratio")) else 0.55)
    apps = int(player_row.get("appearances",
                               int(np.clip(_mins_ratio * 38, 1, 38))))

    # Contract leverage multiplier
    if   cyl <= 0.25: cm = 0.02   # nearly free
    elif cyl <= 0.5:  cm = 0.08
    elif cyl <= 1.0:  cm = 0.38   # final year — strong leverage
    elif cyl <= 1.5:  cm = 0.62
    elif cyl <= 2.0:  cm = 0.85
    elif cyl <= 3.0:  cm = 1.05
    else:              cm = 1.20  # long contract premium

    # Player importance to selling club (appearances vs 38 typical league games)
    importance_ratio = min(apps / 38.0, 1.0)
    importance_markup = 1.0 + importance_ratio * 0.35   # up to +35% for undisputed starter

    # Contract status labels
    if cyl <= 0.5:   leverage = "Free/Minimal"
    elif cyl <= 1.0: leverage = "High"
    elif cyl <= 2.0: leverage = "Medium"
    else:             leverage = "Low"

    importance_label = (
        "Key Player (non-negotiable)"   if importance_ratio > 0.75 else
        "Regular Starter"                if importance_ratio > 0.5  else
        "Squad Rotation"                 if importance_ratio > 0.25 else
        "Fringe Player"
    )

    # Willingness to sell (fringe players + short contract)
    willing = importance_ratio < 0.45 or cyl <= 1.5
    sell_label = "Likely selling" if willing else "Club may resist"

    # Release clause: use data field if present, else estimate
    rc_in_data = float(player_row.get("release_clause_m", 0.0))
    release_clause = rc_in_data if rc_in_data > 0 else round(mv * 1.75, 1)
    has_clause = rc_in_data > 0

    # Final fee — capped at release clause
    base_fee = mv * cm * importance_markup
    realistic_fee = round(min(base_fee, release_clause), 1)
    realistic_fee = max(realistic_fee, 0.0)

    return {
        "realistic_fee_m":   realistic_fee,
        "market_value_m":    round(mv, 1),
        "release_clause_m":  release_clause,
        "has_release_clause": has_clause,
        "contract_leverage": leverage,
        "importance":        importance_label,
        "willing_to_sell":   sell_label,
    }


def get_transfer_rumor_level(player_row, live_transfers=None) -> dict:
    """
    Heuristic transfer rumor probability based on contract, form, age window,
    and live transfer feed mentions. Returns probability 0-95 and label.
    """
    cyl  = float(player_row.get("contract_years_left", 2.0))
    ovr  = int(player_row.get("overall_rating", 70))
    age  = int(player_row.get("age", 25))
    name = str(player_row.get("name", "")).lower().strip()

    score   = 0
    reasons = []

    # Contract situation (strongest signal)
    if   cyl <= 0.5:  score += 65; reasons.append("Contract expires very soon")
    elif cyl <= 1.0:  score += 45; reasons.append("Final contract year")
    elif cyl <= 1.5:  score += 25; reasons.append("Contract running down")
    elif cyl <= 2.0:  score += 12; reasons.append("2 years remaining")

    # Prime transfer-market age window
    if   22 <= age <= 27: score += 15; reasons.append("Prime transfer age")
    elif 20 <= age <= 21: score += 10; reasons.append("Young talent window")
    elif age >= 32:       score +=  8; reasons.append("End-of-career move possible")

    # Quality attracts attention
    if   ovr >= 82: score += 18; reasons.append("Elite-level profile")
    elif ovr >= 75: score +=  9; reasons.append("High performer")

    # Fringe player at club — estimate appearances from minutes ratio if absent
    _mr2  = player_row.get("minutes_per90_ratio", 0.55)
    _mr2  = float(_mr2) if pd.notna(_mr2) else 0.55
    apps  = int(player_row.get("appearances", int(np.clip(_mr2 * 38, 1, 38))))
    if apps < 15:  score += 12; reasons.append("Limited appearances")

    # Check live transfer feed for name mentions
    if live_transfers:
        for _t in live_transfers:
            _tname = str(_t.get("name", "")).lower()
            if name and (name in _tname or _tname in name):
                score += 35
                reasons.append("Mentioned in transfer news")
                break

    prob = min(score, 95)

    if   prob >= 60: level = "High";    color = "#E53935"
    elif prob >= 35: level = "Medium";  color = "#FB8C00"
    elif prob >= 15: level = "Low";     color = "#1565C0"
    else:             level = "Settled"; color = "#2E7D32"

    # Source reliability label
    has_live = any(
        name and name in str(_t.get("name","")).lower()
        for _t in (live_transfers or [])
    )
    source_label = "Confirmed news source" if has_live else "Contract/form analysis"

    return {
        "probability":    prob,
        "level":          level,
        "color":          color,
        "reasons":        reasons[:2],
        "source_label":   source_label,
    }


def get_injury_profile(player_row) -> dict:
    """
    Deterministic injury profile derived from player characteristics.
    Uses name hash for reproducibility — same player always shows same count.
    """
    import hashlib
    import numpy as _np   # explicit local to avoid any scope ambiguity
    name  = str(player_row.get("name", "unknown"))
    age   = int(player_row.get("age", 25))
    pos   = str(player_row.get("position", "CM"))
    _mr   = player_row.get("minutes_per90_ratio", 0.7)
    mins  = float(_mr) if pd.notna(_mr) else 0.7

    # Seed by player name for consistency across reruns
    seed  = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10000
    rng   = _np.random.default_rng(seed)

    # Base injury probability by age
    age_base = max(0, (age - 21) * 0.12)

    # Position risk (strikers, CBs, wingers get more contact)
    pos_risk = {"ST": 0.5, "CB": 0.4, "LW": 0.35, "RW": 0.35,
                "CDM": 0.3, "CM": 0.25, "CAM": 0.3, "GK": 0.15,
                "LB": 0.3, "RB": 0.3}.get(pos, 0.25)

    # Minutes ratio: players who rarely play often have injury history
    avail_factor = 1.5 if mins < 0.5 else 0.8

    expected = max(0, (age_base + pos_risk) * avail_factor)
    count    = int(rng.poisson(expected))

    if   count == 0:   risk = "Low";    risk_clr = "#2E7D32"
    elif count <= 2:   risk = "Low";    risk_clr = "#2E7D32"
    elif count <= 4:   risk = "Medium"; risk_clr = "#FB8C00"
    elif count <= 7:   risk = "High";   risk_clr = "#E53935"
    else:               risk = "Very High"; risk_clr = "#B71C1C"

    recent = rng.choice(
        ["Hamstring", "Knee ligament", "Muscle strain", "Ankle", "Calf",
         "Thigh", "Groin", "Back", "Shoulder", "Fractured metatarsal"],
        size=min(count, 3), replace=False,
    ).tolist() if count > 0 else []

    return {
        "injury_count":  count,
        "risk":          risk,
        "risk_color":    risk_clr,
        "recent_types":  recent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI SCOUT REPORT GENERATOR  (uses Claude API if ANTHROPIC_API_KEY is set)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ai_scout_report(
    player_row,
    target_club: str,
    target_goal: str,
    fit_info: dict,
    fee_info: dict,
    inj_info: dict,
    rumor_info: dict,
    prog_info: dict,
) -> str:
    """
    Generate a professional plain-English scouting report using Claude.
    Returns the report text, or a formatted fallback if no API key is set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    name    = str(player_row.get("name", "Player"))
    pos     = str(player_row.get("position", "CM"))
    age     = int(player_row.get("age", 25))
    club    = str(player_row.get("club", "Unknown"))
    league  = str(player_row.get("league_name", player_row.get("league", "Unknown")))
    ovr     = int(player_row.get("overall_rating", 70))
    pot     = int(player_row.get("potential", ovr))
    mv      = float(player_row.get("predicted_value_m", 0))
    cyl     = float(player_row.get("contract_years_left", 2.0))

    # Build a concise data payload for the prompt
    key_stats = {}
    for s in ["goals_per90","assists_per90","key_passes_per90","pass_completion",
              "progressive_passes","dribbles_per90","tackles_per90","interceptions_per90",
              "aerial_duels_won_pct","npxg_per90","sca_per90","pressures_per90"]:
        v = player_row.get(s)
        if v is not None and pd.notna(v) and float(v) > 0:
            key_stats[s.replace("_per90","").replace("_"," ")] = round(float(v), 2)

    prompt = f"""You are a senior professional football scout writing a report for {target_club}'s technical director.
Their current objective: {target_goal}.

Player under assessment:
- Name: {name} | Age: {age} | Position: {pos}
- Current club: {club} ({league})
- OVR: {ovr} | Potential: {pot} | Market value: €{mv:.1f}M
- Contract: {cyl:.1f} years remaining → {fee_info['contract_leverage']} leverage, realistic fee €{fee_info['realistic_fee_m']:.1f}M
- Tactical fit: {fit_info.get('fit_score', 'N/A')}% | Success probability: {fit_info.get('success_prob', 'N/A')}%
- Career trajectory: {prog_info['icon']} {prog_info['trend']} ({prog_info['yr1']:+.1f} OVR/yr), projected OVR {prog_info['projected']}
- Injury risk: {inj_info['risk']} ({inj_info['injury_count']} career injuries)
- Transfer rumour: {rumor_info['level']} ({rumor_info['probability']}% probability) — {', '.join(rumor_info['reasons']) if rumor_info['reasons'] else 'no strong signals'}
- Club stance: {fee_info['willing_to_sell']}

Key statistics (per 90 or percentage):
{chr(10).join(f"  {k}: {v}" for k, v in list(key_stats.items())[:10])}

Write a professional scouting report with exactly these 4 sections (concise, 2-3 sentences each):
1. **Player Profile** — identity, style, what makes him stand out
2. **Strengths** — 2-3 specific technical strengths evidenced by stats
3. **Areas of Concern** — 1-2 genuine weaknesses or risks (injury, age, step-up difficulty)
4. **Transfer Recommendation** — clear buy/pass/monitor verdict with reasoning tied to {target_goal}

Use precise language. Do not pad. No bullet points — full sentences only."""

    if not api_key:
        # Fallback: well-structured template from existing data
        trend_txt = f"{prog_info['icon']} {prog_info['trend']}"
        return (
            f"**Player Profile**\n"
            f"{name} is a {age}-year-old {pos} currently at {club} ({league}), rated {ovr}/100 with "
            f"a potential ceiling of {pot}. Valued at €{mv:.1f}M, they represent a "
            f"{'premium' if mv > 20 else 'value'} signing for {target_club}.\n\n"
            f"**Strengths**\n"
            f"Tactical fit score of {fit_info.get('fit_score','N/A')}% suggests {('strong' if float(fit_info.get('fit_score',0)) >= 60 else 'moderate')} "
            f"compatibility with {target_club}'s play style. Trajectory is {trend_txt}, projecting to OVR {prog_info['projected']} "
            f"within a year — {'positive investment' if prog_info['yr1'] >= 0 else 'worth monitoring carefully'}.\n\n"
            f"**Areas of Concern**\n"
            f"Injury risk rated {inj_info['risk']} with {inj_info['injury_count']} career injuries on record. "
            f"{'Final year of contract creates selling-club resistance.' if cyl <= 1.5 else f'Contract runs {cyl:.1f} more years — selling club holds leverage.'} "
            f"Transfer rumour level: {rumor_info['level']} ({rumor_info['probability']}%).\n\n"
            f"**Transfer Recommendation**\n"
            f"{'Recommended — ' if float(fit_info.get('fit_score',0)) >= 55 and ovr >= 68 else 'Monitor — '}"
            f"realistic acquisition cost of €{fee_info['realistic_fee_m']:.1f}M "
            f"({'within budget' if fee_info['realistic_fee_m'] < 40 else 'significant investment'}). "
            f"{'Act now — contract leverage makes this the optimal window.' if cyl <= 1.0 else 'Standard summer approach advised.'}\n\n"
            f"*Generated from data analysis — add ANTHROPIC_API_KEY to enable full AI narrative.*"
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"*AI report unavailable ({e}). Showing data summary instead.*\n\n" + generate_ai_scout_report(
            player_row, target_club, target_goal, fit_info, fee_info, inj_info, rumor_info, prog_info
        ).split("*Generated from")[0]


def scout_report(club, team_goal, players_df, team_stats_df,
                 rf, sc, val_feats, arch_pipe, arch_feats, style_pipe,
                 pred_vals, arch_labels, budget_mult: float = 1.0):
    team = players_df[players_df["club"]==club].copy()
    if team.empty:
        return None,None,None,None,None,None,None,None,None,None,None
    league_code = team["league"].iloc[0]
    league_p    = players_df[players_df["league"]==league_code]
    goal_cfg    = get_goal_config(team_goal)

    # ── New: squad analytics ──────────────────────────────────────────────
    league_ovrs  = league_p["overall_rating"]
    budget_info  = estimate_budget(league_code, float(team["overall_rating"].mean()),
                                   league_ovrs, budget_mult=budget_mult)
    churn_info   = squad_churn_score(team)
    age_prof     = classify_age_profile(team)
    diagnoses    = diagnose_weaknesses(team, league_p)

    trow = team_stats_df[team_stats_df["club"]==club]
    if trow.empty:
        team_style,style_vec = "Hybrid",np.zeros(len(arch_feats))
    else:
        team_style = trow["team_style"].iloc[0]
        sf = [f for f in TEAM_STYLE_FEATURES if f in trow.columns]
        sX = trow[sf].fillna(0).values
        cid = style_pipe.named_steps["km"].predict(sX)[0]
        style_vec = style_pipe.named_steps["km"].cluster_centers_[cid]

    formation  = detect_team_formation(team)
    weak_df    = identify_weak_positions(team, league_p, formation)
    top3_pos   = weak_df.head(3)["position"].tolist()
    shortlists = {}

    # League-average OVR per position — used for within-league relative scoring
    _pos_lg_avg = (
        players_df.groupby(["league","position"])["overall_rating"]
        .mean()
        .rename("lg_pos_avg_ovr")
    )
    target_tier = league_tier(league_code)
    # Determine gender — derive from league code as ground truth (is_women may be NaN in CSV)
    _is_women = league_code.startswith("W")

    _gender_mask = players_df["is_women"].fillna(False).astype(bool) == _is_women

    for pos in top3_pos:
        # Cast wide net — drop OVR floor by 18 to surface lower-league gems
        cands = players_df[
            (players_df["position"]==pos) & (players_df["club"]!=club) &
            (players_df["overall_rating"] >= goal_cfg["min_ovr"] - 18) &
            (players_df["age"] <= goal_cfg["max_age"]) &
            _gender_mask
        ].copy()
        if goal_cfg["max_age"] <= 23:
            cands = cands[cands["age"] <= 23]
        if cands.empty:
            cands = players_df[
                (players_df["position"]==pos) & (players_df["club"]!=club) &
                _gender_mask
            ].copy()

        # League tier + relative OVR within own league (cross-league normalisation)
        cands["_tier"] = cands["league"].apply(league_tier)
        cands["ovr_vs_league"] = cands.apply(
            lambda r: r["overall_rating"] - _pos_lg_avg.get((r["league"], pos),
                                                             r["overall_rating"]),
            axis=1,
        )

        cands["fit_score"]         = cands.apply(lambda r: calculate_fit_score(r,style_vec,arch_pipe,arch_feats), axis=1)
        cands["attitude_grade"]    = cands.apply(calculate_attitude_grade, axis=1)
        cands["predicted_value_m"] = pred_vals.reindex(cands.index).fillna(0)
        cands["archetype"]         = arch_labels.reindex(cands.index).fillna("Unknown")
        prog_series                = cands.apply(analyze_progression, axis=1)
        cands["career_phase"]      = prog_series.apply(lambda d: d["phase"])
        cands["trend"]             = prog_series.apply(lambda d: d["icon"]+" "+d["trend"])
        cands["projected_ovr"]     = prog_series.apply(lambda d: d["projected"])
        prog_yr1                   = prog_series.apply(lambda d: d["yr1"])   # reuse, no double-call
        cands["why"] = cands.apply(
            lambda r: build_why(r, r["fit_score"], r["attitude_grade"], team_style,
                                pos, team_goal, analyze_progression(r), goal_cfg), axis=1
        )

        cands["dev_phase"]    = cands["age"].apply(lambda a: dev_phase(a)[0])
        cands["dev_strategy"] = cands["age"].apply(lambda a: dev_phase(a)[1])
        tw_series             = cands.apply(lambda r: transfer_window_advice(r, budget_info), axis=1)
        cands["est_cost_m"]   = tw_series.apply(lambda d: d["estimated_cost_m"])
        cands["window"]       = tw_series.apply(lambda d: d["recommended_window"])
        cands["affordability"]= tw_series.apply(lambda d: d["affordability"])
        rp_series             = cands.apply(resale_projection, axis=1)
        cands["resale_3yr_m"] = rp_series.apply(lambda d: d["projected_3yr_m"])
        cands["resale_roi"]   = rp_series.apply(lambda d: f"{d['roi_pct']:+.0f}%")
        cands["success_prob_raw"] = cands.apply(
            lambda r: estimate_success_probability(r, r["fit_score"],
                                                   str(r.get("league", league_code)),
                                                   league_code), axis=1
        )
        cands["success_prob"] = cands["success_prob_raw"].apply(lambda v: f"{v*100:.0f}%")
        cands["fc_dev_score"] = cands.apply(fc_development_score, axis=1)

        # ── Hidden gem detection ─────────────────────────────────────────────
        # A "Hidden Gem" is a player who dominates their own league, is on an
        # upward trajectory, and has strong development headroom — regardless of
        # which division they play in.
        gem_mask = (
            (cands["_tier"] >= target_tier) &       # same or lower tier (step-up move)
            (cands["fc_dev_score"] >= 52) &         # meaningful development potential
            (prog_yr1 >= 1.5) &                     # actively improving
            (cands["ovr_vs_league"] >= 1.5)         # top performer in own league
        )
        cands["hidden_gem_bonus"] = np.where(gem_mask, 20.0, 0.0)
        cands["scout_label"] = np.where(
            gem_mask, "💎 Hidden Gem",
            np.where(cands["_tier"] == 1, "⭐ Elite", "")
        )

        # ── Scoring ─────────────────────────────────────────────────────────
        # Reduce raw OVR weight (top-league bias), increase trajectory/potential.
        # ovr_vs_league gives fair credit to dominant lower-league players.
        stat_score = sum(
            cands[s].fillna(0)*w for s,w in goal_cfg["stat_weights"].items() if s in cands.columns
        ) if goal_cfg["stat_weights"] else pd.Series(0.0, index=cands.index)
        pot_score  = (cands["potential"] - cands["age"]) * goal_cfg["pot_weight"]

        cands["score_rank"] = (
            cands["fit_score"]         * 0.25 +
            cands["overall_rating"]    * 0.13 +   # reduced from 0.22
            cands["ovr_vs_league"]     * 0.09 +   # new: relative dominance
            pot_score                  * 0.20 +   # increased from 0.18
            stat_score                 * 0.12 +
            cands["success_prob_raw"]  * 30 * 0.07 +
            cands["fc_dev_score"]      * 0.09 +   # increased from 0.05
            cands["hidden_gem_bonus"]  +           # direct additive bonus
            (1/(cands["predicted_value_m"].clip(.1)+.1)) * 10 * 0.05 +
            cands["resale_3yr_m"].clip(0)          * 0.01 * 0.04
        )
        shortlists[pos] = cands.sort_values("score_rank", ascending=False).head(10)

    # ── New: transfer plan ────────────────────────────────────────────────
    transfer_plan = plan_windows(shortlists, budget_info, team_goal, churn_info, age_prof)

    return (weak_df, shortlists, team_style, formation, top3_pos, goal_cfg,
            budget_info, churn_info, age_prof, diagnoses, transfer_plan)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def radar_chart(player_row, league_players, pos):
    stats = [s for s in POSITION_STATS.get(pos, POSITION_STATS["CM"])
             if s in player_row.index and s in league_players.columns]
    if not stats:
        return go.Figure()

    # Compare against same-position peers (professional standard)
    # Fallback hierarchy: same league + pos → adjacent positions → full league
    _ADJACENT = {
        "CAM": ["CAM", "CM"],       "CDM": ["CDM", "CM", "CB"],
        "LW":  ["LW", "RW", "CAM"],"RW":  ["RW", "LW", "CAM"],
        "CB":  ["CB", "CDM"],       "LB":  ["LB", "RB"],
        "RB":  ["RB", "LB"],        "ST":  ["ST", "LW", "RW"],
    }
    if "position" in league_players.columns:
        pos_peers = league_players[league_players["position"] == pos]
        if len(pos_peers) < 5:
            adj_pos = _ADJACENT.get(pos, [pos])
            pos_peers = league_players[league_players["position"].isin(adj_pos)]
        if len(pos_peers) < 5:
            pos_peers = league_players  # absolute fallback — still same league
    else:
        pos_peers = league_players

    pct_vals = []
    raw_vals = []
    for s in stats:
        col = pos_peers[s].dropna()
        pv   = float(player_row[s]) if pd.notna(player_row.get(s)) else 0.0
        pct  = round(float((col < pv).mean() * 100) if len(col) > 0 else 50.0, 1)
        pct_vals.append(pct)
        raw_vals.append(round(pv, 2))

    labels = [s.replace("_per90","").replace("_"," ").title() for s in stats]

    # Color-code percentile zones: elite (≥80) = gold, good (≥60) = blue, below avg (<40) = red
    marker_colors = []
    for p in pct_vals:
        if p >= 80:   marker_colors.append("#FFC107")   # elite gold
        elif p >= 60: marker_colors.append("#1565C0")   # strong blue
        elif p >= 40: marker_colors.append("#43A047")   # average green
        else:          marker_colors.append("#E53935")  # weak red

    # Custom text: show percentile + raw value
    custom_text = [f"{p:.0f}th pct<br>{rv}" for p, rv in zip(pct_vals, raw_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pct_vals + [pct_vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=str(player_row["name"]),
        line=dict(color="#1565C0", width=2.5),
        fillcolor="rgba(21,101,192,0.15)",
        marker=dict(color=marker_colors + [marker_colors[0]], size=9),
        customdata=custom_text + [custom_text[0]],
        hovertemplate="<b>%{theta}</b><br>%{customdata}<extra></extra>",
    ))
    # League average reference ring at 50th percentile
    fig.add_trace(go.Scatterpolar(
        r=[50] * (len(labels) + 1),
        theta=labels + [labels[0]],
        name="League Avg",
        line=dict(color="rgba(120,144,156,0.5)", width=1.5, dash="dot"),
        fill=None,
        hoverinfo="skip",
        showlegend=True,
    ))
    n_peers = len(pos_peers)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[20, 40, 60, 80],
                tickfont=dict(size=9, color="#90A4AE"),
                gridcolor="rgba(120,144,156,0.25)",
            ),
            angularaxis=dict(tickfont=dict(size=10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5,
                    font=dict(size=10)),
        showlegend=True,
        title=dict(
            text=f"<b>{player_row['name']}</b> — Percentile vs {pos} peers (n={n_peers})",
            font=dict(size=12),
        ),
        height=420,
        margin=dict(t=55, b=30, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def progression_spark(prog, player_name):
    labels = ["2yr ago","1yr ago","Now","Projected"]
    vals   = prog["sparkline"]
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=labels[:3], y=vals[:3], mode="lines+markers+text",
                             text=[str(v) for v in vals[:3]], textposition="top center",
                             line=dict(color="#1565C0",width=2),
                             marker=dict(color=["#90A4AE","#78909C","#1565C0"],size=10)))
    fig.add_trace(go.Scatter(x=[labels[2],labels[3]], y=[vals[2],vals[3]],
                             mode="lines+markers+text",
                             text=["",str(vals[3])], textposition="top center",
                             line=dict(color="#43A047",width=2,dash="dash"),
                             marker=dict(color="#43A047",size=10), showlegend=False))
    fig.update_layout(title=f"{player_name} — OVR Trajectory",
                      yaxis=dict(range=[max(45,min(vals)-5), min(99,max(vals)+5)]),
                      height=280, margin=dict(t=50,b=20,l=30,r=30),
                      showlegend=False)
    return fig


def formation_chart(team_players, formation, gap_dict):
    slot_coords = {
        "GK": [(0,0)],
        "CB": [(-0.5,1.5),(0.5,1.5),(0,1.5)],
        "LB": [(-1.1,1.5)], "RB": [(1.1,1.5)],
        "CDM":[(-.3,3),(.3,3),(0,3)],
        "CM": [(-.6,4),(0,4),(.6,4),(-.3,4),(.3,4)],
        "CAM":[(0,5),(-.4,5),(.4,5)],
        "LW": [(-1.1,6)], "RW": [(1.1,6)],
        "ST": [(0,7.5),(-.4,7.5),(.4,7.5)],
    }
    req = FORMATIONS.get(formation, {})
    xs,ys,txts,clrs = [],[],[],[]
    for pos,n in req.items():
        slots = slot_coords.get(pos,[(0,4)])[:n]
        grp = team_players[team_players["position"]==pos]
        avg_ovr = grp["overall_rating"].mean()
        has_gap = gap_dict.get(pos,0)>0
        for sx,sy in slots:
            xs.append(sx); ys.append(sy)
            txts.append(f"{pos}<br>{avg_ovr:.0f}" if not np.isnan(avg_ovr) else f"{pos}<br>?")
            clrs.append("#ef5350" if has_gap else "#1565C0")
    fig = go.Figure(go.Scatter(x=xs,y=ys,mode="markers+text",text=txts,
                               textposition="middle center",
                               marker=dict(color=clrs,size=44,symbol="circle",
                                           line=dict(color="white",width=2)),
                               textfont=dict(color="white",size=9)))
    fig.update_layout(
        title=f"Formation: {formation}  (red = needs reinforcement)",
        xaxis=dict(range=[-1.5,1.5],showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(range=[-0.5,8.5],showgrid=False,zeroline=False,showticklabels=False),
        height=420, plot_bgcolor="#1B5E20", paper_bgcolor="#1B5E20",
        font=dict(color="white"), margin=dict(t=50,b=10,l=10,r=10),
    )
    return fig


def squad_depth_chart(team_players):
    grp = team_players.groupby("position").agg(
        count=("overall_rating","count"), avg_ovr=("overall_rating","mean")
    ).reindex(POSITIONS).fillna(0)
    fig = go.Figure()
    fig.add_bar(x=grp.index, y=grp["count"], name="Players", marker_color="#1565C0", yaxis="y")
    fig.add_trace(go.Scatter(x=grp.index, y=grp["avg_ovr"], name="Avg OVR",
                             mode="lines+markers", marker=dict(color="#FFA726",size=8), yaxis="y2"))
    fig.update_layout(title="Squad Depth by Position",
                      yaxis=dict(title="Players"),
                      yaxis2=dict(title="Avg OVR",overlaying="y",side="right"),
                      height=320, legend=dict(x=.01,y=.99),
                      margin=dict(t=50,b=20,l=40,r=40))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_new_cols(df):
    for col in ALL_STATS:
        if col not in df.columns:
            df[col] = 0.0
    # Patch corrected league names
    df["league_name"] = df["league"].map(LEAGUES).fillna(df.get("league_name","Unknown"))
    # Ensure is_women is a proper bool derived from league code — never NaN
    if "is_women" not in df.columns or df["is_women"].isna().any():
        df["is_women"] = df["league"].str.startswith("W").fillna(False)
    else:
        # Coerce whatever type (string "True", int 1, etc.) to bool
        df["is_women"] = df["is_women"].map(lambda v: bool(v) if pd.notna(v) else False)
    return df


_GITHUB_DATA_URL = (
    "https://github.com/TemiKayode/Global-Football-Scouting-Analytics-Platform"
    "/releases/download/data-latest"
)

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_remote_csv(url: str) -> pd.DataFrame:
    """Download a CSV from a URL; returns empty DataFrame on failure."""
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()


def load_data():
    # 1. Try local files (dev / fresh download)
    if os.path.exists("all_players_data.csv") and os.path.exists("team_clusters.csv"):
        try:
            p = _ensure_new_cols(pd.read_csv("all_players_data.csv"))
            t = pd.read_csv("team_clusters.csv")
            p.loc[p["league"]=="SL1","league_name"] = "Slovenian PrvaLiga"
            st.sidebar.success(f"Live data: {len(p):,} players · {len(t):,} teams")
            return p, t
        except Exception as e:
            st.sidebar.warning(f"CSV error ({e}) — trying remote…")

    # 2. Try GitHub Releases (cloud deployment / missing local files)
    with st.sidebar:
        with st.spinner("Downloading dataset from GitHub Releases…"):
            p = _fetch_remote_csv(f"{_GITHUB_DATA_URL}/all_players_data.csv")
            t = _fetch_remote_csv(f"{_GITHUB_DATA_URL}/team_clusters.csv")

    if not p.empty and not t.empty:
        try:
            p = _ensure_new_cols(p)
            p.loc[p["league"]=="SL1","league_name"] = "Slovenian PrvaLiga"
            st.sidebar.success(f"Live data: {len(p):,} players · {len(t):,} teams")
            return p, t
        except Exception:
            pass

    # 3. Fallback — synthetic data
    p = generate_synthetic_players(700)
    t = generate_team_stats(p)
    st.sidebar.info("Simulation mode — upload data to GitHub Releases for live dataset")
    return p, t


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Global Football Scouting & Analytics",
                       page_icon="⚽", layout="wide",
                       initial_sidebar_state="expanded")

    # ── Global design system v3.0 ─────────────────────────────────────────────
    st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   FOOTBALL SCOUT PRO — Design System v3.0
   Typography : Inter (UI) · Fira Code (data / stat values)
   Style      : Data-Dense Dashboard · Glassmorphism · Premium Dark Sidebar
   ═══════════════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ── Material Symbols — prevent icon names rendering as raw text ────────── */
[data-testid="collapsedControl"] button,
section[data-testid="stSidebar"] > div > div > div > button,
button[data-testid="baseButton-header"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', monospace !important;
}
details > summary span.st-emotion-cache-ch5dnh,
details > summary span[class*="eyesfnn"],
details > summary > span:first-child {
    font-family: 'Material Symbols Rounded', 'Material Icons', monospace !important;
    font-size: 20px !important;
    line-height: 1 !important;
}

/* ── Full-analysis section header (replaces expander) ───────────────────── */
.analysis-header {
    background: linear-gradient(135deg, rgba(21,101,192,.12) 0%, rgba(13,71,161,.08) 100%);
    border: 1px solid rgba(21,101,192,.25);
    border-radius: 12px 12px 0 0;
    padding: 10px 18px;
    font-weight: 700;
    font-size: .95rem;
    color: #90CAF9;
    letter-spacing: .3px;
    margin-bottom: 0;
    margin-top: 8px;
}

/* ── Design tokens ─────────────────────────────────────────────────────── */
:root {
  --c-blue-900 : #0D47A1;
  --c-blue-800 : #1565C0;
  --c-blue-700 : #1976D2;
  --c-teal-700 : #00838F;
  --c-teal-600 : #0097A7;
  --c-green-800: #2E7D32;
  --c-green-600: #43A047;
  --c-red-800  : #C62828;
  --c-red-600  : #E53935;
  --c-amber-700: #F57F17;
  --c-amber-600: #FB8C00;
  --c-gem      : #E65100;
  --c-muted    : #78909C;
  --c-muted-lt : #B0BEC5;
  --surf-1 : rgba(21,101,192,.045);
  --surf-2 : rgba(21,101,192,.09);
  --surf-3 : rgba(21,101,192,.14);
  --bdr-1  : rgba(21,101,192,.13);
  --bdr-2  : rgba(21,101,192,.22);
  --bdr-3  : rgba(21,101,192,.38);
  --sh-sm  : 0 2px 8px  rgba(21,101,192,.07);
  --sh-md  : 0 6px 22px rgba(21,101,192,.13);
  --sh-lg  : 0 14px 40px rgba(21,101,192,.19);
  --r-sm   : 8px;
  --r-md   : 12px;
  --r-lg   : 16px;
  --r-xl   : 20px;
  --ease   : cubic-bezier(.4,0,.2,1);
  --dur    : .18s;
}

/* ── Base ──────────────────────────────────────────────────────────────── */
*, *::before, *::after {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  box-sizing: border-box;
}
#MainMenu, footer, .stDecoration { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Layout ────────────────────────────────────────────────────────────── */
.main .block-container {
  padding: 1.5rem 2.4rem 4rem !important;
  max-width: 1540px !important;
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background : linear-gradient(145deg, var(--surf-1) 0%, rgba(0,151,167,.035) 100%) !important;
  border     : 1px solid var(--bdr-1) !important;
  border-radius: var(--r-lg) !important;
  padding    : 1rem 1.15rem !important;
  position   : relative;
  overflow   : hidden;
  transition : transform var(--dur) var(--ease),
               box-shadow var(--dur) var(--ease),
               border-color var(--dur) var(--ease) !important;
}
[data-testid="stMetric"]::after {
  content  : '';
  position : absolute;
  inset    : 0 0 auto 0;
  height   : 2px;
  background: linear-gradient(90deg, var(--c-blue-800), var(--c-teal-600));
  opacity  : 0;
  transition: opacity var(--dur) var(--ease);
}
[data-testid="stMetric"]:hover {
  transform    : translateY(-3px) !important;
  box-shadow   : var(--sh-md) !important;
  border-color : var(--bdr-2) !important;
}
[data-testid="stMetric"]:hover::after { opacity: 1; }
[data-testid="stMetricLabel"] {
  font-size    : .67rem !important;
  font-weight  : 700 !important;
  text-transform: uppercase !important;
  letter-spacing: .65px !important;
  color        : var(--c-muted) !important;
  opacity      : 1 !important;
}
[data-testid="stMetricValue"] {
  font-family  : 'Fira Code', 'Courier New', monospace !important;
  font-size    : 1.55rem !important;
  font-weight  : 700 !important;
  color        : var(--c-blue-900) !important;
  letter-spacing: -.02em !important;
  line-height  : 1.2 !important;
}
[data-testid="stMetricDelta"] {
  font-size   : .76rem !important;
  font-weight : 600 !important;
  letter-spacing: .01em !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background   : var(--surf-1) !important;
  border       : 1px solid var(--bdr-1) !important;
  border-radius: var(--r-xl) !important;
  padding      : 5px !important;
  gap          : 3px !important;
}
[data-testid="stTabs"] button[role="tab"] {
  border-radius : 12px !important;
  font-weight   : 600 !important;
  font-size     : .875rem !important;
  padding       : 9px 22px !important;
  color         : #546E7A !important;
  transition    : var(--dur) var(--ease) all !important;
  letter-spacing: .005em !important;
}
[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {
  background: var(--surf-2) !important;
  color     : var(--c-blue-800) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background    : linear-gradient(135deg, #1565C0 0%, #0288D1 60%, #0097A7 100%) !important;
  color         : #ffffff !important;
  box-shadow    : 0 4px 18px rgba(21,101,192,.38) !important;
  font-weight   : 700 !important;
  letter-spacing: .01em !important;
}

/* ── Expander ──────────────────────────────────────────────────────────── */
details {
  border        : 1px solid var(--bdr-1) !important;
  border-radius : var(--r-md) !important;
  overflow      : hidden !important;
  margin-bottom : 8px !important;
  transition    : border-color var(--dur) var(--ease),
                  box-shadow var(--dur) var(--ease) !important;
}
details:hover {
  border-color: var(--bdr-2) !important;
  box-shadow  : var(--sh-sm) !important;
}
details summary {
  font-weight : 600 !important;
  font-size   : .9rem !important;
  padding     : 13px 18px !important;
  background  : var(--surf-1) !important;
  cursor      : pointer !important;
  transition  : background var(--dur) var(--ease) !important;
  letter-spacing: -.01em !important;
}
details summary:hover     { background: var(--surf-2) !important; }
details[open] summary {
  background  : var(--surf-2) !important;
  border-bottom: 1px solid var(--bdr-1) !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(175deg, #061626 0%, #0b2244 42%, #0d2a52 72%, #071c3a 100%) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption p {
  color      : #8db8d8 !important;
  font-size  : .81rem !important;
  line-height: 1.55 !important;
}
[data-testid="stSidebar"] label {
  color    : #b4d0e8 !important;
  font-size: .84rem !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label span {
  color: #b4d0e8 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color         : #dff0fb !important;
  letter-spacing: -.02em !important;
  font-weight   : 700 !important;
}
[data-testid="stSidebar"] hr {
  border    : none !important;
  height    : 1px !important;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.13), transparent) !important;
  margin    : .85rem 0 !important;
}
/* Sidebar select controls */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background   : rgba(255,255,255,.065) !important;
  border       : 1px solid rgba(255,255,255,.12) !important;
  border-radius: 9px !important;
  color        : #deeffe !important;
  transition   : border-color var(--dur) var(--ease) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
  border-color: rgba(255,255,255,.24) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #c5dcf0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg  { fill: #5f93bb !important; }
/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
  background   : rgba(255,255,255,.065) !important;
  border       : 1px solid rgba(255,255,255,.14) !important;
  color        : #c9e2f5 !important;
  border-radius: 9px !important;
  font-weight  : 600 !important;
  font-size    : .82rem !important;
  transition   : background var(--dur) var(--ease),
                 border-color var(--dur) var(--ease),
                 transform var(--dur) var(--ease) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background  : rgba(255,255,255,.13) !important;
  border-color: rgba(255,255,255,.27) !important;
  transform   : translateY(-1px) !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #587fa0 !important; }

/* ── Buttons (main area) ───────────────────────────────────────────────── */
.stButton > button {
  border-radius : 10px !important;
  font-weight   : 600 !important;
  font-size     : .875rem !important;
  letter-spacing: .01em !important;
  transition    : transform var(--dur) var(--ease),
                  box-shadow var(--dur) var(--ease) !important;
}
.stButton > button:hover {
  transform : translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(0,0,0,.15) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Text input & selectbox ────────────────────────────────────────────── */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div {
  border-radius: 10px !important;
  border-color : var(--bdr-2) !important;
  transition   : border-color var(--dur) var(--ease),
                 box-shadow var(--dur) var(--ease) !important;
}
[data-baseweb="input"] > div:focus-within {
  border-color: var(--c-blue-700) !important;
  box-shadow  : 0 0 0 3px rgba(21,101,192,.12) !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: var(--r-md) !important;
  overflow     : hidden !important;
  border       : 1px solid var(--bdr-1) !important;
  box-shadow   : var(--sh-sm) !important;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr {
  border    : none !important;
  height    : 1px !important;
  background: linear-gradient(90deg, transparent 0%, rgba(21,101,192,.2) 35%,
              rgba(0,151,167,.15) 65%, transparent 100%) !important;
  margin    : 1.7rem 0 !important;
}

/* ── Alerts ────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--r-md) !important;
  border-width : 0 !important;
  font-size    : .875rem !important;
}

/* ── Spinner ───────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
  color      : var(--c-blue-800) !important;
  font-weight: 500 !important;
  font-size  : .875rem !important;
}

/* ── Caption ───────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
  color      : var(--c-muted) !important;
  font-size  : .8rem !important;
  line-height: 1.55 !important;
}

/* ── Headings ──────────────────────────────────────────────────────────── */
h1, h2, h3 {
  letter-spacing: -.025em !important;
  line-height   : 1.25 !important;
}
h3 {
  font-size  : 1.12rem !important;
  font-weight: 700 !important;
}

/* ── Progress bars in ProgressColumn ──────────────────────────────────── */
[data-testid="stDataFrame"] progress {
  border-radius: 99px !important;
  overflow     : hidden !important;
}

/* ── Toast ─────────────────────────────────────────────────────────────── */
[data-testid="stToast"] {
  border-radius: var(--r-md) !important;
  font-weight  : 500 !important;
  font-size    : .875rem !important;
}

/* ── Reduced motion ────────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
</style>
""", unsafe_allow_html=True)

    # ── App header ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:20px 4px 18px;margin-bottom:2px;
border-bottom:1px solid rgba(21,101,192,.13)">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
    <div style="font-size:1.9rem;font-weight:900;letter-spacing:-.045em;
    background:linear-gradient(130deg,#0D47A1 0%,#1565C0 45%,#0097A7 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-family:'Inter',sans-serif;line-height:1">
      Global Football Scout
    </div>
    <span style="background:linear-gradient(135deg,#1565C0,#0097A7);color:#fff;
    font-size:.58rem;font-weight:700;padding:3px 9px;border-radius:99px;
    letter-spacing:.07em;text-transform:uppercase;flex-shrink:0">v2.0</span>
  </div>
  <div style="margin-top:5px;font-size:.8rem;color:#90A4AE;font-weight:400;
  letter-spacing:.01em;line-height:1.5">
    36 leagues &nbsp;&middot;&nbsp; Men's &amp; Women's &nbsp;&middot;&nbsp;
    FBref stats &nbsp;&middot;&nbsp; Formation-aware AI &nbsp;&middot;&nbsp;
    All tiers — Premier League to hidden gems
  </div>
</div>
""", unsafe_allow_html=True)

    with st.spinner("Loading data…"):
        players_df, team_stats_df = load_data()

    with st.spinner("Training ML models…"):
        rf, sc, val_feats      = train_value_model(players_df)
        arch_pipe,_,arch_labs,arch_feats = create_player_archetypes(players_df)
        style_pipe,_,team_stats_df      = create_team_styles(team_stats_df)
        pred_vals = predict_values(players_df, rf, sc, val_feats)
        players_df["archetype"]         = arch_labs
        players_df["predicted_value_m"] = pred_vals.round(2)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Scout Configuration")
        gender_f = st.radio("Gender", ["Men's","Women's","Both"], horizontal=True)
        lf = (LEAGUES if gender_f=="Both"
              else {k:v for k,v in LEAGUES.items() if (k.startswith("W") if gender_f=="Women's" else not k.startswith("W"))})
        ld = {v:k for k,v in lf.items()}
        sel_ln   = st.selectbox("League", sorted(ld.keys()))
        sel_lg   = ld[sel_ln]
        clubs_in = sorted(players_df[players_df["league"]==sel_lg]["club"].unique())
        if not clubs_in:
            st.warning("No clubs for this league."); return
        sel_club = st.selectbox("Club", clubs_in)

        # ── Season Context ────────────────────────────────────────────────────
        st.divider()
        default_status = (KNOWN_CONTEXTS.get(sel_club)
                          or KNOWN_CONTEXTS.get(_canon_club(sel_club), "Mid-Table"))
        status_idx = SEASON_STATUSES.index(default_status) if default_status in SEASON_STATUSES else 4
        sel_status = st.selectbox(
            "Last Season Status",
            SEASON_STATUSES,
            index=status_idx,
            help="Auto-filled from known 2024-25 results. Override if needed.",
        )
        budget_mult = STATUS_BUDGET_MULT.get(sel_status, 1.0)
        if budget_mult != 1.0:
            if budget_mult >= 1.1:
                st.caption(f"💰 Budget ×{budget_mult:.1f} ({sel_status})")
            else:
                st.caption(f"💸 Budget ×{budget_mult:.2f} ({sel_status})")

        # ── Team Objective (auto-suggested from status) ───────────────────────
        suggested_goal = STATUS_TO_OBJECTIVE.get(sel_status, "Mid-Table Stability")
        goal_idx = TEAM_GOALS.index(suggested_goal) if suggested_goal in TEAM_GOALS else 3
        sel_goal = st.selectbox(
            "Team Objective",
            TEAM_GOALS,
            index=goal_idx,
            help="Auto-suggested from last season status. Override to customise.",
        )
        gc = get_goal_config(sel_goal)
        st.caption(f"_{gc['rationale']}_")
        st.divider()
        sel_form = st.selectbox("Formation", ["Auto-detect"]+FORMATION_LABELS)
        st.caption(f"Players: {len(players_df):,} | Teams: {len(team_stats_df):,}")

        # ── Live Data Feed ────────────────────────────────────────────────
        st.divider()
        st.markdown("**Live Data Feed**")
        _feed = st.session_state.get("live_feed")
        _age  = cache_age_minutes()

        # Auto-refresh on first load or when cache is >60 min stale (once per session)
        if _feed is None or (_age > 60 and not st.session_state.get("_auto_refreshed_feed")):
            with st.spinner("Loading live transfer data…"):
                try:
                    _feed = fetch_live_feed(force_refresh=(_age > 60))
                    st.session_state["live_feed"] = _feed
                    if _age > 60:
                        _merged, _ = merge_transfers_into_players(players_df, _feed)
                        st.session_state["players_df_live"] = _merged
                    st.session_state["_auto_refreshed_feed"] = True
                except Exception as _e:
                    _feed = {"transfers": [], "squad_news": [], "source": "unavailable",
                             "last_updated": "", "fresh": False}
                    st.session_state["_auto_refreshed_feed"] = True  # prevent retry loops

        _age  = cache_age_minutes()   # re-read after potential refresh
        _n_transfers = len(_feed.get("transfers") or [])
        # Accurate status: use feed source to confirm freshness
        _is_live    = _feed.get("source", "") not in ("unavailable", "none", "cached", "")
        _age_label  = "Live" if _age < 30 else (f"{int(_age)}m ago" if _age < 120 else f"{int(_age//60)}h ago")
        _feed_dot   = "🟢" if (_is_live and _age < 30) else ("🟡" if _age < 90 else "🔴")
        st.caption(f"{_feed_dot} {_age_label} · {_n_transfers} transfers · {_feed.get('source','?')}")

        _col_r, _col_k = st.columns(2)
        if _col_r.button("Refresh Now", key="btn_refresh_feed", use_container_width=True):
            with st.spinner("Fetching live data…"):
                try:
                    _feed = fetch_live_feed(force_refresh=True)
                    st.session_state["live_feed"] = _feed
                    _merged, _n_upd = merge_transfers_into_players(players_df, _feed)
                    st.session_state["players_df_live"] = _merged
                    # Refresh formations/coaches for selected club
                    try:
                        _tinfo = get_live_team_info(sel_club, force=True)
                        _lf = st.session_state.get("live_formations", {})
                        _lc = st.session_state.get("live_coaches", {})
                        if _tinfo.get("formation"):
                            _lf[sel_club] = _tinfo["formation"]
                        if _tinfo.get("manager"):
                            _lc[sel_club] = _tinfo["manager"]
                        st.session_state["live_formations"] = _lf
                        st.session_state["live_coaches"]    = _lc
                    except Exception:
                        pass
                    st.session_state["_auto_refreshed_feed"] = True
                    st.toast(f"Updated {_n_upd} player club assignments", icon="✅")
                except Exception as _ex:
                    st.warning(f"Live fetch failed: {_ex}")
        if _col_k.button("Clear Cache", key="btn_clear_feed", use_container_width=True):
            for _k in ("live_feed","players_df_live","live_formations","live_coaches","_auto_refreshed_feed"):
                st.session_state.pop(_k, None)

        # Use live-merged players_df if available
        if "players_df_live" in st.session_state:
            players_df = st.session_state["players_df_live"]

        # Lazy-load coach/formation for selected club in background
        if sel_club and sel_club not in st.session_state.get("live_coaches", {}):
            try:
                _tinfo = get_live_team_info(sel_club)
                _lf = st.session_state.get("live_formations", {})
                _lc = st.session_state.get("live_coaches", {})
                if _tinfo.get("formation"):
                    _lf[sel_club] = _tinfo["formation"]
                if _tinfo.get("manager"):
                    _lc[sel_club] = _tinfo["manager"]
                st.session_state["live_formations"] = _lf
                st.session_state["live_coaches"]    = _lc
            except Exception:
                pass

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 AI Scout","🔎 Player Search","⚡ What-If Transfer","📊 Team Analysis"])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — AI SCOUT
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader(f"AI Scouting Report — {sel_club}")

        # ── Season Context Banner ─────────────────────────────────────────
        status_clr  = STATUS_COLORS.get(sel_status, "#546E7A")
        status_note = STATUS_NOTES.get(sel_status, "")
        bm_label    = (f"×{budget_mult:.1f} budget multiplier (parachute payments)"
                       if budget_mult >= 1.5
                       else f"×{budget_mult:.2f} budget multiplier" if budget_mult != 1.0
                       else "")
        badge_html = (
            f"<div style='background:{status_clr};color:white;padding:8px 16px;"
            f"border-radius:8px;margin-bottom:8px'>"
            f"<strong>{sel_club}</strong> — {sel_status}"
            + (f" &nbsp;|&nbsp; {bm_label}" if bm_label else "")
            + f"</div>"
            f"<div style='color:#555;font-size:0.9em;margin-bottom:12px'>{status_note}</div>"
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        st.caption(f"Objective: **{sel_goal}**")

        with st.spinner("Generating report…"):
            result = scout_report(sel_club, sel_goal, players_df, team_stats_df,
                                  rf, sc, val_feats, arch_pipe, arch_feats,
                                  style_pipe, pred_vals, arch_labs,
                                  budget_mult=budget_mult)
        if result[0] is None:
            st.error("No data for selected club."); st.stop()
        (weak_df, shortlists, team_style, formation, top3_pos, g_cfg,
         budget_info, churn_info, age_prof, diagnoses, transfer_plan) = result

        if sel_form != "Auto-detect":
            formation = sel_form
        gap_dict = formation_gaps(formation, players_df[players_df["club"]==sel_club])

        # Style + formation header
        style_clr = {"Possession-Based":"#1565C0","High-Press":"#C62828",
                     "Counter-Attacking":"#2E7D32","Direct/Long-Ball":"#E65100","Hybrid":"#6A1B9A"}
        clr = style_clr.get(team_style,"#455A64")
        _coach_name = st.session_state.get("live_coaches", {}).get(sel_club, "")
        h1, h2, h3, h4 = st.columns([1.2, 0.9, 1.4, 1.1])
        h1.markdown(
            f"**Play Style:** <span style='background:{clr};color:white;"
            f"padding:3px 12px;border-radius:12px'>{team_style}</span>",
            unsafe_allow_html=True,
        )
        h2.markdown(f"**Formation:** `{formation}`")
        h3.markdown(
            f"**Formation Gaps:** "
            f"{', '.join(f'{p}(+{n})' for p,n in gap_dict.items()) or 'None detected'}"
        )
        if _coach_name:
            h4.markdown(
                f"**Manager:** <span style='color:#1565C0;font-weight:600'>{_coach_name}</span>",
                unsafe_allow_html=True,
            )
        st.write("")

        # ── SQUAD HEALTH DASHBOARD ─────────────────────────────────────────
        st.markdown("### Squad Health Dashboard")
        sh1, sh2, sh3, sh4, sh5 = st.columns(5)

        churn_score = churn_info["score"]
        churn_clr   = ("#2E7D32" if churn_score < 25 else
                       "#FB8C00" if churn_score < 40 else "#E53935")
        tier_clrs = {1:"#1565C0", 2:"#2E7D32", 3:"#6A1B9A"}
        t_clr = tier_clrs.get(budget_info["tier"], "#455A64")

        def _stat_card(label, value, sub, accent):
            return (
                f"<div style='background:{accent}0f;border:1px solid {accent}28;"
                f"border-radius:13px;padding:12px 14px;height:100%'>"
                f"<div style='font-size:.63rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:.55px;color:{accent}aa;margin-bottom:5px'>{label}</div>"
                f"<div style='font-family:\"Fira Code\",monospace;font-size:1.35rem;"
                f"font-weight:700;color:{accent};line-height:1.1'>{value}</div>"
                f"<div style='font-size:.72rem;color:#78909C;margin-top:4px;"
                f"line-height:1.35'>{sub}</div></div>"
            )

        sh1.markdown(_stat_card(
            "Churn Score", f"{churn_score:.0f}%", churn_info['label'], churn_clr),
            unsafe_allow_html=True)
        sh2.markdown(_stat_card(
            "League Tier", f"T{budget_info['tier']}",
            budget_info['tier_label'].split('—')[-1].strip(), t_clr),
            unsafe_allow_html=True)
        sh3.markdown(_stat_card(
            "Transfer Budget",
            f"€{budget_info['budget_lo_m']:.0f}–{budget_info['budget_hi_m']:.0f}M",
            f"Summer €{budget_info['summer_budget_m']}M", "#0097A7"),
            unsafe_allow_html=True)
        sh4.markdown(_stat_card(
            "Age Profile", f"{age_prof['avg_age']} yrs",
            age_prof['profile'], "#F57F17"),
            unsafe_allow_html=True)
        sh5.markdown(_stat_card(
            "Expiring Ctrs", f"{churn_info['departing_pct']:.0f}%",
            "of squad ≤ 1yr left", "#E53935"),
            unsafe_allow_html=True)
        st.caption(f"_{churn_info['interpretation']}_")
        st.write("")

        # Age tier breakdown
        ab1, ab2, ab3, ab4 = st.columns(4)
        ab1.metric("Academy (≤20)",   age_prof["academy_u21"],
                   help="Develop / Loan out")
        ab2.metric("Prime Asset (21-23)", age_prof["prime_asset_21_23"],
                   help="Highest resale value window — integrate or sell at peak")
        ab3.metric("Peak (24-28)",     age_prof["peak_24_28"],
                   help="Core squad / winning now")
        ab4.metric("Experience (29+)", age_prof["experienced_29plus"],
                   help="Leadership / squad depth")

        # Tactical weakness diagnosis
        if diagnoses:
            with st.expander("🔍 Tactical Weakness Diagnosis", expanded=True):
                for d in diagnoses:
                    icon = {"Attacking":"⚽","Defensive":"🛡️","Physical":"💪","Build-Up":"🎯"}.get(d["type"],"🔍")
                    st.markdown(
                        f"**{icon} {d['deficiency']}** — {d['detail']} "
                        f"→ Recruit **{d['recommended_position']}**"
                    )
        st.divider()

        cf, cd = st.columns(2)
        with cf:
            st.plotly_chart(formation_chart(players_df[players_df["club"]==sel_club],
                                            formation, gap_dict), use_container_width=True,
                            key="tab1_formation")
        with cd:
            st.plotly_chart(squad_depth_chart(players_df[players_df["club"]==sel_club]),
                            use_container_width=True, key="tab1_depth")

        # Priority positions
        st.markdown("### Priority Positions to Strengthen")
        fig_pos = px.bar(weak_df, x="position", y="delta",
                         color="delta", color_continuous_scale=["#ef5350","#ffee58","#66bb6a"],
                         labels={"delta":"Team vs League OVR gap","position":"Position"},
                         title="Squad Rating Gap vs League Average")
        fig_pos.update_layout(coloraxis_showscale=False, height=320)
        for _, rg in weak_df.iterrows():
            if rg.get("formation_gap",0)>0:
                fig_pos.add_annotation(x=rg["position"],y=rg["delta"],
                                       text=f"Gap+{int(rg['formation_gap'])}",
                                       showarrow=False,yshift=12,font=dict(color="#C62828",size=10))
        st.plotly_chart(fig_pos, use_container_width=True, key="tab1_priority_pos")
        st.markdown(f"**Top 3 Priority Positions:** `{'` · `'.join(top3_pos)}`")
        st.divider()

        # Per-position shortlists
        _tier_colors = {1: "#1565C0", 2: "#2E7D32", 3: "#6A1B9A"}
        _tier_names  = {1: "T1 Elite", 2: "T2 Competitive", 3: "T3 Development"}

        _live_transfers = st.session_state.get("live_feed", {}).get("transfers", [])

        for pos in top3_pos:
            sl = shortlists.get(pos)
            if sl is None or sl.empty: continue
            sl = sl.reset_index(drop=True)

            # Enrich shortlist with realistic fee, injury risk, rumor level
            _fee_series  = sl.apply(estimate_realistic_fee, axis=1)
            _inj_series  = sl.apply(get_injury_profile, axis=1)
            _rmr_series  = sl.apply(lambda r: get_transfer_rumor_level(r, _live_transfers), axis=1)
            sl["realistic_fee_m"]  = _fee_series.apply(lambda d: d["realistic_fee_m"])
            sl["contract_leverage"]= _fee_series.apply(lambda d: d["contract_leverage"])
            sl["club_stance"]      = _fee_series.apply(lambda d: d["willing_to_sell"])
            sl["injury_count"]     = _inj_series.apply(lambda d: d["injury_count"])
            sl["injury_risk"]      = _inj_series.apply(lambda d: d["risk"])
            sl["rumor_prob"]       = _rmr_series.apply(lambda d: d["probability"])
            sl["rumor_level"]      = _rmr_series.apply(lambda d: d["level"])

            # ── Section header ────────────────────────────────────────────────
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin:8px 0 10px'>"
                f"<div style='width:4px;height:26px;background:linear-gradient(180deg,#1565C0,#0097A7);"
                f"border-radius:2px;flex-shrink:0'></div>"
                f"<span style='font-size:1.12rem;font-weight:700'>{pos} — Recommended Targets</span>"
                f"<span style='font-size:.75rem;background:rgba(21,101,192,.1);color:#1565C0;"
                f"border:1px solid rgba(21,101,192,.2);padding:2px 9px;border-radius:20px;"
                f"font-weight:600'>{len(sl)} candidates · all leagues</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Table ─────────────────────────────────────────────────────────
            disp = ["scout_label","name","club","league_name","age","overall_rating",
                    "projected_ovr","predicted_value_m","realistic_fee_m","est_cost_m",
                    "contract_years_left","contract_leverage","club_stance",
                    "injury_count","injury_risk","rumor_prob","rumor_level",
                    "fit_score","success_prob","attitude_grade","dev_phase",
                    "career_phase","trend","window","affordability",
                    "resale_3yr_m","resale_roi","fc_dev_score","archetype","why"]
            avail = [c for c in disp if c in sl.columns]
            tbl = sl[avail].rename(columns={
                "scout_label":"Tag","name":"Name","club":"Club",
                "league_name":"League","age":"Age",
                "overall_rating":"OVR","projected_ovr":"Proj.OVR",
                "predicted_value_m":"Mkt Value(€M)","realistic_fee_m":"Real Fee(€M)",
                "est_cost_m":"Est.Cost(€M)","contract_years_left":"Ctr.Yrs",
                "contract_leverage":"Leverage","club_stance":"Seller Stance",
                "injury_count":"Injuries","injury_risk":"Inj.Risk",
                "rumor_prob":"Rumor%","rumor_level":"Rumor",
                "fit_score":"Fit%","success_prob":"Success%",
                "attitude_grade":"Grade","dev_phase":"Dev Phase",
                "career_phase":"Phase","trend":"Trend",
                "window":"Transfer Window","affordability":"Afford.",
                "resale_3yr_m":"Resale 3yr(€M)","resale_roi":"ROI",
                "fc_dev_score":"FC Dev","archetype":"Archetype","why":"Why?"})
            st.dataframe(
                tbl,
                column_config={
                    "Tag":          st.column_config.TextColumn("Tag", width="small",
                                    help="Hidden Gem = top performer from lower league | Elite = Tier 1"),
                    "OVR":          st.column_config.ProgressColumn("OVR",  min_value=45, max_value=95, format="%d"),
                    "Proj.OVR":     st.column_config.ProgressColumn("Proj.OVR", min_value=45, max_value=98, format="%d"),
                    "Fit%":         st.column_config.ProgressColumn("Fit%", min_value=0, max_value=100, format="%.0f"),
                    "FC Dev":       st.column_config.ProgressColumn("FC Dev", min_value=0, max_value=100, format="%.0f"),
                    "Rumor%":       st.column_config.ProgressColumn("Rumor%", min_value=0, max_value=100, format="%d%%",
                                    help="Probability of player moving this window"),
                    "Mkt Value(€M)":st.column_config.NumberColumn("Mkt Value(€M)", format="€%.1f M"),
                    "Real Fee(€M)": st.column_config.NumberColumn("Real Fee(€M)",  format="€%.1f M",
                                    help="Realistic transfer fee accounting for contract, importance, release clause"),
                    "Est.Cost(€M)": st.column_config.NumberColumn("Est.Cost(€M)",  format="€%.1f M"),
                    "Injuries":     st.column_config.NumberColumn("Injuries", format="%d",
                                    help="Career injury count estimate"),
                    "Why?":         st.column_config.TextColumn("Why?", width="large"),
                },
                use_container_width=True,
                hide_index=True,
            )

            # ── Player selector ───────────────────────────────────────────────
            # Use index as selectbox value to handle duplicate player names
            _p_labels = [
                f"{r['name']}  ({r.get('club','?')})  OVR {r.get('overall_rating','?')}"
                + (f"  {r.get('scout_label','')}" if r.get("scout_label") else "")
                for _, r in sl.iterrows()
            ]
            _deep_idx = st.selectbox(
                "Select player for Deep Dive",
                options=list(range(len(_p_labels))),
                format_func=lambda i: _p_labels[i],
                key=f"tab1_sel_{pos}",
                help="Select any candidate to update the analysis panel below",
            )
            deep_p = sl.iloc[_deep_idx]

            st.markdown(
                f"<div class='analysis-header'>{deep_p['name']} — Full Analysis</div>",
                unsafe_allow_html=True,
            )
            with st.container():
                # ── Player card ───────────────────────────────────────────────
                _p_tier  = int(deep_p.get("_tier", league_tier(str(deep_p.get("league", sel_lg)))))
                _tc      = _tier_colors.get(_p_tier, "#455A64")
                _tn      = _tier_names.get(_p_tier, "")
                _fit_n   = int(deep_p.get("fit_score", 0))
                _succ_n  = int(float(deep_p.get("success_prob_raw", 0.5)) * 100)
                _ovr_n   = int(deep_p.get("overall_rating", 70))
                _proj_n  = int(deep_p.get("projected_ovr", _ovr_n))
                _gem     = deep_p.get("scout_label", "")
                _val_m   = float(deep_p.get("predicted_value_m", 0))
                _fit_clr = "#1565C0" if _fit_n >= 60 else ("#FB8C00" if _fit_n >= 40 else "#E53935")
                _suc_clr = "#2E7D32" if _succ_n >= 60 else ("#FB8C00" if _succ_n >= 40 else "#E53935")
                _ovr_clr = ("#1565C0" if _ovr_n >= 75 else "#2E7D32" if _ovr_n >= 68 else "#78909C")

                # Enriched data: fee, injury, rumors
                _fee_info   = estimate_realistic_fee(deep_p)
                _inj_info   = get_injury_profile(deep_p)
                _rmr_info   = get_transfer_rumor_level(
                    deep_p,
                    st.session_state.get("live_feed", {}).get("transfers", [])
                )
                _real_fee_m = _fee_info["realistic_fee_m"]
                _inj_cnt    = _inj_info["injury_count"]
                _inj_clr    = _inj_info["risk_color"]
                _rmr_clr    = _rmr_info["color"]
                _rmr_prob   = _rmr_info["probability"]
                _rmr_lvl    = _rmr_info["level"]

                _gem_html = (
                    f'<span style="background:#FFF3E0;color:#E65100;'
                    f'border:1px solid #FFCC80;padding:2px 10px;border-radius:99px;'
                    f'font-size:.67rem;font-weight:700;letter-spacing:.03em">{_gem}</span>'
                    if _gem else ""
                )
                _inj_html = (
                    f'<span style="background:{_inj_clr}14;color:{_inj_clr};'
                    f'border:1px solid {_inj_clr}30;padding:2px 10px;border-radius:99px;'
                    f'font-size:.67rem;font-weight:700;letter-spacing:.03em">'
                    f'{_inj_cnt} inj.</span>'
                ) if _inj_cnt > 0 else ""
                _rmr_html = (
                    f'<span style="background:{_rmr_clr}14;color:{_rmr_clr};'
                    f'border:1px solid {_rmr_clr}30;padding:2px 10px;border-radius:99px;'
                    f'font-size:.67rem;font-weight:700;letter-spacing:.03em">'
                    f'Rumor: {_rmr_lvl}</span>'
                )
                _card_html = (
                    f'<div style="background:linear-gradient(145deg,rgba(21,101,192,.055) 0%,rgba(0,151,167,.025) 100%);'
                    f'border:1px solid rgba(21,101,192,.18);border-radius:18px;padding:18px 22px;'
                    f'display:flex;align-items:flex-start;gap:18px;margin-bottom:16px;'
                    f'box-shadow:0 3px 14px rgba(21,101,192,.09)">'
                    f'<div style="background:linear-gradient(145deg,{_ovr_clr},{_ovr_clr}cc);color:#fff;'
                    f'font-family:\'Fira Code\',monospace;font-size:1.9rem;font-weight:700;'
                    f'min-width:68px;height:68px;border-radius:14px;display:flex;flex-direction:column;'
                    f'align-items:center;justify-content:center;line-height:1;'
                    f'box-shadow:0 5px 16px {_ovr_clr}44;flex-shrink:0">'
                    f'<span>{_ovr_n}</span>'
                    f'<span style="font-size:.52rem;font-weight:600;letter-spacing:.05em;opacity:.8;margin-top:2px">OVR</span>'
                    f'</div>'
                    f'<div style="flex:1;min-width:0">'
                    f'<div style="font-size:1.2rem;font-weight:800;letter-spacing:-.02em;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#0D2340">{deep_p["name"]}</div>'
                    f'<div style="font-size:.79rem;color:#78909C;margin-top:3px;font-weight:400">'
                    f'{deep_p.get("club","—")} &nbsp;&middot;&nbsp; '
                    f'{deep_p.get("league_name", deep_p.get("league",""))} &nbsp;&middot;&nbsp; '
                    f'{pos} &nbsp;&middot;&nbsp; Age&nbsp;{int(deep_p.get("age",25))}</div>'
                    f'<div style="margin-top:9px;display:flex;gap:5px;flex-wrap:wrap;align-items:center">'
                    f'<span style="background:{_tc}14;color:{_tc};border:1px solid {_tc}30;'
                    f'padding:2px 10px;border-radius:99px;font-size:.67rem;font-weight:700;'
                    f'letter-spacing:.03em">{_tn}</span>'
                    f'{_gem_html}'
                    f'<span style="background:rgba(21,101,192,.09);color:#1565C0;'
                    f'border:1px solid rgba(21,101,192,.22);padding:2px 10px;border-radius:99px;'
                    f'font-size:.67rem;font-weight:700;letter-spacing:.03em">{deep_p.get("dev_phase","—")}</span>'
                    f'{_inj_html}'
                    f'{_rmr_html}'
                    f'</div>'
                    f'<div style="margin-top:12px;display:flex;gap:20px">'
                    f'<div style="flex:1">'
                    f'<div style="font-size:.62rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.55px;color:#90A4AE;margin-bottom:5px">Tactical Fit</div>'
                    f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<div style="flex:1;height:6px;background:rgba(0,0,0,.07);border-radius:99px;overflow:hidden">'
                    f'<div style="width:{_fit_n}%;height:100%;border-radius:99px;'
                    f'background:linear-gradient(90deg,{_fit_clr},{_fit_clr}bb)"></div></div>'
                    f'<span style="font-family:\'Fira Code\',monospace;font-size:.78rem;'
                    f'font-weight:700;color:{_fit_clr};min-width:34px">{_fit_n}%</span>'
                    f'</div></div>'
                    f'<div style="flex:1">'
                    f'<div style="font-size:.62rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:.55px;color:#90A4AE;margin-bottom:5px">Success Prob.</div>'
                    f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<div style="flex:1;height:6px;background:rgba(0,0,0,.07);border-radius:99px;overflow:hidden">'
                    f'<div style="width:{_succ_n}%;height:100%;border-radius:99px;'
                    f'background:linear-gradient(90deg,{_suc_clr},{_suc_clr}bb)"></div></div>'
                    f'<span style="font-family:\'Fira Code\',monospace;font-size:.78rem;'
                    f'font-weight:700;color:{_suc_clr};min-width:34px">{_succ_n}%</span>'
                    f'</div></div></div></div>'
                    f'<div style="text-align:right;flex-shrink:0;display:flex;flex-direction:column;gap:8px">'
                    f'<div><div style="font-family:\'Fira Code\',monospace;font-size:1.35rem;font-weight:700;'
                    f'color:#1565C0;letter-spacing:-.02em">€{_val_m:.1f}M</div>'
                    f'<div style="font-size:.63rem;color:#B0BEC5;margin-top:1px;letter-spacing:.02em">MARKET VALUE</div></div>'
                    f'<div><div style="font-family:\'Fira Code\',monospace;font-size:1.05rem;font-weight:700;'
                    f'color:#E65100;letter-spacing:-.01em">€{_real_fee_m:.1f}M</div>'
                    f'<div style="font-size:.63rem;color:#B0BEC5;margin-top:1px;letter-spacing:.02em">REALISTIC FEE</div></div>'
                    f'<div><div style="font-family:\'Fira Code\',monospace;font-size:1rem;font-weight:700;'
                    f'color:#0097A7">{_proj_n}</div>'
                    f'<div style="font-size:.63rem;color:#B0BEC5;margin-top:1px;letter-spacing:.02em">PROJ. OVR</div></div>'
                    f'<div style="font-size:.82rem;font-weight:700;color:#2E7D32">{deep_p.get("trend","—")}</div>'
                    f'</div></div>'
                )
                st.markdown(_card_html, unsafe_allow_html=True)

                # ── Fee / Injury / Rumor Intelligence row ─────────────────────
                _fi1, _fi2, _fi3 = st.columns(3)
                with _fi1:
                    _rc_label = (f" (release clause: €{_fee_info['release_clause_m']:.1f}M)"
                                 if _fee_info["has_release_clause"] else
                                 f" (est. clause: €{_fee_info['release_clause_m']:.1f}M)")
                    st.markdown(
                        f"<div style='background:rgba(230,81,0,.06);border:1px solid rgba(230,81,0,.2);"
                        f"border-radius:12px;padding:12px 14px'>"
                        f"<div style='font-size:.62rem;font-weight:700;text-transform:uppercase;"
                        f"letter-spacing:.5px;color:#E65100aa;margin-bottom:6px'>Transfer Fee Analysis</div>"
                        f"<div style='font-size:.82rem;color:#37474F;line-height:1.55'>"
                        f"<b>Contract leverage:</b> {_fee_info['contract_leverage']}<br>"
                        f"<b>Importance:</b> {_fee_info['importance']}<br>"
                        f"<b>Seller stance:</b> {_fee_info['willing_to_sell']}<br>"
                        f"<b style='color:#E65100'>Realistic fee:</b> €{_fee_info['realistic_fee_m']:.1f}M{_rc_label}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                with _fi2:
                    _inj_types_str = (", ".join(_inj_info["recent_types"])
                                      if _inj_info["recent_types"] else "None on record")
                    st.markdown(
                        f"<div style='background:{_inj_clr}08;border:1px solid {_inj_clr}22;"
                        f"border-radius:12px;padding:12px 14px'>"
                        f"<div style='font-size:.62rem;font-weight:700;text-transform:uppercase;"
                        f"letter-spacing:.5px;color:{_inj_clr}aa;margin-bottom:6px'>Injury Record</div>"
                        f"<div style='font-size:.82rem;color:#37474F;line-height:1.55'>"
                        f"<b>Total injuries:</b> <span style='color:{_inj_clr};font-weight:700'>{_inj_cnt}</span><br>"
                        f"<b>Risk level:</b> <span style='color:{_inj_clr};font-weight:700'>{_inj_info['risk']}</span><br>"
                        f"<b>Types:</b> {_inj_types_str}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                with _fi3:
                    _rmr_reason_str = (" · ".join(_rmr_info["reasons"])
                                       if _rmr_info["reasons"] else "No strong signals")
                    st.markdown(
                        f"<div style='background:{_rmr_clr}08;border:1px solid {_rmr_clr}22;"
                        f"border-radius:12px;padding:12px 14px'>"
                        f"<div style='font-size:.62rem;font-weight:700;text-transform:uppercase;"
                        f"letter-spacing:.5px;color:{_rmr_clr}aa;margin-bottom:6px'>Transfer Rumor Intelligence</div>"
                        f"<div style='font-size:.82rem;color:#37474F;line-height:1.55'>"
                        f"<b>Move probability:</b> <span style='color:{_rmr_clr};font-weight:700'>{_rmr_prob}% — {_rmr_lvl}</span><br>"
                        f"<b>Signals:</b> {_rmr_reason_str}<br>"
                        f"<b>Source:</b> {_rmr_info['source_label']}"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                st.write("")

                col_r, col_p = st.columns(2)
                with col_r:
                    lg4p = players_df[players_df["league"] == deep_p.get("league", "")]
                    st.plotly_chart(
                        radar_chart(deep_p, lg4p, pos),
                        use_container_width=True,
                        key=f"tab1_radar_{pos}_{_deep_idx}",
                    )
                with col_p:
                    pr_info = analyze_progression(deep_p)
                    st.plotly_chart(
                        progression_spark(pr_info, deep_p["name"]),
                        use_container_width=True,
                        key=f"tab1_spark_{pos}_{_deep_idx}",
                    )
                    st.markdown(
                        f"**Phase:** {pr_info['phase']} | "
                        f"**Trend:** {pr_info['icon']} {pr_info['trend']} | "
                        f"**Projected OVR:** {pr_info['projected']}"
                    )
                    st.caption(deep_p.get("why",""))

                # FBref-style full stat table
                stat_rows = []
                for cat, sts in [
                    ("Standard",  ["goals_per90","assists_per90","shots_on_target_pct",
                                   "yellow_cards_per90","red_cards_per90","minutes_per90_ratio"]),
                    ("Shooting",  ["shots_total_per90","npxg_per90","npxg_per_shot","xg_per90"]),
                    ("Passing",   ["pass_completion","pass_completion_short","pass_completion_medium",
                                   "pass_completion_long","key_passes_per90","progressive_passes",
                                   "xa_per90","through_balls_per90"]),
                    ("Creation",  ["sca_per90","gca_per90","crosses_per90"]),
                    ("Defence",   ["tackles_per90","tackles_won_pct","interceptions_per90",
                                   "blocks_per90","clearances_per90","pressures_per90",
                                   "pressure_success_pct","aerial_duels_won_pct","duels_won_pct"]),
                    ("Possession",["dribbles_per90","progressive_carries","touches_per90",
                                   "touches_att3rd_per90","progressive_passes_received_per90"]),
                    ("Misc",      ["fouls_committed_per90","fouls_drawn_per90","offsides_per90"]),
                ]:
                    for s in sts:
                        if s in deep_p.index and pd.notna(deep_p[s]):
                            stat_rows.append({
                                "Category": cat,
                                "Stat": s.replace("_per90","").replace("_"," ").title(),
                                "Value": round(float(deep_p[s]), 3),
                            })
                if stat_rows:
                    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

                # Transfer-specific summary
                tw_adv = transfer_window_advice(deep_p, budget_info)
                rp_adv = resale_projection(deep_p)
                _club_lg = players_df.loc[players_df["club"]==sel_club, "league"]
                _club_lg = _club_lg.iloc[0] if not _club_lg.empty else sel_lg
                sp_adv = estimate_success_probability(
                    deep_p, float(deep_p.get("fit_score", 50)),
                    str(deep_p.get("league", _club_lg)), _club_lg,
                )
                ci_adv = calc_churn_impact(deep_p, players_df[players_df["club"]==sel_club])
                tfc1, tfc2, tfc3, tfc4 = st.columns(4)
                tfc1.metric("Est. Transfer Cost", f"€{tw_adv['estimated_cost_m']}M",
                            tw_adv["affordability"])
                tfc2.metric("Recommended Window", tw_adv["recommended_window"][:20])
                tfc3.metric("Success Probability", f"{sp_adv*100:.0f}%",
                            help="Probability of meaningful contribution in Season 1")
                tfc4.metric("3yr Resale Value", f"€{rp_adv['projected_3yr_m']}M",
                            f"{rp_adv['roi_pct']:+.0f}% ROI")
                st.caption(
                    f"Churn impact: {ci_adv['impact']} "
                    f"(squad churn {ci_adv['churn_before']:.0f}% → "
                    f"{ci_adv['churn_after']:.0f}% after signing)"
                )

                # ── AI Scout Report ───────────────────────────────────────────
                st.markdown("---")
                _ai_key  = f"ai_report_{pos}_{_deep_idx}"
                _has_key = bool(os.getenv("ANTHROPIC_API_KEY", ""))
                if not _has_key:
                    st.markdown(
                        "<div style='background:rgba(21,101,192,.07);border-left:3px solid #1565C0;"
                        "border-radius:0 8px 8px 0;padding:8px 14px;font-size:.8rem;color:#1565C0;"
                        "margin-bottom:8px'>"
                        "<b>Claude AI Reports</b> — set <code>ANTHROPIC_API_KEY</code> as an "
                        "environment variable before launching to enable full AI narrative. "
                        "Without it, a structured data summary is generated instead.</div>",
                        unsafe_allow_html=True,
                    )
                if st.button("Generate Scout Report", key=f"btn_{_ai_key}",
                             use_container_width=True):
                    with st.spinner("Writing scout report…"):
                        _ai_txt = generate_ai_scout_report(
                            player_row  = deep_p,
                            target_club = sel_club,
                            target_goal = sel_goal,
                            fit_info    = {
                                "fit_score":   int(deep_p.get("fit_score", 0)),
                                "success_prob":int(float(deep_p.get("success_prob_raw",0.5))*100),
                            },
                            fee_info    = _fee_info,
                            inj_info    = _inj_info,
                            rumor_info  = _rmr_info,
                            prog_info   = analyze_progression(deep_p),
                        )
                        st.session_state[_ai_key] = _ai_txt
                if _ai_key in st.session_state:
                    st.markdown(
                        f"<div style='background:rgba(21,101,192,.05);border:1px solid rgba(21,101,192,.18);"
                        f"border-radius:14px;padding:18px 22px;line-height:1.7;font-size:.9rem'>"
                        f"{st.session_state[_ai_key].replace(chr(10), '<br>')}</div>",
                        unsafe_allow_html=True,
                    )
            st.divider()

        # ── LIVE TRANSFER ACTIVITY ───────────────────────────────────────
        _live = st.session_state.get("live_feed", {})
        _tf_df = transfers_to_df(_live)
        if not _tf_df.empty:
            with st.expander(f"🔴 Live Transfer Activity — {len(_tf_df)} recent moves "
                             f"({_live.get('source','?')})", expanded=False):
                # Filter to clubs relevant to selected club if possible
                _canon = _canon_club(sel_club)
                _club_moves = _tf_df[
                    _tf_df["from_club"].str.contains(_canon, case=False, na=False) |
                    _tf_df["to_club"].str.contains(_canon, case=False, na=False)
                ]
                if not _club_moves.empty:
                    st.markdown(f"**Moves involving {sel_club}:**")
                    st.dataframe(_club_moves, use_container_width=True, hide_index=True)
                    st.write("")
                st.markdown("**All recent transfers in this league:**")
                # Filter to same league if possible
                _league_clubs = set(players_df[players_df["league"] == sel_lg]["club"].unique())
                _canon_clubs  = {_canon_club(c) for c in _league_clubs}
                _league_mask  = (
                    _tf_df["from_club"].apply(lambda c: _canon_club(str(c)) in _canon_clubs) |
                    _tf_df["to_club"].apply(lambda c: _canon_club(str(c)) in _canon_clubs)
                )
                _league_moves = _tf_df[_league_mask]
                st.dataframe(_league_moves if not _league_moves.empty else _tf_df.head(50),
                             use_container_width=True, hide_index=True)
                st.caption(f"Last updated: {_live.get('last_updated','?')[:16]} UTC  "
                           f"· Refresh in sidebar to update")

        # ── TRANSFER WINDOW PLAN ─────────────────────────────────────────
        st.markdown("### Transfer Window Plan")

        # Strategy summary
        strat_lines = strategy_summary(sel_goal, churn_info, budget_info, age_prof, diagnoses)
        for line in strat_lines:
            st.markdown(f"- {line}")
        st.write("")

        feasible_clr = "#43A047" if transfer_plan["feasible"] else "#E53935"
        spend = transfer_plan["summer_est_spend_m"]
        bgt   = transfer_plan["summer_budget_m"]
        st.markdown(
            f"**Estimated Summer Spend:** "
            f"<span style='color:{feasible_clr};font-weight:700'>"
            f"€{spend}M</span> vs Budget €{bgt}M  "
            f"{'✅ Feasible' if transfer_plan['feasible'] else '⚠️ Over Budget'}",
            unsafe_allow_html=True,
        )
        st.write("")

        tp1, tp2 = st.columns([3, 1])
        with tp1:
            if transfer_plan["summer"]:
                st.markdown("#### Summer Window Targets")
                summer_df = pd.DataFrame(transfer_plan["summer"])[
                    ["position","name","age","ovr","value_m","est_cost_m",
                     "dev_phase","fit_score","window","league"]
                ].rename(columns={
                    "position":"Pos","name":"Player","age":"Age","ovr":"OVR",
                    "value_m":"Value €M","est_cost_m":"Est.Cost €M",
                    "dev_phase":"Dev Phase","fit_score":"Fit%",
                    "window":"Timing","league":"League",
                })
                st.dataframe(summer_df, use_container_width=True, hide_index=True)

        with tp2:
            if transfer_plan["loans"]:
                st.markdown("#### Loan Candidates")
                loan_df = pd.DataFrame(transfer_plan["loans"])[
                    ["position","name","age","ovr","dev_phase","fit_score"]
                ].rename(columns={
                    "position":"Pos","name":"Player","age":"Age","ovr":"OVR",
                    "dev_phase":"Phase","fit_score":"Fit%",
                })
                st.dataframe(loan_df, use_container_width=True, hide_index=True)

        if transfer_plan["winter"]:
            with st.expander("January Window Options"):
                w_df = pd.DataFrame(transfer_plan["winter"])[
                    ["position","name","age","ovr","value_m","est_cost_m","window"]
                ].rename(columns={
                    "position":"Pos","name":"Player","age":"Age","ovr":"OVR",
                    "value_m":"Value €M","est_cost_m":"Est.Cost €M","window":"Timing",
                })
                st.dataframe(w_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — PLAYER SEARCH
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Player Search & Profile")
        c1,c2,c3 = st.columns([2,1,1])
        with c1: q = st.text_input("Search name", placeholder="e.g. Erling Haaland")
        with c2: pf = st.selectbox("Position", ["All"]+POSITIONS)
        with c3: lf2= st.selectbox("League", ["All"]+sorted(LEAGUES.values()))

        flt = players_df.copy()
        if q:     flt = flt[flt["name"].str.contains(q,case=False,na=False)]
        if pf!="All": flt = flt[flt["position"]==pf]
        if lf2!="All":
            lc2 = next((k for k,v in LEAGUES.items() if v==lf2),None)
            if lc2: flt = flt[flt["league"]==lc2]

        if flt.empty:
            st.info("No players found.")
        else:
            prv = ["name","club","league_name","age","position","overall_rating",
                   "potential","predicted_value_m","archetype"]
            st.dataframe(flt[[c for c in prv if c in flt.columns]].head(50)
                         .reset_index(drop=True), use_container_width=True)

            sel_name = st.selectbox("Select player for full profile", flt["name"].tolist()[:100])
            pr = flt[flt["name"]==sel_name].iloc[0]
            st.markdown(f"### {pr['name']}")
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Club",pr["club"]); m2.metric("Age",pr["age"])
            m3.metric("Position",pr["position"]); m4.metric("OVR",pr["overall_rating"])
            m5.metric("Potential",pr["potential"])
            m6.metric("Value €M", f"{pr['predicted_value_m']:.1f}")
            m7,m8,m9,m10 = st.columns(4)
            m7.metric("Ctr Yrs",f"{pr['contract_years_left']}y")
            m8.metric("Archetype",str(pr.get("archetype","—")))
            m9.metric("Grade",calculate_attitude_grade(pr))
            prog_pr = analyze_progression(pr)
            m10.metric("Trend",prog_pr["icon"]+" "+prog_pr["trend"])

            # Fit for currently selected club
            tr2 = team_stats_df[team_stats_df["club"]==sel_club]
            if not tr2.empty:
                sf2 = [f for f in TEAM_STYLE_FEATURES if f in tr2.columns]
                if sf2:
                    tc2 = style_pipe.named_steps["km"].predict(tr2[sf2].fillna(0).values)[0]
                    tv2 = style_pipe.named_steps["km"].cluster_centers_[tc2]
                    st.metric(f"Fit Score for {sel_club}", f"{calculate_fit_score(pr,tv2,arch_pipe,arch_feats)}%")

            cr2, cp2 = st.columns(2)
            with cr2:
                st.plotly_chart(radar_chart(pr, players_df[players_df["league"]==pr["league"]], pr["position"]),
                                use_container_width=True,
                                key=f"tab2_radar_{pr.get('name','x')}")
            with cp2:
                st.plotly_chart(progression_spark(prog_pr, pr["name"]), use_container_width=True,
                                key=f"tab2_spark_{pr.get('name','x')}")
                st.markdown(f"**Phase:** {prog_pr['phase']} | **2yr:** {prog_pr['total']:+.1f} OVR | "
                            f"**Projected:** {prog_pr['projected']}")

            # FBref stat tabs
            st.markdown("#### Full Stats Breakdown (FBref categories)")
            stabs = st.tabs(["Standard","Shooting","Passing","Creation","Defence","Possession","Misc"])
            cat_sts = [
                ["goals_per90","assists_per90","shots_on_target_pct","minutes_per90_ratio"],
                ["shots_total_per90","npxg_per90","npxg_per_shot","xg_per90"],
                ["pass_completion","pass_completion_short","pass_completion_medium",
                 "pass_completion_long","key_passes_per90","progressive_passes","xa_per90"],
                ["sca_per90","gca_per90","through_balls_per90","crosses_per90"],
                ["tackles_per90","tackles_won_pct","interceptions_per90","blocks_per90",
                 "clearances_per90","pressures_per90","pressure_success_pct",
                 "aerial_duels_won_pct","duels_won_pct"],
                ["dribbles_per90","progressive_carries","touches_per90",
                 "touches_att3rd_per90","progressive_passes_received_per90"],
                ["fouls_committed_per90","fouls_drawn_per90","offsides_per90",
                 "yellow_cards_per90","red_cards_per90"],
            ]
            lg_p = players_df[(players_df["league"]==pr["league"])&(players_df["position"]==pr["position"])]
            for stab, sts in zip(stabs, cat_sts):
                with stab:
                    rows_ = []
                    for s in sts:
                        if s not in pr.index: continue
                        val = pr[s]
                        if pd.isna(val): continue
                        col_lg = lg_p[s].dropna() if s in lg_p.columns else pd.Series(dtype=float)
                        pct = round(float((col_lg<val).mean()*100)) if len(col_lg)>0 else 50
                        rows_.append({"Stat":s.replace("_per90","").replace("_"," ").title(),
                                      "Value":round(float(val),3),"League Pct":f"{pct}th"})
                    if rows_:
                        st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — WHAT-IF TRANSFER
    # ═══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader(f"What-If Transfer Simulator — {sel_club}")
        others = sorted(players_df[players_df["club"]!=sel_club]["name"].unique())[:300]
        if not others:
            st.info("No other players available for comparison."); st.stop()
        tgt_name = st.selectbox("Player to sign", others)
        tgt_matches = players_df[players_df["name"]==tgt_name]
        if tgt_matches.empty:
            st.warning(f"Player '{tgt_name}' not found in dataset."); st.stop()
        tgt  = tgt_matches.iloc[0]
        my_sq= players_df[players_df["club"]==sel_club].copy()

        def _sv(squad):
            def _n(col):
                return pd.to_numeric(squad[col], errors="coerce").fillna(0) if col in squad.columns else pd.Series([0.0])
            return {
                "avg_pass_completion":    float(_n("pass_completion").mean()),
                "avg_pressing_actions":   float((_n("tackles_per90") + _n("interceptions_per90")).mean()),
                "avg_progressive_passes": float(_n("progressive_passes").mean()),
                "avg_progressive_carries":float(_n("progressive_carries").mean()),
                "avg_key_passes":         float(_n("key_passes_per90").mean()),
                "avg_dribbles":           float(_n("dribbles_per90").mean()),
                "avg_crosses":            float(_n("crosses_per90").mean()),
                "avg_aerial_duels_won":   float(_n("aerial_duels_won_pct").mean()),
                "squad_age":              float(_n("age").mean()),
                "squad_rating":           float(_n("overall_rating").mean()),
            }

        cur = _sv(my_sq)
        same = my_sq[my_sq["position"]==tgt["position"]]
        if same.empty:
            sim = pd.concat([my_sq, tgt.to_frame().T], ignore_index=True); replaced="(new slot)"
        else:
            widx = same["overall_rating"].idxmin()
            replaced = my_sq.loc[widx,"name"]
            sim = pd.concat([my_sq.drop(widx), tgt.to_frame().T], ignore_index=True)
        new = _sv(sim)

        st.markdown(f"**Signing:** {tgt['name']} ({tgt.get('club','')}) · OVR {tgt['overall_rating']} · "
                    f"{tgt['position']} · €{tgt['predicted_value_m']:.1f}M")
        st.markdown(f"**Replacing:** {replaced}")

        metrics_keys = ["avg_pass_completion","avg_pressing_actions","avg_progressive_passes",
                        "avg_key_passes","avg_dribbles","squad_age","squad_rating"]
        ca,cb_ = st.columns(2)
        with ca:
            st.markdown("**Current Squad**")
            for m in metrics_keys:
                st.metric(m.replace("avg_","").replace("_"," ").title(), f"{cur.get(m,0):.3f}")
        with cb_:
            st.markdown("**After Transfer**")
            for m in metrics_keys:
                d = new.get(m,0)-cur.get(m,0)
                st.metric(m.replace("avg_","").replace("_"," ").title(), f"{new.get(m,0):.3f}", f"{d:+.3f}")

        cur_f = detect_team_formation(my_sq)
        new_f = detect_team_formation(sim)
        st.markdown(f"**Formation:** `{cur_f}` → `{new_f}`")

        prog_t = analyze_progression(tgt)
        st.plotly_chart(progression_spark(prog_t, tgt["name"]), use_container_width=True,
                        key=f"tab3_spark_{tgt.get('name','x')}")

        tr3 = team_stats_df[team_stats_df["club"]==sel_club]
        if not tr3.empty:
            sf3 = [f for f in TEAM_STYLE_FEATURES if f in tr3.columns]
            if sf3:
                tc3 = style_pipe.named_steps["km"].predict(tr3[sf3].fillna(0).values)[0]
                tv3 = style_pipe.named_steps["km"].cluster_centers_[tc3]
                st.metric(f"Tactical Fit for {sel_club}", f"{calculate_fit_score(tgt,tv3,arch_pipe,arch_feats)}%")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 — TEAM ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader(f"Team Analysis — {sel_club}")
        sq4 = players_df[players_df["club"]==sel_club].copy()
        if sq4.empty:
            st.warning("No squad data."); st.stop()

        f4    = detect_team_formation(sq4) if sel_form=="Auto-detect" else sel_form
        g4    = formation_gaps(f4, sq4)
        prog4 = sq4.apply(analyze_progression, axis=1)

        # ── Squad health & financial summary ─────────────────────────────
        churn4   = squad_churn_score(sq4)
        age4     = classify_age_profile(sq4)
        league4  = sq4["league"].iloc[0]
        budget4  = estimate_budget(league4, float(sq4["overall_rating"].mean()),
                                   players_df[players_df["league"]==league4]["overall_rating"],
                                   budget_mult=budget_mult)
        diags4   = diagnose_weaknesses(sq4, players_df[players_df["league"]==league4])

        s1,s2,s3,s4,s5,s6 = st.columns(6)
        s1.metric("Squad Size",    len(sq4))
        s2.metric("Avg OVR",       f"{sq4['overall_rating'].mean():.1f}")
        s3.metric("Avg Age",       f"{age4['avg_age']}")
        s4.metric("Avg Value €M",  f"{sq4['market_value_m'].mean():.1f}")
        rising = sum(1 for p in prog4 if p["trend"] in ("Rising Star","Improving"))
        s5.metric("Rising Players", rising)
        s6.metric("Churn Score",   f"{churn4['score']:.0f}%",
                  delta=churn4["label"], delta_color="off")

        # Churn + budget card
        st.write("")
        ta1, ta2, ta3 = st.columns(3)
        churn_clr4 = ("#43A047" if churn4["score"] < 25 else
                      "#FB8C00" if churn4["score"] < 40 else "#E53935")
        ta1.markdown(
            f"<div style='background:{churn_clr4}22;border-left:4px solid {churn_clr4};"
            f"padding:10px;border-radius:6px'>"
            f"<b>Squad Churn</b><br>"
            f"<span style='font-size:1.5rem;font-weight:700;color:{churn_clr4}'>{churn4['score']:.0f}%</span><br>"
            f"<small>{churn4['label']} — {churn4['interpretation']}</small></div>",
            unsafe_allow_html=True,
        )
        ta2.markdown(
            f"<div style='background:#0097A722;border-left:4px solid #0097A7;"
            f"padding:10px;border-radius:6px'>"
            f"<b>Transfer Budget Est.</b><br>"
            f"<span style='font-size:1.1rem;font-weight:700;color:#006064'>"
            f"€{budget4['budget_lo_m']}M–€{budget4['budget_hi_m']}M</span><br>"
            f"<small>{budget4['tier_label']} · {budget4['squad_position'].title()} squad</small></div>",
            unsafe_allow_html=True,
        )
        tier_clr4 = {1:"#1565C0",2:"#2E7D32",3:"#6A1B9A"}.get(budget4["tier"],"#455A64")
        ta3.markdown(
            f"<div style='background:{tier_clr4}22;border-left:4px solid {tier_clr4};"
            f"padding:10px;border-radius:6px'>"
            f"<b>Age Profile</b><br>"
            f"<span style='font-size:1.1rem;font-weight:700;color:{tier_clr4}'>"
            f"Avg {age4['avg_age']} yrs — {age4['profile']}</span><br>"
            f"<small>Academy {age4['academy_u21']} · "
            f"Prime {age4['prime_asset_21_23']} · "
            f"Peak {age4['peak_24_28']} · "
            f"Exp {age4['experienced_29plus']}</small></div>",
            unsafe_allow_html=True,
        )
        st.write("")

        # Age tier bar chart
        age_bar = go.Figure()
        age_labels = ["Academy\n(≤20)","Prime Asset\n(21-23)","Peak\n(24-28)","Experience\n(29+)"]
        age_counts = [age4["academy_u21"],age4["prime_asset_21_23"],
                      age4["peak_24_28"],age4["experienced_29plus"]]
        age_bar.add_bar(x=age_labels, y=age_counts,
                        marker_color=["#9E9E9E","#43A047","#1565C0","#F57F17"],
                        text=age_counts, textposition="auto")
        age_bar.update_layout(title="Squad Age Matrix", height=280,
                              showlegend=False, margin=dict(t=50,b=20,l=30,r=30))
        st.plotly_chart(age_bar, use_container_width=True, key="tab4_age_bar")

        if diags4:
            with st.expander("🔍 Tactical Weaknesses", expanded=False):
                for d in diags4:
                    icon = {"Attacking":"⚽","Defensive":"🛡️","Physical":"💪","Build-Up":"🎯"}.get(d["type"],"🔍")
                    st.markdown(
                        f"**{icon} {d['deficiency']}** ({d['type']}) — {d['detail']} "
                        f"→ Recruit **{d['recommended_position']}**"
                    )
        st.divider()

        c4f, c4d = st.columns(2)
        with c4f: st.plotly_chart(formation_chart(sq4, f4, g4), use_container_width=True, key="tab4_formation")
        with c4d: st.plotly_chart(squad_depth_chart(sq4), use_container_width=True, key="tab4_depth")

        # Progression table
        st.markdown("### Progression Overview")
        prows = []
        for (_,pr4), p4d in zip(sq4.iterrows(), prog4):
            prows.append({"Name":pr4["name"],"Pos":pr4["position"],"OVR":pr4["overall_rating"],
                          "Proj.OVR":p4d["projected"],"Phase":p4d["phase"],
                          "Trend":p4d["icon"]+" "+p4d["trend"],"2yr Δ":f"{p4d['total']:+.1f}"})
        st.dataframe(pd.DataFrame(prows).sort_values("2yr Δ",ascending=False),
                     use_container_width=True, hide_index=True)

        # Age vs OVR scatter
        st.markdown("### Age & OVR Distribution")
        fig_age = px.scatter(sq4, x="age", y="overall_rating", color="position",
                             size="market_value_m", hover_data=["name","potential"],
                             title="Age vs OVR (bubble = market value)")
        fig_age.add_vline(x=28,line_dash="dash",line_color="gray",
                          annotation_text="Peak ~28",annotation_position="top right")
        fig_age.update_layout(height=420)
        st.plotly_chart(fig_age, use_container_width=True, key="tab4_age_scatter")

        # Team radar vs league
        st.markdown("### Team Style Profile vs League Average")
        lg4 = players_df[players_df["league"]==sq4["league"].iloc[0]]
        kts = ["goals_per90","assists_per90","pass_completion","tackles_per90",
               "interceptions_per90","progressive_carries","sca_per90","pressures_per90"]
        avail_kts = [s for s in kts if s in sq4.columns and s in lg4.columns]
        tm_means = sq4[avail_kts].mean()
        pct4 = [round(float((lg4[s].dropna()<tm_means[s]).mean()*100)) for s in avail_kts]
        labs4 = [s.replace("_per90","").replace("_"," ").title() for s in avail_kts]
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatterpolar(r=pct4+[pct4[0]],theta=labs4+[labs4[0]],
                                         fill="toself",line_color="#1565C0",
                                         fillcolor="rgba(21,101,192,0.2)",name=sel_club))
        fig_tr.add_trace(go.Scatterpolar(r=[50]*len(labs4)+[50],theta=labs4+[labs4[0]],
                                         mode="lines",line=dict(dash="dash",color="gray"),
                                         name="League avg (50th pct)"))
        fig_tr.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                             title=f"{sel_club} — Team Style vs League",height=420)
        st.plotly_chart(fig_tr, use_container_width=True, key="tab4_radar_team")

        # Contract watch
        st.markdown("### Contract Watch (≤18 months remaining)")
        exp = sq4[sq4["contract_years_left"]<=1.5].sort_values("contract_years_left")
        if exp.empty:
            st.success("No contracts expiring within 18 months.")
        else:
            st.dataframe(exp[["name","position","age","overall_rating",
                               "contract_years_left","market_value_m"]].rename(columns={
                "name":"Name","position":"Pos","age":"Age","overall_rating":"OVR",
                "contract_years_left":"Ctr Yrs","market_value_m":"Value €M"}),
                use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
