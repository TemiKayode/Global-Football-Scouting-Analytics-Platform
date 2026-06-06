"""
Squad Analytics Module — v1.0
League tier classification · Squad churn · Age profile · Weakness diagnosis ·
Financial reality engine · Transfer success probability
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE TIER CLASSIFICATION
# Tier 1 = Elite (regular CL, massive TV revenue)
# Tier 2 = Competitive (occasional CL/EL, significant scouting base)
# Tier 3 = Development (talent incubators, lower budgets)
# ─────────────────────────────────────────────────────────────────────────────
LEAGUE_TIERS: dict[str, int] = {
    # Tier 1 — Elite
    "GB1": 1, "ES1": 1, "IT1": 1, "L1": 1, "FR1": 1,
    # Tier 2 — Competitive
    "PO1": 2, "GB2": 2, "BE1": 2, "TR1": 2,
    "SC1": 2, "AR1N": 2, "BRA1": 2, "NO1": 2,
    "ES2": 2, "IT2": 2, "L2":  2, "FR2": 2,
    # Tier 3 — Development
    "A1":   3, "SE1":  3, "PL1":  3, "RO1":  3, "C1":   3,
    "GR1":  3, "BU1":  3, "KR1":  3, "TS1":  3, "SL1":  3,
    "UNG1": 3, "RU1":  3,
    # Women's
    "WWSL": 2, "WGBL": 2, "WFRD1": 2, "WNWSL": 2,
    "WAUS": 3, "WBRA": 3, "WITA":  3,
}

TIER_LABELS: dict[int, str] = {
    1: "Tier 1 — Elite",
    2: "Tier 2 — Competitive",
    3: "Tier 3 — Development",
}

# Transfer budget ranges €M  (lo, hi)  by tier × squad_position
_BUDGETS: dict[int, dict[str, tuple]] = {
    1: {"top": (80, 200), "mid": (30, 80),  "bottom": (10, 40)},
    2: {"top": (15, 50),  "mid":  (5, 20),  "bottom":  (1, 10)},
    3: {"top":  (3, 12),  "mid":  (1,  4),  "bottom": (0.1, 1.5)},
}


def league_tier(league_code: str) -> int:
    return LEAGUE_TIERS.get(league_code, 3)


def estimate_budget(league_code: str,
                    team_avg_ovr: float,
                    league_ovrs: "pd.Series",
                    budget_mult: float = 1.0) -> dict:
    """
    Estimate a club's realistic transfer budget based on league tier and
    where they sit in their league by squad OVR.
    budget_mult is applied from season status (e.g. 2.4 for relegated clubs
    with parachute payments, 1.3 for title winners flush with prize money).
    Summer window takes ~75% of annual spend; winter ~25%.
    """
    tier = league_tier(league_code)
    pct  = float((league_ovrs < team_avg_ovr).mean()) if len(league_ovrs) > 0 else 0.5
    pos  = "top" if pct >= 0.67 else ("mid" if pct >= 0.33 else "bottom")
    lo, hi = _BUDGETS[tier][pos]
    lo = round(lo * budget_mult, 1)
    hi = round(hi * budget_mult, 1)
    return {
        "tier":            tier,
        "tier_label":      TIER_LABELS[tier],
        "squad_position":  pos,
        "budget_mult":     budget_mult,
        "budget_lo_m":     lo,
        "budget_hi_m":     hi,
        "summer_budget_m": round(hi * 0.75, 1),
        "winter_budget_m": round(hi * 0.25, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SQUAD CHURN SCORE
# Research: Champions avg 27.5% churn; promoted clubs avg 50%+.
# New signings contribute ~40% of available minutes in Season 1.
# ─────────────────────────────────────────────────────────────────────────────
def squad_churn_score(team_players: pd.DataFrame) -> dict:
    """
    Proxy churn (0–100) using:
    - % contracts expiring ≤ 1yr  (likely departures)   → 40% weight
    - % players on OVR decline ≥ 2  (candidates to sell) → 30% weight
    - % young players rising ≥ 2    (academy pushes)     → 15% weight
    - OVR volatility (std dev)                           → 15% weight
    """
    n = len(team_players)
    if n == 0:
        return {"score": 0.0, "label": "Unknown", "interpretation": "",
                "departing_pct": 0.0, "declining_pct": 0.0, "rising_young_pct": 0.0}

    cyl_col = team_players["contract_years_left"] if "contract_years_left" in team_players.columns \
              else pd.Series(2.0, index=team_players.index)
    dep = float((cyl_col.fillna(2.0) <= 1.0).mean())

    pr1_col = (team_players["past_rating_1yr"]
               if "past_rating_1yr" in team_players.columns
               else team_players["overall_rating"])
    yr1     = (team_players["overall_rating"] - pr1_col.fillna(team_players["overall_rating"])).fillna(0.0)
    dec     = float((yr1 <= -2).mean())
    ry      = float(((team_players["age"] <= 23) & (yr1 >= 2)).mean())
    vol     = float(yr1.std(ddof=0)) if n > 1 else 0.0

    score = float(dep * 40 + dec * 30 + ry * 15 + min(vol * 2, 15))
    score = round(min(score, 100.0), 1)

    if   score < 20: label, interp = "Stable Champion", "Very low churn — title-contender stability (research avg 27.5%)"
    elif score < 30: label, interp = "Stable",          "Good squad continuity — minimal disruption risk"
    elif score < 40: label, interp = "Moderate",        "Average churn — some settling-in period expected"
    elif score < 50: label, interp = "High Turnover",   "Above-average churn — disruption risk, limit to 4 signings max"
    else:            label, interp = "Rebuilding",      "High churn — major transition; new signings play ~40% minutes in Season 1"

    return {
        "score":            score,
        "label":            label,
        "interpretation":   interp,
        "departing_pct":    round(dep * 100, 1),
        "declining_pct":    round(dec * 100, 1),
        "rising_young_pct": round(ry  * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGE PROFILE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def classify_age_profile(team_players: pd.DataFrame) -> dict:
    """Break squad into four age tiers mirroring the research matrix."""
    ages = team_players["age"].dropna()
    if ages.empty:
        return {"avg_age": 0.0, "profile": "Unknown",
                "academy_u21": 0, "prime_asset_21_23": 0,
                "peak_24_28": 0, "experienced_29plus": 0}

    avg  = round(float(ages.mean()), 1)
    u21  = int((ages <= 20).sum())
    pa   = int(((ages >= 21) & (ages <= 23)).sum())
    pk   = int(((ages >= 24) & (ages <= 28)).sum())
    exp  = int((ages >= 29).sum())

    if   avg < 23: profile = "Very Young"
    elif avg < 25: profile = "Young & Developing"
    elif avg < 27: profile = "Balanced"
    elif avg < 29: profile = "Experienced"
    else:          profile = "Aging Squad"

    return {
        "avg_age":            avg,
        "profile":            profile,
        "academy_u21":        u21,
        "prime_asset_21_23":  pa,
        "peak_24_28":         pk,
        "experienced_29plus": exp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TACTICAL WEAKNESS DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
def _col_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    return float(df[col].fillna(0).mean()) if col in df.columns else default


def diagnose_weaknesses(team_players: pd.DataFrame,
                         league_players: pd.DataFrame) -> list:
    """
    Identify specific tactical deficiencies by comparing team against
    league averages. Returns list of dicts sorted by priority (1=most urgent).
    """
    diags = []

    # ── 1. Great shots creation (npxG/shot > 0.10 = "great shot")
    if ("npxg_per_shot" in team_players.columns and
            "shots_total_per90" in team_players.columns):
        gs_tm = float(
            (team_players["npxg_per_shot"].fillna(0) *
             team_players["shots_total_per90"].fillna(0)).sum()
        )
        gs_lg = float(
            (league_players["npxg_per_shot"].fillna(0) *
             league_players["shots_total_per90"].fillna(0)).mean()
        ) * max(len(team_players), 1)
        if gs_lg > 0 and gs_tm < gs_lg * 0.65:
            diags.append({
                "type":                 "Attacking",
                "deficiency":           "Great Shots Creation",
                "detail":               f"High-quality shot volume ~{gs_tm:.0f} vs league norm ~{gs_lg:.0f}",
                "priority":             1,
                "recommended_position": "ST or CAM",
            })

    # ── 2. Transition defence / pressing
    tm_pr = _col_mean(team_players, "pressures_per90")
    lg_pr = _col_mean(league_players, "pressures_per90")
    if lg_pr > 0 and tm_pr < lg_pr * 0.75:
        diags.append({
            "type":                 "Defensive",
            "deficiency":           "Transition Defence / Pressing",
            "detail":               f"Pressing {tm_pr:.1f}/90 vs league avg {lg_pr:.1f}/90",
            "priority":             2,
            "recommended_position": "CDM or CM",
        })

    # ── 3. Aerial dominance
    tm_ae = _col_mean(team_players, "aerial_duels_won_pct", 0.5)
    lg_ae = _col_mean(league_players, "aerial_duels_won_pct", 0.5)
    if lg_ae > 0 and tm_ae < lg_ae * 0.85:
        diags.append({
            "type":                 "Physical",
            "deficiency":           "Aerial Dominance",
            "detail":               f"Aerial win rate {tm_ae:.0%} vs league {lg_ae:.0%}",
            "priority":             3,
            "recommended_position": "CB or ST",
        })

    # ── 4. Progressive passing (build-up play)
    tm_pp = _col_mean(team_players, "progressive_passes")
    lg_pp = _col_mean(league_players, "progressive_passes")
    if lg_pp > 0 and tm_pp < lg_pp * 0.75:
        diags.append({
            "type":                 "Build-Up",
            "deficiency":           "Progressive Passing",
            "detail":               f"Prog. passes {tm_pp:.1f}/90 vs league {lg_pp:.1f}/90",
            "priority":             4,
            "recommended_position": "CM or CB",
        })

    # ── 5. Goalscoring threat
    tm_xg = _col_mean(team_players, "npxg_per90")
    lg_xg = _col_mean(league_players, "npxg_per90")
    if lg_xg > 0 and tm_xg < lg_xg * 0.70:
        diags.append({
            "type":                 "Attacking",
            "deficiency":           "Goalscoring Threat (npxG)",
            "detail":               f"npxG/90 {tm_xg:.2f} vs league {lg_xg:.2f}",
            "priority":             5,
            "recommended_position": "ST or LW/RW",
        })

    return sorted(diags, key=lambda d: d["priority"])[:4]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER SUCCESS PROBABILITY
# Research: new signings average ~40% of available minutes in Season 1.
# Step-up penalty is significant; fit score, age, and experience help.
# ─────────────────────────────────────────────────────────────────────────────
def estimate_success_probability(player_row, fit_score: float,
                                  league_from: str, league_to: str) -> float:
    tier_from = league_tier(league_from)
    tier_to   = league_tier(league_to)

    base = 0.55  # baseline: ~55% chance of meaningful contribution in Season 1
    step = tier_from - tier_to  # positive = stepping up to harder league
    if step > 0:  base -= 0.10 * step
    elif step < 0: base += 0.04

    base += (fit_score - 50) * 0.003

    age = int(player_row.get("age", 25))
    if   21 <= age <= 23: base += 0.08   # peak adaptability
    elif 24 <= age <= 27: base += 0.04
    elif age >= 32:       base -= 0.10

    intl = int(player_row.get("international_reputation", 1))
    base += (intl - 1) * 0.03

    return round(float(np.clip(base, 0.10, 0.95)), 2)
