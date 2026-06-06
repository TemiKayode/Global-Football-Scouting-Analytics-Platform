"""
Recommendation Engine — v1.0
Development phase tagging · Transfer window timing · Resale value projection ·
Squad churn impact assessment · FC-style development score
"""

import pandas as pd
import numpy as np
from squad_analytics import (
    league_tier, squad_churn_score, estimate_success_probability, LEAGUE_TIERS,
)

# ─────────────────────────────────────────────────────────────────────────────
# AGE DEVELOPMENT PHASE MATRIX
# Based on research: players 21-23 command highest resale values at peak.
# ─────────────────────────────────────────────────────────────────────────────
_DEV_PHASES: list = [
    ((16, 20), "Academy",      "Develop / Loan out",           0.5),
    ((21, 23), "Prime Asset",  "Integrate or sell at peak",    2.0),
    ((24, 28), "Peak",         "Core squad — win now",         1.0),
    ((29, 32), "Experience",   "Leadership / squad depth",     0.3),
    ((33, 99), "Veteran",      "Short-term / free agent only", 0.1),
]


def dev_phase(age: int) -> tuple:
    """Return (label, strategy, resale_multiplier) for a given age."""
    for (lo, hi), label, strategy, mult in _DEV_PHASES:
        if lo <= age <= hi:
            return label, strategy, mult
    return "Peak", "Core squad — win now", 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER WINDOW ADVICE
# ─────────────────────────────────────────────────────────────────────────────
def transfer_window_advice(player_row, budget_info: dict) -> dict:
    """
    Determine optimal transfer window and estimated cost, factoring in
    contract situation, age (loan candidacy), and available budget.
    """
    mv      = float(player_row.get("predicted_value_m",
                                    player_row.get("market_value_m", 5.0)))
    cyl     = float(player_row.get("contract_years_left", 2.0))
    age     = int(player_row.get("age", 25))
    summer  = float(budget_info.get("summer_budget_m", 20.0))
    winter  = float(budget_info.get("winter_budget_m", 7.0))

    is_loan = (age <= 22 and mv <= 6.0)

    if   is_loan:      window = "Either (loan candidate)";            cost = 0.0
    elif cyl <= 0.5:   window = "Either (free — act immediately!)";   cost = round(mv * 0.05, 2)
    elif cyl <= 1.0:   window = "Summer ⚡ (contract leverage)";      cost = round(mv * 0.40, 2)
    elif cyl <= 1.5:   window = "Summer (negotiate discount now)";    cost = round(mv * 0.65, 2)
    else:              window = "Summer (standard)";                   cost = round(mv * 1.10, 2)

    if   cost == 0.0:          afford = "✅ Free (loan)"
    elif cost <= summer * 1.1: afford = "✅ Affordable"
    elif cost <= summer * 1.7: afford = "⚠️ Stretch"
    else:                      afford = "❌ Over Budget"

    return {
        "estimated_cost_m":    cost,
        "recommended_window":  window,
        "is_loan":             is_loan,
        "affordable_summer":   cost <= summer * 1.25,
        "affordable_winter":   cost <= winter * 1.25,
        "affordability":       afford,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESALE VALUE PROJECTION
# Players 21-23 grow fastest; post-28 depreciate ~8-9% per year.
# ─────────────────────────────────────────────────────────────────────────────
def resale_projection(player_row, years: int = 3) -> dict:
    mv  = float(player_row.get("predicted_value_m",
                                player_row.get("market_value_m", 5.0)))
    age = int(player_row.get("age", 25))
    pot = int(player_row.get("potential", player_row.get("overall_rating", 70)))
    ovr = int(player_row.get("overall_rating", 70))
    _, _, mult = dev_phase(age)

    gap       = max(0, pot - ovr)
    dev_lift  = min(gap * 0.18 * mult, mv * 1.6)
    age_dec   = max(0.15, 1.0 - max(0, age + years - 27) * 0.09)
    projected = round(mv * age_dec + dev_lift, 1)
    roi       = round((projected - mv) / max(mv, 0.1) * 100, 1)

    return {
        "current_m":      round(mv, 1),
        "projected_3yr_m": projected,
        "roi_pct":         roi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SQUAD CHURN IMPACT
# ─────────────────────────────────────────────────────────────────────────────
def churn_impact(player_row, team_players: pd.DataFrame) -> dict:
    """
    Simulate adding this player (and replacing weakest same-position player)
    and measure change in squad churn score.
    """
    pos      = player_row.get("position", "CM")
    same_pos = team_players[team_players["position"] == pos]

    if same_pos.empty:
        new_sq   = pd.concat([team_players,
                               pd.DataFrame([player_row])], ignore_index=True)
        replaced = "(new slot)"
    else:
        widx     = same_pos["overall_rating"].idxmin()
        replaced = str(team_players.loc[widx, "name"])
        new_sq   = pd.concat(
            [team_players.drop(widx), pd.DataFrame([player_row])],
            ignore_index=True,
        )

    before = squad_churn_score(team_players)["score"]
    after  = squad_churn_score(new_sq)["score"]
    delta  = after - before

    impact = "Stabilising" if delta < -2 else ("Neutral" if abs(delta) <= 2 else "Disruptive")

    return {
        "replaces":      replaced,
        "churn_before":  before,
        "churn_after":   after,
        "churn_delta":   round(delta, 1),
        "impact":        impact,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FC-STYLE DEVELOPMENT SCORE  (no actual EA data required)
# Models the same concept: how much developmental headroom does a player have?
# Mirrors EA FC 24 Evolution / FC 26 Archetype versatility research patterns.
# ─────────────────────────────────────────────────────────────────────────────
def fc_development_score(player_row) -> float:
    """
    Compute a 0-100 Development Score analogous to EA FC progression metrics.
    Components:
    - Potential gap (OVR → POT)  — 40%  like FC 24 Evolution eligibility
    - Age factor (21-23 optimal) — 25%  peak asset value window
    - Trajectory (OVR trend)     — 25%  like FC 26 Archetype progress
    - Positional versatility     — 10%  FC 26 multi-position value
    """
    ovr  = float(player_row.get("overall_rating", 70))
    pot  = float(player_row.get("potential", ovr))
    age  = int(player_row.get("age", 25))
    pr1  = float(player_row.get("past_rating_1yr", ovr))
    pr2  = float(player_row.get("past_rating_2yr", ovr))

    # 1. Potential gap (0-40)
    gap_score = min((pot - ovr) / 20.0, 1.0) * 40

    # 2. Age factor (0-25) — peaks 21-23, rises from 16, declines after 29
    if   21 <= age <= 23: age_score = 25
    elif 24 <= age <= 26: age_score = 18
    elif age == 20:       age_score = 20
    elif 27 <= age <= 29: age_score = 10
    elif age < 20:        age_score = min(20, max(0, 6 + (age - 16) * 3))   # 16→6, 17→9, 18→12, 19→15
    else:                  age_score = max(0, 10 - (age - 29) * 3)           # 30→7, 31→4, 32→1, 33+→0

    # 3. Trajectory (0-25)
    yr1  = ovr - pr1
    yr2  = pr1 - pr2
    traj = np.clip(yr1 * 5 + yr2 * 2.5, -10, 25)

    # 4. Versatility proxy via intl rep (0-10)
    intl  = int(player_row.get("international_reputation", 1))
    vers  = min((intl - 1) * 2.5, 10)

    score = round(float(np.clip(gap_score + age_score + traj + vers, 0, 100)), 1)
    return score
