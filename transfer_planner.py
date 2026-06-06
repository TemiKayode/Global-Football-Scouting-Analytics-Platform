"""
Transfer Window Planner — v1.0
Two-window planning · Budget allocation · Loan market identification ·
Plain-language strategy summary · Feasibility check
"""

import pandas as pd
import numpy as np
from recommendation_engine import dev_phase


# ─────────────────────────────────────────────────────────────────────────────
# TWO-WINDOW TRANSFER PLAN
# ─────────────────────────────────────────────────────────────────────────────
def plan_windows(shortlists: dict,
                 budget_info: dict,
                 team_goal: str,
                 churn_info: dict,
                 age_profile: dict) -> dict:
    """
    Produce a structured Summer + Loan plan from the scouting shortlists.
    Winter targets are inferred from contract situations.
    """
    summer: list = []
    winter: list = []
    loans:  list = []

    for pos, sl in shortlists.items():
        for _, p in sl.head(3).iterrows():
            age  = int(p.get("age", 25))
            mv   = float(p.get("predicted_value_m", p.get("market_value_m", 5.0)))
            cyl  = float(p.get("contract_years_left", 2.0))
            fs   = float(p.get("fit_score", 50.0))
            ph, strategy, _ = dev_phase(age)
            is_loan = age <= 22 and mv <= 6.0

            entry = {
                "name":       p["name"],
                "position":   pos,
                "age":        age,
                "ovr":        int(p["overall_rating"]),
                "value_m":    round(mv, 1),
                "ctr_yrs":    cyl,
                "dev_phase":  ph,
                "strategy":   strategy,
                "fit_score":  round(fs, 1),
                "league":     str(p.get("league_name", p.get("league", "—"))),
            }

            if is_loan:
                loans.append({**entry, "window": "Loan",
                               "est_cost_m": 0.0, "priority": 3})
            elif cyl <= 1.0:
                summer.append({**entry,
                               "window":     "Summer ⚡ (contract leverage)",
                               "est_cost_m": round(mv * 0.40, 1),
                               "priority":   1})
                winter.append({**entry,
                               "window":     "Winter (if not secured in Summer)",
                               "est_cost_m": round(mv * 0.35, 1),
                               "priority":   1})
            else:
                summer.append({**entry,
                               "window":     "Summer",
                               "est_cost_m": round(mv * 1.05, 1),
                               "priority":   2})

    summer.sort(key=lambda x: (x["priority"], -x["fit_score"]))
    loans.sort(key=lambda x: -x["fit_score"])

    summer_spend = sum(t["est_cost_m"] for t in summer[:3])
    budget_hi    = float(budget_info.get("budget_hi_m", 20.0))
    summer_budget = float(budget_info.get("summer_budget_m", budget_hi * 0.75))

    return {
        "summer":                summer[:6],
        "winter":                winter[:3],
        "loans":                 loans[:5],
        "summer_est_spend_m":    round(summer_spend, 1),
        "summer_budget_m":       summer_budget,
        "budget_hi_m":           budget_hi,
        "feasible":              summer_spend <= summer_budget * 1.20,
        "priority_signing_pos":  summer[0]["position"] if summer else "—",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY SUMMARY  (plain-language bullets)
# ─────────────────────────────────────────────────────────────────────────────
def strategy_summary(team_goal:  str,
                     churn_info: dict,
                     budget_info: dict,
                     age_profile: dict,
                     diagnoses:  list) -> list:
    """
    Return a list of strategy bullet strings informed by churn research,
    age matrix, budget tier, and tactical weakness diagnoses.
    """
    lines: list = []
    churn   = churn_info["score"]
    tier    = budget_info["tier"]
    blo     = budget_info["budget_lo_m"]
    bhi     = budget_info["budget_hi_m"]
    avg_age = age_profile["avg_age"]
    sq_pos  = budget_info["squad_position"].title()

    # ── Squad churn guidance ──
    if churn >= 45:
        lines.append(
            f"⚠️ **High Churn ({churn:.0f}%)** — Research shows new signings contribute only ~40% "
            "of available minutes in Season 1. Cap at **3-4 signings** to avoid disruption."
        )
    elif churn <= 20:
        lines.append(
            f"✅ **Stable Squad ({churn:.0f}% churn)** — Title-contender stability. "
            "Target **1-2 elite upgrades** rather than wholesale changes."
        )
    else:
        lines.append(
            f"📊 **Moderate Churn ({churn:.0f}%)** — **3-5 targeted signings** recommended. "
            "Prioritise players who hit the ground running."
        )

    # ── Age profile guidance ──
    if avg_age > 29:
        lines.append(
            f"🔴 **Aging Squad** (avg {avg_age:.1f} yrs) — Prioritise players aged **21-26** "
            "for long-term value; they hold highest resale value at 21-23."
        )
    elif avg_age < 23:
        lines.append(
            f"🟡 **Very Young Squad** (avg {avg_age:.1f} yrs) — Add **2-3 experienced players** "
            "(age 28-32) for leadership, set-pieces, and first-season stability."
        )
    else:
        lines.append(
            f"✅ **Balanced Age Profile** (avg {avg_age:.1f} yrs) — Maintain development/peak mix. "
            f"Academy: {age_profile['academy_u21']} | Prime Asset: {age_profile['prime_asset_21_23']} | "
            f"Peak: {age_profile['peak_24_28']} | Exp.: {age_profile['experienced_29plus']}"
        )

    # ── Budget context ──
    if tier == 1:
        lines.append(
            f"💰 **Tier 1 Budget** ({sq_pos} squad): €{blo}M–€{bhi}M — "
            "Can attract established elite talent. Summer window takes ~75% of spend."
        )
    elif tier == 2:
        lines.append(
            f"💰 **Tier 2 Budget** ({sq_pos} squad): €{blo}M–€{bhi}M — "
            "Follow the **Brighton/Brentford model**: sell 1 star for €40-60M, reinvest in 2-3 players."
        )
    else:
        lines.append(
            f"💰 **Tier 3 Budget** ({sq_pos} squad): €{blo}M–€{bhi}M — "
            "Focus on **loan deals** and players aged 21-23 in Tier 3 for future profit margin."
        )

    # ── Team goal specific ──
    _goal_advice = {
        "Win the League Title":
            "🏆 **Champions evolve, don't revolve.** Max 2 signings to maintain winning culture. "
            "Average title-winning churn is 27.5% — flag anything above 35%.",
        "Top 4 / Champions League":
            "🌟 **50-game squad depth is critical.** Prioritise versatile players covering 2+ positions. "
            "High-press system demands fit, high-energy profiles.",
        "Top Half / Europa League":
            "⚽ **Smart value signings.** Europa League adds 15+ fixtures — depth and versatility are key.",
        "Mid-Table Stability":
            "🛡️ **Reliability over brilliance.** Experienced players who stay fit and minimise errors.",
        "Avoid Relegation":
            "🛡️ **6-8 signings may be needed for promoted/struggling clubs.** "
            "Prioritise defensive solidity and set-piece specialists first.",
        "Develop Youth (U23)":
            "🌱 **Buy 21-23 yr olds in Tier 2/3 on upward trajectory.** Loan out for experience. "
            "Development Score > 60 is the filter.",
        "Maximize Transfer Revenue":
            "💎 **Identify Rising Stars in Tier 3 aged 21-23.** "
            "Buy low, develop 1-2 seasons, sell at peak (typically +150% ROI for high-potential players).",
    }
    if team_goal in _goal_advice:
        lines.append(_goal_advice[team_goal])

    # ── Tactical weaknesses ──
    for d in diagnoses[:2]:
        lines.append(
            f"🔍 **Tactical Gap — {d['deficiency']}**: {d['detail']} "
            f"→ Recruit **{d['recommended_position']}**"
        )

    return lines
