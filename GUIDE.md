# Global Football Scouting & Analytics Platform — User Guide

## What This Tool Does

A web-based football scouting dashboard that covers **36 leagues across men's and women's football**.
It recommends players based on your team's formation, objective, and squad gaps — not just raw ratings.

---

## Quick Start (3 Steps)

### Step 1 — Install Python dependencies

```
pip install -r requirements.txt
```

Requires Python 3.10 or higher. Tested on Python 3.12 and 3.14.

### Step 2 — Build the data (one-time, takes ~2 min)

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File build_data.ps1
```

This reads `raw_players.csv` and `raw_appearances.csv` (Transfermarkt data included) and outputs:
- `all_players_data.csv` — 20,000+ players, 45 columns
- `team_clusters.csv` — 621 clubs with aggregate style metrics

> If you don't have the raw CSV files, the app still runs in **Simulation Mode** using synthetic data.

### Step 3 — Launch the app

```
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

---

## File Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app (1 500+ lines, 4 tabs) |
| `squad_analytics.py` | League tiers, churn score, age profile, weakness diagnosis, budget engine |
| `recommendation_engine.py` | Development phase, transfer window advice, resale projection, FC dev score |
| `transfer_planner.py` | Two-window transfer plan, strategy summary bullets |
| `build_data.ps1` | PowerShell data pipeline (no Python required to build data) |
| `prepare_data.py` | Python alternative data pipeline with StatsBomb enrichment |
| `all_players_data.csv` | Built output — loaded by the app |
| `team_clusters.csv` | Built output — club style/aggregate metrics |
| `data_adapters.py` | Provider-agnostic data layer — adapters for FBref, StatsBomb, Wyscout, Opta |
| `requirements.txt` | Python dependencies |

---

## Sidebar Controls

| Control | What it does |
|---------|-------------|
| **Gender** | Filter to Men's, Women's, or both |
| **League** | Select one of 36 leagues |
| **Club** | Select a specific club within the league |
| **Last Season Status** | Auto-filled from 2024-25 data. Drives budget multiplier and objective suggestion |
| **Team Objective** | Auto-suggested from status. Override to customise recommendation weights |
| **Formation** | Override the auto-detected formation, or leave on Auto-detect |

---

## Season Context System

The tool auto-detects each club's last-season status from a 300+ club database covering all 36 leagues. Status drives two things:

1. **Budget multiplier** — applied to tier budget ranges (e.g. ×2.4 for relegated clubs with parachute payments)
2. **Objective suggestion** — automatically pre-selects the most appropriate Team Objective

### Status → Budget Multiplier Mapping

| Last Season Status | Budget Mult | Typical Use Case |
|--------------------|------------|-----------------|
| Title Winners | ×1.30 | Reinforce without disrupting champion DNA |
| Title Challengers (2nd/3rd) | ×1.20 | 1-2 elite upgrades to close the gap |
| Top 4 / Champions League | ×1.10 | CL depth, rotation options |
| Europa League Qualification | ×1.00 | Baseline tier budget |
| Mid-Table | ×0.90 | Targeted upgrades only |
| Survived Relegation Battle | ×0.75 | Emergency reinforcement |
| Relegated — Seeking Promotion | ×2.40 | Parachute payments create atypical budget |
| Newly Promoted — Consolidating | ×0.70 | Limited first-season resources |

**West Ham example**: Relegated club in the Championship. Parachute payments give ~£40-50M Year 1. Budget multiplier ×2.4 is applied to the Championship (Tier 2) base range, producing an atypically large budget vs Championship competitors. The system auto-selects "Secure Promotion" objective which prioritises physical, aerial, and goalscoring profiles proven at this level.

**Arsenal example**: Title winners. ×1.30 multiplier on Tier 1 top-club range. System auto-selects "Win the League Title" objective prioritising elite attacking output and press-resistant passing.

---

## Team Objectives Explained

The objective you choose changes the minimum OVR filter, maximum age cap, potential weighting, and stat weights for every recommendation in the session.

| Objective | Min OVR | Max Age | Focus |
|-----------|---------|---------|-------|
| Win the League Title | 77 | 32 | Goals, npxG, xA, passing — elite proven performers |
| Top 4 / Champions League | 74 | 31 | SCA, GCA, depth for 50-game seasons |
| Top Half / Europa League | 70 | 32 | Key passes, progressive actions, versatility |
| Mid-Table Stability | 67 | 33 | Pass completion, duels, reliability |
| Avoid Relegation | 63 | 36 | Tackles, interceptions, aerial duels, clearances |
| **Secure Promotion** | **65** | **32** | **Goals, aerial duels, physical dominance — Championship level** |
| **Consolidate After Promotion** | **67** | **33** | **Top-flight experience, pass completion, physicality** |
| Develop Youth (U23) | 62 | 23 | Potential gap, dribbles, progressive carries |
| Maximize Transfer Revenue | 67 | 25 | Rising trajectory, high potential gap, resale value |

---

## Tab 1 — AI Scout

The core tab. Select a club and objective, and the system auto-generates a complete scouting report.

### Squad Health Dashboard (top section)

Five cards are shown immediately:

- **Churn Score** — How much squad turnover is expected. Based on: % contracts expiring ≤ 1yr (40%), % players declining (30%), % young risers (15%), OVR volatility (15%).
  - Below 20% = Stable Champion
  - 20–30% = Stable
  - 30–40% = Moderate
  - 40–50% = High Turnover
  - Above 50% = Rebuilding
  - Research note: Champions average 27.5% churn; new signings contribute only ~40% of minutes in Season 1.

- **League Tier** — Tier 1 (Elite: PL/La Liga/Serie A/Bundesliga/Ligue 1), Tier 2 (Competitive), or Tier 3 (Development). Affects budget estimates.

- **Transfer Budget** — Estimated range based on tier + where the club sits in its league by squad OVR. Summer takes 75%, Winter 25%.

- **Age Profile** — Average squad age with category: Very Young / Young & Developing / Balanced / Experienced / Aging.

- **Expiring Contracts** — % of squad with ≤ 1yr remaining (departure risk).

### Age Tier Breakdown (four metrics)

Shows how many players fall in each development phase:
- **Academy ≤20** — Develop or loan out
- **Prime Asset 21-23** — Highest resale value window; integrate or sell at peak
- **Peak 24-28** — Core squad, winning now
- **Experience 29+** — Leadership, squad depth

### Tactical Weakness Diagnosis

Up to 4 tactical deficiencies flagged by comparing the team to its league average:
1. **Great Shots Creation** — npxG-per-shot volume below 65% of league norm
2. **Transition Defence/Pressing** — pressures/90 below 75% of league average
3. **Aerial Dominance** — aerial duel win rate below 85% of league average
4. **Progressive Passing** — progressive passes below 75% of league average
5. **Goalscoring Threat (npxG)** — team npxG/90 below 70% of league average

### Formation Chart + Squad Depth

Pitch visualisation showing current formation with red circles where gaps exist (formation slots not filled). Squad depth bar chart shows player count and average OVR per position.

### Priority Positions Bar Chart

Positions ranked by gap vs league average OVR. Formation gaps (unfilled slots) add an extra penalty to the score.

### Player Shortlist Tables

For each of the top 3 priority positions, a ranked list of 8 candidates showing:

| Column | Meaning |
|--------|---------|
| OVR | Current overall rating |
| Proj.OVR | Projected rating in ~1 year |
| Value €M | Predicted market value |
| Est.Cost €M | Estimated transfer fee (accounts for contract situation) |
| Ctr.Yrs | Contract years remaining |
| Fit% | Tactical fit score vs team's play style (0–100) |
| Success% | Probability of meaningful Season 1 contribution |
| Grade | Attitude/character grade (A–F) based on progression, discipline, availability |
| Dev Phase | Academy / Prime Asset / Peak / Experience / Veteran |
| Phase | Career phase: Emerging / Pre-Peak / Peak / Post-Peak / Veteran |
| Trend | OVR trajectory: ↑↑ Rising Star / ↑ Improving / → Stable / ↓ Regressing / ↓↓ Sharp Decline |
| Transfer Window | Recommended window: Summer / Loan / Either (free) |
| Afford. | ✅ Affordable / ⚠️ Stretch / ❌ Over Budget vs estimated budget |
| Resale 3yr €M | Projected value in 3 years |
| ROI | Return on investment vs current fee |
| FC Dev | Development score 0–100 (potential gap + age window + trajectory) |
| Why? | Plain-English rationale combining all factors |

### Deep Dive Expander

Click any top target to see:
- **Radar chart** — percentile vs league average across position-specific stats
- **OVR Trajectory chart** — 2yr history + projected
- **Full FBref stat breakdown** across 7 categories: Standard, Shooting, Passing, Creation, Defence, Possession, Misc
- **4 transfer metrics**: estimated cost, window timing, success probability, 3yr resale with ROI
- **Churn impact**: how the signing changes squad churn score

### Transfer Window Plan (bottom of Tab 1)

After all shortlists:

- **Strategy Summary** — 5–8 plain-English bullet points covering churn guidance, age strategy, budget context, and team-goal-specific advice
- **Feasibility indicator** — estimated summer spend vs budget
- **Summer Targets table** — top candidates ranked by priority (contract leverage = priority 1, standard = priority 2)
- **Loan Candidates** — players aged ≤22 with value ≤€6M suitable for loan arrangements
- **January Window Options** (collapsible) — targets where contract expiry makes a January move viable

---

## Tab 2 — Player Search

Search the full database by name, position, or league.

- Full stat breakdown across 7 FBref categories with **league percentile** for every stat
- Fit score for the currently selected club (from sidebar)
- OVR trajectory chart
- Progression phase and trend badge

---

## Tab 3 — What-If Transfer Simulator

Simulate signing any player and see the squad-level impact:

- Squad average changes across 7 metrics (pass completion, pressing, progressive passes, key passes, dribbles, age, OVR)
- Identifies the weakest same-position player who would be replaced
- Formation change indicator
- OVR trajectory of the target
- Tactical fit score for the current club

---

## Tab 4 — Team Analysis

Full squad health dashboard for the selected club:

- **Churn score card** with interpretation
- **Budget estimate card** for the club's league tier and position
- **Age matrix bar chart** — visual of academy/prime/peak/experience breakdown
- **Tactical Weaknesses** (collapsible) — same diagnosis as Tab 1
- **Formation chart** with gap indicators
- **Progression table** — every player with OVR trend, phase, projected OVR, 2yr delta
- **Age vs OVR scatter** with bubble size = market value
- **Team radar vs league** — 8 key metrics as percentiles vs league average
- **Contract Watch** — players with ≤ 18 months remaining

---

## League Codes Reference

### Men's Leagues

| Code | League | Tier |
|------|--------|------|
| GB1 | Premier League | 1 |
| ES1 | La Liga | 1 |
| IT1 | Serie A | 1 |
| L1 | Bundesliga | 1 |
| FR1 | Ligue 1 | 1 |
| PO1 | Primeira Liga | 2 |
| TR1 | Süper Lig | 2 |
| BE1 | Belgian Pro League | 2 |
| SC1 | Scottish Premiership | 2 |
| AR1N | Argentine Primera División | 2 |
| BRA1 | Brasileirão Série A | 2 |
| NO1 | Eliteserien | 2 |
| GB2 | Championship | 2 |
| ES2 | Segunda División | 2 |
| IT2 | Serie B | 2 |
| L2 | 2. Bundesliga | 2 |
| FR2 | Ligue 2 | 2 |
| A1 | Austrian Bundesliga | 3 |
| C1 | Swiss Super League | 3 |
| GR1 | Super League Greece | 3 |
| PL1 | PKO BP Ekstraklasa | 3 |
| RO1 | Superliga Romania | 3 |
| SE1 | Allsvenskan | 3 |
| BU1 | Bulgarian First League | 3 |
| KR1 | Croatian HNL | 3 |
| TS1 | Czech Fortuna Liga | 3 |
| SL1 | Slovenian PrvaLiga | 3 |
| UNG1 | OTP Bank Liga (Hungary) | 3 |
| RU1 | Russian Premier League | 3 |

### Women's Leagues

| Code | League | Tier |
|------|--------|------|
| WWSL | WSL | 2 |
| WGBL | Frauen-Bundesliga | 2 |
| WFRD1 | D1 Féminine | 2 |
| WNWSL | NWSL | 2 |
| WAUS | A-League Women | 3 |
| WBRA | Brazilian Women's Série A | 3 |
| WITA | Women's Serie A | 3 |

---

## Stat Categories (FBref-style)

| Category | Key Stats |
|----------|----------|
| Standard | goals/90, assists/90, shots on target %, yellow/red cards/90 |
| Shooting | shots total/90, npxG/90, npxG per shot, xG/90 |
| Passing | pass completion (total/short/medium/long), key passes/90, progressive passes, xA/90, through balls/90 |
| Goal & Shot Creation | SCA/90, GCA/90, crosses/90 |
| Defensive Actions | tackles/90, tackles won %, interceptions/90, blocks/90, clearances/90, pressures/90, pressure success %, aerial duels won %, duels won % |
| Possession | dribbles/90, progressive carries, touches/90, touches in att. third/90, progressive passes received/90 |
| Miscellaneous | fouls committed/90, fouls drawn/90, offsides/90 |

---

## How Scores Are Calculated

### Fit Score (0–100)
Euclidean distance between the player's normalised feature vector and the team's cluster centroid in archetype space. Transformed via `100 / (1 + distance)`.

### Score Rank (determines shortlist order)
```
score_rank = fit_score×0.28 + overall_rating×0.22 + potential_gap×0.18
           + weighted_stats×0.14 + success_probability×30×0.10
           + fc_dev_score×0.05 + value_efficiency×0.08 + resale_value×0.05
```

### Success Probability (0–10 to 95%)
Base 55%, adjusted for:
- League step-up (−10% per tier climbed)
- Fit score (±3% per 10 points from 50)
- Age (21-23 = +8%, 24-27 = +4%, 32+ = −10%)
- International reputation (+3% per star above 1)

### FC Development Score (0–100)
No EA data required — derived from your data:
- Potential gap (0–40 pts): how far below potential the player is
- Age window (0–25 pts): peaks at 21-23, graduated curve from 16 onwards
- Trajectory (0–25 pts): weighted OVR trend over 2 years
- Versatility (0–10 pts): proxy via international reputation

### Squad Churn Score (0–100%)
- 40%: % players with ≤1yr contract
- 30%: % players showing OVR decline ≥2
- 15%: % young players (≤23) showing OVR rise ≥2
- 15%: OVR volatility (std dev)

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: squad_analytics` | Running from wrong directory | `cd` into the Soccer folder first |
| App shows "Simulation mode" | `all_players_data.csv` not found | Run `build_data.ps1` first |
| `KeyError: 'npxg_per90'` | Old CSV missing new columns | Re-run `build_data.ps1` |
| Streamlit not found | Not installed | `pip install streamlit` |
| Very slow first load | ML model training (one-time) | Subsequent loads are cached |
| No clubs showing for a league | League code not in CSV | Rebuild data; check league selector |

---

## Rebuild / Update Data

To get fresh data or re-run after code changes:

```powershell
# From the Soccer folder:
powershell -ExecutionPolicy Bypass -File build_data.ps1
```

The script:
1. Reads `raw_players.csv` (47 701 TM players)
2. Reads `raw_appearances.csv` (1.8M rows, 2022–2025)
3. Patches real goals/assists/cards/minutes from appearances
4. Generates synthetic players for leagues not in Transfermarkt
5. Generates synthetic women's league players
6. Writes `all_players_data.csv` (20 120 players, 45 columns)
7. Writes `team_clusters.csv` (621 teams)

Typical runtime: 90–120 seconds.

---

## Development Notes

### Adding a New League
1. Add to `LEAGUES` dict in `app.py`
2. Add clubs to `CLUBS_BY_LEAGUE` in `app.py`
3. Add tier to `LEAGUE_TIERS` in `squad_analytics.py`
4. Add to `$missingLeagues` or `$womenLeagues` in `build_data.ps1`
5. Rebuild data

### Adding a New Stat Column
1. Add stat to `ALL_STATS` in `app.py`
2. Add to the correct `POSITION_STATS` entry in `app.py`
3. Add base ranges to `$statBases` in `build_data.ps1`
4. Add computation and `$row` assignment in both the TM player loop and `New-SyntheticPlayer` function
5. Rebuild data

### Changing Recommendation Weights
Edit `get_goal_config()` in `app.py` — each team goal has its own `stat_weights` dict. Increase a weight to boost that stat's influence on the score rank.

### Adding a New Data Source
Use `data_adapters.py` — the provider-agnostic data layer:

1. Subclass `DataAdapter` and implement `load() -> pd.DataFrame`
2. Map your source's columns to the standard schema (`STANDARD_COLUMNS`)
3. Return a DataFrame — it will be padded with NaN for any missing standard columns
4. Use `merge_sources(base_df, new_df, on="name")` to blend it with the Transfermarkt base

**Available adapters:**
| Class | Source | Key | Notes |
|-------|--------|-----|-------|
| `TransfermarktAdapter` | TM CSV | None | Default — always available |
| `FBrefAdapter` | FBref | None (free scrape) | Needs `pip install soccerdata` + Chrome |
| `StatsBombAdapter` | StatsBomb Open | None (free tier) | Limited to free competitions |
| `WyscoutAdapter` | Wyscout | `WYSCOUT_API_KEY` | Paid subscription |
| `OptaAdapter` | Opta/StatsPerform | `OPTA_API_KEY` | Paid subscription |

### Enabling the AI Scout Report
The "Generate AI Scout Report" button appears in every player deep-dive. With no API key, it produces a structured data summary. With a key, it calls Claude to write a professional narrative.

Set the key before launching the app:
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
streamlit run app.py
```

The report covers: Player Profile · Strengths · Areas of Concern · Transfer Recommendation. Uses `claude-haiku-4-5-20251001` for fast, cost-effective generation.

### Live Formation Detection
Formation detection uses a three-tier priority system:
1. **Live API** — most recent match lineups from API-Football (needs `RAPIDAPI_KEY`)
2. **Persisted cache** — formations fetched in previous sessions (2h TTL in `live_cache.json`)
3. **Static knowledge base** — `KNOWN_FORMATIONS` dict in `app.py` covering 80+ clubs for 2025-26

To force a refresh: click "Refresh Now" in the Live Data Feed sidebar section. This re-fetches both transfers and the selected club's formation/manager.
