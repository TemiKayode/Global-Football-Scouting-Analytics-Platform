# Global Football Scouting & Analytics Platform

A full-stack football intelligence tool built with Streamlit. Covers 36 leagues across men's and women's football, 20,000+ players, and FBref-style per-90 statistics. Uses machine learning to rank transfer targets, surface hidden gems from lower leagues, and model squad health — all in a single dashboard.

---

## What It Does

| Tab | Purpose |
|-----|---------|
| **AI Scout** | Full scouting report for any club: squad health, formation gaps, priority positions, AI-ranked shortlists, player deep-dives, transfer window plan |
| **Player Search** | Filter 20k+ players by name, position, or league. Full FBref stat breakdown with percentile rankings |
| **What-If Transfer** | Simulate signing any player — see how squad metrics, formation, and style change |
| **Team Analysis** | Age matrix, OVR distribution, contract watch, team radar vs league average |

---

## Features

### AI Scouting Engine
- Identifies the 3 weakest positions in your formation via OVR gap analysis
- Recommends up to 10 candidates per position from **all 36 leagues** — not just top divisions
- **Cross-league normalisation**: `ovr_vs_league` scores players relative to their own league, so a dominant lower-league midfielder ranks fairly against a fringe Premier League player
- **Hidden Gem detection**: players who dominate their league (`ovr_vs_league ≥ 1.5`), are actively improving (`prog_yr1 ≥ 1.5`), and have development headroom (`fc_dev_score ≥ 52`) receive a +20 scoring bonus and a gem badge
- Score formula balances fit (25%), trajectory (20%), potential gap (20%), stat production (12%), development score (9%), relative OVR (9%), and value (9%)

### Squad Analytics
- **Churn score** — proxy for squad turnover using contract expiry, OVR decline, and young risers
- **Budget estimation** — realistic transfer budget ranges by league tier and squad standing
- **Age matrix** — Academy / Prime Asset / Peak / Experience breakdown
- **Tactical weakness diagnosis** — compares team stats vs league averages across pressing, shot creation, aerial duels, progressive passing, and goalscoring

### Transfer Planning
- Summer and winter window targets with per-player cost estimates
- Loan candidate identification (age ≤ 22, value ≤ €6M)
- Budget feasibility check against estimated transfer spend
- Contract leverage flags (players within 12 months of expiry get automatic discount pricing)

### Live Data Feed
- Pulls recent transfer news from TheSportsDB, Transfermarkt endpoints, and optionally API-Football / football-data.org
- **Auto-refreshes** on startup if cached data is older than 60 minutes — no manual click needed
- Caches to `live_cache.json` (transfers: 1h TTL, formations/coaches: 2h TTL)
- Merges confirmed club changes directly into the player dataset for the session
- Fetches current **manager/coach** name per club from TheSportsDB (no API key)
- Fetches **live formation** from API-Football fixture lineups (requires `RAPIDAPI_KEY`) — falls back to 2025-26 knowledge base

### Realistic Transfer Fee Engine
- Goes beyond raw market value to model the actual cost of signing a player:
  - **Contract leverage**: final-year contracts cost 35–40% of market value; long contracts add a 20% premium
  - **Player importance**: undisputed starters command up to 35% extra; fringe players are discounted
  - **Release clause**: uses the actual clause from data or estimates at 175% of market value; fee is capped at the clause
  - **Club willingness to sell**: flagged as "Likely selling" or "Club may resist" based on appearances + contract situation

### Transfer Rumor Intelligence
- Estimates the probability a player moves this window (0–95%) using:
  - Contract expiry signals (strongest indicator)
  - Prime transfer age windows (22–27)
  - Live transfer feed mentions
  - Appearances at current club
- Labels: **Settled / Low / Medium / High** with source reliability tag

### Injury Record Profiling
- Deterministic injury history estimate per player using age, position, and availability (minutes ratio)
- Risk levels: Low / Medium / High / Very High
- Shows common injury types (hamstring, knee, muscle strain, etc.)
- Displayed in both the recommendation table and the deep-dive player card

### ML Models
- **Value model** — Random Forest trained on OVR, age, position, league tier, stats
- **Archetype clustering** — KMeans on per-90 stats to label player roles (e.g. Press Machine, Chance Creator, Aerial Threat)
- **Style clustering** — KMeans on team aggregates to classify playing style (High-Press, Possession-Based, Counter-Attacking, Direct, Hybrid)

---

## Leagues Covered

### Men's — Tier 1 (Elite)
Premier League · La Liga · Serie A · Bundesliga · Ligue 1

### Men's — Tier 2 (Competitive)
Championship · Primeira Liga · Belgian Pro League · Süper Lig · Scottish Premiership · Argentine Primera · Brasileirão · Eliteserien · Segunda División · Serie B · 2. Bundesliga · Ligue 2

### Men's — Tier 3 (Development)
Austrian Bundesliga · Allsvenskan · PKO BP Ekstraklasa · Superliga Romania · Swiss Super League · Super League Greece · Bulgarian First League · Croatian HNL · Czech Fortuna Liga · Slovenian PrvaLiga · OTP Bank Liga · Russian Premier League

### Women's
WSL · Frauen-Bundesliga · D1 Féminine · NWSL · A-League Women · Brazilian Women's Série A · Women's Serie A

---

## Setup

### Requirements
- Python 3.10+
- PowerShell 7+ (for data build script)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Build the dataset

The app needs two CSV files: `all_players_data.csv` and `team_clusters.csv`. Generate them from Transfermarkt open data:

**Step 1 — Download raw data**

Get the free Transfermarkt dataset from [transfermarkt-datasets on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) and place these two files in the project folder:
- `raw_players.csv`
- `raw_appearances.csv`

Compressed `.gz` versions are also accepted — the build script handles both.

**Step 2 — Run the build script**
```powershell
pwsh -File build_data.ps1
```

This processes ~20,000 real players from Transfermarkt, supplements missing leagues with synthetic players using position-specific FBref-style stat distributions, aggregates real goals/assists/minutes from appearances, and exports both CSVs.

To skip the download step if you already have the raw CSVs:
```powershell
pwsh -File build_data.ps1 -SkipDownload
```

### Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Optional: Live Data API Keys

The live feed works without any API key using public endpoints. For higher rate limits, set these environment variables before running:

```bash
# API-Football (RapidAPI) — enables live formation detection from recent fixtures
export RAPIDAPI_KEY=your_key_here

# football-data.org — optional squad data
export FOOTBALL_DATA_KEY=your_key_here

# Transfer cache TTL in hours (default: 1 hour — reduced from 6)
export LIVE_CACHE_TTL_HOURS=1

# Formation/coach cache TTL in hours (default: 2 hours)
export FORMATION_CACHE_TTL_HOURS=2
```

On Windows:
```powershell
$env:RAPIDAPI_KEY = "your_key_here"
```

---

## Project Structure

```
Soccer/
├── app.py                    # Main Streamlit app (~3100 lines)
├── squad_analytics.py        # League tiers, budget estimation, churn score,
│                             #   age profile, tactical weakness diagnosis,
│                             #   transfer success probability
├── recommendation_engine.py  # Dev phase tagging, transfer window advice,
│                             #   resale projection, churn impact, FC dev score
├── transfer_planner.py       # Two-window plan builder, strategy summary
├── live_feed.py              # Live transfer/market data fetcher + cache
├── build_data.ps1            # PowerShell data pipeline (Transfermarkt → CSV)
├── prepare_data.py           # Python alternative to build_data.ps1
├── requirements.txt
├── .gitignore
├── all_players_data.csv      # Generated — not in repo (see .gitignore)
└── team_clusters.csv         # Generated — not in repo
```

---

## How the Recommendation Score Works

Each candidate is scored as:

```
score = fit_score         × 0.25   # tactical fit to team style
      + ovr_vs_league     × 0.09   # dominance in own league (cross-league fairness)
      + overall_rating    × 0.13   # raw quality
      + pot_score         × 0.20   # (potential − age) × position weight
      + stat_score        × 0.12   # position-weighted per-90 stats
      + success_prob      × 0.07   # step-up penalty model
      + fc_dev_score      × 0.09   # development headroom
      + hidden_gem_bonus           # +20 additive for verified gems
      + value_efficiency  × 0.05   # cheaper relative to quality
      + resale_3yr        × 0.04   # projected 3-year resale upside
```

**Hidden Gem conditions** (all four must be met):
1. Player's league tier ≥ target club's league tier (same or lower division)
2. FC development score ≥ 52
3. Year-1 OVR progression ≥ +1.5
4. OVR vs league position average ≥ +1.5

---

## Stat Categories

The app uses FBref-style per-90 statistics across 7 categories:

| Category | Key Stats |
|----------|-----------|
| Standard | Goals, assists, shots on target %, minutes ratio |
| Shooting | npxG/90, xG/90, shots/90, npxG per shot |
| Passing | Completion % (short/medium/long), key passes, progressive passes, xA |
| Creation | SCA/90, GCA/90, through balls, crosses |
| Defence | Tackles, interceptions, blocks, clearances, pressures, aerial %, duels % |
| Possession | Dribbles, progressive carries, touches, att-third touches |
| Misc | Fouls committed/drawn, offsides, yellow/red cards |

---

## Simulation Mode

If `all_players_data.csv` is missing, the app enters simulation mode and generates a small synthetic dataset automatically. A sidebar notice will appear. All features work in simulation mode but data is randomised — run `build_data.ps1` for real analysis.

---

## Configuration

All key constants live at the top of `app.py`:

| Constant | Description |
|----------|-------------|
| `LEAGUES` | League code → display name mapping |
| `POSITIONS` | Supported position codes |
| `FORMATIONS` | Formation templates (4-3-3, 4-2-3-1, etc.) |
| `TEAM_GOALS` | Scouting objective presets |
| `SEASON_STATUSES` | Last-season context (Title Winners, Relegated, etc.) |
| `KNOWN_CONTEXTS` | Pre-filled 2024-25 season status for major clubs |

League tier mappings are in `squad_analytics.py` under `LEAGUE_TIERS`.
