---
title: Global Football Scouting Analytics
emoji: ⚽
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: true
license: mit
short_description: AI-powered football scouting — 36 leagues, 20k+ players
---

# Global Football Scouting & Analytics Platform

A full-stack football intelligence tool built with Streamlit. Covers 36 leagues across men's and women's football, 20,000+ players, and FBref-style per-90 statistics. Uses machine learning to rank transfer targets, surface hidden gems from lower leagues, and model squad health — all in a single dashboard.

## Features

- **AI Scout Reports** — ML-ranked transfer shortlists with fit scores, attitude grades, and career phase analysis
- **Live Transfer Feed** — Real-time player movement and transfer rumor intelligence with probability scores
- **Realistic Transfer Fee Engine** — Contract leverage, release clauses, and club willingness-to-sell modelling
- **Injury Profile** — Historical injury record profiling per player
- **Formation Detection** — Live formation lookup via API-Football + TheSportsDB
- **Percentile Radar Charts** — Position-peer comparison with colour-coded performance zones
- **Claude AI Reports** — Full narrative scout reports via Claude (requires `ANTHROPIC_API_KEY`)
- **StatsBomb Integration** — Free open event data via statsbombpy (Bundesliga 2023/24)
- **What-If Transfer Simulator** — Model squad changes before committing
- **Team Analysis** — Style clustering, age profiles, squad health dashboards

## Secrets (set in Space settings)

```
ANTHROPIC_API_KEY = "sk-ant-..."   # Claude AI reports (optional)
RAPIDAPI_KEY      = "..."          # Live formations + transfers (optional)
```

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```
