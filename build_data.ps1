# build_data.ps1
# Builds all_players_data.csv and team_clusters.csv from Transfermarkt data
# Run from the Soccer directory: pwsh -File build_data.ps1

param([switch]$SkipDownload)

Set-Location $PSScriptRoot
$rng = [System.Random]::new(42)
$now = Get-Date

# ── League maps ────────────────────────────────────────────────────────────────
# Transfermarkt code → our app code
$tmToApp = @{
    "A1"  = "A1";   "ARG1"= "AR1N"; "BE1" = "BE1";  "BRA1"= "BRA1"
    "C1"  = "C1";   "ES1" = "ES1";  "FR1" = "FR1";  "GB1" = "GB1"
    "GR1" = "GR1";  "IT1" = "IT1";  "KR1" = "KR1";  "L1"  = "L1"
    "NO1" = "NO1";  "PL1" = "PL1";  "PO1" = "PO1";  "RO1" = "RO1"
    "RU1" = "RU1";  "SC1" = "SC1";  "SE1" = "SE1";  "TR1" = "TR1"
    "TS1" = "TS1"
}

$leagueNames = @{
    "A1"  = "Austrian Bundesliga";   "AR1N"= "Argentine Primera Division"
    "BE1" = "Belgian Pro League";    "BRA1"= "Brasileirao Serie A"
    "BU1" = "Bulgarian First League";"C1"  = "Swiss Super League"
    "ES1" = "La Liga";               "ES2" = "Segunda Division"
    "FR1" = "Ligue 1";               "FR2" = "Ligue 2"
    "GB1" = "Premier League";        "GB2" = "Championship"
    "GR1" = "Super League Greece";   "IT1" = "Serie A"
    "IT2" = "Serie B";               "KR1" = "Croatian HNL"
    "L1"  = "Bundesliga";            "L2"  = "2. Bundesliga"
    "NO1" = "Eliteserien";           "PL1" = "PKO BP Ekstraklasa"
    "PO1" = "Primeira Liga";         "RO1" = "Superliga Romania"
    "RU1" = "Russian Premier League";"SC1" = "Scottish Premiership"
    "SE1" = "Allsvenskan";           "SL1" = "Slovenian PrvaLiga"
    "TR1" = "Super Lig";             "TS1" = "Czech Fortuna Liga"
    "UNG1"= "OTP Bank Liga (Hungary)"
    "WGBL"= "Frauen-Bundesliga";     "WWSL"= "WSL"
    "WFRD1"="D1 Feminine";           "WNWSL"="NWSL"
    "WAUS"= "A-League Women";        "WBRA"= "Brazilian Womens Serie A"
    "WITA"= "Womens Serie A"
}

# Sub-position to position code
$posMap = @{
    "Goalkeeper"        = "GK"
    "Centre-Back"       = "CB"
    "Left-Back"         = "LB"
    "Right-Back"        = "RB"
    "Defensive Midfield"= "CDM"
    "Central Midfield"  = "CM"
    "Left Midfield"     = "CM"
    "Right Midfield"    = "CM"
    "Attacking Midfield"= "CAM"
    "Left Winger"       = "LW"
    "Right Winger"      = "RW"
    "Centre-Forward"    = "ST"
    "Second Striker"    = "ST"
}

# ── Per-90 base stats per position [midpoint, half-range] ─────────────────────
# Format: stat = @(base, scale_factor)   final = base + scale_factor * quality
$statBases = @{
    # Each entry: stat=@(base, scale_factor)  =>  value = base + scale*quality + noise*scale*0.3
    # New FBref columns added: npxg_per90, xa_per90, shots_total_per90, npxg_per_shot,
    #   pass_completion_short/medium/long, sca_per90, gca_per90,
    #   tackles_won_pct, blocks_per90, clearances_per90, pressures_per90, pressure_success_pct,
    #   touches_per90, touches_att3rd_per90, progressive_passes_received_per90,
    #   fouls_committed_per90, fouls_drawn_per90, offsides_per90

    "GK"  = @{ goals_per90=@(0.00,0.01); assists_per90=@(0.00,0.01); shots_on_target_pct=@(0.30,0.10)
               pass_completion=@(0.60,0.25); pass_completion_short=@(0.82,0.12); pass_completion_medium=@(0.72,0.14); pass_completion_long=@(0.52,0.22)
               key_passes_per90=@(0.05,0.10); progressive_passes=@(1.0,2.0); through_balls_per90=@(0.0,0.02)
               progressive_passes_received_per90=@(0.5,1.0)
               tackles_per90=@(0.1,0.2); interceptions_per90=@(0.05,0.15); dribbles_per90=@(0.05,0.10)
               aerial_duels_won_pct=@(0.45,0.20); duels_won_pct=@(0.45,0.15); progressive_carries=@(0.2,0.5)
               crosses_per90=@(0.0,0.05); xg_per90=@(0.00,0.01); npxg_per90=@(0.00,0.01); xa_per90=@(0.00,0.01)
               shots_total_per90=@(0.02,0.05); npxg_per_shot=@(0.0,0.0)
               sca_per90=@(0.2,0.4); gca_per90=@(0.02,0.05)
               tackles_won_pct=@(0.40,0.15); blocks_per90=@(0.02,0.05); clearances_per90=@(0.2,0.5)
               pressures_per90=@(1.5,3.0); pressure_success_pct=@(0.25,0.10)
               touches_per90=@(38,22); touches_att3rd_per90=@(0.5,1.5)
               fouls_committed_per90=@(0.05,0.10); fouls_drawn_per90=@(0.05,0.10); offsides_per90=@(0.0,0.0)
               saves_per90=@(2.5,2.5); clean_sheets_pct=@(0.20,0.25); sweeper_actions=@(1.0,2.0)
               goals_conceded_per90=@(1.2,1.0) }

    "CB"  = @{ goals_per90=@(0.02,0.04); assists_per90=@(0.01,0.03); shots_on_target_pct=@(0.30,0.20)
               pass_completion=@(0.75,0.18); pass_completion_short=@(0.88,0.08); pass_completion_medium=@(0.78,0.12); pass_completion_long=@(0.58,0.20)
               key_passes_per90=@(0.15,0.20); progressive_passes=@(3.0,5.0); through_balls_per90=@(0.02,0.05)
               progressive_passes_received_per90=@(1.0,2.0)
               tackles_per90=@(1.8,2.5); interceptions_per90=@(1.2,2.0); dribbles_per90=@(0.1,0.3)
               aerial_duels_won_pct=@(0.50,0.25); duels_won_pct=@(0.50,0.20); progressive_carries=@(0.5,1.5)
               crosses_per90=@(0.1,0.3); xg_per90=@(0.02,0.03); npxg_per90=@(0.02,0.03); xa_per90=@(0.01,0.03)
               shots_total_per90=@(0.3,0.4); npxg_per_shot=@(0.06,0.04)
               sca_per90=@(0.6,1.0); gca_per90=@(0.06,0.10)
               tackles_won_pct=@(0.50,0.18); blocks_per90=@(0.6,1.2); clearances_per90=@(3.5,3.5)
               pressures_per90=@(8.0,8.0); pressure_success_pct=@(0.28,0.12)
               touches_per90=@(60,25); touches_att3rd_per90=@(2.0,3.0)
               fouls_committed_per90=@(0.6,0.6); fouls_drawn_per90=@(0.4,0.4); offsides_per90=@(0.0,0.05)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "LB"  = @{ goals_per90=@(0.02,0.05); assists_per90=@(0.05,0.15); shots_on_target_pct=@(0.28,0.20)
               pass_completion=@(0.73,0.18); pass_completion_short=@(0.86,0.09); pass_completion_medium=@(0.76,0.13); pass_completion_long=@(0.55,0.22)
               key_passes_per90=@(0.2,0.4); progressive_passes=@(2.5,4.0); through_balls_per90=@(0.03,0.08)
               progressive_passes_received_per90=@(1.5,2.5)
               tackles_per90=@(1.5,2.5); interceptions_per90=@(0.8,1.5); dribbles_per90=@(0.3,0.8)
               aerial_duels_won_pct=@(0.40,0.25); duels_won_pct=@(0.45,0.20); progressive_carries=@(1.5,3.5)
               crosses_per90=@(0.8,2.5); xg_per90=@(0.02,0.04); npxg_per90=@(0.02,0.04); xa_per90=@(0.04,0.12)
               shots_total_per90=@(0.5,0.7); npxg_per_shot=@(0.05,0.04)
               sca_per90=@(1.2,1.8); gca_per90=@(0.12,0.18)
               tackles_won_pct=@(0.48,0.18); blocks_per90=@(0.4,0.8); clearances_per90=@(1.5,2.0)
               pressures_per90=@(12.0,9.0); pressure_success_pct=@(0.28,0.12)
               touches_per90=@(55,20); touches_att3rd_per90=@(8.0,12.0)
               fouls_committed_per90=@(0.8,0.7); fouls_drawn_per90=@(0.7,0.7); offsides_per90=@(0.02,0.05)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "RB"  = @{ goals_per90=@(0.02,0.05); assists_per90=@(0.05,0.15); shots_on_target_pct=@(0.28,0.20)
               pass_completion=@(0.73,0.18); pass_completion_short=@(0.86,0.09); pass_completion_medium=@(0.76,0.13); pass_completion_long=@(0.55,0.22)
               key_passes_per90=@(0.2,0.4); progressive_passes=@(2.5,4.0); through_balls_per90=@(0.03,0.08)
               progressive_passes_received_per90=@(1.5,2.5)
               tackles_per90=@(1.5,2.5); interceptions_per90=@(0.8,1.5); dribbles_per90=@(0.3,0.8)
               aerial_duels_won_pct=@(0.40,0.25); duels_won_pct=@(0.45,0.20); progressive_carries=@(1.5,3.5)
               crosses_per90=@(0.8,2.5); xg_per90=@(0.02,0.04); npxg_per90=@(0.02,0.04); xa_per90=@(0.04,0.12)
               shots_total_per90=@(0.5,0.7); npxg_per_shot=@(0.05,0.04)
               sca_per90=@(1.2,1.8); gca_per90=@(0.12,0.18)
               tackles_won_pct=@(0.48,0.18); blocks_per90=@(0.4,0.8); clearances_per90=@(1.5,2.0)
               pressures_per90=@(12.0,9.0); pressure_success_pct=@(0.28,0.12)
               touches_per90=@(55,20); touches_att3rd_per90=@(8.0,12.0)
               fouls_committed_per90=@(0.8,0.7); fouls_drawn_per90=@(0.7,0.7); offsides_per90=@(0.02,0.05)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "CDM" = @{ goals_per90=@(0.03,0.07); assists_per90=@(0.03,0.08); shots_on_target_pct=@(0.28,0.18)
               pass_completion=@(0.80,0.14); pass_completion_short=@(0.90,0.07); pass_completion_medium=@(0.82,0.10); pass_completion_long=@(0.60,0.18)
               key_passes_per90=@(0.3,0.6); progressive_passes=@(4.0,6.0); through_balls_per90=@(0.05,0.12)
               progressive_passes_received_per90=@(2.0,3.0)
               tackles_per90=@(2.5,3.0); interceptions_per90=@(1.8,2.5); dribbles_per90=@(0.3,0.8)
               aerial_duels_won_pct=@(0.45,0.22); duels_won_pct=@(0.52,0.18); progressive_carries=@(1.0,2.5)
               crosses_per90=@(0.1,0.3); xg_per90=@(0.03,0.06); npxg_per90=@(0.03,0.06); xa_per90=@(0.03,0.08)
               shots_total_per90=@(0.8,0.8); npxg_per_shot=@(0.05,0.04)
               sca_per90=@(1.5,2.0); gca_per90=@(0.15,0.20)
               tackles_won_pct=@(0.52,0.18); blocks_per90=@(0.8,1.2); clearances_per90=@(1.2,1.5)
               pressures_per90=@(18.0,12.0); pressure_success_pct=@(0.30,0.12)
               touches_per90=@(68,22); touches_att3rd_per90=@(5.0,6.0)
               fouls_committed_per90=@(1.0,0.8); fouls_drawn_per90=@(0.6,0.6); offsides_per90=@(0.02,0.05)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "CM"  = @{ goals_per90=@(0.05,0.12); assists_per90=@(0.05,0.18); shots_on_target_pct=@(0.30,0.20)
               pass_completion=@(0.78,0.15); pass_completion_short=@(0.89,0.08); pass_completion_medium=@(0.80,0.11); pass_completion_long=@(0.58,0.18)
               key_passes_per90=@(0.6,1.5); progressive_passes=@(4.0,6.0); through_balls_per90=@(0.08,0.18)
               progressive_passes_received_per90=@(3.0,4.0)
               tackles_per90=@(1.5,2.0); interceptions_per90=@(0.8,1.5); dribbles_per90=@(0.5,1.2)
               aerial_duels_won_pct=@(0.40,0.22); duels_won_pct=@(0.48,0.18); progressive_carries=@(1.5,3.5)
               crosses_per90=@(0.2,0.5); xg_per90=@(0.05,0.10); npxg_per90=@(0.05,0.10); xa_per90=@(0.05,0.15)
               shots_total_per90=@(1.0,1.0); npxg_per_shot=@(0.06,0.04)
               sca_per90=@(2.0,2.5); gca_per90=@(0.20,0.25)
               tackles_won_pct=@(0.49,0.18); blocks_per90=@(0.5,0.9); clearances_per90=@(0.6,0.9)
               pressures_per90=@(14.0,10.0); pressure_success_pct=@(0.29,0.12)
               touches_per90=@(72,25); touches_att3rd_per90=@(8.0,9.0)
               fouls_committed_per90=@(0.9,0.7); fouls_drawn_per90=@(0.8,0.7); offsides_per90=@(0.03,0.06)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "CAM" = @{ goals_per90=@(0.08,0.20); assists_per90=@(0.10,0.28); shots_on_target_pct=@(0.32,0.22)
               pass_completion=@(0.76,0.16); pass_completion_short=@(0.87,0.09); pass_completion_medium=@(0.78,0.12); pass_completion_long=@(0.55,0.20)
               key_passes_per90=@(1.2,2.5); progressive_passes=@(3.5,5.5); through_balls_per90=@(0.15,0.40)
               progressive_passes_received_per90=@(4.0,5.0)
               tackles_per90=@(0.8,1.2); interceptions_per90=@(0.4,0.8); dribbles_per90=@(0.8,2.5)
               aerial_duels_won_pct=@(0.35,0.22); duels_won_pct=@(0.44,0.18); progressive_carries=@(2.0,5.0)
               crosses_per90=@(0.3,0.8); xg_per90=@(0.08,0.18); npxg_per90=@(0.07,0.18); xa_per90=@(0.10,0.28)
               shots_total_per90=@(1.5,1.5); npxg_per_shot=@(0.08,0.05)
               sca_per90=@(3.5,3.0); gca_per90=@(0.35,0.35)
               tackles_won_pct=@(0.45,0.18); blocks_per90=@(0.3,0.5); clearances_per90=@(0.3,0.5)
               pressures_per90=@(10.0,8.0); pressure_success_pct=@(0.28,0.11)
               touches_per90=@(65,22); touches_att3rd_per90=@(18.0,14.0)
               fouls_committed_per90=@(0.8,0.7); fouls_drawn_per90=@(1.2,0.9); offsides_per90=@(0.08,0.12)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "LW"  = @{ goals_per90=@(0.12,0.35); assists_per90=@(0.10,0.28); shots_on_target_pct=@(0.34,0.25)
               pass_completion=@(0.72,0.18); pass_completion_short=@(0.85,0.10); pass_completion_medium=@(0.74,0.14); pass_completion_long=@(0.52,0.22)
               key_passes_per90=@(0.8,2.0); progressive_passes=@(2.5,4.0); through_balls_per90=@(0.05,0.15)
               progressive_passes_received_per90=@(4.5,5.0)
               tackles_per90=@(0.6,1.0); interceptions_per90=@(0.3,0.6); dribbles_per90=@(1.5,3.5)
               aerial_duels_won_pct=@(0.35,0.22); duels_won_pct=@(0.42,0.18); progressive_carries=@(2.5,5.5)
               crosses_per90=@(0.5,1.5); xg_per90=@(0.12,0.30); npxg_per90=@(0.11,0.30); xa_per90=@(0.08,0.22)
               shots_total_per90=@(2.0,2.0); npxg_per_shot=@(0.09,0.06)
               sca_per90=@(3.0,3.0); gca_per90=@(0.30,0.35)
               tackles_won_pct=@(0.44,0.18); blocks_per90=@(0.2,0.4); clearances_per90=@(0.2,0.4)
               pressures_per90=@(9.0,8.0); pressure_success_pct=@(0.27,0.11)
               touches_per90=@(55,20); touches_att3rd_per90=@(20.0,16.0)
               fouls_committed_per90=@(0.7,0.6); fouls_drawn_per90=@(1.4,1.0); offsides_per90=@(0.12,0.18)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "RW"  = @{ goals_per90=@(0.12,0.35); assists_per90=@(0.10,0.28); shots_on_target_pct=@(0.34,0.25)
               pass_completion=@(0.72,0.18); pass_completion_short=@(0.85,0.10); pass_completion_medium=@(0.74,0.14); pass_completion_long=@(0.52,0.22)
               key_passes_per90=@(0.8,2.0); progressive_passes=@(2.5,4.0); through_balls_per90=@(0.05,0.15)
               progressive_passes_received_per90=@(4.5,5.0)
               tackles_per90=@(0.6,1.0); interceptions_per90=@(0.3,0.6); dribbles_per90=@(1.5,3.5)
               aerial_duels_won_pct=@(0.35,0.22); duels_won_pct=@(0.42,0.18); progressive_carries=@(2.5,5.5)
               crosses_per90=@(0.5,1.5); xg_per90=@(0.12,0.30); npxg_per90=@(0.11,0.30); xa_per90=@(0.08,0.22)
               shots_total_per90=@(2.0,2.0); npxg_per_shot=@(0.09,0.06)
               sca_per90=@(3.0,3.0); gca_per90=@(0.30,0.35)
               tackles_won_pct=@(0.44,0.18); blocks_per90=@(0.2,0.4); clearances_per90=@(0.2,0.4)
               pressures_per90=@(9.0,8.0); pressure_success_pct=@(0.27,0.11)
               touches_per90=@(55,20); touches_att3rd_per90=@(20.0,16.0)
               fouls_committed_per90=@(0.7,0.6); fouls_drawn_per90=@(1.4,1.0); offsides_per90=@(0.12,0.18)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }

    "ST"  = @{ goals_per90=@(0.18,0.55); assists_per90=@(0.06,0.20); shots_on_target_pct=@(0.38,0.28)
               pass_completion=@(0.68,0.20); pass_completion_short=@(0.82,0.12); pass_completion_medium=@(0.70,0.16); pass_completion_long=@(0.50,0.22)
               key_passes_per90=@(0.4,1.0); progressive_passes=@(1.5,3.0); through_balls_per90=@(0.03,0.08)
               progressive_passes_received_per90=@(5.0,5.5)
               tackles_per90=@(0.4,0.8); interceptions_per90=@(0.2,0.5); dribbles_per90=@(0.8,2.0)
               aerial_duels_won_pct=@(0.45,0.28); duels_won_pct=@(0.46,0.20); progressive_carries=@(1.2,3.0)
               crosses_per90=@(0.1,0.3); xg_per90=@(0.18,0.55); npxg_per90=@(0.17,0.52); xa_per90=@(0.05,0.18)
               shots_total_per90=@(2.5,2.5); npxg_per_shot=@(0.10,0.08)
               sca_per90=@(2.5,2.5); gca_per90=@(0.28,0.35)
               tackles_won_pct=@(0.44,0.18); blocks_per90=@(0.2,0.4); clearances_per90=@(0.2,0.5)
               pressures_per90=@(7.0,7.0); pressure_success_pct=@(0.26,0.11)
               touches_per90=@(48,18); touches_att3rd_per90=@(18.0,15.0)
               fouls_committed_per90=@(1.0,0.7); fouls_drawn_per90=@(1.8,1.2); offsides_per90=@(0.20,0.25)
               saves_per90=@(0.0,0.0); clean_sheets_pct=@(0.0,0.0); sweeper_actions=@(0.0,0.0)
               goals_conceded_per90=@(0.0,0.0) }
}

# ── Helper functions ───────────────────────────────────────────────────────────
function Get-AgeFromDOB($dobStr) {
    try {
        $dob = [datetime]::Parse($dobStr)
        return [int](($now - $dob).TotalDays / 365.25)
    } catch { return 25 }
}

function Get-ContractYearsLeft($expiryStr) {
    try {
        if ([string]::IsNullOrWhiteSpace($expiryStr)) { return 1.0 }
        $exp = [datetime]::Parse($expiryStr)
        $years = ($exp - $now).TotalDays / 365.25
        if ($years -lt 0) { return 0.0 }
        if ($years -gt 6) { return 5.0 }
        return [Math]::Round($years, 1)
    } catch { return 1.0 }
}

function Get-OverallRating($marketValueEur) {
    try {
        $val = [double]$marketValueEur
        if ($val -le 0) { return 58 }
        $log = [Math]::Log10($val)
        $ovr = [int][Math]::Round(55 + 8 * ($log - 4))
        return [Math]::Max(50, [Math]::Min(97, $ovr))
    } catch { return 60 }
}

function Get-IntlRep($caps) {
    try {
        $c = [int]$caps
        if ($c -ge 60) { return 5 }
        if ($c -ge 31) { return 4 }
        if ($c -ge 11) { return 3 }
        if ($c -ge 1)  { return 2 }
        return 1
    } catch { return 1 }
}

function Get-Stat($base, $scale, $quality, $noise) {
    # quality is 0-1 (0=worst, 1=best), noise is -0.5 to 0.5
    # Use 0.0 (Double literal) so [Math]::Max picks the Double overload, not Int
    $val = [double]$base + [double]$scale * ([double]$quality + [double]$noise * 0.3)
    $rounded = [Math]::Round($val, 3)
    if ($rounded -lt 0.0) { return 0.0 }
    return $rounded
}

function rnd { return ($rng.NextDouble() - 0.5) }

# ── Process Transfermarkt players ─────────────────────────────────────────────
Write-Host "Loading raw_players.csv..."
$rawPlayers = Import-Csv "raw_players.csv"
Write-Host "Loaded $($rawPlayers.Count) players. Filtering..."

$tmCodes = $tmToApp.Keys
$filtered = $rawPlayers | Where-Object {
    $_.last_season -ge 2023 -and
    $_.current_club_domestic_competition_id -in $tmCodes -and
    $_.sub_position -ne "" -and
    $_.current_club_name -ne "" -and
    $_.name -ne ""
}
Write-Host "Filtered to $($filtered.Count) recent players in target leagues."

$outputRows = [System.Collections.Generic.List[PSCustomObject]]::new()
$playerIdx = 0

foreach ($p in $filtered) {
    $playerIdx++
    $tmCode  = $p.current_club_domestic_competition_id
    $appCode = $tmToApp[$tmCode]
    $pos     = if ($posMap.ContainsKey($p.sub_position)) { $posMap[$p.sub_position] } else { "CM" }
    $age     = Get-AgeFromDOB $p.date_of_birth
    $cyl     = Get-ContractYearsLeft $p.contract_expiration_date
    $mvEur   = try { [double]$p.market_value_in_eur } catch { 500000 }
    if ($mvEur -le 0) { $mvEur = 200000 }
    $mvM     = [Math]::Round($mvEur / 1000000.0, 2)
    $ovr     = Get-OverallRating $mvEur
    $pot     = [Math]::Min(97, $ovr + $rng.Next(0, [Math]::Max(1, [int]([Math]::Max(0, (30 - $age)) * 0.8))))
    $intlRep = Get-IntlRep $p.international_caps

    # Per-90 stats
    $qualityNorm = [Math]::Max(0, [Math]::Min(1, ($ovr - 50.0) / 47.0))
    $bases = $statBases[$pos]
    $noise = rnd

    $goals_p90    = Get-Stat $bases.goals_per90[0]           $bases.goals_per90[1]         $qualityNorm (rnd)
    $assists_p90  = Get-Stat $bases.assists_per90[0]         $bases.assists_per90[1]       $qualityNorm (rnd)
    $sot_pct      = Get-Stat $bases.shots_on_target_pct[0]  $bases.shots_on_target_pct[1] $qualityNorm (rnd)
    $_pc          = Get-Stat $bases.pass_completion[0]       $bases.pass_completion[1]     $qualityNorm (rnd)
    $pass_comp    = [Math]::Min(0.98, $_pc)
    $kp_p90       = Get-Stat $bases.key_passes_per90[0]     $bases.key_passes_per90[1]    $qualityNorm (rnd)
    $prog_pass    = Get-Stat $bases.progressive_passes[0]   $bases.progressive_passes[1]  $qualityNorm (rnd)
    $tackles_p90  = Get-Stat $bases.tackles_per90[0]         $bases.tackles_per90[1]       $qualityNorm (rnd)
    $inter_p90    = Get-Stat $bases.interceptions_per90[0]  $bases.interceptions_per90[1] $qualityNorm (rnd)
    $drib_p90     = Get-Stat $bases.dribbles_per90[0]        $bases.dribbles_per90[1]      $qualityNorm (rnd)
    $_ad          = Get-Stat $bases.aerial_duels_won_pct[0] $bases.aerial_duels_won_pct[1] $qualityNorm (rnd)
    $aerial_pct   = [Math]::Min(0.95, $_ad)
    $_dw          = Get-Stat $bases.duels_won_pct[0]         $bases.duels_won_pct[1]       $qualityNorm (rnd)
    $duels_pct    = [Math]::Min(0.80, $_dw)
    $prog_carries = Get-Stat $bases.progressive_carries[0]  $bases.progressive_carries[1] $qualityNorm (rnd)
    $crosses_p90  = Get-Stat $bases.crosses_per90[0]         $bases.crosses_per90[1]       $qualityNorm (rnd)
    $thru_p90     = Get-Stat $bases.through_balls_per90[0]  $bases.through_balls_per90[1] $qualityNorm (rnd)
    $xg_p90       = Get-Stat $bases.xg_per90[0]              $bases.xg_per90[1]            $qualityNorm (rnd)
    $saves_p90    = Get-Stat $bases.saves_per90[0]           $bases.saves_per90[1]         $qualityNorm (rnd)
    $cs_pct       = Get-Stat $bases.clean_sheets_pct[0]     $bases.clean_sheets_pct[1]    $qualityNorm (rnd)
    $sweeper      = Get-Stat $bases.sweeper_actions[0]       $bases.sweeper_actions[1]     $qualityNorm (rnd)
    $gcph90       = if ($pos -eq "GK") { Get-Stat $bases.goals_conceded_per90[0] $bases.goals_conceded_per90[1] (1-$qualityNorm) (rnd) } else { 0 }

    # ── New FBref-style stats ──────────────────────────────────────────────────
    $npxg_p90    = Get-Stat $bases.npxg_per90[0]            $bases.npxg_per90[1]           $qualityNorm (rnd)
    $xa_p90      = Get-Stat $bases.xa_per90[0]              $bases.xa_per90[1]             $qualityNorm (rnd)
    $shots_tot   = Get-Stat $bases.shots_total_per90[0]     $bases.shots_total_per90[1]    $qualityNorm (rnd)
    $npxg_shot   = Get-Stat $bases.npxg_per_shot[0]         $bases.npxg_per_shot[1]        $qualityNorm (rnd)
    $_pcs        = Get-Stat $bases.pass_completion_short[0]  $bases.pass_completion_short[1]  $qualityNorm (rnd)
    $pass_short  = [Math]::Min(0.99, $_pcs)
    $_pcm        = Get-Stat $bases.pass_completion_medium[0] $bases.pass_completion_medium[1] $qualityNorm (rnd)
    $pass_med    = [Math]::Min(0.98, $_pcm)
    $_pcl        = Get-Stat $bases.pass_completion_long[0]   $bases.pass_completion_long[1]   $qualityNorm (rnd)
    $pass_long   = [Math]::Min(0.92, $_pcl)
    $sca_p90     = Get-Stat $bases.sca_per90[0]             $bases.sca_per90[1]            $qualityNorm (rnd)
    $gca_p90     = Get-Stat $bases.gca_per90[0]             $bases.gca_per90[1]            $qualityNorm (rnd)
    $_twp        = Get-Stat $bases.tackles_won_pct[0]        $bases.tackles_won_pct[1]      $qualityNorm (rnd)
    $tackles_won = [Math]::Min(0.85, $_twp)
    $blocks_p90  = Get-Stat $bases.blocks_per90[0]          $bases.blocks_per90[1]         $qualityNorm (rnd)
    $clear_p90   = Get-Stat $bases.clearances_per90[0]      $bases.clearances_per90[1]     $qualityNorm (rnd)
    $press_p90   = Get-Stat $bases.pressures_per90[0]       $bases.pressures_per90[1]      $qualityNorm (rnd)
    $_psp        = Get-Stat $bases.pressure_success_pct[0]  $bases.pressure_success_pct[1] $qualityNorm (rnd)
    $press_succ  = [Math]::Min(0.55, $_psp)
    $touches_p90 = Get-Stat $bases.touches_per90[0]         $bases.touches_per90[1]        $qualityNorm (rnd)
    $touch_att   = Get-Stat $bases.touches_att3rd_per90[0]  $bases.touches_att3rd_per90[1] $qualityNorm (rnd)
    $prog_rcv    = Get-Stat $bases.progressive_passes_received_per90[0] $bases.progressive_passes_received_per90[1] $qualityNorm (rnd)
    $fouls_c     = Get-Stat $bases.fouls_committed_per90[0] $bases.fouls_committed_per90[1] $qualityNorm (rnd)
    $fouls_d     = Get-Stat $bases.fouls_drawn_per90[0]     $bases.fouls_drawn_per90[1]    $qualityNorm (rnd)
    $offsides_p9 = Get-Stat $bases.offsides_per90[0]        $bases.offsides_per90[1]       $qualityNorm (rnd)

    # Card and minutes estimates
    $matches  = $rng.Next(10, 38)
    $minBase  = if ($ovr -ge 75) { 60 } elseif ($ovr -ge 65) { 40 } else { 25 }
    $minutes  = [Math]::Min($matches * 90, $rng.Next($matches * $minBase, $matches * 90))
    $yCards   = $rng.Next(0, 12)
    $rCards   = $rng.Next(0, 2)
    $minRatio = [Math]::Round($minutes / ([Math]::Max(1, $matches) * 90.0), 3)
    $yp90     = [Math]::Round($yCards / [Math]::Max(1, $minutes / 90.0), 3)
    $rp90     = [Math]::Round($rCards / [Math]::Max(1, $minutes / 90.0), 3)

    # Progression ratings
    $pastRating2 = [Math]::Max(45, [Math]::Min($ovr, $ovr - $rng.Next(0, [Math]::Max(1, [int]([Math]::Max(0, (27 - $age)) * 0.7 + 1)))))
    $pastRating1 = [Math]::Max(45, [int](($ovr + $pastRating2) / 2))

    $row = [PSCustomObject]@{
        player_id             = "TM$($p.player_id)"
        name                  = $p.name
        age                   = $age
        position              = $pos
        club                  = $p.current_club_name
        league                = $appCode
        league_name           = $leagueNames[$appCode]
        overall_rating        = $ovr
        potential             = $pot
        contract_years_left   = $cyl
        international_reputation = $intlRep
        market_value_m        = $mvM
        past_rating_2yr       = $pastRating2
        past_rating_1yr       = $pastRating1
        yellow_cards          = $yCards
        red_cards             = $rCards
        matches_in_squad      = $matches
        minutes_played        = $minutes
        is_women              = $false
        goals_per90           = $goals_p90
        assists_per90         = $assists_p90
        shots_on_target_pct   = $sot_pct
        pass_completion       = $pass_comp
        key_passes_per90      = $kp_p90
        progressive_passes    = $prog_pass
        tackles_per90         = $tackles_p90
        interceptions_per90   = $inter_p90
        dribbles_per90        = $drib_p90
        aerial_duels_won_pct  = $aerial_pct
        duels_won_pct         = $duels_pct
        progressive_carries   = $prog_carries
        crosses_per90         = $crosses_p90
        through_balls_per90               = $thru_p90
        xg_per90                          = $xg_p90
        npxg_per90                        = $npxg_p90
        xa_per90                          = $xa_p90
        shots_total_per90                 = $shots_tot
        npxg_per_shot                     = $npxg_shot
        pass_completion_short             = $pass_short
        pass_completion_medium            = $pass_med
        pass_completion_long              = $pass_long
        sca_per90                         = $sca_p90
        gca_per90                         = $gca_p90
        tackles_won_pct                   = $tackles_won
        blocks_per90                      = $blocks_p90
        clearances_per90                  = $clear_p90
        pressures_per90                   = $press_p90
        pressure_success_pct              = $press_succ
        touches_per90                     = $touches_p90
        touches_att3rd_per90              = $touch_att
        progressive_passes_received_per90 = $prog_rcv
        fouls_committed_per90             = $fouls_c
        fouls_drawn_per90                 = $fouls_d
        offsides_per90                    = $offsides_p9
        saves_per90                       = $saves_p90
        clean_sheets_pct                  = $cs_pct
        sweeper_actions                   = $sweeper
        goals_conceded_per90              = $gcph90
        minutes_per90_ratio               = $minRatio
        yellow_cards_per90                = $yp90
        red_cards_per90                   = $rp90
    }
    $outputRows.Add($row)
}
Write-Host "Processed $playerIdx Transfermarkt players."

# ── Synthetic supplement: missing leagues ─────────────────────────────────────
$missingLeagues = @{
    "AR1N" = @{ name="Argentine Primera"; clubs=@("Boca Juniors","River Plate","Racing Club","Independiente","San Lorenzo","Estudiantes","Talleres","Velez Sarsfield","Huracan","Lanus") }
    "BU1"  = @{ name="Bulgarian First League"; clubs=@("Ludogorets","CSKA Sofia","Levski Sofia","Lokomotiv Plovdiv","Botev Plovdiv","Slavia Sofia","Beroe","Arda Kardzhali","Etar","Montana") }
    "ES2"  = @{ name="La Liga 2"; clubs=@("Valladolid","Mirandes","Levante","Huesca","Eibar","Zaragoza","Racing Santander","Oviedo","Alcorcon","Elche") }
    "FR2"  = @{ name="Ligue 2"; clubs=@("Strasbourg","Metz","Caen","Rodez","Grenoble","Amiens","Troyes","Valenciennes","Pau FC","Concarneau") }
    "GB2"  = @{ name="Championship"; clubs=@("Leeds United","Leicester City","Middlesbrough","Sunderland","West Brom","Swansea","Burnley","Sheffield United","Watford","QPR") }
    "IT2"  = @{ name="Serie B"; clubs=@("Parma","Como","Venezia","Sampdoria","Genoa","Pisa","Palermo","Bari","Catanzaro","Ascoli") }
    "L2"   = @{ name="2. Bundesliga"; clubs=@("Hamburger SV","Schalke 04","Hannover 96","Kaiserslautern","Hertha BSC","Fortuna Dusseldorf","FC Nurnberg","FC Magdeburg","Greuther Furth","Karlsruher SC") }
    "SL1"  = @{ name="Slovenian PrvaLiga"; clubs=@("NK Olimpija Ljubljana","NK Maribor","NK Celje","NK Koper","NK Mura","NK Domzale","NK Bravo","NK Radomlje","NK Nafta 1903","ND Gorica") }
    "UNG1" = @{ name="OTP Bank Liga"; clubs=@("Ferencvaros","MOL Fehervár","Paks","Ujpest","Kecskemet","Puskas Akademia","Debrecen","MTK Budapest","Zalaegerszeg","Honved") }
}

$womenLeagues = @{
    "WGBL"  = @{ name="Womens Bundesliga"; clubs=@("Bayern Munich W","Wolfsburg W","Frankfurt W","Freiburg W","Hoffenheim W","Turbine Potsdam","Koln W","RB Leipzig W","Duisburg W","Essen W") }
    "WWSL"  = @{ name="WSL"; clubs=@("Chelsea W","Arsenal W","Manchester City W","Manchester United W","Aston Villa W","Liverpool W","Brighton W","West Ham W","Tottenham W","Leicester W") }
    "WFRD1" = @{ name="D1 Feminine"; clubs=@("Lyon W","PSG W","Paris FC W","Bordeaux W","Montpellier W","Guingamp W","Dijon W","Nice W","Reims W","Fleury W") }
    "WNWSL" = @{ name="NWSL"; clubs=@("Portland Thorns","NC Courage","Chicago Red Stars","OL Reign","Washington Spirit","San Diego Wave","Angel City","Houston Dash","NJ/NY Gotham","Racing Louisville") }
    "WAUS"  = @{ name="A-League Women"; clubs=@("Melbourne City W","Sydney FC W","Western Sydney W","Brisbane Roar W","Perth Glory W","Adelaide United W","Wellington Phoenix W","Canberra United","Newcastle Jets W","Central Coast W") }
    "WBRA"  = @{ name="Brazilian Womens Serie A"; clubs=@("Corinthians W","Palmeiras W","Flamengo W","Santos W","Sao Paulo W","Cruzeiro W","Gremio W","Internacional W","Ferroviaria","Avai Kindermann") }
    "WITA"  = @{ name="Womens Serie A"; clubs=@("Roma W","Juventus W","Milan W","Inter W","Fiorentina W","Sassuolo W","Sampdoria W","Lazio W","Napoli W","Hellas Verona W") }
}

$firstNames = @("Lucas","Marco","Luca","Joao","Carlos","Diego","Ahmed","Mohamed","Pierre","Theo","Kai","Jamal","Phil","Mason","Bukayo","Erling","Kylian","Vinicius","Rodri","Federico","Pedri","Gavi","Jude","Florian","Leroy","Alphonso","Cody","Lamine","Aitana","Sam","Caroline","Vivianne","Ada","Pernille","Kadidiatou","Asisat","Trinity","Sophia","Elena","Lars","Erik","Ivan","Aleksandr","Tomas","Krzysztof","Rui","Bruno","Bernardo","Rafael","Gabriel","Sandro","Roberto","Raul","Antonio","Jose","Manuel","Sergio","Leon","Julian","Thomas","Toni","Joshua","Mats","Niklas","Ali","Omar","Yusuf","Mehmet","Burak","Emre","Hakan","Aleksandar","Stefan","Nemanja","Dusan","Luka","Mateo")
$lastNames  = @("Silva","Costa","Santos","Ferreira","Oliveira","Rodrigues","Mueller","Schmidt","Schneider","Fischer","Weber","Meyer","Garcia","Martinez","Lopez","Hernandez","Gonzalez","Perez","Dupont","Martin","Bernard","Rossi","Ferrari","Russo","Esposito","Romano","Kowalski","Nowak","Petrov","Ivanov","Sidorov","Park","Kim","Lee","Choi","Mbappe","Diallo","Traore","Diarra","Haaland","Odegaard","Pedersen","Nielsen","Hansen","Jensen","Moura","Nunes","Pinto","Carvalho","Mendes","Rashford","Sterling","Walker","Trippier","Yilmaz","Ozil","Calhanoglu","Mitrovic","Jovic","Vlahovic","Modric","Kovacic","Kramaric")

$positions = @("GK","CB","CB","LB","RB","CDM","CM","CM","CAM","LW","RW","ST","ST")

function New-SyntheticPlayer($leagueCode, $leagueName, $club, $idx, $isWomen) {
    $pos     = $positions[$rng.Next(0, $positions.Length)]
    $baseOvr = if ($leagueCode -in @("ES2","FR2","GB2","IT2","L2")) { 63 }
               elseif ($leagueCode -in @("BU1","UNG1","TS1")) { 60 }
               elseif ($isWomen) { 68 }
               else { 65 }
    $ovr     = [Math]::Max(50, [Math]::Min(90, $baseOvr + $rng.Next(-8, 12)))
    $age     = $rng.Next(18, 36)
    $pot     = [Math]::Min(95, $ovr + $rng.Next(0, [Math]::Max(1, (30-$age))))
    $mvM     = [Math]::Round([Math]::Pow(10, ($ovr - 55) / 8.0 + 4) / 1000000.0, 2)
    $mvM     = [Math]::Max(0.1, [Math]::Min(25, $mvM))
    $cyl     = [Math]::Round($rng.NextDouble() * 4 + 0.5, 1)
    $intlRep = $rng.Next(1, 4)
    $qualityNorm = [Math]::Max(0, [Math]::Min(1, ($ovr - 50.0) / 47.0))

    $bases       = $statBases[$pos]
    $goals_p90   = Get-Stat $bases.goals_per90[0]           $bases.goals_per90[1]         $qualityNorm (rnd)
    $assists_p90 = Get-Stat $bases.assists_per90[0]         $bases.assists_per90[1]       $qualityNorm (rnd)
    $sot_pct     = Get-Stat $bases.shots_on_target_pct[0]  $bases.shots_on_target_pct[1] $qualityNorm (rnd)
    $_pc2        = Get-Stat $bases.pass_completion[0]       $bases.pass_completion[1]     $qualityNorm (rnd)
    $pass_comp   = [Math]::Min(0.97, $_pc2)
    $kp_p90      = Get-Stat $bases.key_passes_per90[0]     $bases.key_passes_per90[1]    $qualityNorm (rnd)
    $prog_pass   = Get-Stat $bases.progressive_passes[0]   $bases.progressive_passes[1]  $qualityNorm (rnd)
    $tackles_p90 = Get-Stat $bases.tackles_per90[0]         $bases.tackles_per90[1]       $qualityNorm (rnd)
    $inter_p90   = Get-Stat $bases.interceptions_per90[0]  $bases.interceptions_per90[1] $qualityNorm (rnd)
    $drib_p90    = Get-Stat $bases.dribbles_per90[0]        $bases.dribbles_per90[1]      $qualityNorm (rnd)
    $_ad2        = Get-Stat $bases.aerial_duels_won_pct[0] $bases.aerial_duels_won_pct[1] $qualityNorm (rnd)
    $aerial_pct  = [Math]::Min(0.95, $_ad2)
    $_dw2        = Get-Stat $bases.duels_won_pct[0]         $bases.duels_won_pct[1]       $qualityNorm (rnd)
    $duels_pct   = [Math]::Min(0.80, $_dw2)
    $prog_car    = Get-Stat $bases.progressive_carries[0]  $bases.progressive_carries[1] $qualityNorm (rnd)
    $crosses_p90 = Get-Stat $bases.crosses_per90[0]         $bases.crosses_per90[1]       $qualityNorm (rnd)
    $thru_p90    = Get-Stat $bases.through_balls_per90[0]  $bases.through_balls_per90[1] $qualityNorm (rnd)
    $xg_p90      = Get-Stat $bases.xg_per90[0]              $bases.xg_per90[1]            $qualityNorm (rnd)
    $saves_p90   = Get-Stat $bases.saves_per90[0]           $bases.saves_per90[1]         $qualityNorm (rnd)
    $cs_pct      = Get-Stat $bases.clean_sheets_pct[0]     $bases.clean_sheets_pct[1]    $qualityNorm (rnd)
    $sweeper     = Get-Stat $bases.sweeper_actions[0]       $bases.sweeper_actions[1]     $qualityNorm (rnd)
    $gcph90      = if ($pos -eq "GK") { Get-Stat $bases.goals_conceded_per90[0] $bases.goals_conceded_per90[1] (1-$qualityNorm) (rnd) } else { 0 }

    # New FBref stats for synthetic players
    $npxg_p90s   = Get-Stat $bases.npxg_per90[0]            $bases.npxg_per90[1]           $qualityNorm (rnd)
    $xa_p90s     = Get-Stat $bases.xa_per90[0]              $bases.xa_per90[1]             $qualityNorm (rnd)
    $shots_tots  = Get-Stat $bases.shots_total_per90[0]     $bases.shots_total_per90[1]    $qualityNorm (rnd)
    $npxg_shots  = Get-Stat $bases.npxg_per_shot[0]         $bases.npxg_per_shot[1]        $qualityNorm (rnd)
    $_pcs2       = Get-Stat $bases.pass_completion_short[0]  $bases.pass_completion_short[1]  $qualityNorm (rnd)
    $pass_shorts = [Math]::Min(0.99, $_pcs2)
    $_pcm2       = Get-Stat $bases.pass_completion_medium[0] $bases.pass_completion_medium[1] $qualityNorm (rnd)
    $pass_meds   = [Math]::Min(0.98, $_pcm2)
    $_pcl2       = Get-Stat $bases.pass_completion_long[0]   $bases.pass_completion_long[1]   $qualityNorm (rnd)
    $pass_longs  = [Math]::Min(0.92, $_pcl2)
    $sca_p90s    = Get-Stat $bases.sca_per90[0]             $bases.sca_per90[1]            $qualityNorm (rnd)
    $gca_p90s    = Get-Stat $bases.gca_per90[0]             $bases.gca_per90[1]            $qualityNorm (rnd)
    $_twps       = Get-Stat $bases.tackles_won_pct[0]        $bases.tackles_won_pct[1]      $qualityNorm (rnd)
    $tack_wons   = [Math]::Min(0.85, $_twps)
    $blocks_p90s = Get-Stat $bases.blocks_per90[0]          $bases.blocks_per90[1]         $qualityNorm (rnd)
    $clear_p90s  = Get-Stat $bases.clearances_per90[0]      $bases.clearances_per90[1]     $qualityNorm (rnd)
    $press_p90s  = Get-Stat $bases.pressures_per90[0]       $bases.pressures_per90[1]      $qualityNorm (rnd)
    $_psps       = Get-Stat $bases.pressure_success_pct[0]  $bases.pressure_success_pct[1] $qualityNorm (rnd)
    $press_succs = [Math]::Min(0.55, $_psps)
    $touch_p90s  = Get-Stat $bases.touches_per90[0]         $bases.touches_per90[1]        $qualityNorm (rnd)
    $touch_atts  = Get-Stat $bases.touches_att3rd_per90[0]  $bases.touches_att3rd_per90[1] $qualityNorm (rnd)
    $prog_rcvs   = Get-Stat $bases.progressive_passes_received_per90[0] $bases.progressive_passes_received_per90[1] $qualityNorm (rnd)
    $fouls_cs    = Get-Stat $bases.fouls_committed_per90[0] $bases.fouls_committed_per90[1] $qualityNorm (rnd)
    $fouls_ds    = Get-Stat $bases.fouls_drawn_per90[0]     $bases.fouls_drawn_per90[1]    $qualityNorm (rnd)
    $offs_p90s   = Get-Stat $bases.offsides_per90[0]        $bases.offsides_per90[1]       $qualityNorm (rnd)

    $matches  = $rng.Next(10, 38)
    $minBase  = if ($ovr -ge 75) { 60 } elseif ($ovr -ge 65) { 40 } else { 25 }
    $minutes  = [Math]::Min($matches * 90, $rng.Next($matches * $minBase, $matches * 90))
    $yCards   = $rng.Next(0, 10)
    $rCards   = $rng.Next(0, 2)
    $minRatio = [Math]::Round($minutes / ([Math]::Max(1, $matches) * 90.0), 3)
    $yp90     = [Math]::Round($yCards / [Math]::Max(1, $minutes / 90.0), 3)
    $rp90     = [Math]::Round($rCards / [Math]::Max(1, $minutes / 90.0), 3)
    $pastR2   = [Math]::Max(45, $ovr - $rng.Next(0, [Math]::Max(1, [int]([Math]::Max(0,(27-$age))*0.7+1))))
    $pastR1   = [Math]::Max(45, [int](($ovr + $pastR2) / 2))

    $fn = $firstNames[$rng.Next(0, $firstNames.Length)]
    $ln = $lastNames[$rng.Next(0, $lastNames.Length)]

    return [PSCustomObject]@{
        player_id             = "SYN${leagueCode}${idx}"
        name                  = "$fn $ln"
        age                   = $age
        position              = $pos
        club                  = $club
        league                = $leagueCode
        league_name           = $leagueName
        overall_rating        = $ovr
        potential             = $pot
        contract_years_left   = $cyl
        international_reputation = $intlRep
        market_value_m        = $mvM
        past_rating_2yr       = $pastR2
        past_rating_1yr       = $pastR1
        yellow_cards          = $yCards
        red_cards             = $rCards
        matches_in_squad      = $matches
        minutes_played        = $minutes
        is_women              = $isWomen
        goals_per90           = $goals_p90
        assists_per90         = $assists_p90
        shots_on_target_pct   = $sot_pct
        pass_completion       = $pass_comp
        key_passes_per90      = $kp_p90
        progressive_passes    = $prog_pass
        tackles_per90         = $tackles_p90
        interceptions_per90   = $inter_p90
        dribbles_per90        = $drib_p90
        aerial_duels_won_pct  = $aerial_pct
        duels_won_pct         = $duels_pct
        progressive_carries   = $prog_car
        crosses_per90         = $crosses_p90
        through_balls_per90   = $thru_p90
        xg_per90              = $xg_p90
        saves_per90           = $saves_p90
        clean_sheets_pct      = $cs_pct
        sweeper_actions       = $sweeper
        goals_conceded_per90  = $gcph90
        minutes_per90_ratio   = $minRatio
        yellow_cards_per90    = $yp90
        red_cards_per90       = $rp90
        npxg_per90                        = $npxg_p90s
        xa_per90                          = $xa_p90s
        shots_total_per90                 = $shots_tots
        npxg_per_shot                     = $npxg_shots
        pass_completion_short             = $pass_shorts
        pass_completion_medium            = $pass_meds
        pass_completion_long              = $pass_longs
        sca_per90                         = $sca_p90s
        gca_per90                         = $gca_p90s
        tackles_won_pct                   = $tack_wons
        blocks_per90                      = $blocks_p90s
        clearances_per90                  = $clear_p90s
        pressures_per90                   = $press_p90s
        pressure_success_pct              = $press_succs
        touches_per90                     = $touch_p90s
        touches_att3rd_per90              = $touch_atts
        progressive_passes_received_per90 = $prog_rcvs
        fouls_committed_per90             = $fouls_cs
        fouls_drawn_per90                 = $fouls_ds
        offsides_per90                    = $offs_p90s
    }
}

# Generate ~20 players per club for missing men's leagues
$synIdx = 0
foreach ($lCode in $missingLeagues.Keys) {
    $lInfo = $missingLeagues[$lCode]
    foreach ($club in $lInfo.clubs) {
        for ($i = 0; $i -lt 20; $i++) {
            $synIdx++
            $row = New-SyntheticPlayer $lCode $lInfo.name $club $synIdx $false
            $outputRows.Add($row)
        }
    }
}
Write-Host "Added synthetic men's players for missing leagues."

# Generate ~18 players per club for women's leagues
foreach ($lCode in $womenLeagues.Keys) {
    $lInfo = $womenLeagues[$lCode]
    foreach ($club in $lInfo.clubs) {
        for ($i = 0; $i -lt 18; $i++) {
            $synIdx++
            $row = New-SyntheticPlayer $lCode $lInfo.name $club $synIdx $true
            $outputRows.Add($row)
        }
    }
}
Write-Host "Added synthetic women's players."

# ── Aggregate real stats from appearances.csv ─────────────────────────────────
# appearances columns: appearance_id(0),game_id(1),player_id(2),player_club_id(3),
#   player_current_club_id(4),date(5),player_name(6),competition_id(7),
#   yellow_cards(8),red_cards(9),goals(10),assists(11),minutes_played(12)

Write-Host "Aggregating real stats from appearances.csv (2022-2025)..."
$appStats = @{}  # key = player_id (string), value = hashtable of aggregated stats

if (Test-Path "raw_appearances.csv") {
    $appReader = [System.IO.StreamReader]::new("raw_appearances.csv", [System.Text.Encoding]::UTF8)
    $null = $appReader.ReadLine()  # skip header
    $appCount = 0
    $appKept  = 0
    while (!$appReader.EndOfStream) {
        $line  = $appReader.ReadLine()
        $appCount++
        $parts = $line -split ","
        if ($parts.Count -lt 13) { continue }
        $date  = $parts[5]
        if ($date -lt "2022-07-01") { continue }
        $appKept++
        $tmid  = $parts[2]
        $yc    = try { [int]$parts[8] } catch { 0 }
        $rc    = try { [int]$parts[9] } catch { 0 }
        $g     = try { [int]$parts[10] } catch { 0 }
        $a     = try { [int]$parts[11] } catch { 0 }
        $mins  = try { [int]$parts[12] } catch { 0 }
        if (!$appStats.ContainsKey($tmid)) {
            $appStats[$tmid] = @{ goals=0; assists=0; yellow_cards=0; red_cards=0; minutes=0; matches=0 }
        }
        $appStats[$tmid].goals        += $g
        $appStats[$tmid].assists      += $a
        $appStats[$tmid].yellow_cards += $yc
        $appStats[$tmid].red_cards    += $rc
        $appStats[$tmid].minutes      += $mins
        $appStats[$tmid].matches      += 1
    }
    $appReader.Close()
    Write-Host "Processed $appCount appearance rows; kept $appKept (2022+); $($appStats.Count) unique players."
} else {
    Write-Host "raw_appearances.csv not found - using estimated stats only."
}

# Patch player rows with real appearance data where available
$patched = 0
foreach ($row in $outputRows) {
    # strip "TM" prefix to get raw Transfermarkt player_id
    $rawId = $row.player_id -replace "^TM", ""
    if (-not $appStats.ContainsKey($rawId)) { continue }
    $s = $appStats[$rawId]
    if ($s.minutes -lt 90) { continue }  # ignore tiny samples
    $nineties = [double]$s.minutes / 90.0

    $row.goals_per90         = [Math]::Round([double]$s.goals   / $nineties, 3)
    $row.assists_per90       = [Math]::Round([double]$s.assists  / $nineties, 3)
    $row.yellow_cards_per90  = [Math]::Round([double]$s.yellow_cards / $nineties, 3)
    $row.red_cards_per90     = [Math]::Round([double]$s.red_cards    / $nineties, 3)
    $row.yellow_cards        = $s.yellow_cards
    $row.red_cards           = $s.red_cards
    $row.minutes_played      = $s.minutes
    $row.matches_in_squad    = $s.matches
    $row.minutes_per90_ratio = [Math]::Round([double]$s.minutes / ([double]$s.matches * 90.0), 3)

    # Also update xg estimate from goals (as rough proxy)
    if ($row.xg_per90 -gt 0) {
        $row.xg_per90 = [Math]::Round($row.goals_per90 * 1.05, 3)
    }
    $patched++
}
Write-Host "Patched $patched players with real goals/assists/cards/minutes."

# ── Export all_players_data.csv ────────────────────────────────────────────────
Write-Host "Exporting all_players_data.csv ($($outputRows.Count) players)..."
$outputRows | Export-Csv -Path "all_players_data.csv" -NoTypeInformation -Encoding UTF8
Write-Host "Saved all_players_data.csv"

# ── Build team_clusters.csv ────────────────────────────────────────────────────
Write-Host "Building team_clusters.csv..."
$teamRows = [System.Collections.Generic.List[PSCustomObject]]::new()

$byTeam = $outputRows | Group-Object -Property club, league

foreach ($grp in $byTeam) {
    $teamPlayers = $grp.Group
    $club   = $teamPlayers[0].club
    $league = $teamPlayers[0].league
    $n      = $teamPlayers.Count

    $avgPassComp     = [Math]::Round(($teamPlayers | Measure-Object pass_completion     -Average).Average, 4)
    $avgPressAct     = [Math]::Round((($teamPlayers | Measure-Object tackles_per90      -Average).Average + ($teamPlayers | Measure-Object interceptions_per90 -Average).Average), 4)
    $avgProgPass     = [Math]::Round(($teamPlayers | Measure-Object progressive_passes  -Average).Average, 4)
    $avgProgCar      = [Math]::Round(($teamPlayers | Measure-Object progressive_carries -Average).Average, 4)
    $avgKeyPasses    = [Math]::Round(($teamPlayers | Measure-Object key_passes_per90    -Average).Average, 4)
    $avgDribbles     = [Math]::Round(($teamPlayers | Measure-Object dribbles_per90      -Average).Average, 4)
    $avgCrosses      = [Math]::Round(($teamPlayers | Measure-Object crosses_per90       -Average).Average, 4)
    $avgAerialDuels  = [Math]::Round(($teamPlayers | Measure-Object aerial_duels_won_pct -Average).Average, 4)
    $squadAge        = [Math]::Round(($teamPlayers | Measure-Object age                 -Average).Average, 1)
    $squadRating     = [Math]::Round(($teamPlayers | Measure-Object overall_rating      -Average).Average, 1)
    $squadSize       = $n

    $teamRows.Add([PSCustomObject]@{
        club                 = $club
        league               = $league
        avg_pass_completion  = $avgPassComp
        avg_pressing_actions = $avgPressAct
        avg_progressive_passes = $avgProgPass
        avg_progressive_carries = $avgProgCar
        avg_key_passes       = $avgKeyPasses
        avg_dribbles         = $avgDribbles
        avg_crosses          = $avgCrosses
        avg_aerial_duels_won = $avgAerialDuels
        squad_age            = $squadAge
        squad_rating         = $squadRating
        squad_size           = $squadSize
        team_style           = ""
    })
}

$teamRows | Export-Csv -Path "team_clusters.csv" -NoTypeInformation -Encoding UTF8
Write-Host "Saved team_clusters.csv ($($teamRows.Count) teams)."
Write-Host ""
Write-Host "Done! Files ready:"
Write-Host "  all_players_data.csv  - $($outputRows.Count) players"
Write-Host "  team_clusters.csv     - $($teamRows.Count) teams"
