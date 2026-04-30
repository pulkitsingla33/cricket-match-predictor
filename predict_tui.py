import pandas as pd
import numpy as np
import joblib
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, IntPrompt
from rich import print as rprint
from pathlib import Path
from par_score import get_venue_batting_stats

console = Console()

# Home cities mapping (must match ipl_feature_engineering.py)
TEAM_HOME_CITIES = {
    "Mumbai Indians": ["Mumbai"],
    "Chennai Super Kings": ["Chennai"],
    "Royal Challengers Bengaluru": ["Bangalore", "Bengaluru"],
    "Kolkata Knight Riders": ["Kolkata"],
    "Delhi Capitals": ["Delhi"],
    "Punjab Kings": ["Chandigarh", "Mohali", "Mullanpur"],
    "Rajasthan Royals": ["Jaipur"],
    "Sunrisers Hyderabad": ["Hyderabad"],
    "Lucknow Super Giants": ["Lucknow"],
    "Gujarat Titans": ["Ahmedabad"],
    "Deccan Chargers": ["Hyderabad"],
    "Kochi Tuskers Kerala": ["Kochi"],
    "Pune Warriors": ["Pune"],
    "Rising Pune Supergiants": ["Pune"],
    "Gujarat Lions": ["Rajkot", "Ahmedabad"],
}

def display_numbered_menu(title, options, columns=2):
    """Prints a numbered menu for a list of options."""
    table = Table(show_header=False, box=None, pad_edge=False)
    for _ in range(columns):
        table.add_column()
    for i in range(0, len(options), columns):
        row_items = []
        for j, option in enumerate(options[i:i + columns], start=i + 1):
            row_items.append(f"[bold yellow]{j:2d}[/bold yellow] {option}")
        while len(row_items) < columns:
            row_items.append("")
        table.add_row(*row_items)
    console.print(Panel(table, title=title, expand=False, border_style="green"))


def select_numbered_option(prompt_text, options):
    """Prompts the user to choose an option by its number."""
    while True:
        choice = IntPrompt.ask(prompt_text)
        if 1 <= choice <= len(options):
            return options[choice - 1]
        rprint(f"[red]Invalid selection:[/red] enter a number between 1 and {len(options)}.")


def load_latest_stats(team1, team2, venue):
    """
    Fetches the latest pre-match stats for the two teams.
    Pulls actual computed stats from the feature CSVs.
    """
    data_path = Path(__file__).parent
    df = pd.read_csv(data_path / "features_match_level.csv")
    batting_df = pd.read_csv(data_path / "features_batting.csv")
    bowling_df = pd.read_csv(data_path / "features_bowling.csv")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    def get_team_stats(team):
        """Pull the most recent stats for a team from feature CSVs."""
        recent = df[(df['team1'] == team) | (df['team2'] == team)].sort_values('date', ascending=False)
        if recent.empty:
            return {
                'form': 0.5, 'venue_wr': 0.5, 'bat_first_wr': 0.5,
                'avg_pp_runs': 45.0, 'avg_death_wkts': 2.0,
                'bat_sr': 130.0, 'bat_avg': 25.0,
                'bowl_econ': 8.5, 'bowl_wkts': 1.5,
            }
        
        latest = recent.iloc[0]
        prefix = 'team1' if latest['team1'] == team else 'team2'
        
        stats = {
            'form': latest.get(f'{prefix}_recent_form', 0.5),
            'venue_wr': latest.get(f'{prefix}_venue_win_rate', 0.5),
            'bat_first_wr': latest.get(f'{prefix}_win_after_batting_first', 0.5),
            'avg_pp_runs': latest.get(f'{prefix}_avg_pp_runs', 45.0),
            'avg_death_wkts': latest.get(f'{prefix}_avg_death_wkts', 2.0),
        }
        
        # Pull actual player strength from batting/bowling CSVs
        # Get most recent matches for team's batters and bowlers
        recent_match_ids = recent['match_id'].head(5).tolist()
        
        team_batting = batting_df[
            (batting_df['match_id'].isin(recent_match_ids)) & 
            (batting_df['batting_team'] == team)
        ]
        stats['bat_sr'] = team_batting['strike_rate'].mean() if len(team_batting) > 0 else 130.0
        stats['bat_avg'] = team_batting['runs'].mean() if len(team_batting) > 0 else 25.0
        
        team_bowling = bowling_df[
            (bowling_df['match_id'].isin(recent_match_ids)) & 
            (bowling_df['bowling_team'] == team)
        ]
        stats['bowl_econ'] = team_bowling['economy'].mean() if len(team_bowling) > 0 else 8.5
        stats['bowl_wkts'] = team_bowling['wickets'].mean() if len(team_bowling) > 0 else 1.5
        
        return stats

    # Fetching H2H
    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) | 
             ((df['team1'] == team2) & (df['team2'] == team1))].sort_values('date', ascending=False)
    
    if not h2h.empty:
        latest_h2h = h2h.iloc[0]
        h2h_matches = latest_h2h.get('h2h_matches', 0)
        if latest_h2h['team1'] == team1:
            h2h_wr = latest_h2h['h2h_team1_win_rate']
        else:
            h2h_wr = 1 - latest_h2h['h2h_team1_win_rate']
    else:
        h2h_wr = 0.5
        h2h_matches = 0
        
    t1_stats = get_team_stats(team1)
    t2_stats = get_team_stats(team2)
    
    # Home advantage
    venue_city = df[df['venue'] == venue]['city'].mode()
    city = venue_city.iloc[0] if len(venue_city) > 0 else None
    is_home_t1 = int(city in TEAM_HOME_CITIES.get(team1, [])) if city else 0
    is_home_t2 = int(city in TEAM_HOME_CITIES.get(team2, [])) if city else 0
    
    # Construct feature row matching the model's expected features
    # All new matches use the Impact Player rule (IPL 2023+: 12-player squads)
    features = {
        'team1': team1,
        'team2': team2,
        # Toss
        'toss_winner_is_team1': 1,  # Default assumption for TUI
        'toss_bat_first': 0,
        # Historical rates
        'h2h_team1_win_rate': h2h_wr,
        'h2h_matches': h2h_matches,
        'team1_venue_win_rate': t1_stats['venue_wr'],
        'team2_venue_win_rate': t2_stats['venue_wr'],
        'team1_recent_form': t1_stats['form'],
        'team2_recent_form': t2_stats['form'],
        # Player strength
        'team1_batting_sr': t1_stats['bat_sr'],
        'team2_batting_sr': t2_stats['bat_sr'],
        'team1_batting_avg': t1_stats['bat_avg'],
        'team2_batting_avg': t2_stats['bat_avg'],
        'team1_bowling_econ': t1_stats['bowl_econ'],
        'team2_bowling_econ': t2_stats['bowl_econ'],
        'team1_bowling_wkts': t1_stats['bowl_wkts'],
        'team2_bowling_wkts': t2_stats['bowl_wkts'],
        # New features
        'team1_win_after_batting_first': t1_stats['bat_first_wr'],
        'team2_win_after_batting_first': t2_stats['bat_first_wr'],
        'team1_avg_pp_runs': t1_stats['avg_pp_runs'],
        'team2_avg_pp_runs': t2_stats['avg_pp_runs'],
        'team1_avg_death_wkts': t1_stats['avg_death_wkts'],
        'team2_avg_death_wkts': t2_stats['avg_death_wkts'],
        'is_home_team1': is_home_t1,
        'is_home_team2': is_home_t2,
        'season_progress': 0.5,  # Default mid-season
        # Differential features (computed here to match training)
        'form_diff': t1_stats['form'] - t2_stats['form'],
        'batting_sr_diff': t1_stats['bat_sr'] - t2_stats['bat_sr'],
        'batting_avg_diff': t1_stats['bat_avg'] - t2_stats['bat_avg'],
        'bowling_econ_diff': t2_stats['bowl_econ'] - t1_stats['bowl_econ'],
        'bowling_wkts_diff': t1_stats['bowl_wkts'] - t2_stats['bowl_wkts'],
        'venue_wr_diff': t1_stats['venue_wr'] - t2_stats['venue_wr'],
        'pp_runs_diff': t1_stats['avg_pp_runs'] - t2_stats['avg_pp_runs'],
        'death_wkts_diff': t1_stats['avg_death_wkts'] - t2_stats['avg_death_wkts'],
        # Impact Player era: all current IPL matches use 12-player squads (IPL 2023+)
        'is_impact_player_era': 1,
    }
    
    return pd.DataFrame([features]), t1_stats, t2_stats


def build_score_features(batting_team, bowling_team, venue, t_bat_stats, t_bowl_stats, 
                         toss_bat_first, is_home_bat, is_home_bowl, match_features_df):
    """
    Build the feature row for the score prediction model.
    Features must match train_score_model.py's expected columns.
    All predictions are assumed to be for the Impact Player era (IPL 2023+).
    """
    features = {
        'innings1_team': batting_team,
        'innings2_team': bowling_team,
        'venue': venue,
        # Batting team strength
        'bat_team_sr': t_bat_stats['bat_sr'],
        'bat_team_avg': t_bat_stats['bat_avg'],
        # Bowling team strength (team fielding first)
        'bowl_team_econ': t_bowl_stats['bowl_econ'],
        'bowl_team_wkts': t_bowl_stats['bowl_wkts'],
        # Historical rolling averages (use whichever team's stats apply)
        'team1_avg_pp_runs': t_bat_stats['avg_pp_runs'],
        'team2_avg_pp_runs': t_bowl_stats['avg_pp_runs'],
        'team1_avg_death_wkts': t_bat_stats['avg_death_wkts'],
        'team2_avg_death_wkts': t_bowl_stats['avg_death_wkts'],
        'team1_win_after_batting_first': t_bat_stats['bat_first_wr'],
        'team2_win_after_batting_first': t_bowl_stats['bat_first_wr'],
        # Context
        'toss_bat_first': toss_bat_first,
        'is_home_team1': is_home_bat,
        'is_home_team2': is_home_bowl,
        'season_progress': 0.5,
        # Impact Player era: all current IPL matches use 12-player squads (IPL 2023+)
        'is_impact_player_era': 1,
    }
    return pd.DataFrame([features])


def main():
    console.clear()
    rprint(Panel.fit(
        "[bold green]🏏 IPL Match Predictor [white]Ensemble Edition[/white] [/bold green]",
        subtitle="v3.0 — Winner · Score · Par Score"
    ))
    
    # Load Models
    data_path = Path(__file__).parent
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Loading model pipelines...", total=None)
        winner_model = joblib.load(data_path / "ipl_ensemble_predictor.joblib")
        
        # Score model (optional — may not be trained yet)
        score_model = None
        score_meta = None
        try:
            score_model = joblib.load(data_path / "ipl_score_predictor.joblib")
            score_meta = joblib.load(data_path / "ipl_score_predictor_meta.joblib")
        except FileNotFoundError:
            pass
        time.sleep(1)
        
    # Get available list for selection
    df = pd.read_csv(data_path / "features_match_level.csv")
    teams = sorted(df['team1'].unique().tolist())
    venues = sorted(df['venue'].unique().tolist())
    
    display_numbered_menu("Available Teams", teams)
    display_numbered_menu("Available Venues", venues)
    
    # ── Inputs ────────────────────────────────────────────
    t1 = select_numbered_option("Choose [bold red]Team 1[/bold red] (enter number)", teams)
    remaining_teams = [team for team in teams if team != t1]
    display_numbered_menu(f"Teams excluding {t1}", remaining_teams)
    t2 = select_numbered_option("Choose [bold blue]Team 2[/bold blue] (enter number)", remaining_teams)
    display_numbered_menu("Available Venues", venues)
    venue = select_numbered_option("Choose [bold yellow]Venue[/bold yellow] (enter number)", venues)
    
    # Ask who bats first
    bat_first = select_numbered_option(
        f"\nWho [bold green]bats first[/bold green]? (enter number)", [t1, t2]
    )
    bowl_first = t2 if bat_first == t1 else t1
    
    # ── Predictions ───────────────────────────────────────
    prediction_start = time.perf_counter()
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Analyzing {t1} vs {t2} at {venue}...", total=None)
        input_data, t1_stats, t2_stats = load_latest_stats(t1, t2, venue)
        
        # Winner prediction
        prob = winner_model.predict_proba(input_data)[0]
        winner_idx = winner_model.predict(input_data)[0]
        
        # Score prediction
        predicted_score = None
        score_range = None
        if score_model is not None:
            bat_stats = t1_stats if bat_first == t1 else t2_stats
            bowl_stats = t2_stats if bat_first == t1 else t1_stats
            
            is_home_bat = int(input_data.iloc[0]['is_home_team1']) if bat_first == t1 else int(input_data.iloc[0]['is_home_team2'])
            is_home_bowl = int(input_data.iloc[0]['is_home_team2']) if bat_first == t1 else int(input_data.iloc[0]['is_home_team1'])
            
            score_features = build_score_features(
                batting_team=bat_first,
                bowling_team=bowl_first,
                venue=venue,
                t_bat_stats=bat_stats,
                t_bowl_stats=bowl_stats,
                toss_bat_first=1,  # they chose to bat
                is_home_bat=is_home_bat,
                is_home_bowl=is_home_bowl,
                match_features_df=df,
            )
            predicted_score = score_model.predict(score_features)[0]
            # Apply impact-player-era bias correction:
            # The model trains on pre-2023 data and tends to underpredict higher post-2023 scores.
            # We add the mean residual (bias) measured on the 2023+ test set.
            if score_meta:
                bias = score_meta.get('impact_era_bias', 0.0)
                predicted_score += bias
                residual_std = score_meta.get('residual_std_impact_era',
                                              score_meta.get('residual_std', 20.0))
            else:
                residual_std = 20.0
            score_range = (int(round(predicted_score - residual_std)), 
                          int(round(predicted_score + residual_std)))
        
        # Par score
        venue_stats = get_venue_batting_stats(venue)
        
        time.sleep(2)
    
    # ══════════════════════════════════════════════════════
    #  DISPLAY RESULTS
    # ══════════════════════════════════════════════════════
    console.print("\n")
    
    # ── 1. Match Analysis Breakdown ───────────────────────
    res_table = Table(title="Match Analysis Breakdown")
    res_table.add_column("Factor", style="cyan")
    res_table.add_column(t1, style="red", justify="center")
    res_table.add_column(t2, style="blue", justify="center")
    
    input_row = input_data.iloc[0]
    res_table.add_row("Recent Form (Win %)", f"{input_row['team1_recent_form']*100:.1f}%", f"{input_row['team2_recent_form']*100:.1f}%")
    res_table.add_row("Venue Record (Win %)", f"{input_row['team1_venue_win_rate']*100:.1f}%", f"{input_row['team2_venue_win_rate']*100:.1f}%")
    res_table.add_row("Head-to-Head Win Rate", f"{input_row['h2h_team1_win_rate']*100:.1f}%", f"{(1-input_row['h2h_team1_win_rate'])*100:.1f}%")
    res_table.add_row("Batting Strike Rate", f"{input_row['team1_batting_sr']:.1f}", f"{input_row['team2_batting_sr']:.1f}")
    res_table.add_row("Bowling Economy", f"{input_row['team1_bowling_econ']:.2f}", f"{input_row['team2_bowling_econ']:.2f}")
    res_table.add_row("Bat-First Win Rate", f"{input_row['team1_win_after_batting_first']*100:.1f}%", f"{input_row['team2_win_after_batting_first']*100:.1f}%")
    res_table.add_row("Avg Powerplay Runs", f"{input_row['team1_avg_pp_runs']:.1f}", f"{input_row['team2_avg_pp_runs']:.1f}")
    res_table.add_row("Home Advantage", "✓" if input_row['is_home_team1'] else "✗", "✓" if input_row['is_home_team2'] else "✗")
    
    console.print(res_table)
    
    # ── 2. Winner Prediction ──────────────────────────────
    t1_prob = prob[1] * 100
    t2_prob = prob[0] * 100
    
    winner = t1 if winner_idx == 1 else t2
    winner_color = "red" if winner == t1 else "blue"
    win_prob = t1_prob if winner == t1 else t2_prob
    
    win_panel = Panel(
        f"[bold yellow]PREDICTED WINNER:[/bold yellow]\n\n"
        f"[bold {winner_color} underline]{winner}[/bold {winner_color} underline]\n"
        f"[white]Win Probability: {win_prob:.1f}%[/white]",
        expand=False,
        border_style="yellow",
        padding=(1, 5)
    )
    console.print(win_panel, justify="center")
    
    # ── 3. First Innings Score Prediction ─────────────────
    if predicted_score is not None:
        score_int = int(round(predicted_score))
        
        # Compare with par score
        par = venue_stats['par_score']
        if score_int > par:
            score_verdict = f"[green]Above par ({par}) — favors {bat_first}[/green]"
        elif score_int < par:
            score_verdict = f"[red]Below par ({par}) — favors {bowl_first}[/red]"
        else:
            score_verdict = f"[yellow]Right at par ({par}) — evenly balanced[/yellow]"
        
        score_panel = Panel(
            f"[bold cyan]PREDICTED 1ST INNINGS SCORE:[/bold cyan]\n\n"
            f"[bold white]{bat_first}[/bold white] batting first\n"
            f"[bold green]{score_int}[/bold green] [dim]({score_range[0]}–{score_range[1]} range)[/dim]\n\n"
            f"{score_verdict}",
            expand=False,
            border_style="cyan",
            padding=(1, 5)
        )
        console.print(score_panel, justify="center")
    else:
        rprint("\n [dim]⚠ Score prediction model not found. Run train_score_model.py first.[/dim]")
    
    # ── 4. Par Score & Venue Insights ─────────────────────
    par_table = Table(title=f"Venue Insights — {venue}")
    par_table.add_column("Metric", style="cyan")
    par_table.add_column("Value", style="white", justify="center")
    
    par_note = "" if venue_stats['is_venue_specific'] else " [dim](global avg)[/dim]"
    par_table.add_row("Par Score", f"[bold]{venue_stats['par_score']}[/bold]{par_note}")
    par_table.add_row("Avg 1st Innings", f"{venue_stats['avg_first_innings']}")
    if venue_stats['avg_second_innings']:
        par_table.add_row("Avg 2nd Innings", f"{venue_stats['avg_second_innings']}")
    par_table.add_row("Bat-First Win %", f"{venue_stats['bat_first_win_pct']:.1f}%")
    par_table.add_row("Matches at Venue", f"{venue_stats['venue_matches']}")
    
    if not venue_stats['is_venue_specific']:
        par_table.add_row("", "[dim italic]< 10 matches — using global averages[/dim italic]")
    
    console.print(par_table)
    
    prediction_end = time.perf_counter()
    elapsed_seconds = prediction_end - prediction_start
    rprint(f"\n[dim]Prediction completed in [bold]{elapsed_seconds:.2f} seconds[/bold] using Stacking Ensemble models (RF + HGB + ET).[/dim]")

if __name__ == "__main__":
    main()
