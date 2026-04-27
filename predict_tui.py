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
    }
    
    return pd.DataFrame([features])

def main():
    console.clear()
    rprint(Panel.fit("[bold green]🏏 IPL Match Outcome Predictor [white]Ensemble Edition[/white] [/bold green]", subtitle="v2.0"))
    
    # Load Model
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Loading model pipeline...", total=None)
        model = joblib.load("ipl_ensemble_predictor.joblib")
        time.sleep(1)
        
    # Get available list for selection
    df = pd.read_csv("features_match_level.csv")
    teams = sorted(df['team1'].unique().tolist())
    venues = sorted(df['venue'].unique().tolist())
    
    rprint("[bold cyan]Available Teams:[/bold cyan]")
    # Print teams in columns
    table = Table(show_header=False, box=None)
    for i in range(0, len(teams), 2):
        row = teams[i:i+2]
        table.add_row(*row)
    console.print(table)

    rprint("[bold cyan]Available Venues:[/bold cyan]")
    # Print venues in columns
    table = Table(show_header=False, box=None)
    for i in range(0, len(venues), 2):
        row = venues[i:i+2]
        table.add_row(*row)
    console.print(table)
    
    # Inputs
    t1 = Prompt.ask("\nChoose [bold red]Team 1[/bold red]", choices=teams)
    t2 = Prompt.ask("Choose [bold blue]Team 2[/bold blue]", choices=[t for t in teams if t != t1])
    venue = Prompt.ask("Select [bold yellow]Venue[/bold yellow]", choices=venues, default=venues[0])
    
    # Predictions
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Analyzing match statistics for {t1} vs {t2}...", total=None)
        input_data = load_latest_stats(t1, t2, venue)
        prob = model.predict_proba(input_data)[0]
        winner_idx = model.predict(input_data)[0]
        time.sleep(2)
    
    # Display Result
    t1_prob = prob[1] * 100
    t2_prob = prob[0] * 100
    
    winner = t1 if winner_idx == 1 else t2
    winner_color = "red" if winner == t1 else "blue"
    win_prob = t1_prob if winner == t1 else t2_prob
    
    console.print("\n")
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
    
    win_panel = Panel(
        f"[bold yellow]PREDICTED WINNER:[/bold yellow]\n\n"
        f"[bold {winner_color} underline]{winner}[/bold {winner_color} underline]\n"
        f"[white]Probability: {win_prob:.1f}%[/white]",
        expand=False,
        border_style="yellow",
        padding=(1, 5)
    )
    
    console.print(win_panel, justify="center")
    
    # Final Graphic
    rprint(f"\n [dim]Prediction completed using a Stacking Ensemble of RF, HGB, and ET models.[/dim]")

if __name__ == "__main__":
    main()
