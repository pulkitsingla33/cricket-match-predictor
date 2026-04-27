import pandas as pd
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

def load_latest_stats(team1, team2, venue):
    """
    Simulates fetching the latest pre-match stats for the two teams.
    In a real app, this would pull from a live database.
    """
    data_path = Path("/home/ayush/Documents/SEM_2/SES/cricket-match-predictor/features_match_level.csv")
    df = pd.read_csv(data_path)
    
    # Pre-calculate player strengths (simplified for TUI)
    # In practice, we'd use the same logic as train_model.py
    # Here we'll just mock it or pull representative values if possible
    # For now, let's grab the most recent match for each team to get their form
    
    def get_team_stats(team):
        recent = df[(df['team1'] == team) | (df['team2'] == team)].sort_values('date', ascending=False).iloc[0]
        if recent['team1'] == team:
            return {
                'form': recent['team1_recent_form'],
                'venue_wr': recent['team1_venue_win_rate'], # Not perfect as venue might differ
                'bat_sr': 135.0, # Placeholder averages if not in main CSV
                'bowl_econ': 8.5
            }
        else:
            return {
                'form': recent['team2_recent_form'],
                'venue_wr': recent['team2_venue_win_rate'],
                'bat_sr': 135.0,
                'bowl_econ': 8.5
            }

    # Fetching H2H
    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) | 
             ((df['team1'] == team2) & (df['team2'] == team1))].sort_values('date', ascending=False)
    
    if not h2h.empty:
        latest_h2h = h2h.iloc[0]
        if latest_h2h['team1'] == team1:
            h2h_wr = latest_h2h['h2h_team1_win_rate']
        else:
            h2h_wr = 1 - latest_h2h['h2h_team1_win_rate']
    else:
        h2h_wr = 0.5
        
    t1_stats = get_team_stats(team1)
    t2_stats = get_team_stats(team2)
    
    # Construct feature row
    features = {
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'toss_winner_is_team1': 1, # Default assumption for TUI
        'toss_bat_first': 0,
        'h2h_team1_win_rate': h2h_wr,
        'team1_venue_win_rate': t1_stats['venue_wr'],
        'team2_venue_win_rate': t2_stats['venue_wr'],
        'team1_recent_form': t1_stats['form'],
        'team2_recent_form': t2_stats['form'],
        'team1_batting_sr': t1_stats['bat_sr'],
        'team2_batting_sr': t2_stats['bat_sr'],
        'team1_bowling_econ': t1_stats['bowl_econ'],
        'team2_bowling_econ': t2_stats['bowl_econ']
    }
    
    return pd.DataFrame([features])

def main():
    console.clear()
    rprint(Panel.fit("[bold green]🏏 IPL Match Outcome Predictor [white]Ensemble Edition[/white] [/bold green]", subtitle="v1.0"))
    
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
