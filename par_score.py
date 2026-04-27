"""
IPL Par Score Calculator
========================
Computes the "par score" at each IPL venue — the first-innings total at which
the batting-first team has a ~50 % win rate historically.

Approach:
  - Group completed matches by venue
  - For each venue with enough data (≥10 matches), fit a logistic curve
    to (innings1_runs → batting_first_won) and find the 50 % crossover
  - Fall back to the global par score for venues with sparse data

Usage:
    from par_score import get_par_score, compute_all_par_scores
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_all_par_scores(min_matches: int = 10) -> dict:
    """
    Returns a dict: venue_name → par_score (int).
    Also returns the global par score used as fallback.
    """
    data_path = Path(__file__).parent
    df = pd.read_csv(data_path / "features_match_level.csv")
    df = df.dropna(subset=['innings1_runs', 'winner'])

    # Did the batting-first team win?
    df['batting_first_won'] = (df['innings1_team'] == df['winner']).astype(int)

    # ── Global par score ───────────────────────────────
    global_par = _find_par_score(df)

    # ── Per-venue par scores ───────────────────────────
    venue_pars = {}
    for venue, vdf in df.groupby('venue'):
        if len(vdf) >= min_matches:
            par = _find_par_score(vdf)
            venue_pars[venue] = par
        else:
            venue_pars[venue] = global_par  # fallback

    return venue_pars, global_par


def _find_par_score(df: pd.DataFrame) -> int:
    """
    Find the first-innings score at which the batting-first team wins ~50 % 
    of the time, using a sliding-window approach over sorted scores.

    Uses a rolling window of matches sorted by innings1_runs:
    par score = score where cumulative bat-first win rate crosses 50 %.
    """
    sorted_df = df.sort_values('innings1_runs').reset_index(drop=True)
    scores = sorted_df['innings1_runs'].values
    wins = sorted_df['batting_first_won'].values

    n = len(scores)
    if n == 0:
        return 167  # sensible IPL default

    # Use a sliding approach: for each score threshold, compute
    # win rate for batting first when scoring >= threshold
    # We want: the score at which scoring >= that score gives ~50% wins
    # 
    # But more intuitive: the score at which cumulative win rate = 50%
    # As we go from low to high scores, win rate should increase.
    # Actually, let's compute: at each score bucket, what's the win rate?

    # Bucket approach: group by 10-run buckets
    sorted_df['score_bucket'] = (sorted_df['innings1_runs'] // 10) * 10
    bucket_stats = sorted_df.groupby('score_bucket').agg(
        matches=('batting_first_won', 'count'),
        wins=('batting_first_won', 'sum'),
    ).reset_index()
    bucket_stats['win_rate'] = bucket_stats['wins'] / bucket_stats['matches']

    # Cumulative from the bottom: what's the win rate if you score AT LEAST X?
    # We want the score X where: P(batting first wins | score >= X) ≈ 50%
    # Actually, the simpler interpretation: the median winning score
    # 
    # More robust: find where interpolated win rate crosses 0.5
    # using cumulative "at least" approach
    buckets = bucket_stats['score_bucket'].values
    
    # For each bucket, compute: win rate when scoring in [bucket, bucket+10)
    # Then interpolate to find 50% crossover
    win_rates = bucket_stats['win_rate'].values
    
    # Find the crossover point
    for i in range(len(win_rates) - 1):
        if win_rates[i] <= 0.5 <= win_rates[i + 1]:
            # Linear interpolation
            frac = (0.5 - win_rates[i]) / (win_rates[i + 1] - win_rates[i]) if win_rates[i + 1] != win_rates[i] else 0.5
            par = buckets[i] + frac * 10
            return int(round(par))

    # If no clean crossover found, use the weighted median approach
    # Median score where batting first wins
    winning_scores = sorted_df[sorted_df['batting_first_won'] == 1]['innings1_runs']
    if len(winning_scores) > 0:
        return int(round(winning_scores.median()))

    return 167  # ultimate fallback


def get_par_score(venue: str) -> tuple:
    """
    Get the par score for a specific venue.
    
    Returns:
        (par_score: int, is_venue_specific: bool, venue_match_count: int)
    """
    data_path = Path(__file__).parent
    df = pd.read_csv(data_path / "features_match_level.csv")
    df = df.dropna(subset=['innings1_runs', 'winner'])
    df['batting_first_won'] = (df['innings1_team'] == df['winner']).astype(int)

    venue_df = df[df['venue'] == venue]
    venue_matches = len(venue_df)

    if venue_matches >= 10:
        par = _find_par_score(venue_df)
        return par, True, venue_matches
    else:
        global_par = _find_par_score(df)
        return global_par, False, venue_matches


def get_venue_batting_stats(venue: str) -> dict:
    """
    Get comprehensive batting-first stats for a venue.
    Returns dict with par score, avg score, bat-first win %, etc.
    """
    data_path = Path(__file__).parent
    df = pd.read_csv(data_path / "features_match_level.csv")
    df = df.dropna(subset=['innings1_runs', 'winner'])
    df['batting_first_won'] = (df['innings1_team'] == df['winner']).astype(int)

    venue_df = df[df['venue'] == venue]
    all_df = df  # for global fallback

    venue_matches = len(venue_df)
    is_venue_specific = venue_matches >= 10

    source_df = venue_df if is_venue_specific else all_df

    par, _, _ = get_par_score(venue)
    avg_first_inn = source_df['innings1_runs'].mean()
    avg_second_inn = source_df['innings2_runs'].mean() if 'innings2_runs' in source_df.columns else None
    bat_first_win_pct = source_df['batting_first_won'].mean() * 100

    return {
        'par_score': par,
        'is_venue_specific': is_venue_specific,
        'venue_matches': venue_matches,
        'avg_first_innings': round(avg_first_inn, 1),
        'avg_second_innings': round(avg_second_inn, 1) if avg_second_inn else None,
        'bat_first_win_pct': round(bat_first_win_pct, 1),
    }


def print_all_par_scores():
    """Utility: Print par scores for all venues (for verification)."""
    venue_pars, global_par = compute_all_par_scores()
    
    print("-" * 70)
    print(f"{'VENUE':<50s} {'PAR':>6s} {'NOTE':>10s}")
    print("-" * 70)

    data_path = Path(__file__).parent
    df = pd.read_csv(data_path / "features_match_level.csv")
    venue_counts = df['venue'].value_counts()

    for venue, par in sorted(venue_pars.items(), key=lambda x: -venue_counts.get(x[0], 0)):
        count = venue_counts.get(venue, 0)
        note = "" if count >= 10 else "(global)"
        print(f"  {venue:<48s} {par:>4d}   {note:>8s}  ({count} matches)")

    print(f"\n  Global par score: {global_par}")
    print("-" * 70)


if __name__ == "__main__":
    print_all_par_scores()
