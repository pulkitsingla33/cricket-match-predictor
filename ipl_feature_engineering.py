"""
IPL Feature Engineering
=======================
Reads matches.csv + deliveries.csv and produces:
    - features_match_level.csv  :   one row per match, ready for ML
    - features_batting.csv      :   per-batter per-match aggrgates
    - features_bowling.csv      :   per-bowler per-match aggrgates

Usage:
    python ipl_feature_engineering.py \
        --matches matches.csv \
        --deliveries deliveries.csv \
        --output_dir ./feature_output
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

#LOAD
def load_data(matches_path: str, deliveries_path: str):
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)


    #Normalize types
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    #Coerce numeric columns that may have read as object
    for col in ["innings1_runs", "innings1_wickets", "innings2_runs", "innings2_wickets", "win_by_runs", "win_by_wickets", "match_number"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    print(f"Loaded {len(matches)} matches and {len(deliveries)} deliveries")

    return matches, deliveries


#HELPER UTILITIES
def safe_div(a, b, fill=0.0):
    """Element-wise safe division, returns fill where b==0"""
    return np.where(b == 0, fill, a / b)


#MATCH-LEVEL FEATURE ENGINEERING
def build_match_features(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Produces one feature row per match combining:
        - toss features
        - target / run-rate features
        - powerplay / death-over features per innings
        - historical head-to-head and venue win-rates (look-ahead safe)
    """
    df = matches.copy().sort_values("date").reset_index(drop=True)

    #Toss
    df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["toss_bat_first"]       = (df["toss_decision"] == "bat").astype(int)

    #Did toss winner choose to bat?
    df["toss_winner_batted"] = (
        ((df["toss_decision"] == "bat") & (df["toss_winner"] == df["innings_1_team"])) |
        ((df["toss_decision"] == "field") & (df["toss_winner"] == df["innings_2_team"]))
    ).astype(int)

    #Required Run Rate
    df["run_rate_inn1"] = safe_div(df["innings1_runs"].values, (df["overs"].fillna(20)).values)
    df["run_rate_inn2"] = safe_div(df["innings2_runs"].values, (df["overs"].fillna(20)).values)
    df["sore_diff"]     = df["innings1_runs"] - df["innings2_runs"]

    #Target
    df["target"] = df["innings1_runs"] + 1

    # Winner binary
    # 1 = team1 won, 0 = team2 won, NaN = no result / tie
    def winner_bnary(row):
        if pd.isna(row["winner"]):
            return np.nan
        if row["winner"] == row["team1"]:
            return 1
        if row["winner"] == row["team2"]:
            return 0
        return np.nan
    
    df["team1_won"] = df.apply(winner_bnary, axis=1)

    # Win method
    df["win_method"] = np.where(df["win_by_runs"].notna(), "runs", np.where(df["win_by_wickets"].notna(), "wickets", "other"))

    # Powerplay features from deliveries
    pp = deliveries[deliveries["in_powerplay"] == 1].copy()
    pp_agg = (
        pp.groupby(["match_id", "innings"]).agg(pp_runs=("total_runs", "sum"),pp_wickets=("is_wicket", "sum")).reset_index())
    
    def pivot_pp(inn_num, suffix):
        sub = pp_agg[pp_agg["innings"] == inn_num].drop(columns=["innings"])
        sub = sub.rename(columns={"pp_runs": f"pp_runs_inn{suffix}", "pp_wickets": f"pp_wickets_inn{suffix}"})
        return sub
    
    df = df.merge(pivot_pp(1, "1"), how="left", on="match_id")
    df = df.merge(pivot_pp(2, "2"), how="left", on="match_id")

    # Death-overs  (overs 17-20)
    death = deliveries[deliveries["over"].isin([17, 18, 19, 20])].copy()
    death_agg = (
        death.groupby(["match_id", "innings"]).agg(death_runs=("total_runs", "sum"), death_wickets=("is_wicket", "sum")).reset_index())

    def pivot_death(inn_num, suffix):
        sub = death_agg[death_agg["innings"] == inn_num].drop(columns=["innings"])
        sub = sub.rename(columns={"death_runs": f"death_runs_inn{suffix}", "death_wickets": f"death_wickets_inn{suffix}"})
        return sub
    
    df = df.merge(pivot_death(1, "1"), how="left", on="match_id")
    df = df.merge(pivot_death(2, "2"), how="left", on="match_id")

    # Middle overs (overs 7-16)
    mid = deliveries[deliveries["over"].isin(range(7, 17))].copy()
    mid_agg = (
        mid.groupby(["match_id", "innings"]).agg(mid_runs=("total_runs", "sum"), mid_wickets=("is_wicket", "sum")).reset_index())

    def pivot_mid(inn_num, suffix):
        sub = mid_agg[mid_agg["innings"] == inn_num].drop(columns=["innings"])
        sub = sub.rename(columns={"mid_runs": f"mid_runs_inn{suffix}", "mid_wickets": f"mid_wickets_inn{suffix}"})
        return sub
    
    df = df.merge(pivot_mid(1, "1"), how="left", on="match_id")
    df = df.merge(pivot_mid(2, "2"), how="left", on="match_id")

    #Boundaries (4s and 6s)
    inn1_del = deliveries[deliveries["innings"] == 1].copy()
    inn2_del = deliveries[deliveries["innings"] == 2].copy()

    def boundary_agg(del_df, suffix):
        agg = del_df.groupby("match_id").agg(
            fours=("batter_runs", lambda x: (x == 4).sum()),
            sixes=("batter_runs", lambda x: (x == 6).sum()),
            dot_balls=("batter_runs", lambda x: (x == 0).sum()),
        ).reset_index()
        agg.columns = ["match_id"] + [f"{col}_inn{suffix}" for col in ["fours", "sixes", "dot_balls"]]
        return agg

    df = df.merge(boundary_agg(inn1_del, "1"), how="left", on="match_id")
    df = df.merge(boundary_agg(inn2_del, "2"), how="left", on="match_id")

    # Win rate (lookahead safe)
    # For each match, count past H2H results between the two teams
    h2h_records = []
    for idx, row in df.iterrows():
        past = df.iloc[:idx]
        h2h = past[
            ((past["team1"] == row["team1"]) & (past["team2"] == row["team2"]))
            | ((past["team1"] == row["team2"]) & (past["team2"] == row["team1"]))
        ]
        total = len(h2h)
        if total == 0:
            h2h_records.append({"match_id": row["match_id"], "h2h_matches":0, "h2h_team1_wins":0, "h2h_team1_win_rate":0.5})
            continue

        t1_wins = ((h2h["team1"] == row["team1"]) & (h2h["team1_won"] == 1)).sum() + \
                    ((h2h["team2"] == row["team1"]) & (h2h["team1_won"] == 0)).sum()

        h2h_records.append({"match_id": row["match_id"], "h2h_matches":total, "h2h_team1_wins":t1_wins, "h2h_team1_win_rate":safe_div(t1_wins, total)})

    h2h_df = pd.DataFrame(h2h_records)
    df = df.merge(h2h_df, how="left", on="match_id")

    # Venue win rate (lookahead safe)
    venue_records = []
    for idx, row in df.iterrows():
        past = df.iloc[:idx]
        venue_past = past[past["venue"] == row["venue"]]
        
        def team_venue_winrate(team):
            played = venue_past[(venue_past["team1"] == team) | (venue_past["team2"] == team)]
            if(len(played) == 0):
                return 0.5
            wins = played[(played["team1"] == team) & (played["team1_won"] == 1)].sum() + played[(played["team2"] == team) & (played["team1_won"] == 0)].sum()
            return wins / len(played)

        venue_records.append({
            "match_id": row["match_id"],
            "team1_venue_win_rate": team_venue_winrate(row["team1"]),
            "team2_venue_win_rate": team_venue_winrate(row["team2"])
        })

        venue_df = pd.DataFrame(venue_records)
        df = df.merge(venue_df, how="left", on="match_id")

        # Recent form: last 5 match win rate
        form_records = []
        for idx, row in df.iterrows():
            past = df.iloc[:idx]
            
            def recent_form(team, n=5):
                team_games = past[(past["team1"] == team) | (past["team2"] == team)]
                if len(team_games) == 0:
                    return 0.5
                wins = ((team_games["team1"] == team) & (team_games["team1_won"] == 1)).sum() + ((team_games["team2"] == team) & (team_games["team1_won"] == 0)).sum()
                return wins / len(team_games)

            form_records.append({
                "match_id": row["match_id"],
                "team1_recent_form": recent_form(row["team1"]),
                "team2_recent_form": recent_form(row["team2"])
            })

        form_df = pd.DataFrame(form_records)
        df = df.merge(form_df, how="left", on="match_id")

        df["season_year"] = df["season"].str.extract(r"(\d{4})").astype(float)

    return df

# BATTER FEATURES (per match)

def build_batting_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-batter stats for each match"""

    legal = deliveries[deliveries["wides"] == 0].copy()

    agg = legal.groupby(["match_id", "innings", "batting_team", "bowling_team"]).agg(
        runs=("batter_runs", "sum"),
        balls_faced=("batter_runs", "count"),
        fours=("batter_runs", lambda x: (x == 4).sum()),
        sixes=("batter_runs", lambda x: (x == 6).sum()),
        dot_balls=("batter_runs", lambda x: (x == 0).sum()),).reset_index()

    agg["strike_rate"] = safe_div((agg["runs"].values * 100), agg["balls_faced"].values)
    agg["boundary_pct"] = safe_div((agg["fours"] + agg["sixes"]).values * 100, agg["balls_faced"].values)

    # Dismissal flag
    dismissed = (deliveries[deliveries["is_wicket"] == 1].groupby(["match_id", "innings", "player_out"]).size().reset_index(name="_d"))
    dismissed = dismissed.rename(columns={"player_out": "batter"})
    dismissed["dismissed"] = 1

    agg = agg.merge(dismissed[["match_id", "innings", "batter", "dismissed"]], how="left", on=["match_id", "innings", "batter"])
    agg["dismissed"] = agg["dismissed"].fillna(0).astype(int)
    
    return agg

    
# BOWLER FEATURES (per match)

def build_bowling_features(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-bowler stats for each match"""
    agg = deliveries.groupby(["match_id", "innings", "bowling_team", "bowler"]).agg(
        runs_conceded = ("total_runs", "sum"),
        balls_bowled = ("total_runs", "count"),
        wickets = ("is_wicket", "sum"),
        wides = ("wides", "sum"),
        no_balls = ("noballs", "sum"),
        dot_balls = ("total_runs", lambda x: (x == 0).sum()),
        fours_conceded = ("batter_runs", lambda x: (x == 4).sum()),
        sixes_conceded = ("batter_runs", lambda x: (x == 6).sum()),
    ).reset_index()

    agg["overs_bowled"] = agg["balls_bowled"] / 6
    agg["economy"] = safe_div(agg["runs_conceded"].values * 6, agg["balls_bowled"].values)
    agg["bowling_avg"] = safe_div(agg["runs_conceded"].values, agg["wickets"].values, fill=np.nan)
    agg["dot_ball_pct"] = safe_div(agg["dot_balls"].values * 100, agg["balls_bowled"].values)

    return agg


# MAIN
def main():
    parser = argparse.ArgumentParser(description="IPL Feature Engineering")
    parser.add_argument("--matches", required=True, help="Path to matches.csv")
    parser.add_argument("--deliveries", required=True, help="Path to deliveries.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    matches, deliveries = load_data(args.matches, args.deliveries)

    print("\n[1/3] Building match-level features...")
    match_features = build_match_features(matches, deliveries)
    match_out = out / "features_match_level.csv"
    match_features.to_csv(match_out, index=False)

    print("\n[2/3] Building batting features...")
    batting_features = build_batting_features(deliveries)
    batting_out = out / "features_batting.csv"
    batting_features.to_csv(batting_out, index=False)

    print("\n[3/3] Building bowling features...")
    bowling_features = build_bowling_features(deliveries)
    bowling_out = out / "features_bowling.csv"
    bowling_features.to_csv(bowling_out, index=False)

    print("\n" + "─"*55)
    print("MATCH-LEVEL FEATURE SUMMARY")
    print("─"*55)
    feature_groups = {
        "Target / Result":     ["target","score_diff","run_rate_inn1","run_rate_inn2",
                                 "team1_won","win_method"],
        "Toss":                ["toss_winner_is_team1","toss_bat_first","toss_winner_batted"],
        "Powerplay":           ["pp_runs_inn1","pp_wkts_inn1","pp_runs_inn2","pp_wkts_inn2"],
        "Middle Overs":        ["mid_runs_inn1","mid_wkts_inn1","mid_runs_inn2","mid_wkts_inn2"],
        "Death Overs":         ["death_runs_inn1","death_wkts_inn1","death_runs_inn2","death_wkts_inn2"],
        "Boundaries":          ["fours_inn1","sixes_inn1","dot_balls_inn1",
                                 "fours_inn2","sixes_inn2","dot_balls_inn2"],
        "Head-to-Head":        ["h2h_matches","h2h_team1_wins","h2h_team1_win_rate"],
        "Venue":               ["team1_venue_win_rate","team2_venue_win_rate"],
        "Recent Form":         ["team1_recent_form","team2_recent_form"],
    }
    for group, cols in feature_groups.items():
        present = [c for c in cols if c in match_features.columns]
        print(f"  {group:<20s}: {', '.join(present)}")
 
    print("\nDone! 🎯")
 
 
if __name__ == "__main__":
    main()
    
