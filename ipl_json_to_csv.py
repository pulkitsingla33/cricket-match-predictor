"""
IPL JSON → CSV Converter
========================
Converts Cricsheet-format JSON match files into two CSVs:
  - matches.csv    : one row per match (match-level metadata)
  - deliveries.csv : one row per ball  (ball-by-ball data, linked by match_id)

Usage:
    python ipl_json_to_csv.py --input_dir ./json_files --output_dir ./csv_output

    # Or process a single file:
    python ipl_json_to_csv.py --input_dir ./json_files --output_dir ./csv_output --single 335982.json
"""

import json
import csv
import os
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# MATCH-LEVEL EXTRACTION
# ─────────────────────────────────────────────

def extract_match_row(match_id: str, data: dict) -> dict:
    """Flatten top-level match info into a single dict (one row in matches.csv)."""
    info = data.get("info", {})
    meta = data.get("meta", {})

    # Basic info
    dates     = info.get("dates", [])
    teams     = info.get("teams", [])
    toss      = info.get("toss", {})
    outcome   = info.get("outcome", {})
    event     = info.get("event", {})
    officials = info.get("officials", {})

    # Outcome fields – handle win by runs OR wickets OR tie/no result
    outcome_by     = outcome.get("by", {})
    win_by_runs    = outcome_by.get("runs", None)
    win_by_wickets = outcome_by.get("wickets", None)
    winner         = outcome.get("winner", outcome.get("result", None))  # "tie" / "no result" stored in result

    # Player of match (can be a list)
    pom_list = info.get("player_of_match", [])
    player_of_match = ", ".join(pom_list) if pom_list else None

    # Umpires
    umpires = officials.get("umpires", [])

    # Build innings summary (score, wickets) for each innings
    innings_data = data.get("innings", [])
    inn_scores = {}
    for idx, inn in enumerate(innings_data, start=1):
        runs = 0
        wkts = 0
        for over in inn.get("overs", []):
            for d in over.get("deliveries", []):
                runs += d.get("runs", {}).get("total", 0)
                wkts += len(d.get("wickets", []))
        inn_scores[idx] = {"runs": runs, "wickets": wkts, "team": inn.get("team")}

    row = {
        "match_id":            match_id,
        "season":              info.get("season"),
        "date":                dates[0] if dates else None,
        "venue":               info.get("venue"),
        "city":                info.get("city"),
        "team1":               teams[0] if len(teams) > 0 else None,
        "team2":               teams[1] if len(teams) > 1 else None,
        "toss_winner":         toss.get("winner"),
        "toss_decision":       toss.get("decision"),
        "innings1_team":       inn_scores.get(1, {}).get("team"),
        "innings1_runs":       inn_scores.get(1, {}).get("runs"),
        "innings1_wickets":    inn_scores.get(1, {}).get("wickets"),
        "innings2_team":       inn_scores.get(2, {}).get("team"),
        "innings2_runs":       inn_scores.get(2, {}).get("runs"),
        "innings2_wickets":    inn_scores.get(2, {}).get("wickets"),
        "winner":              winner,
        "win_by_runs":         win_by_runs,
        "win_by_wickets":      win_by_wickets,
        "player_of_match":     player_of_match,
        "umpire1":             umpires[0] if len(umpires) > 0 else None,
        "umpire2":             umpires[1] if len(umpires) > 1 else None,
        "match_number":        event.get("match_number"),
        "event_name":          event.get("name"),
        "match_type":          info.get("match_type"),
        "overs":               info.get("overs"),
        "balls_per_over":      info.get("balls_per_over"),
        "data_version":        meta.get("data_version"),
    }
    return row


# ─────────────────────────────────────────────
# DELIVERY-LEVEL EXTRACTION
# ─────────────────────────────────────────────

def extract_delivery_rows(match_id: str, data: dict) -> list[dict]:
    """
    Flatten every ball in every innings into one row each.
    Returns a list of dicts (rows for deliveries.csv).
    """
    rows = []
    innings_data = data.get("innings", [])

    for inn_idx, inn in enumerate(innings_data, start=1):
        batting_team  = inn.get("team")
        info_teams    = data.get("info", {}).get("teams", [])
        bowling_team  = next((t for t in info_teams if t != batting_team), None)

        # Powerplay overs lookup
        powerplays = inn.get("powerplays", [])

        # Track running totals across the innings
        cumulative_runs     = 0
        cumulative_wickets  = 0
        ball_number_in_inn  = 0  # counts legal + extras

        for over_obj in inn.get("overs", []):
            over_number = over_obj.get("over")  # 0-indexed

            for del_idx, delivery in enumerate(over_obj.get("deliveries", []), start=1):
                runs_obj   = delivery.get("runs", {})
                extras_obj = delivery.get("extras", {})
                wickets    = delivery.get("wickets", [])

                batter_runs  = runs_obj.get("batter", 0)
                extras_runs  = runs_obj.get("extras", 0)
                total_runs   = runs_obj.get("total", 0)

                # Extra types
                wides    = extras_obj.get("wides", 0)
                noballs  = extras_obj.get("noballs", 0)
                legbyes  = extras_obj.get("legbyes", 0)
                byes     = extras_obj.get("byes", 0)
                penalty  = extras_obj.get("penalty", 0)

                # Is this delivery a legal ball (not wide / no ball)?
                is_legal = (wides == 0 and noballs == 0)
                if is_legal:
                    ball_number_in_inn += 1

                cumulative_runs    += total_runs
                cumulative_wickets += len(wickets)

                # Wicket details (can be multiple, e.g. obstructing + run-out)
                wicket_kind     = None
                player_out      = None
                fielder         = None
                is_wicket       = 0
                if wickets:
                    is_wicket   = 1
                    wkt         = wickets[0]
                    wicket_kind = wkt.get("kind")
                    player_out  = wkt.get("player_out")
                    fielders    = wkt.get("fielders", [])
                    if fielders:
                        fielder = fielders[0].get("name")

                # Powerplay flag
                in_powerplay = int(any(
                    pp["from"] <= (over_number + del_idx / 10) <= pp["to"]
                    for pp in powerplays
                ))

                row = {
                    "match_id":            match_id,
                    "innings":             inn_idx,
                    "batting_team":        batting_team,
                    "bowling_team":        bowling_team,
                    "over":                over_number + 1,          # 1-indexed for readability
                    "ball":                del_idx,                  # delivery index in this over
                    "batter":              delivery.get("batter"),
                    "non_striker":         delivery.get("non_striker"),
                    "bowler":              delivery.get("bowler"),
                    "batter_runs":         batter_runs,
                    "extras":              extras_runs,
                    "total_runs":          total_runs,
                    "wides":               wides,
                    "noballs":             noballs,
                    "legbyes":             legbyes,
                    "byes":                byes,
                    "penalty":             penalty,
                    "is_wicket":           is_wicket,
                    "wicket_kind":         wicket_kind,
                    "player_out":          player_out,
                    "fielder":             fielder,
                    "cumulative_runs":     cumulative_runs,
                    "cumulative_wickets":  cumulative_wickets,
                    "in_powerplay":        in_powerplay,
                }
                rows.append(row)

    return rows


# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────

MATCH_FIELDS = [
    "match_id", "season", "date", "venue", "city",
    "team1", "team2",
    "toss_winner", "toss_decision",
    "innings1_team", "innings1_runs", "innings1_wickets",
    "innings2_team", "innings2_runs", "innings2_wickets",
    "winner", "win_by_runs", "win_by_wickets",
    "player_of_match", "umpire1", "umpire2",
    "match_number", "event_name", "match_type", "overs", "balls_per_over",
    "data_version",
]

DELIVERY_FIELDS = [
    "match_id", "innings", "batting_team", "bowling_team",
    "over", "ball", "batter", "non_striker", "bowler",
    "batter_runs", "extras", "total_runs",
    "wides", "noballs", "legbyes", "byes", "penalty",
    "is_wicket", "wicket_kind", "player_out", "fielder",
    "cumulative_runs", "cumulative_wickets", "in_powerplay",
]


def process_files(input_dir: str, output_dir: str, single: str):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    matches_csv_path    = output_path / "matches.csv"
    deliveries_csv_path = output_path / "deliveries.csv"

    json_files = [input_path / single] if single else sorted(input_path.glob("*.json"))

    match_count    = 0
    delivery_count = 0
    errors         = []

    with open(matches_csv_path, "w", newline="", encoding="utf-8") as mf, \
         open(deliveries_csv_path, "w", newline="", encoding="utf-8") as df:

        match_writer    = csv.DictWriter(mf, fieldnames=MATCH_FIELDS)
        delivery_writer = csv.DictWriter(df, fieldnames=DELIVERY_FIELDS)
        match_writer.writeheader()
        delivery_writer.writeheader()

        for jfile in json_files:
            try:
                with open(jfile, "r", encoding="utf-8") as f:
                    data = json.load(f)

                match_id = jfile.stem  # filename without extension = match_id

                match_row = extract_match_row(match_id, data)
                match_writer.writerow(match_row)
                match_count += 1

                delivery_rows = extract_delivery_rows(match_id, data)
                delivery_writer.writerows(delivery_rows)
                delivery_count += len(delivery_rows)

                print(f"  ✓ {jfile.name:30s} → {len(delivery_rows):>4d} deliveries")

            except Exception as e:
                errors.append((jfile.name, str(e)))
                print(f"  ✗ {jfile.name}: {e}")

    print(f"\n{'─'*50}")
    print(f"Done. Processed {match_count} match(es), {delivery_count} deliveries.")
    print(f"Output files:")
    print(f"  {matches_csv_path}")
    print(f"  {deliveries_csv_path}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for fname, err in errors:
            print(f"  {fname}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IPL Cricsheet JSON files to CSV.")
    parser.add_argument("--input_dir",  required=True, help="Directory containing .json match files")
    parser.add_argument("--output_dir", required=True, help="Directory to write matches.csv and deliveries.csv")
    parser.add_argument("--single",     default=None,  help="Process a single file (filename only, e.g. 335982.json)")
    args = parser.parse_args()

    process_files(args.input_dir, args.output_dir, args.single)