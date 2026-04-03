import pandas as pd
import numpy as np

# Load data
matches = pd.read_csv("data/all_ipl_matches_data.csv")
teams   = pd.read_csv("data/all_teams_data.csv")

# Map team IDs to names
team_map = dict(zip(teams["team_id"], teams["team_name"]))
matches["team1_name"]  = matches["team1"].map(team_map)
matches["team2_name"]  = matches["team2"].map(team_map)
matches["winner_name"] = matches["match_winner"].map(team_map)
matches["toss_name"]   = matches["toss_winner"].map(team_map)

# Keep only completed matches
matches = matches[matches["result"] == "win"].copy()

# Current active teams only
active_teams = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
    "Delhi Capitals", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
]

features = []

for team in active_teams:
    team_matches = matches[(matches["team1_name"] == team) | (matches["team2_name"] == team)]
    
    # Overall win rate
    wins = (team_matches["winner_name"] == team).sum()
    total = len(team_matches)
    win_rate = wins / total if total > 0 else 0

    # Recent form — last 3 seasons
    recent = team_matches[team_matches["season"].isin(["2023", "2024", "2025"])]
    recent_wins = (recent["winner_name"] == team).sum()
    recent_total = len(recent)
    recent_win_rate = recent_wins / recent_total if recent_total > 0 else 0

    # Toss win rate
    toss_wins = (team_matches["toss_name"] == team).sum()
    toss_win_rate = toss_wins / total if total > 0 else 0

    # Toss + match win rate (won toss AND won match)
    toss_match_wins = team_matches[
        (team_matches["toss_name"] == team) &
        (team_matches["winner_name"] == team)
    ]
    toss_match_rate = len(toss_match_wins) / total if total > 0 else 0

    # Finals appearances & wins
    finals_played = matches[
        ((matches["team1_name"] == team) | (matches["team2_name"] == team)) &
        (matches["match_number"] == "Final")
    ]
    finals_won = (finals_played["winner_name"] == team).sum()

    features.append({
        "team": team,
        "total_matches": total,
        "total_wins": wins,
        "win_rate": round(win_rate, 3),
        "recent_win_rate": round(recent_win_rate, 3),
        "toss_win_rate": round(toss_win_rate, 3),
        "toss_match_win_rate": round(toss_match_rate, 3),
        "finals_won": finals_won,
    })

df = pd.DataFrame(features)
df = df.sort_values("win_rate", ascending=False).reset_index(drop=True)

print("=== TEAM FEATURES ===")
print(df.to_string())

df.to_csv("data/team_features.csv", index=False)
print("\n✅ Saved to data/team_features.csv")
