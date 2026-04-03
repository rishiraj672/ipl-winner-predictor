import pandas as pd

matches = pd.read_csv("data/all_ipl_matches_data.csv")
teams   = pd.read_csv("data/all_teams_data.csv")
finals  = pd.read_csv("data/IPL_finals.csv")

team_map = dict(zip(teams["team_id"], teams["team_name"]))

matches["team1_name"]  = matches["team1"].map(team_map)
matches["team2_name"]  = matches["team2"].map(team_map)
matches["winner_name"] = matches["match_winner"].map(team_map)
matches["toss_name"]   = matches["toss_winner"].map(team_map)

matches = matches[matches["result"] == "win"].copy()

print("=== SHAPE ===")
print(matches.shape)

print("\n=== SEASONS ===")
print(sorted(matches["season"].unique()))

print("\n=== TOTAL WINS PER TEAM ===")
print(matches["winner_name"].value_counts())

print("\n=== FINALS ===")
final_matches = matches[matches["match_id"].isin(finals["id"])]
print(final_matches[["season", "team1_name", "team2_name", "winner_name"]])
