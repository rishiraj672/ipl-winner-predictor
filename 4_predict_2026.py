import pandas as pd
import numpy as np
import pickle
import itertools

# Load model
with open("data/model.pkl", "rb") as f:
    saved = pickle.load(f)
model    = saved["model"]
feat_cols = saved["features"]

# Load features
features = pd.read_csv("data/team_features.csv")
feat_map  = features.set_index("team").to_dict("index")

active_teams = features["team"].tolist()

def predict_match(team1, team2, toss_winner):
    f1 = feat_map[team1]
    f2 = feat_map[team2]
    toss_advantage = 1 if toss_winner == team1 else 0
    row = {
        "win_rate_diff":      f1["win_rate"] - f2["win_rate"],
        "recent_form_diff":   f1["recent_win_rate"] - f2["recent_win_rate"],
        "toss_win_rate_diff": f1["toss_win_rate"] - f2["toss_win_rate"],
        "finals_won_diff":    f1["finals_won"] - f2["finals_won"],
        "toss_advantage":     toss_advantage,
        "team1_win_rate":     f1["win_rate"],
        "team2_win_rate":     f2["win_rate"],
        "team1_recent":       f1["recent_win_rate"],
        "team2_recent":       f2["recent_win_rate"],
    }
    X = pd.DataFrame([row])[feat_cols]
    prob = model.predict_proba(X)[0][1]
    return prob

# ── Simulate full tournament ─────────────────────────────────
print("=" * 55)
print("   🏏  IPL 2026 WINNER PREDICTION")
print("=" * 55)

# Each team plays every other team twice (home & away)
# Score = sum of win probabilities across all matchups
scores = {team: 0.0 for team in active_teams}

for team1, team2 in itertools.permutations(active_teams, 2):
    # Assume toss is 50/50 — simulate both scenarios
    prob_t1_toss = predict_match(team1, team2, team1)
    prob_t2_toss = predict_match(team1, team2, team2)
    avg_prob = (prob_t1_toss + prob_t2_toss) / 2
    scores[team1] += avg_prob
    scores[team2] += (1 - avg_prob)

# Normalize to percentage
total = sum(scores.values())
win_prob = {team: (s / total) * 100 for team, s in scores.items()}
win_prob = dict(sorted(win_prob.items(), key=lambda x: x[1], reverse=True))

# Print results
print("\n🏆 PREDICTED WIN PROBABILITIES FOR IPL 2026\n")
medals = ["🥇", "🥈", "🥉"]
for i, (team, prob) in enumerate(win_prob.items()):
    medal = medals[i] if i < 3 else "  "
    bar   = "█" * int(prob * 2)
    print(f"{medal} {team:<35} {prob:.1f}%  {bar}")

winner = list(win_prob.keys())[0]
runner = list(win_prob.keys())[1]

print("\n" + "=" * 55)
print(f"🏆  PREDICTED IPL 2026 WINNER  : {winner}")
print(f"🥈  PREDICTED RUNNER UP        : {runner}")
print("=" * 55)

# ── Head to Head: Top 4 teams ────────────────────────────────
top4 = list(win_prob.keys())[:4]
print("\n📊 HEAD-TO-HEAD: TOP 4 TEAMS\n")
print(f"{'Matchup':<45} {'Win Prob'}")
print("-" * 55)
for team1, team2 in itertools.combinations(top4, 2):
    prob = predict_match(team1, team2, team1) * 100
    print(f"{team1} vs {team2:<25} {prob:.1f}% — {team1}")
