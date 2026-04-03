import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Load data
matches  = pd.read_csv("data/all_ipl_matches_data.csv")
teams    = pd.read_csv("data/all_teams_data.csv")
features = pd.read_csv("data/team_features.csv")

# Map team IDs to names
team_map = dict(zip(teams["team_id"], teams["team_name"]))
matches["team1_name"]  = matches["team1"].map(team_map)
matches["team2_name"]  = matches["team2"].map(team_map)
matches["winner_name"] = matches["match_winner"].map(team_map)
matches["toss_name"]   = matches["toss_winner"].map(team_map)
matches = matches[matches["result"] == "win"].copy()

# Active teams only
active_teams = features["team"].tolist()
matches = matches[
    matches["team1_name"].isin(active_teams) &
    matches["team2_name"].isin(active_teams)
].copy()

# Build feature map
feat_map = features.set_index("team").to_dict("index")

def get_features(row):
    t1 = row["team1_name"]
    t2 = row["team2_name"]
    if t1 not in feat_map or t2 not in feat_map:
        return None
    f1 = feat_map[t1]
    f2 = feat_map[t2]
    toss_advantage = 1 if row["toss_name"] == t1 else 0
    return {
        "win_rate_diff":        f1["win_rate"] - f2["win_rate"],
        "recent_form_diff":     f1["recent_win_rate"] - f2["recent_win_rate"],
        "toss_win_rate_diff":   f1["toss_win_rate"] - f2["toss_win_rate"],
        "finals_won_diff":      f1["finals_won"] - f2["finals_won"],
        "toss_advantage":       toss_advantage,
        "team1_win_rate":       f1["win_rate"],
        "team2_win_rate":       f2["win_rate"],
        "team1_recent":         f1["recent_win_rate"],
        "team2_recent":         f2["recent_win_rate"],
        "label": 1 if row["winner_name"] == t1 else 0
    }

rows = [get_features(row) for _, row in matches.iterrows()]
rows = [r for r in rows if r is not None]
df   = pd.DataFrame(rows)

X = df.drop("label", axis=1)
y = df["label"]

print(f"Training samples: {len(df)}")
print(f"Class balance — team1 wins: {y.mean():.2%}\n")

# Train & compare models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

best_score = 0
best_model_name = ""
best_model = None

print("=== MODEL COMPARISON ===")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name:25s} → Accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_model_name = name
        best_model = model

print(f"\n✅ Best Model: {best_model_name} ({best_score:.2%})")

# Train best model on full data
best_model.fit(X, y)

# Save model info
import pickle
with open("data/model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": list(X.columns)}, f)

print("✅ Model saved to data/model.pkl")
