import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools

st.set_page_config(page_title="IPL 2026 Predictor", page_icon="🏏", layout="wide")

@st.cache_resource
def load_model():
    with open("data/model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    return pd.read_csv("data/team_features.csv")

saved     = load_model()
model     = saved["model"]
feat_cols = saved["features"]
features  = load_features()
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
    return round(prob * 100, 1)

@st.cache_data
def simulate_tournament():
    scores = {team: 0.0 for team in active_teams}
    for team1, team2 in itertools.permutations(active_teams, 2):
        p1 = predict_match(team1, team2, team1) / 100
        p2 = predict_match(team1, team2, team2) / 100
        avg = (p1 + p2) / 2
        scores[team1] += avg
        scores[team2] += (1 - avg)
    total = sum(scores.values())
    result = {t: round((s / total) * 100, 1) for t, s in scores.items()}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

st.markdown("""
<style>
.big-title {
    font-size: 3rem; font-weight: 900; text-align: center;
    background: linear-gradient(90deg, #f78166, #e3b341, #39d353);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.team-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 12px; padding: 1rem; margin: 0.4rem 0;
}
.stButton > button {
    background: linear-gradient(90deg, #f78166, #e3b341);
    color: black; font-weight: 800; font-size: 1.1rem;
    border: none; border-radius: 10px; width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🏏 IPL 2026 Winner Predictor</div>', unsafe_allow_html=True)
st.caption("Machine Learning predictions based on IPL data 2008–2025")
st.divider()

tab1, tab2, tab3 = st.tabs(["🏆 Tournament Prediction", "⚔️ Head to Head", "📊 Team Stats"])

with tab1:
    st.subheader("🏆 IPL 2026 Full Tournament Simulation")
    with st.spinner("Simulating..."):
        win_probs = simulate_tournament()
    teams_list = list(win_probs.keys())
    probs_list = list(win_probs.values())

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🥇 **{teams_list[0]}** — {probs_list[0]}%\n\nPredicted IPL 2026 Champion")
    with col2:
        st.warning(f"🥈 **{teams_list[1]}** — {probs_list[1]}%\n\nPredicted Runner Up")

    st.markdown("### All Teams Ranking")
    medals = ["🥇","🥈","🥉"] + ["  "]*10
    for i, (team, prob) in enumerate(win_probs.items()):
        bar = "█" * int(prob * 8)
        st.markdown(f"""
        <div class="team-card">
            {medals[i]} <b style="color:#c9d1d9">{team}</b>
            <span style="float:right;color:#e3b341;font-weight:700">{prob}%</span>
            <div style="color:#39d353;font-size:0.8rem;margin-top:4px">{bar}</div>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.subheader("⚔️ Head to Head Predictor")
    col1, col2, col3 = st.columns(3)
    with col1:
        team1 = st.selectbox("🏏 Team 1", active_teams, index=0)
    with col2:
        team2 = st.selectbox("🏏 Team 2", [t for t in active_teams if t != team1], index=1)
    with col3:
        toss = st.selectbox("🪙 Toss Winner", [team1, team2])

    if st.button("⚡ Predict Winner"):
        prob1 = predict_match(team1, team2, toss)
        prob2 = round(100 - prob1, 1)
        winner = team1 if prob1 > prob2 else team2
        loser  = team2 if winner == team1 else team1
        st.markdown("---")
        col1, col2, col3 = st.columns([2,1,2])
        with col1:
            if winner == team1:
                st.success(f"### 🏆 {team1}\n**{prob1}%**")
            else:
                st.error(f"### {team1}\n**{prob1}%**")
        with col2:
            st.markdown("<h2 style='text-align:center;color:#e3b341'>VS</h2>", unsafe_allow_html=True)
        with col3:
            if winner == team2:
                st.success(f"### 🏆 {team2}\n**{prob2}%**")
            else:
                st.error(f"### {team2}\n**{prob2}%**")
        st.info(f"🏆 **{winner}** wins with **{max(prob1,prob2)}% probability**")

with tab3:
    st.subheader("📊 Team Statistics (2008–2025)")
    df_display = features.copy()
    df_display["win_rate"]
