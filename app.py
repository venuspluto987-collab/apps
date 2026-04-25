import streamlit as st
import pandas as pd
import requests
import pickle

# -------------------------------
# Load ML Model
# -------------------------------
model = pickle.load(open("ipl_model.pkl", "rb"))

st.set_page_config(layout="wide")
st.title("🏏 IPL AI Dashboard (LIVE)")

# -------------------------------
# LIVE MATCH DATA (API)
# -------------------------------
API_KEY = "YOUR_API_KEY"

url = "https://cricket-live-data.p.rapidapi.com/matches"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "cricket-live-data.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    matches = response.json()["results"]
else:
    matches = []

# -------------------------------
# SHOW LIVE MATCHES
# -------------------------------
st.subheader("🔴 Live Matches")

for match in matches[:3]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {match['team1']} vs {match['team2']}")
        st.write(match['status'])

    with col2:
        st.metric("Score", match.get("score", "N/A"))

# -------------------------------
# AI PREDICTOR
# -------------------------------
st.subheader("🤖 AI Win Predictor")

col1, col2, col3, col4 = st.columns(4)

with col1:
    target = st.number_input("Target", 100, 300, 180)

with col2:
    score = st.number_input("Score", 0, 300, 100)

with col3:
    overs = st.slider("Overs", 0, 20, 10)

with col4:
    wickets = st.slider("Wickets", 0, 10, 7)

runs_left = max(target - score, 0)
balls_left = max(120 - overs * 6, 0)

input_df = pd.DataFrame(
    [[runs_left, balls_left, wickets]],
    columns=["runs_left", "balls_left", "wickets"]
)

prob = model.predict_proba(input_df)[0][1]

st.success(f"🔥 Winning Chance: {round(prob*100,2)}%")
st.progress(int(prob * 100))

# -------------------------------
# VISUAL DASHBOARD
# -------------------------------
st.subheader("📊 Match Analysis")

chart_data = pd.DataFrame({
    "Metric": ["Runs Left", "Balls Left", "Wickets"],
    "Value": [runs_left, balls_left, wickets]
})

st.bar_chart(chart_data.set_index("Metric"))
