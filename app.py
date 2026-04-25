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
