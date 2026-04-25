import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------
# 🎯 Title
# -------------------------------
st.title("🏏 IPL Win Predictor")
st.caption("AI predicts match winner in real-time 🔥")

# -------------------------------
# 📊 Sample Training Data
# -------------------------------
data = {
    "runs_left": [100, 80, 60, 40, 20, 10],
    "balls_left": [60, 50, 40, 30, 20, 10],
    "wickets": [10, 9, 8, 7, 6, 5],
    "win": [0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["runs_left", "balls_left", "wickets"]]
y = df["win"]

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# 🎛 User Input
# -------------------------------
st.sidebar.header("Match Situation")

target = st.sidebar.number_input("Target Score", 100, 300, 180)
current_score = st.sidebar.number_input("Current Score", 0, 300, 100)
overs = st.sidebar.slider("Overs Completed", 0, 20, 10)
wickets = st.sidebar.slider("Wickets Remaining", 0, 10, 7)

# -------------------------------
# 🧮 Calculations
# -------------------------------
runs_left = target - current_score
balls_left = 120 - (overs * 6)

input_data = pd.DataFrame([[runs_left, balls_left, wickets]],
                          columns=["runs_left", "balls_left", "wickets"])

# -------------------------------
# 🔮 Prediction
# -------------------------------
win_prob = model.predict_proba(input_data)[0][1]

# -------------------------------
# 📢 Output
# -------------------------------
st.subheader("📊 Win Probability")

st.success(f"Winning Chance: {round(win_prob * 100, 2)} %")

# -------------------------------
# 📈 Dashboard Chart
# -------------------------------
chart_data = pd.DataFrame({
    "Scenario": ["Current"],
    "Win Probability": [win_prob * 100]
})

st.bar_chart(chart_data.set_index("Scenario"))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.write("Built with Machine Learning 🚀")
