import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="IPL AI Dashboard", layout="wide")
st.title("🏏 IPL Win Predictor (AI + ML)")

# -------------------------------
# TRAIN MODEL (NO PKL)
# -------------------------------
X = [
    [50, 30, 5],
    [20, 10, 2],
    [80, 50, 7],
    [10, 5, 1],
    [60, 40, 6],
    [30, 20, 4]
]

y = [1, 0, 1, 0, 1, 0]

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📊 Match Input")

col1, col2, col3, col4 = st.columns(4)

with col1:
    target = st.number_input("Target Score", 100, 300, 180)

with col2:
    score = st.number_input("Current Score", 0, 300, 100)

with col3:
    overs = st.slider("Overs Completed", 0, 20, 10)

with col4:
    wickets = st.slider("Wickets Remaining", 0, 10, 7)

# -------------------------------
# CALCULATIONS
# -------------------------------
runs_left = max(target - score, 0)
balls_left = max(120 - (overs * 6), 1)

required_rr = (runs_left / balls_left) * 6

# Prepare ML input
input_data = pd.DataFrame(
    [[runs_left, balls_left, wickets]],
    columns=["runs_left", "balls_left", "wickets"]
)

# -------------------------------
# PREDICTION (HYBRID LOGIC)
# -------------------------------
ml_prob = model.predict_proba(input_data)[0][1]

if required_rr > 12:
    final_prob = ml_prob * 0.5
elif required_rr > 9:
    final_prob = ml_prob * 0.7
elif wickets <= 2:
    final_prob = ml_prob * 0.6
else:
    final_prob = ml_prob

final_prob = max(0, min(final_prob, 1))

# -------------------------------
# OUTPUT
# -------------------------------
st.subheader("🤖 Prediction Result")

st.success(f"🔥 Winning Probability: {round(final_prob * 100, 2)}%")
st.progress(int(final_prob * 100))

# Extra metrics
colA, colB, colC = st.columns(3)

with colA:
    st.metric("Runs Left", runs_left)

with colB:
    st.metric("Balls Left", balls_left)

with colC:
    st.metric("Required Run Rate", round(required_rr, 2))

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("📈 Match Analysis")

chart_data = pd.DataFrame({
    "Metric": ["Runs Left", "Balls Left", "Wickets"],
    "Value": [runs_left, balls_left, wickets]
})

st.bar_chart(chart_data.set_index("Metric"))
