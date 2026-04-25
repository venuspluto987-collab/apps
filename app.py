import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="IPL AI Dashboard", layout="wide")
st.title("🏏 IPL Win Predictor (Real ML 2008–2025)")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")
    return matches, deliveries

matches, deliveries = load_data()

# -------------------------------
# TRAIN MODEL (REAL DATA)
# -------------------------------
@st.cache_resource
def train_model():
    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    # Feature engineering
    df['runs_left'] = df['total_runs_y'] - df['total_runs_x']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
    df['wickets'] = 10 - df['player_dismissed'].notnull().cumsum()

    # Clean data
    df = df[(df['balls_left'] > 0) & (df['runs_left'] >= 0)]

    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    X = df[['runs_left', 'balls_left', 'wickets']]
    y = df['result']

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)

    return model

model = train_model()

# -------------------------------
# TEAM SELECTION
# -------------------------------
teams = sorted(matches['team1'].dropna().unique())

st.subheader("🏟️ Match Setup")

colT1, colT2 = st.columns(2)

with colT1:
    batting_team = st.selectbox("Batting Team", teams)

with colT2:
    bowling_team = st.selectbox("Bowling Team", teams)

if batting_team == bowling_team:
    st.warning("⚠️ Choose different teams!")

# -------------------------------
# INPUT
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

input_data = pd.DataFrame(
    [[runs_left, balls_left, wickets]],
    columns=["runs_left", "balls_left", "wickets"]
)

# -------------------------------
# PREDICTION
# -------------------------------
ml_prob = model.predict_proba(input_data)[0][1]

# Cricket logic tweak
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

st.success(f"🔥 {batting_team} Winning Probability: {round(final_prob * 100, 2)}%")
st.progress(int(final_prob * 100))

# -------------------------------
# METRICS
# -------------------------------
colA, colB, colC = st.columns(3)

with colA:
    st.metric("Runs Left", runs_left)

with colB:
    st.metric("Balls Left", balls_left)

with colC:
    st.metric("Required RR", round(required_rr, 2))

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("📈 Match Analysis")

chart_data = pd.DataFrame({
    "Metric": ["Runs Left", "Balls Left", "Wickets"],
    "Value": [runs_left, balls_left, wickets]
})

st.bar_chart(chart_data.set_index("Metric"))
