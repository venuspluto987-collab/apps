import pandas as pd

# Load dataset
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# Merge
df = deliveries.merge(matches, left_on='match_id', right_on='id')

# Only 2nd innings (chasing team)
df = df[df['inning'] == 2]

# Target score
df['total_runs_y'] = df.groupby('match_id')['total_runs_x'].transform('sum')

# Current score
df['current_score'] = df.groupby('match_id')['total_runs_x'].cumsum()

# Balls left
df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])

# Runs left
df['runs_left'] = df['total_runs_y'] - df['current_score']

# Wickets left
df['wickets'] = 10 - df.groupby('match_id')['player_dismissed'].cumcount()

# Result
df['result'] = df['batting_team'] == df['winner']

# Clean
df = df[['runs_left', 'balls_left', 'wickets', 'result']].dropna()

df.to_csv("processed_data.csv", index=False)
