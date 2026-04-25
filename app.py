import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------------
# 📊 Dataset
# -------------------------------
data = {
    "Area": [600, 800, 1000, 1200, 1500],
    "Bedrooms": [1, 2, 2, 3, 3],
    "Price": [3000, 4000, 5000, 6000, 7500]
}

df = pd.DataFrame(data)

# -------------------------------
# 🤖 Train model FIRST
# -------------------------------
X = df[["Area", "Bedrooms"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

# -------------------------------
# 🎛 Input
# -------------------------------
area = 1100
bedrooms = 2

new_data = pd.DataFrame([[area, bedrooms]], columns=["Area", "Bedrooms"])

# -------------------------------
# 🔮 Predict AFTER model
# -------------------------------
prediction = model.predict(new_data)

st.write("Predicted Price:", prediction[0])
