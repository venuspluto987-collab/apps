import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# 📊 Sample dataset
# -------------------------------
data = {
    "Area": [600, 800, 1000, 1200, 1500],
    "Bedrooms": [1, 2, 2, 3, 3],
    "Price": [3000, 4000, 5000, 6000, 7500]
}

df = pd.DataFrame(data)

# -------------------------------
# 🎯 Features & Target
# -------------------------------
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# -------------------------------
# 🤖 Train Model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# 🔮 Predict
# -------------------------------
prediction = model.predict([[1100, 2]])

print("Predicted Price:", prediction[0])

# -------------------------------
# 📈 Visualization
# -------------------------------
plt.scatter(df["Area"], df["Price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()
