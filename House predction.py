import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------
# 1. Dataset
# ----------------------------
data = {
    "X": [1, 2, 3, 4, 5],
    "Y": [2, 4, 6, 8, 10]
}

df = pd.DataFrame(data)

# ----------------------------
# 2. Model Training
# ----------------------------
X = df[["X"]]
y = df["Y"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# 3. Prediction
# ----------------------------
x_new = 6
y_pred = model.predict([[x_new]])

print("Predicted value for X =", x_new, "is:", y_pred[0])

# ----------------------------
# 4. Graph Visualization
# ----------------------------
plt.scatter(df["X"], df["Y"], label="Actual Data")   # points
plt.plot(df["X"], model.predict(X), label="Regression Line")  # line

# Plot predicted point
plt.scatter(x_new, y_pred, color='red', label="Predicted Point")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Prediction")
plt.legend()
plt.show()
