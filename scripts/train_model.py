import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# ---------------------------
# 1️⃣ Define dataset
# ---------------------------
data = {
    "Study_Hours": [1, 2, 3, 4, 5],
    "Exam_Score": [30, 40, 50, 60, 70]
}
df = pd.DataFrame(data)
X = df[["Study_Hours"]]
Y = df["Exam_Score"]

# ---------------------------
# 2️⃣ Train model
# ---------------------------
model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X)

# ---------------------------
# 3️⃣ Evaluate model
# ---------------------------
mse = mean_squared_error(Y, y_pred)
r2 = r2_score(Y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# ---------------------------
# 4️⃣ Save model
# ---------------------------
os.makedirs("../models", exist_ok=True)
with open("../models/model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model saved as models/model.pkl")

# ---------------------------
# 5️⃣ Create graphs
# ---------------------------
os.makedirs("../graphs", exist_ok=True)

# Actual vs Predicted
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Actual vs Predicted Exam Scores')
plt.legend()
plt.savefig('../graphs/actual_vs_predicted.png')
plt.close()

# Example prediction
example_hours = [[6]]
pred_score = model.predict(example_hours)
plt.bar([6], [pred_score[0]], color='green')
plt.xlabel('Study Hours')
plt.ylabel('Predicted Score')
plt.title('Prediction for 6 Study Hours')
plt.savefig('../graphs/prediction_example.png')
plt.close()

print("Graphs saved in graphs/ folder")

