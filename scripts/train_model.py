import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Make folders if not exist
os.makedirs("../models", exist_ok=True)
os.makedirs("../graphs", exist_ok=True)

# Dataset
data = {
    "Study_Hours": [1,2,3,4,5],
    "Exam_Score": [30,40,50,60,70]
}
df = pd.DataFrame(data)
X = df[["Study_Hours"]]
Y = df["Exam_Score"]

# Train model
model = LinearRegression()
model.fit(X,Y)

# Predict for training data
y_pred = model.predict(X)

# Evaluation
mse = mean_squared_error(Y, y_pred)
r2 = r2_score(Y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Plot actual vs predicted
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Actual vs Predicted Exam Scores')
plt.legend()
plt.savefig('../graphs/actual_vs_predicted.png')
plt.show()

# Predict example
hours = [[6]]
pred_score = model.predict(hours)
plt.bar([6], [pred_score[0]], color='green')
plt.xlabel('Study Hours')
plt.ylabel('Predicted Score')
plt.title('Prediction for 6 Study Hours')
plt.savefig('../graphs/prediction_example.png')
plt.show()

# Save model
with open("../models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model and graphs saved!")

