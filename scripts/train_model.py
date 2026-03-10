import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# 1️⃣ Ensure models folder exists
models_dir = "../models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created folder: {models_dir}")

# 2️⃣ Dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5],
    "Exam_Score": [30, 40, 50, 60, 70]
}
df = pd.DataFrame(data)
X = df[["Study_Hours"]]
Y = df["Exam_Score"]

# 3️⃣ Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4️⃣ Train model
model = LinearRegression()
model.fit(X_train, Y_train)
print("Model trained successfully!")

# 5️⃣ Evaluate
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# 6️⃣ Save model
#model_path = os.path.join(models_dir, "linear_model.pkl")
with open("models/linear_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model saved at models/linear_model.pkl")

# 7️⃣ Test prediction
prediction = model.predict([[6]])
print(f"Predicted score for 6 study hours: {prediction[0]}")

