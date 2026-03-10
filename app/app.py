from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__, template_folder="templates")

# Load model
with open("../models/linear_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    hours = data.get("Study_Hours") if data else None
    if hours is None:
        return jsonify({"error": "Provide Study_Hours in JSON"}), 400
    try:
        hours = float(hours)
    except ValueError:
        return jsonify({"error": "Study_Hours must be a number"}), 400
    prediction = model.predict(pd.DataFrame([[hours]], columns=["Study_Hours"]))
    return jsonify({"Predicted_Score": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)