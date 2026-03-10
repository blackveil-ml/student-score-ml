# Student Score Prediction (Machine Learning)

This project predicts a student's exam score based on the number of hours studied using a simple Machine Learning model.

## Project Overview

The goal of this project is to demonstrate a basic ML workflow:

1. Load dataset
2. Train a Linear Regression model
3. Save the trained model
4. Serve the model through an API

## Project Structure

student-score-ml/
│
├── data/                # Dataset
├── models/              # Saved ML models
│   └── model.pkl
│
├── scripts/             # Training scripts
│   └── train_model.py
│
├── api/                 # API for predictions
│   └── app.py
│
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

## Model

### Algorithm used:
Linear Regression (Scikit-Learn)

### Input:  
- Study hours

### Output:  
- Predicted exam score

## Example API Request

### POST request:

`json
{
  "Study_Hours": 6
}
## Response:
{
  "Predicted_Score": 80
}

## How to Run the Project

### Install dependencies:
pip install -r requirements.txt

### Train the model:
python scripts/train_model.py

### Run the API:
python api/app.py

### The API will run at:
http://127.0.0.1:5000

## Author
Zaynab
Machine Learning Engineer (Learning Journey)