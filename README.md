# Student Score Prediction ML Project

This project predicts student exam scores based on study hours using Linear Regression.

## Project Structure

data/ → dataset  
models/ → trained ML model  
src/ → training scripts  
api/ → prediction API  

## Install Dependencies

pip install -r requirements.txt

## Train Model

python src/train_model.py

## Run API

python api/app.py

## API Endpoint

POST /predict

Example Request

{
 "study_hours": 6
}

Example Response

{
 "predicted_score": 80
}