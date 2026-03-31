from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.preprocess import preprocess_input

from src.logger import logging
from src.exception import CustomException
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# create app
app = FastAPI()


# load model
try:
    logging.info("Loading model")

    
    MODEL_PATH = os.path.join(BASE_DIR, "../models/loan_model.pkl")

    model = joblib.load(MODEL_PATH)

    logging.info("Model loaded successfully")

except Exception as e:
    logging.error("Error occurred while loading model")
    raise CustomException(e, sys)

    

# request schema
class LoanInput(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

# home route
@app.get("/")
def home():
    return {"message": "Loan Prediction API is running"}

# prediction route
@app.post("/predict")
def predict(data: LoanInput):
    try:
        logging.info("request recievde at /predict endpoint")

        input_dict = data.dict()
        logging.info("Preprocessing the input dict")
        processed_data = preprocess_input(input_dict)
        logging.info(" preprocessing done successfully")

        logging.info("Model prediction started")
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        logging.info(" Model prediction is done")

        result = "Loan Approved" if prediction == 1 else "Loan Rejected"

        return {
            "prediction": result,
            "probability": round(float(probability), 3)
        }
    except Exception as e:
        logging.error("Error occured in prediction endpoint")
        raise CustomException(e,sys)

