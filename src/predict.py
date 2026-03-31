import joblib
from src.preprocess import preprocess_input

# for logging and exception
from src.logger import logging
from src.exception import CustomException
import sys

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# load trained model
try:
    logging.info("Loading the model")

    MODEL_PATH = os.path.join(BASE_DIR, "../models/loan_model.pkl")
    model = joblib.load(MODEL_PATH)
    
    logging.info("model loaded successfully")
except Exception as e:
    logging.error("Error occured in loading the model")
    raise CustomException(e,sys)


# prediction function
def predict(input_dict):
    try:
        logging.info("Prediction pipeline has started ")
        logging.info(" preprocessing input data ")

        processed_data = preprocess_input(input_dict) 

        logging .info("preprocessing done successfully") 
        logging.info("model prediction ")
        
        prediction = model.predict(processed_data)[0]

        logging.info("prediction has done successfully")
    
    

        if prediction == 1:
          return "Loan Approved"
        else:
          return "Loan Rejected"
        
    except Exception as e:
        logging.error("Error occured in prediction pipeline")
        raise CustomException(e,sys)
    
    


# test run
if __name__ == "__main__":
    
    sample_input = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 4000,
        'CoapplicantIncome': 1500,
        'LoanAmount': 180,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 'Semiurban'
    }

    result = predict(sample_input)
    print("Prediction:", result)