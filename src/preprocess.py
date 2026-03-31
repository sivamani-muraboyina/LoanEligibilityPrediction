import pandas as pd
import numpy as np
import joblib

from src.logger import logging
from src.exception import CustomException
import sys

# load saved artifacts (trained during model building)
try:
    logging.info(" loading the standard scaler")
    scaler = joblib.load("models/scaler.pkl")
    logging.info(" scaler loading done succesfully")
except Exception as e:
    logging.error("Error while loading standard scler")
    raise CustomException(e,sys)
try:
    logging.info("loading the columns required for model")
    model_columns = joblib.load("models/columns.pkl")
    logging.info("Model_columns loaded sucessfully")
except Exception as e:
    logging.error("error while loading model _columns")
    raise CustomException(e,sys)

encoder = joblib.load("models/encoder.pkl")
try:
    logging.info("loading the encoder has started")
    encoder = joblib.load("models/encoder.pkl")
    logging.info("Encoder loaded sucessfully")
except Exception as e:
    logging.error("error while loading encoder")
    raise CustomException(e,sys)


# create dataframe from user input
def create_input_df(input_dict):
    try:
        logging.info("Creating the input dataframe")
        df = pd.DataFrame([input_dict])
        logging.info("Data frame created successfully for input")
    
        return df
    except Exception as e:
        logging.error("error while creating the dataframe")
        raise CustomException(e,sys)
        



# basic cleaning (handle messy user input)
def clean_input(df):
    try:
        logging.info("Cleaning the input")
        cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        for col in cat_cols:
             df[col] = df[col].astype(str).str.strip().str.lower()
        df['Dependents'] = df['Dependents'].astype(str).str.strip()
        df['Dependents'] = df['Dependents'].replace('3+', '3')
        logging.info("Cleaning done successfully")
        return df
        
    
    except Exception as e:
        logging.error("Error while cleaning the input")
        raise CustomException(e,sys)
        


# feature engineering (same logic as training)   
def create_features(df):
    try:
        logging.info("Feature engineering started")

        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['Loan_Income_Ratio'] = (df['LoanAmount'] * 1000) / df['TotalIncome']

        logging.info("Feature engineering completed")

        return df

    except Exception as e:
        logging.error("Error occurred during feature engineering")
        raise CustomException(e, sys)




#  encoding for binary categorical columns
def encode_input(df):
    try:
        logging.info("Encoding categorical features started")

        df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
        df['Married'] = df['Married'].map({'yes': 1, 'no': 0})
        df['Education'] = df['Education'].map({'graduate': 1, 'not graduate': 0})
        df['Self_Employed'] = df['Self_Employed'].map({'yes': 1, 'no': 0})

        logging.info("Encoding categorical features completed")

        return df

    except Exception as e:
        logging.error("Error occurred during encoding")
        raise CustomException(e, sys)

# one-hot encoding using trained encoder
def one_hot_encode(df):
    try:
        logging.info("One-hot encoding started")

        cols = ['Dependents', 'Property_Area', 'Loan_Amount_Term']

        encoded = encoder.transform(df[cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(cols)
        )

        df = df.drop(columns=cols)
        df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

        logging.info("One-hot encoding completed")

        return df

    except Exception as e:
        logging.error("Error occurred during one-hot encoding")
        raise CustomException(e, sys)


# align columns to match training data exactly
def align_columns(df):
    try:
        logging.info("Aligning columns with trained model")

        # add missing columns
        for col in model_columns:
            if col not in df:
                df[col] = 0

        # keep only required columns and correct order
        df = df[model_columns]

        logging.info("Column alignment completed")

        return df

    except Exception as e:
        logging.error("Error occurred during column alignment")
        raise CustomException(e, sys)


# scale only numerical columns
def scale_data(df):
    try:
        logging.info("Scaling numerical features started")

        num_cols = [
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'TotalIncome',
            'Loan_Income_Ratio'
        ]

        df[num_cols] = scaler.transform(df[num_cols])

        logging.info("Scaling completed")

        return df

    except Exception as e:
        logging.error("Error occurred during scaling")
        raise CustomException(e, sys)


# complete preprocessing pipeline
def preprocess_input(input_dict):
    try:
        logging.info("Preprocessing pipeline started")

        df = create_input_df(input_dict)

        df = clean_input(df)

        df = create_features(df)

        df = encode_input(df)

        df = one_hot_encode(df)

        df = align_columns(df)

        df = df.fillna(0)

        df = scale_data(df)

        logging.info("Preprocessing pipeline completed successfully")

        return df

    except Exception as e:
        logging.error("Error occurred in preprocessing pipeline")
        raise CustomException(e, sys)


# testing
if __name__ == "__main__":
    sample_input = {
        'Gender': ' Male ',
        'Married': 'Yes',
        'Dependents': '3+',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 0,
        'LoanAmount': 200,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 'Urban'
    }
    processed = preprocess_input(sample_input)
    print("Final shape:", processed.shape)