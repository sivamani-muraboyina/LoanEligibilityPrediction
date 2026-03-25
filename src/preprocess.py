import pandas as pd
import numpy as np
import joblib

# load saved artifacts (trained during model building)
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/columns.pkl")
encoder = joblib.load("models/encoder.pkl")


# create dataframe from user input
def create_input_df(input_dict):
    df = pd.DataFrame([input_dict])
    return df


# basic cleaning (handle messy user input)
def clean_input(df):
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df['Dependents'] = df['Dependents'].astype(str).str.strip()
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    return df


# feature engineering (same logic as training)
def create_features(df):
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Loan_Income_Ratio'] = (df['LoanAmount'] * 1000) / df['TotalIncome']
    return df


# label encoding for binary categorical columns
def encode_input(df):
    df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
    df['Married'] = df['Married'].map({'yes': 1, 'no': 0})
    df['Education'] = df['Education'].map({'graduate': 1, 'not graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'yes': 1, 'no': 0})
    return df


# one-hot encoding using trained encoder
def one_hot_encode(df):
    cols = ['Dependents', 'Property_Area', 'Loan_Amount_Term']
    encoded = encoder.transform(df[cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols))
    df = df.drop(columns=cols)
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    return df


# align columns to match training data exactly
def align_columns(df):
    for col in model_columns:
        if col not in df:
            df[col] = 0
    df = df[model_columns]
    return df


# scale only numerical columns
def scale_data(df):
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome', 'Loan_Income_Ratio']
    df[num_cols] = scaler.transform(df[num_cols])
    return df


# complete preprocessing pipeline
def preprocess_input(input_dict):
    df = create_input_df(input_dict)
    df = clean_input(df)
    df = create_features(df)
    df = encode_input(df)
    df = one_hot_encode(df)
    df = align_columns(df)
    df = df.fillna(0)
    df = scale_data(df)
    return df


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