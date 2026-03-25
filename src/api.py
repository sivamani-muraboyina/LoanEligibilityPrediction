from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.preprocess import preprocess_input

# create app
app = FastAPI()

# load model
model = joblib.load("models/loan_model.pkl")

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
    input_dict = data.dict()

    processed_data = preprocess_input(input_dict)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    result = "Loan Approved" if prediction == 1 else "Loan Rejected"

    return {
        "prediction": result,
        "probability": round(float(probability), 3)
    }