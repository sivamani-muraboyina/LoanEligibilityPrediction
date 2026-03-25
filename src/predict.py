import joblib
from src.preprocess import preprocess_input


# load trained model
model = joblib.load("models/loan_model.pkl")


# prediction function
def predict(input_dict):
    # preprocess input
    processed_data = preprocess_input(input_dict)
    
    # get prediction
    prediction = model.predict(processed_data)[0]
    
    # convert to readable output
    if prediction == 1:
        return "Loan Approved"
    else:
        return "Loan Rejected"


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