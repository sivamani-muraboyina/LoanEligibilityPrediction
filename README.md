🏦 Loan Eligibility Prediction System

A machine learning-based web application that predicts whether a loan application will be approved or not based on user inputs such as income, credit history, and loan details.

🚀 Demo

📌 Live App: (Add after deployment)
📌 GitHub Repo: [(your repo link)](https://github.com/sivamani-muraboyina/LoanEligibilityPrediction)

📌 Overview

This project uses a K-Nearest Neighbors (KNN) model to predict loan approval status.
It provides a simple and interactive Streamlit UI for users to input details and get instant predictions.

🧠 Problem Statement

Loan approval is a critical process in financial institutions.
This project aims to automate decision-making using machine learning to:

Reduce manual effort
Improve consistency
Provide quick predictions
⚙️ Tech Stack
Python
Pandas, NumPy – Data processing
Scikit-learn – Model building (KNN)
Streamlit – Frontend UI
Joblib – Model persistence
Docker – Containerization
📊 Machine Learning Pipeline
Data Cleaning & Preprocessing
Feature Encoding (Categorical → Numerical)
Feature Scaling (StandardScaler)
Model Training (KNN Classifier)
Model Evaluation
Model Saving (.pkl files)
📈 Model Performance
Metric	Value
Accuracy	(fill yours)
Precision	(fill yours)
Recall	(fill yours)
F1 Score	(fill yours)
🖥️ Application UI
🔹 Input Form
Loan Amount
Loan Term
Credit History
Property Area
🔹 Output
Loan Approved / Not Approved

📸 (Add screenshot here)

🐳 Docker Support

This project is fully containerized using Docker.

🔹 Build Image
docker build -t loan-app .
🔹 Run Container
docker run -p 8501:8501 loan-app
🔹 Access App
http://localhost:8501
📁 Project Structure
LoanEligibilityPrediction/
│
├── app.py                 # Streamlit app
├── requirements.txt
├── Dockerfile
│
├── models/
│   ├── loan_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│
├── src/
│   ├── predict.py
│   ├── preprocess.py
│
├── notebooks/
│   └── model_building.ipynb
▶️ How to Run Locally
# Clone repo
git clone <your-repo-link>

# Create virtual env
python -m venv venv

# Activate
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
💡 Key Highlights

✔ End-to-end ML project
✔ Clean UI with Streamlit
✔ Dockerized deployment
✔ Modular code structure
✔ Real-time predictions

🔮 Future Improvements
Add FastAPI backend (optional microservice architecture)
Improve UI design
Deploy on Hugging Face / Render
Add model explainability (SHAP)
👤 Author

SivaManiM

