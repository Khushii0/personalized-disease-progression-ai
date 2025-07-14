# personalized-disease-progression-ai
An end-to-end AI solution that predicts disease progression using patient time-series data and provides personalized insights with explainability.

Features
Predicts next-step glucose levels using LSTM-based deep learning on patient visit sequences.
Patient-specific SHAP explainability for feature importance (e.g., BMI, Blood Pressure, Glucose).
Generates a personalized health risk report with actionable lifestyle recommendations.
Trained on synthetic patient time-series data (extensible to real medical datasets).
Modular design for scalability to other diseases and health metrics.

Problem Statement
Chronic diseases like Diabetes require continuous monitoring. This project predicts future glucose levels based on past health records (Glucose, BMI, BP) and offers:
Predictive monitoring
Model transparency (Explainable AI)
Customized patient reports

Tech Stack
Python
PyTorch (LSTM for Time-Series Forecasting)
SHAP (Explainability)
RandomForestRegressor (Surrogate Model for SHAP)
Pandas, NumPy, Matplotlib

Project Structure
graphql
Copy
Edit
data/
    diabetes.csv                # Raw static data (not directly used)
    synthetic_timeseries.csv    # Main time-series dataset
notebooks/
    preprocessing.py            # Data preprocessing & scaling
    lstm_model.py               # LSTM model training & saving
    explainability.py           # Global feature importance with SHAP
    patient_report.py           # Generates per-patient predictions & insights
model.pth                       # Trained LSTM model weights
patient_1_shap_report.png       # Sample patient SHAP plot
