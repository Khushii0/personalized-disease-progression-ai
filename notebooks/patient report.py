import pandas as pd
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from lstm_model import LSTMModel as LSTMRegressor  # Make sure the model matches what was trained

# Load dataset
DATA_PATH = 'D:/codes/personalized disease progression/data/synthetic_timeseries.csv'
df = pd.read_csv(DATA_PATH)
features = ['Glucose', 'BMI', 'BloodPressure']
group_col = 'Patient_ID'

# Normalize features (same as during training)
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Load model
model = LSTMRegressor(input_size=len(features), hidden_size=64, num_layers=2, output_size=1)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Function to generate patient report
def generate_patient_report(pid):
    patient_data = df[df[group_col] == pid]
    if patient_data.empty:
        print(f"Patient ID {pid} not found.")
        return

    # We'll take the last known visit for prediction (or the only row)
    x_input = patient_data[features].values[-3:]  # use last 3 if available
    if x_input.shape[0] < 3:
        print(f"Not enough data points for Patient {pid} (needs at least 3).")
        return

    x_input_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)  # shape (1, 3, features)
    with torch.no_grad():
        prediction = model(x_input_tensor).item()

    # SHAP Explanation using surrogate model
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(df[features], df['Glucose'])  # mimic general trends for SHAP

    explainer = shap.Explainer(rf.predict, df[features])
    shap_vals = explainer(patient_data[features])

    # Plot SHAP summary for this patient
    shap.plots.bar(shap_vals[-1], max_display=3, show=False)
    plt.title(f"SHAP Explanation for Patient {pid}")
    plt.tight_layout()
    plt.savefig(f'patient_{pid}_shap_report.png')
    plt.show()

    print(f"\nPatient {pid} Risk Prediction: {prediction:.2f}")
    top_feature = np.array(features)[np.argsort(-np.abs(shap_vals[-1].values))][0]
    print(f"Primary contributing factor: {top_feature}")

    # Give suggestions
    print("Suggestions to lower risk:")
    if top_feature == 'Glucose':
        print("- Maintain blood sugar levels through diet and exercise.")
    elif top_feature == 'BMI':
        print("- Aim for a healthier BMI through regular physical activity.")
    elif top_feature == 'BloodPressure':
        print("- Monitor blood pressure and reduce salt intake.")
    else:
        print("- General lifestyle improvements recommended.")

# ==== Run ====
if __name__ == '__main__':
    pid = int(input("Enter Patient ID to generate report: "))
    generate_patient_report(pid)
