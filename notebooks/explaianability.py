import pandas as pd
import numpy as np
import torch
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from lstm_model import LSTMModel  # Same name now
import torch.nn as nn

# Load model
input_size = 3
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load data
df = pd.read_csv('D:/codes/personalized disease progression/data/synthetic_timeseries.csv')
features = ['Glucose', 'BMI', 'BloodPressure']
group_col = 'Patient_ID'
sequence_length = 3

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Explainer Dataset
class ExplainerDataset:
    def __init__(self, df, features, group_col, seq_len):
        self.X_seq = []
        self.X_last_step = []
        for pid in df[group_col].unique():
            patient_data = df[df[group_col] == pid].sort_values('Visit')
            data = patient_data[features].values
            if len(data) >= seq_len + 1:
                for i in range(len(data) - seq_len):
                    seq = data[i:i+seq_len]
                    self.X_seq.append(seq)
                    self.X_last_step.append(seq[-1])  # Last step for surrogate
        self.X_seq = np.array(self.X_seq)
        self.X_last_step = np.array(self.X_last_step)

# Load explainer data
ds = ExplainerDataset(df, features, group_col, sequence_length)

# LSTM predictions
X_tensor = torch.tensor(ds.X_seq, dtype=torch.float32)
with torch.no_grad():
    preds = model(X_tensor).numpy().flatten()

# Train surrogate model (Random Forest)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(ds.X_last_step, preds)

# SHAP Explainer
explainer = shap.Explainer(rf.predict, ds.X_last_step)
shap_values = explainer(ds.X_last_step)

# Plot
shap.summary_plot(shap_values, pd.DataFrame(ds.X_last_step, columns=features))
