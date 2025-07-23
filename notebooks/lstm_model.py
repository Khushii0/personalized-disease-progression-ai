import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('D:/codes/personalized disease progression/data/synthetic_timeseries.csv')
features = ['Glucose', 'BMI', 'BloodPressure']
target = 'Glucose'
group_col = 'Patient_ID'

# Normalize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Dataset
class DiabetesDataset(Dataset):
    def __init__(self, df, features, target, group_col, seq_len):
        self.X, self.y = [], []
        for pid in df[group_col].unique():
            patient_data = df[df[group_col] == pid].sort_values('Visit')
            data = patient_data[features].values
            target_vals = patient_data[target].values
            if len(data) >= seq_len + 1:
                for i in range(len(data) - seq_len):
                    self.X.append(data[i:i+seq_len])
                    self.y.append(target_vals[i+seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).view(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use last time step

# Prepare data
sequence_length = 3
dataset = DiabetesDataset(df, features, target, group_col, sequence_length)
train_size = int(0.8 * len(dataset))
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Initialize
input_size = len(features)
model = LSTMModel(input_size, hidden_size=64, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
