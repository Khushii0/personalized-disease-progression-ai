import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('D:/codes/personalized disease progression/data/diabetes.csv')

# Add Patient_IDs
df['Patient_ID'] = df.index

# Number of synthetic visits to create per patient
n_visits = 5

synthetic_rows = []

for _, row in df.iterrows():
    base = row.copy()
    for visit in range(n_visits):
        new_row = base.copy()
        # Slight realistic changes to simulate progression
        new_row['Glucose'] = base['Glucose'] + np.random.normal(visit * 1.5, 2)
        new_row['BMI'] = base['BMI'] + np.random.normal(visit * 0.3, 1)
        new_row['BloodPressure'] = base['BloodPressure'] + np.random.normal(visit * 0.5, 1.5)
        new_row['Visit'] = visit
        synthetic_rows.append(new_row)

# Combine
ts_df = pd.DataFrame(synthetic_rows)

# Save it
ts_df.to_csv('D:/codes/personalized disease progression/data/synthetic_timeseries.csv', index=False)

# Preview
print(ts_df.head(10))
