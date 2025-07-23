import pandas as pd

# Load the data
df = pd.read_csv('D:/codes/personalized disease progression/data/diabetes.csv')
print(df.head())
print(df.info())

# Check for missing or zero values in important columns
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_to_check:
    print(f"{col} zero values: {(df[col] == 0).sum()}")

# Replace 0s with NaN for these columns
df[columns_to_check] = df[columns_to_check].replace(0, pd.NA)

# Impute missing values with median
df.fillna(df.median(), inplace=True)

# Sanity check after imputation
print(df.isnull().sum())
