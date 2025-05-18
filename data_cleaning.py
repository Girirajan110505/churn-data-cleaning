import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Drop customerID as it's not useful for modeling
df.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert Yes/No columns
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encode other categorical variables
df = pd.get_dummies(df, drop_first=True)

# Save cleaned data
os.makedirs("cleaned_data", exist_ok=True)
df.to_csv("cleaned_data/cleaned_customer_churn.csv", index=False)
print("✅ Cleaned data saved to cleaned_data/cleaned_customer_churn.csv")
