import pandas as pd

df = pd.read_csv('hospital_readmission_dataset.csv')
print(f"Dataset Shape: {df.shape}")
print("\nData Types and Missing Values:")
print(df.info())
print("\nFirst 5 Rows of Raw Data:")
print(df.head())
if 'patient_id' in df.columns:
    df = df.drop(columns=['patient_id'])
if 'admission_date' in df.columns:
    df = df.drop(columns=['admission_date'])
categorical_cols = ['season', 'gender', 'region', 'primary_diagnosis', 'treatment_type', 'insurance_type', 'discharge_disposition']
existing_categorical = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=existing_categorical, drop_first=True)
df.to_csv('cleaned_hospital_readmission_dataset.csv', index=False)
print("\nData cleaning complete! 'cleaned_hospital_readmission_dataset.csv' has been saved.")
