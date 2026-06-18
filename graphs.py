import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_raw = pd.read_csv('hospital_readmission_dataset.csv')
df_clean = pd.read_csv('cleaned_hospital_readmission_dataset.csv')

X = df_clean.drop(columns=['label'])
y = df_clean['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
pie_labels = ['No', 'Yes'] if 0 in df_raw['label'].values else ['Yes', 'No']
df_raw['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90, labels=pie_labels)
plt.title('Overall Patient Readmission Rate', fontsize=12, fontweight='bold')
plt.ylabel('')

plt.subplot(1, 3, 2)
importances = model.feature_importances_
indices = np.argsort(importances)[-5:]

clean_labels = []
for i in indices:
    label = X.columns[i]
    label = label.replace('readmission_risk_score', 'Risk Score')
    label = label.replace('length_of_stay', 'Length of Stay')
    label = label.replace('medications_count', 'Medication Count')
    label = label.replace('followup_visits_last_year', 'Followup Visits')
    label = label.replace('age', 'Age')
    label = label.replace('primary_diagnosis_', 'Diag: ')
    label = label.replace('treatment_type_', 'Treat: ')
    label = label.replace('discharge_disposition_', 'Discharge: ')
    label = label.replace('insurance_type_', 'Ins: ')
    label = label.replace('_', ' ').title()
    clean_labels.append(label)

plt.title('Top 5 Factors Driving Risk', fontsize=12, fontweight='bold')
plt.barh(range(len(indices)), importances[indices], color='teal', align='center', height=0.5)
plt.yticks(range(len(indices)), clean_labels, fontsize=10)
plt.xlabel('Relative Importance', fontsize=10)

plt.subplot(1, 3, 3)
sns.histplot(data=df_raw, x='age', hue='label', multiple='stack', palette='coolwarm', kde=True)
plt.title('Patient Age Distribution vs Readmission', fontsize=12, fontweight='bold')
plt.xlabel('Age', fontsize=10)
plt.ylabel('Count', fontsize=10)

plt.tight_layout()
plt.savefig('project_analysis_graphs.png', dpi=300)
print("\nSuccess! The final polished graphs have been saved.")
plt.show()