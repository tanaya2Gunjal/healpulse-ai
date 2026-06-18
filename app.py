import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="HealPulse AI", layout="wide")

st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-weight: 800 !important;
        margin-bottom: 2rem;
        font-size: 3rem !important;
    }
    
    .metric-card {
        background-color: rgb(31, 41, 55);
        padding: 25px;
        border-radius: 10px;
        border: 1px solid rgb(55, 65, 81);
        text-align: center;
    }
    .metric-card p {
        color: rgb(156, 163, 175);
        margin: 0;
        font-size: 1.3rem !important; 
        font-weight: bold;
        text-transform: uppercase;
    }
    .metric-card h1 {
        font-size: 3.2rem !important;
        margin: 15px 0 0 0 !important;
        font-weight: 800;
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 2.8rem !important; 
        font-weight: 800 !important;
        line-height: 1.2 !important;
    }
    
    [data-testid="stHeaderBlock"] h2, h2 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }
    
    .diagnostic-question {
        font-size: 2.2rem !important; 
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        color: white;
    }
    
    .diagnostic-text {
        font-size: 1.5rem !important; 
        line-height: 1.7 !important;
        margin-bottom: 1rem !important;
        color: rgb(229, 231, 235);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_clean_data():
    return pd.read_csv('cleaned_hospital_readmission_dataset.csv')

@st.cache_resource
def train_model():
    df_clean = load_clean_data()
    X = df_clean.drop(columns=['label'])
    y = df_clean['label']   
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y) 
    return model, X.columns.tolist()

model, model_features = train_model()

st.sidebar.title("Patient Metrics")
st.sidebar.markdown("Adjust the patient's information here:")

age = st.sidebar.slider("Patient Age", min_value=0, max_value=120, value=65, key="sb_age")
length_of_stay = st.sidebar.slider("Length of Stay (Days)", min_value=1, max_value=30, value=5, key="sb_los")
medications_count = st.sidebar.slider("Total Medications Count", min_value=0, max_value=50, value=8, key="sb_meds")
readmission_risk_score = st.sidebar.slider("Calculated Baseline Risk Score", min_value=0, max_value=100, value=50, key="sb_risk")
followup_visits_last_year = st.sidebar.number_input("Followup Visits (Past Year)", min_value=0, max_value=20, value=1, key="sb_visits")

base_inputs = {
    'age': age,
    'length_of_stay': length_of_stay,
    'medications_count': medications_count,
    'readmission_risk_score': readmission_risk_score,
    'followup_visits_last_year': followup_visits_last_year
}
input_dict = {col: base_inputs.get(col, 0) for col in model_features}
input_df = pd.DataFrame([input_dict])[model_features]

probabilities = model.predict_proba(input_df)[0]
risk_percentage = probabilities[1] * 100

if risk_percentage >= 70:
    status_text = "HIGH RISK"
    status_color = "rgb(239, 68, 68)"
    decision_text = "Critical Review"
    decision_color = "rgb(248, 113, 113)"
elif 45 <= risk_percentage < 70:
    status_text = "MEDIUM RISK"
    status_color = "rgb(251, 191, 36)"
    decision_text = "Observation"
    decision_color = "rgb(251, 191, 36)"
else:
    status_text = "LOW RISK"
    status_color = "rgb(16, 185, 129)"
    decision_text = "Standard Discharge"
    decision_color = "rgb(96, 165, 250)"

st.markdown('<h1 class="centered-title">HealPulse: Patient Readmission AI Classifier</h1>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p>Readmission Risk</p>
        <h1 style="color:rgb(248, 113, 113);">{risk_percentage:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p>Prediction Status</p>
        <h1 style="color:{status_color};">{status_text}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <p>Clinical Decision</p>
        <h1 style="color:{decision_color};">{decision_text}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.subheader("Clinical Diagnostic Summary")

st.markdown(f'<p class="diagnostic-question">Why is the patient flagged as {status_text}?</p>', unsafe_allow_html=True)

if status_text == "HIGH RISK":
    st.markdown(f'<p class="diagnostic-text">• <b>Primary Reason:</b> The combined clinical metrics show an urgent pattern. Specifically, the baseline risk score of <b>{readmission_risk_score}</b> and a stay duration of <b>{length_of_stay} days</b> heavily push the total probability into the red zone.</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="diagnostic-text">• <b>Contributing Factors:</b> Managing <b>{medications_count} different medications</b> at an age of <b>{age}</b> introduces significant tracking difficulties after leaving the facility.</p>', unsafe_allow_html=True)

elif status_text == "MEDIUM RISK":
    st.markdown(f'<p class="diagnostic-text">• <b>Primary Reason:</b> The metrics reflect a borderline scenario. While some parameters look stable, the patient\'s baseline risk score of <b>{readmission_risk_score}</b> indicates they require close observation.</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="diagnostic-text">• <b>Contributing Factors:</b> A stay length of <b>{length_of_stay} days</b> paired with <b>{followup_visits_last_year} previous visits</b> over the past year means their recovery tracker is volatile and needs monitoring.</p>', unsafe_allow_html=True)

else:
    st.markdown(f'<p class="diagnostic-text">• <b>Primary Reason:</b> All core numbers are safely within the normal target baseline range, leading to an optimal safety rating.</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="diagnostic-text">• <b>Contributing Factors:</b> The short stay length of <b>{length_of_stay} days</b> and low medical complexity parameters suggest a straightforward recovery period at home.</p>', unsafe_allow_html=True)