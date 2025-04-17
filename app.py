import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier

st.set_page_config(page_title="ICU Readmission Predictor", layout="wide")

# ------------------- CACHE -------------------

@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\varsh\OneDrive\Desktop\minor_project\preprocessed_hospital_readmissions.csv")

@st.cache_resource
def load_model():
    with open("catboost_model.pkl", "rb") as f:
        return pickle.load(f)

# ------------------- LOAD DATA -------------------

df = load_data()
model = load_model()
categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                       'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

# ------------------- SIDEBAR INPUT -------------------

st.sidebar.header("🧾 Enter Patient Details")

with st.sidebar.expander("🔧 Patient Info", expanded=True):
    user_input = {
        'age': st.selectbox("Age Range", sorted(df['age'].unique())),
        'time_in_hospital': st.number_input("Time in hospital (days)", min_value=1, max_value=50, value=5),
        'n_lab_procedures': st.number_input("Number of lab procedures", 0, 100, 40),
        'n_procedures': st.number_input("Number of procedures", 0, 10, 1),
        'n_medications': st.number_input("Number of medications", 0, 100, 20),
        'n_outpatient': st.number_input("Number of outpatient visits", 0, 10, 0),
        'n_emergency': st.number_input("Number of emergency visits", 0, 10, 0),
        'n_inpatient': st.number_input("Number of inpatient visits", 0, 10, 0),
        'medical_specialty': st.selectbox("Medical Specialty", sorted(df['medical_specialty'].dropna().unique())),
        'diag_1': st.selectbox("Diagnosis 1", sorted(df['diag_1'].dropna().unique())),
        'diag_2': st.selectbox("Diagnosis 2", sorted(df['diag_2'].dropna().unique())),
        'diag_3': st.selectbox("Diagnosis 3", sorted(df['diag_3'].dropna().unique())),
        'glucose_test': st.selectbox("Glucose Test Result", sorted(df['glucose_test'].dropna().unique())),
        'A1Ctest': st.selectbox("A1C Test Done?", sorted(df['A1Ctest'].dropna().unique())),
        'change': st.selectbox("Change in Medications?", sorted(df['change'].dropna().unique())),
        'diabetes_med': st.selectbox("Diabetes Medication Prescribed?", sorted(df['diabetes_med'].dropna().unique()))
    }

# ------------------- DATAFRAME & PREDICTION -------------------

input_df = pd.DataFrame([user_input])
for col in categorical_columns:
    input_df[col] = input_df[col].astype('category')
input_df = input_df[model.feature_names_]

prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

# ------------------- TABS -------------------

tab1, tab2 = st.tabs(["📊 Prediction", "📥 Download"])

with tab1:
    st.title("🏥 ICU Readmission Prediction")
    if prediction == 1:
        st.markdown("🔴 **Prediction: Readmitted**")
    else:
        st.markdown("🟢 **Prediction: Not Readmitted**")
    st.metric(label="📉 Risk Score", value=f"{proba:.2f}")

with tab2:
    st.markdown("📋 Download Entered Data:")
    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "patient_input.csv", "text/csv")

# ------------------- END -------------------
