import streamlit as st
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostClassifier

# 🔹 Load the pre-trained model and column names
with open("catboost_model_smote_tomek.pkl", "rb") as f:
    model = pickle.load(f)

with open("X_train_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# Sidebar input for user data
st.sidebar.header("Patient Information")

# Sidebar inputs
age = st.sidebar.selectbox("Age Range", ['<30', '30-40', '40-50', '50-60', '60-70', '70+'])
time_in_hospital = st.sidebar.number_input("Time in hospital (days)", min_value=1, max_value=100, value=6)
num_lab_procedures = st.sidebar.number_input("Number of lab procedures", min_value=0, max_value=100, value=10)
num_procedures = st.sidebar.number_input("Number of procedures", min_value=0, max_value=10, value=2)
num_medications = st.sidebar.number_input("Number of medications", min_value=0, max_value=50, value=10)
num_outpatient_visits = st.sidebar.number_input("Number of outpatient visits", min_value=0, max_value=50, value=1)
num_emergency_visits = st.sidebar.number_input("Number of emergency visits", min_value=0, max_value=50, value=0)
num_inpatient_visits = st.sidebar.number_input("Number of inpatient visits", min_value=0, max_value=50, value=0)

medical_specialty = st.sidebar.selectbox("Medical specialty", ['Cardiology', 'Neurology', 'Orthopedics', 'Pediatrics'])
diag_1 = st.sidebar.selectbox("Diagnosis 1 code", ['428', '250.02', '401.9', '535.50'])
diag_2 = st.sidebar.selectbox("Diagnosis 2 code", ['428', '250.02', '401.9', '535.50'])
diag_3 = st.sidebar.selectbox("Diagnosis 3 code", ['428', '250.02', '401.9', '535.50'])

glucose_test = st.sidebar.selectbox("Glucose test result", ['Normal', 'Abnormal'])
A1Ctest = st.sidebar.selectbox("A1C test done?", ['Yes', 'No'])
change = st.sidebar.selectbox("Change in medications?", ['Yes', 'No'])
diabetes_med = st.sidebar.selectbox("Is diabetes medication prescribed?", ['Yes', 'No'])

# Prepare input data for prediction
input_data = {
    "age": age,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "num_outpatient_visits": num_outpatient_visits,
    "num_emergency_visits": num_emergency_visits,
    "num_inpatient_visits": num_inpatient_visits,
    "medical_specialty": medical_specialty,
    "diag_1": diag_1,
    "diag_2": diag_2,
    "diag_3": diag_3,
    "glucose_test": glucose_test,
    "A1Ctest": A1Ctest,
    "change": change,
    "diabetes_med": diabetes_med
}

# Create DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Ensure input columns match expected columns
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# Prediction result
if st.sidebar.button("Predict Readmission"):
    # Predict using the model
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.write("🚨 The patient is likely to be readmitted to the hospital.")
    else:
        st.write("✅ The patient is unlikely to be readmitted to the hospital.")
