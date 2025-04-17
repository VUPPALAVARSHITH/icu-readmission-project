import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, Pool

# 🔹 Load the trained model
model_path = "catboost_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 🔹 Define categorical columns (same as your training)
categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                       'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

# 🔹 Function to preprocess user inputs
def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])
    
    # Convert categorical columns to 'category' dtype
    for col in categorical_columns:
        input_df[col] = input_df[col].astype('category')
    
    # Ensure the columns match the model's expected order
    input_df = input_df[['age', 'time_in_hospital', 'n_lab_procedures', 'n_procedures',
                         'n_medications', 'n_outpatient', 'n_emergency', 'n_inpatient',
                         'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test',
                         'A1Ctest', 'change', 'diabetes_med']]
    
    return input_df

# 🔹 Sidebar Input for user data
st.sidebar.title("Patient Details for ICU Readmission Prediction")

age = st.sidebar.selectbox("Age Range", ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90+'])
time_in_hospital = st.sidebar.number_input("Time in hospital (days)", min_value=0, max_value=365)
n_lab_procedures = st.sidebar.number_input("Number of lab procedures", min_value=0, max_value=100)
n_procedures = st.sidebar.number_input("Number of procedures", min_value=0, max_value=100)
n_medications = st.sidebar.number_input("Number of medications", min_value=0, max_value=50)
n_outpatient = st.sidebar.number_input("Number of outpatient visits", min_value=0, max_value=100)
n_emergency = st.sidebar.number_input("Number of emergency visits", min_value=0, max_value=100)
n_inpatient = st.sidebar.number_input("Number of inpatient visits", min_value=0, max_value=100)

medical_specialty = st.sidebar.selectbox("Medical Specialty", ['Cardiology', 'Oncology', 'Endocrinology', 'Neurology'])
diag_1 = st.sidebar.text_input("Diagnosis 1 code")
diag_2 = st.sidebar.text_input("Diagnosis 2 code")
diag_3 = st.sidebar.text_input("Diagnosis 3 code")

glucose_test = st.sidebar.selectbox("Glucose test result", ['Normal', 'Abnormal'])
A1Ctest = st.sidebar.selectbox("A1C test done?", ['Yes', 'No'])
change = st.sidebar.selectbox("Change in medications?", ['Yes', 'No'])
diabetes_med = st.sidebar.selectbox("Is diabetes medication prescribed?", ['Yes', 'No'])

# Gather the inputs into a dictionary
user_input = {
    'age': age,
    'time_in_hospital': time_in_hospital,
    'n_lab_procedures': n_lab_procedures,
    'n_procedures': n_procedures,
    'n_medications': n_medications,
    'n_outpatient': n_outpatient,
    'n_emergency': n_emergency,
    'n_inpatient': n_inpatient,
    'medical_specialty': medical_specialty,
    'diag_1': diag_1,
    'diag_2': diag_2,
    'diag_3': diag_3,
    'glucose_test': glucose_test,
    'A1Ctest': A1Ctest,
    'change': change,
    'diabetes_med': diabetes_med
}

# 🔹 Preprocess input data to match model
input_df = preprocess_input(user_input)

# 🔹 Predict
proba = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

# 🔹 Display Prediction and Risk Score
st.title("ICU Readmission Prediction")
st.write(f"**Prediction:** {'Readmitted' if prediction == 1 else 'Not Readmitted'}")
st.write(f"**Risk Score:** {proba:.2f}")

