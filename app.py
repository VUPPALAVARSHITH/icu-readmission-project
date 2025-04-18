import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ Caching model and column loading
@st.cache_resource
def load_model():
    with open("catboost_model_smote_tomek.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_columns():
    with open("X_train_columns.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
expected_columns = load_columns()

# ✅ Sidebar: User Inputs
st.sidebar.title("🩺 Patient Information")

time_in_hospital = st.sidebar.number_input(
    label="Time in hospital (days)",
    min_value=1,  # Minimum of 1 day
    max_value=10000,  # Optional max value, you can adjust
    value=6,  # Default value
    step=1  # Step of 1 day
)

age = st.sidebar.selectbox("Age Range", ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90'])
num_lab_procedures = st.sidebar.slider("Number of lab procedures", 0, 200, 41)
num_procedures = st.sidebar.slider("Number of procedures", 0, 100, 2)
num_medications = st.sidebar.slider("Number of medications", 1, 100, 12)
number_outpatient = st.sidebar.slider("Outpatient visits", 0, 20, 0)
number_emergency = st.sidebar.slider("Emergency visits", 0, 20, 1)
number_inpatient = st.sidebar.slider("Inpatient visits", 0, 20, 0)

medical_specialty = st.sidebar.selectbox("Medical Specialty", ['Cardiology', 'InternalMedicine', 'Surgery', 'Orthopedics', 'Emergency', 'GeneralPractice'])
diag_1 = st.sidebar.text_input("Diagnosis 1 code", "428")
diag_2 = st.sidebar.text_input("Diagnosis 2 code", "250.02")
diag_3 = st.sidebar.text_input("Diagnosis 3 code", "401.9")

glucose_test = st.sidebar.radio("Glucose test result", ["Normal", "Abnormal"])
A1Ctest = st.sidebar.radio("A1C test done?", ["Yes", "No"])
change = st.sidebar.radio("Change in medications?", ["Yes", "No"])
diabetes_med = st.sidebar.radio("Is diabetes medication prescribed?", ["Yes", "No"])

# ✅ Prepare input
input_dict = {
    "age": age,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "medical_specialty": medical_specialty,
    "diag_1": diag_1,
    "diag_2": diag_2,
    "diag_3": diag_3,
    "glucose_test": glucose_test,
    "A1Ctest": A1Ctest,
    "change": change,
    "diabetes_med": diabetes_med
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# ✅ Main UI
st.title("🏥 ICU Readmission Predictor")

tab1, tab2 = st.tabs(["🔍 Prediction Result", "⬇️ Download Input"])

with tab1:
    with st.expander("📋 View Prediction", expanded=True):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown("🔴 **Prediction: Patient is likely to be readmitted**")
        else:
            st.markdown("🟢 **Prediction: Patient is unlikely to be readmitted**")

        st.markdown(f"📊 **Readmission Probability:** `{proba:.2f}`")

with tab2:
    csv = input_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Patient Data",
        data=csv,
        file_name="patient_input.csv",
        mime="text/csv"
    )
