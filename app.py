import streamlit as st
import pandas as pd
import pickle

# 📦 Load model and column info with caching
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

# 🌟 Sidebar Inputs
st.sidebar.title("📝 Patient Information")

age = st.sidebar.selectbox("Age Range", ['0-10', '10-20', '20-30', '30-40', '40-50',
                                         '50-60', '60-70', '70-80', '80-90', '90-100'])
time_in_hospital = st.sidebar.slider("Time in hospital (days)", 1, 14, 6)
num_lab_procedures = st.sidebar.slider("Number of lab procedures", 1, 132, 41)
num_procedures = st.sidebar.slider("Number of procedures", 0, 6, 2)
num_medications = st.sidebar.slider("Number of medications", 1, 81, 12)
number_outpatient = st.sidebar.slider("Outpatient visits", 0, 20, 0)
number_emergency = st.sidebar.slider("Emergency visits", 0, 20, 1)
number_inpatient = st.sidebar.slider("Inpatient visits", 0, 20, 0)

medical_specialty = st.sidebar.selectbox("Medical Specialty", [
    'Cardiology', 'InternalMedicine', 'Family/GeneralPractice', 'Surgery-General',
    'Nephrology', 'Emergency/Trauma', 'Orthopedics', 'Radiologist', 'Other'
])

diag_1 = st.sidebar.text_input("Diagnosis 1 code", "428")
diag_2 = st.sidebar.text_input("Diagnosis 2 code", "250.02")
diag_3 = st.sidebar.text_input("Diagnosis 3 code", "401.9")

glucose_test = st.sidebar.selectbox("Glucose test result", ['Normal', 'Abnormal'])
A1Ctest = st.sidebar.selectbox("A1C test done?", ['Yes', 'No'])
change = st.sidebar.selectbox("Change in medications?", ['Yes', 'No'])
diabetes_med = st.sidebar.selectbox("Is diabetes medication prescribed?", ['Yes', 'No'])

# 🔢 Prepare input data
input_data = pd.DataFrame([{
    'age': age,
    'time_in_hospital': time_in_hospital,
    'num_lab_procedures': num_lab_procedures,
    'num_procedures': num_procedures,
    'num_medications': num_medications,
    'number_outpatient': number_outpatient,
    'number_emergency': number_emergency,
    'number_inpatient': number_inpatient,
    'medical_specialty': medical_specialty,
    'diag_1': diag_1,
    'diag_2': diag_2,
    'diag_3': diag_3,
    'glucose_test': glucose_test,
    'A1Ctest': A1Ctest,
    'change': change,
    'diabetes_med': diabetes_med
}])

# 🔄 One-hot encode and align with training columns
input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

# ⏳ Prediction section
with st.expander("🔍 Prediction Result"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.markdown("### 🔴 Prediction: **Patient likely to be readmitted**")
    else:
        st.markdown("### 🟢 Prediction: **Patient unlikely to be readmitted**")

    st.markdown(f"**Probability of readmission:** `{probability:.2%}`")

# 📥 Download Prediction Input (optional for logs)
st.download_button("📤 Download Your Input Data", input_data.to_csv(index=False), file_name="patient_input.csv")

