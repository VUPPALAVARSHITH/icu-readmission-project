import streamlit as st
import pickle
import pandas as pd

# Load the trained model and columns
with open("catboost_model_smote_tomek.pkl", "rb") as f:
    model = pickle.load(f)

with open("X_train_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# Sidebar input for user data
st.sidebar.header("Patient Data")

# Static inputs (fixed values)
age_range = st.sidebar.selectbox("Age Range", ["60-70", "70-80", "80-90", "90-100"])
medical_specialty = st.sidebar.selectbox("Medical Specialty", ["Cardiology", "Neurology", "Orthopedics", "Oncology", "Pediatrics"])
diagnosis_1 = st.sidebar.text_input("Diagnosis 1 Code", "428")
diagnosis_2 = st.sidebar.text_input("Diagnosis 2 Code", "250.02")
diagnosis_3 = st.sidebar.text_input("Diagnosis 3 Code", "401.9")
glucose_test = st.sidebar.selectbox("Glucose Test Result", ["Normal", "Abnormal"])
A1C_test = st.sidebar.selectbox("A1C Test Done?", ["Yes", "No"])
change_in_medications = st.sidebar.selectbox("Change in Medications?", ["Yes", "No"])
diabetes_med = st.sidebar.selectbox("Is Diabetes Medication Prescribed?", ["Yes", "No"])

# Dynamic input for Time in Hospital (days)
time_in_hospital = st.sidebar.slider("Time in Hospital (days)", min_value=0, max_value=30, value=5, step=1)

# Prepare the input data for prediction
input_data = {
    "age": age_range,
    "medical_specialty": medical_specialty,
    "diag_1": diagnosis_1,
    "diag_2": diagnosis_2,
    "diag_3": diagnosis_3,
    "glucose_test": glucose_test,
    "A1Ctest": A1C_test,
    "change": change_in_medications,
    "diabetes_med": diabetes_med,
    "time_in_hospital": time_in_hospital  # Dynamic input for "Time in Hospital"
}

# Convert input data to DataFrame and one-hot encode it
input_df = pd.DataFrame([input_data])

# Ensure the input columns match the model's expected columns
input_df_encoded = pd.get_dummies(input_df)

# Align the columns of input data with the model's training data
input_df_encoded = input_df_encoded.reindex(columns=expected_columns, fill_value=0)

# Make prediction
prediction = model.predict(input_df_encoded)[0]

# Display the prediction result
st.subheader("Prediction Result")
if prediction == 1:
    st.markdown("🟢 **Patient is likely to be readmitted**.")
else:
    st.markdown("🔴 **Patient is not likely to be readmitted**.")
