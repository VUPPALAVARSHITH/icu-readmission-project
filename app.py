import streamlit as st
import pandas as pd
import shap
import lime
import lime.lime_tabular
import numpy as np
import joblib
from catboost import CatBoostClassifier, Pool

# Load trained model
model = joblib.load("catboost_model.pkl")

# Define categorical features
cat_features = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

# Sidebar input form
st.sidebar.title("Patient Data Input")

age = st.sidebar.selectbox("Age", ['0-10', '10-20', '20-30', '30-40', '40-50',
                                    '50-60', '60-70', '70-80', '80-90', '90-100'])

time_in_hospital = st.sidebar.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=10)

n_lab_procedures = st.sidebar.slider("Number of Lab Procedures", min_value=0, max_value=100, value=45)
n_procedures = st.sidebar.slider("Number of Procedures", min_value=0, max_value=10, value=6)
n_medications = st.sidebar.slider("Number of Medications", min_value=0, max_value=100, value=18)

n_outpatient = st.sidebar.number_input("Outpatient Visits", min_value=0, max_value=50, value=3)
n_emergency = st.sidebar.number_input("Emergency Visits", min_value=0, max_value=50, value=2)
n_inpatient = st.sidebar.number_input("Inpatient Visits", min_value=0, max_value=50, value=1)

medical_specialty = st.sidebar.selectbox("Medical Specialty", ['Cardiology', 'InternalMedicine', 'Surgery-General',
                                                               'Family/GeneralPractice', 'Emergency/Trauma'])

diag_1 = st.sidebar.text_input("Diagnosis 1", value='428.0')
diag_2 = st.sidebar.text_input("Diagnosis 2", value='250.00')
diag_3 = st.sidebar.text_input("Diagnosis 3", value='401.9')

glucose_test = st.sidebar.selectbox("Glucose Test Result", ['None', 'Normal', 'Abnormal'])
A1Ctest = st.sidebar.checkbox("A1C Test Performed?")
change = st.sidebar.checkbox("Change in Medications?")
diabetes_med = st.sidebar.checkbox("Diabetes Medication Given?")

# Format checkbox values
A1Ctest_val = "Yes" if A1Ctest else "No"
change_val = "Yes" if change else "No"
diabetes_med_val = "Yes" if diabetes_med else "No"

# Ordered input as expected by model
input_dict = {
    'time_in_hospital': time_in_hospital,
    'n_lab_procedures': n_lab_procedures,
    'n_procedures': n_procedures,
    'n_medications': n_medications,
    'n_outpatient': n_outpatient,
    'n_emergency': n_emergency,
    'n_inpatient': n_inpatient,
    'age': age,
    'medical_specialty': medical_specialty,
    'diag_1': diag_1,
    'diag_2': diag_2,
    'diag_3': diag_3,
    'glucose_test': glucose_test,
    'A1Ctest': A1Ctest_val,
    'change': change_val,
    'diabetes_med': diabetes_med_val
}

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Prediction section
st.title("🔍 ICU Readmission Prediction")

try:
    pool = Pool(input_df, cat_features=cat_features)
    proba = model.predict_proba(pool)[0][1]
    prediction = model.predict(pool)[0]

    if prediction == 1:
        st.error(f"🔴 Prediction: Patient is **likely to be readmitted** (Probability: {proba:.2f})")
    else:
        st.success(f"🟢 Prediction: Patient is **not likely to be readmitted** (Probability: {proba:.2f})")

except Exception as e:
    st.error(f"❌ Error during prediction: {e}")

# Explainability section
st.header("📊 Model Explainability")

# SHAP Explanation
try:
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st.subheader("🔎 SHAP Force Plot")
    st_shap = st.empty()
    st_shap.html(shap.force_plot(explainer.expected_value, shap_values, input_df).data, height=300)

except Exception as e:
    st.warning(f"⚠️ SHAP explanation not available: {e}")

# LIME Explanation
try:
    st.subheader("🔍 LIME Explanation")

    # Encoding categorical variables for LIME
    lime_input = pd.get_dummies(input_df)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=lime_input.values,
        feature_names=lime_input.columns,
        mode="classification"
    )

    exp = explainer.explain_instance(
        lime_input.values[0],
        model.predict_proba
    )

    st.components.v1.html(exp.as_html(), height=400)

except Exception as e:
    st.warning(f"⚠️ LIME explanation not available: {e}")
