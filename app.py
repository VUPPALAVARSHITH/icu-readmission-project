import streamlit as st
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import pickle
import numpy as np
import base64
import os

# -------------------- Caching --------------------
@st.cache_resource
def load_model():
    with open("catboost_model_smote_tomek.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("hospital_readmissions.csv").dropna(subset=["readmitted"])
    return df.drop("readmitted", axis=1)

# -------------------- App Layout --------------------
st.set_page_config(page_title="ICU Readmission Predictor", layout="wide")
st.title("🏥 ICU Readmission Risk Dashboard")

model = load_model()
X = load_data()

categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                    'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("📋 Patient Information")

age = st.sidebar.selectbox("Age", sorted(X["age"].unique()))
time_in_hospital = st.sidebar.slider("Time in hospital", 1, 30, 5)
n_lab_procedures = st.sidebar.number_input("Number of lab procedures", min_value=0, value=30)
n_procedures = st.sidebar.number_input("Number of procedures", min_value=0, value=1)
n_medications = st.sidebar.number_input("Number of medications", min_value=0, value=10)
n_outpatient = st.sidebar.number_input("Outpatient visits", min_value=0, value=0)
n_emergency = st.sidebar.number_input("Emergency visits", min_value=0, value=0)
n_inpatient = st.sidebar.number_input("Inpatient visits", min_value=0, value=0)
medical_specialty = st.sidebar.selectbox("Medical Specialty", sorted(X["medical_specialty"].dropna().unique()))
diag_1 = st.sidebar.text_input("Diagnosis 1", "428")
diag_2 = st.sidebar.text_input("Diagnosis 2", "250.02")
diag_3 = st.sidebar.text_input("Diagnosis 3", "401.9")
glucose_test = st.sidebar.selectbox("Glucose Test Result", ["Normal", "Abnormal"])
A1Ctest = st.sidebar.selectbox("A1C Test Done?", ["Yes", "No"])
change = st.sidebar.selectbox("Change in Medication?", ["Yes", "No"])
diabetes_med = st.sidebar.selectbox("Diabetes Medication Prescribed?", ["Yes", "No"])

# -------------------- Prediction --------------------
input_data = pd.DataFrame([{
    'age': age, 'time_in_hospital': time_in_hospital, 'n_lab_procedures': n_lab_procedures,
    'n_procedures': n_procedures, 'n_medications': n_medications, 'n_outpatient': n_outpatient,
    'n_emergency': n_emergency, 'n_inpatient': n_inpatient, 'medical_specialty': medical_specialty,
    'diag_1': diag_1, 'diag_2': diag_2, 'diag_3': diag_3, 'glucose_test': glucose_test,
    'A1Ctest': A1Ctest, 'change': change, 'diabetes_med': diabetes_med
}])

for col in categorical_cols:
    input_data[col] = input_data[col].astype('category')
    X[col] = X[col].astype('category')

input_data = input_data[X.columns]

proba = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]
status_icon = "🟢" if prediction == 1 else "🔴"
label = "Readmitted" if prediction == 1 else "Not Readmitted"

# -------------------- Output --------------------
st.subheader("📊 Prediction Result")
st.markdown(f"### {status_icon} {label}")
st.markdown(f"**Risk Score:** `{proba:.2f}`")

# -------------------- SHAP + LIME Tabs --------------------
tabs = st.tabs(["🧠 SHAP Explanation", "🔍 LIME Explanation"])

with tabs[0]:
    with st.expander("📈 SHAP Force Plot", expanded=False):
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        st_shap = shap.force_plot(explainer.expected_value, shap_values, input_data)
        shap.save_html("shap_explanation.html", st_shap)
        st.components.v1.html(open("shap_explanation.html").read(), height=300, scrolling=True)

with tabs[1]:
    with st.expander("📊 LIME Explanation", expanded=False):
        X_lime = X.copy()
        for col in categorical_cols:
            X_lime[col] = X_lime[col].astype(str)

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_lime),
            feature_names=X_lime.columns.tolist(),
            class_names=["Not Readmitted", "Readmitted"],
            categorical_features=[X_lime.columns.get_loc(c) for c in categorical_cols],
            mode='classification'
        )

        input_lime = input_data.copy()
        for col in categorical_cols:
            input_lime[col] = input_lime[col].astype(str)

        lime_exp = lime_explainer.explain_instance(
            data_row=input_lime.iloc[0].values,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=input_data.columns))
        )

        fig = lime_exp.as_pyplot_figure()
        fig.savefig("lime_explanation.png", bbox_inches="tight")
        st.image("lime_explanation.png", caption="LIME Explanation", use_column_width=True)

        # Download button
        with open("lime_explanation.png", "rb") as f:
            btn = st.download_button(
                label="📥 Download LIME Explanation",
                data=f,
                file_name="lime_explanation.png",
                mime="image/png"
            )
