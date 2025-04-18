import streamlit as st
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import pickle
import numpy as np
import os

# Load model
with open("catboost_model_smote_tomek.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get column order
df = pd.read_csvdf = pd.read_csv("preprocessed_hospital_readmissions.csv")

df = df.dropna(subset=["readmitted"])

# Select the same features used in training
categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                    'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

X = df.drop("readmitted", axis=1)
for col in categorical_cols:
    X[col] = X[col].astype('category')

# ========== Sidebar Layout ==========
with st.sidebar:
    st.header("🧾 Enter patient details:")

    # Grouping inputs with logical components
    st.subheader("🔢 Age & Time in Hospital")
    age = st.selectbox("Age Group", options=['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+'], index=5)
    time_in_hospital = st.number_input("Time in hospital (days)", min_value=0, max_value=100, value=10)

    st.subheader("🧑‍⚕️ Medical Info")
    medical_specialty = st.text_input("Medical specialty", value='Cardiology')
    diag_1 = st.text_input("Diagnosis 1 (ICD-9 Code)", value='428.0')
    diag_2 = st.text_input("Diagnosis 2 (ICD-9 Code)", value='250.00')
    diag_3 = st.text_input("Diagnosis 3 (ICD-9 Code)", value='401.9')

    st.subheader("💉 Medical Tests & Medications")
    glucose_test = st.selectbox("Glucose Test Result", options=['Normal', 'Abnormal'], index=1)
    A1Ctest = st.selectbox("A1C Test (Yes/No)", options=['Yes', 'No'], index=0)
    change = st.selectbox("Medication Change (Yes/No)", options=['Yes', 'No'], index=0)
    diabetes_med = st.selectbox("Diabetes Med Prescribed? (Yes/No)", options=['Yes', 'No'], index=0)

    st.subheader("⚕️ Procedures & Visits")
    n_lab_procedures = st.number_input("Number of lab procedures", min_value=0, value=45)
    n_procedures = st.number_input("Number of procedures", min_value=0, value=6)
    n_medications = st.number_input("Number of medications", min_value=0, value=18)
    n_outpatient = st.number_input("Number of outpatient visits", min_value=0, value=3)
    n_emergency = st.number_input("Number of emergency visits", min_value=0, value=2)
    n_inpatient = st.number_input("Number of inpatient visits", min_value=0, value=1)

# Format user input into a dataframe
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

input_df = pd.DataFrame([user_input])
for col in categorical_cols:
    input_df[col] = input_df[col].astype('category')

# Ensure same order of columns as training data
input_df = input_df[X.columns]

# Predict
proba = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]
label = "Readmitted" if prediction == 1 else "Not Readmitted"

# Display Prediction
st.header("🔴 Prediction: Patient is likely to be readmitted" if label == "Readmitted" else "✅ Prediction: Patient is not likely to be readmitted")
st.write(f"📊 Risk Score: {proba:.2f}")

# ========== SHAP Explanation ==========
with st.expander("📈 SHAP Explanation"):
    try:
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # Save SHAP force plot as HTML
        shap_html = shap.force_plot(explainer.expected_value, shap_values, input_df)
        shap.save_html("shap_explanation.html", shap_html)

        if os.path.exists("shap_explanation.html"):
            st.markdown("#### SHAP Explanation:")
            st.components.v1.html(open("shap_explanation.html", "r").read(), width=800, height=400)
            with open("shap_explanation.html", "rb") as f:
                st.download_button("⬇️ Download SHAP (HTML)", f, "shap_explanation.html", "application/html")
        else:
            st.error("SHAP explanation not generated!")

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

# ========== LIME Explanation ==========
with st.expander("🔍 LIME Explanation"):
    try:
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

        input_lime = input_df.copy()
        for col in categorical_cols:
            input_lime[col] = input_lime[col].astype(str)

        lime_exp = lime_explainer.explain_instance(
            data_row=input_lime.iloc[0].values,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=input_df.columns))
        )

        fig = lime_exp.as_pyplot_figure()
        fig.savefig("lime_explanation.png", bbox_inches="tight")
        
        if os.path.exists("lime_explanation.png"):
            st.image("lime_explanation.png", caption="LIME Explanation", use_column_width=True)
            with open("lime_explanation.png", "rb") as f:
                st.download_button("⬇️ Download LIME (PNG)", f, "lime_explanation.png", "image/png")
        else:
            st.error("LIME explanation image not generated!")

    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
