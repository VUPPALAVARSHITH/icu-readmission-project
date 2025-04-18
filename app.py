import streamlit as st
import pandas as pd
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import base64

# Streamlit page config
st.set_page_config(page_title="ICU Readmission Predictor", layout="wide")

# 📦 Load model and columns with caching
@st.cache_resource
def load_model():
    with open("catboost_model_smote_tomek.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_columns():
    df = pd.read_csv("hospital_readmissions.csv")
    df = df.dropna(subset=["readmitted"])
    X = df.drop("readmitted", axis=1)
    categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                        'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
    return X, categorical_cols

model = load_model()
X_train, categorical_cols = load_columns()

# Initialize session state for prediction and refresh
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'refresh' not in st.session_state:
    st.session_state.refresh = False

# 🧾 Sidebar Input
st.sidebar.header("Patient Input")

# Create a form for all inputs
with st.sidebar.form("patient_form"):
    user_input = {
        'age': st.selectbox("Age Range", sorted(X_train['age'].unique())),
        'time_in_hospital': st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=6),
        'n_lab_procedures': st.number_input("Number of Lab Procedures", min_value=0, value=41),
        'n_procedures': st.number_input("Number of Procedures", min_value=0, value=2),
        'n_medications': st.number_input("Number of Medications", min_value=0, value=12),
        'n_outpatient': st.number_input("Number of Outpatient Visits", min_value=0, value=0),
        'n_emergency': st.number_input("Number of Emergency Visits", min_value=0, value=1),
        'n_inpatient': st.number_input("Number of Inpatient Visits", min_value=0, value=0),
        'medical_specialty': st.selectbox("Medical Specialty", sorted(X_train['medical_specialty'].unique())),
        'diag_1': st.text_input("Diagnosis 1 Code", value='428'),
        'diag_2': st.text_input("Diagnosis 2 Code", value='250.02'),
        'diag_3': st.text_input("Diagnosis 3 Code", value='401.9'),
        'glucose_test': st.selectbox("Glucose Test", ['Normal', 'Abnormal']),
        'A1Ctest': st.selectbox("A1C Test", ['Yes', 'No']),
        'change': st.selectbox("Change in Medications", ['Yes', 'No']),
        'diabetes_med': st.selectbox("Diabetes Medication", ['Yes', 'No'])
    }
    
    # Create two columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        predict_button = st.form_submit_button("Predict")
    
    with col2:
        refresh_button = st.form_submit_button("Refresh")

# Handle refresh
if refresh_button:
    st.session_state.prediction_made = False
    st.session_state.refresh = True
    st.experimental_rerun()

# Handle prediction
if predict_button and not st.session_state.refresh:
    st.session_state.prediction_made = True
    st.session_state.refresh = False
    
    # 🔄 Format input
    input_df = pd.DataFrame([user_input])
    for col in categorical_cols:
        input_df[col] = input_df[col].astype('category')
    input_df = input_df[X_train.columns]

    # 🔮 Prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.session_state.prediction = prediction
    st.session_state.proba = proba

# 📢 Display prediction result only after prediction is made
if st.session_state.prediction_made:
    st.markdown("## 🏥 ICU Readmission Prediction")
    label = "✅ The patient is not likely to be readmitted" if st.session_state.prediction == 0 else "🔴 The patient is likely to be readmitted"
    st.markdown(f"### {label}")
    st.markdown(f"### Risk Score: **{st.session_state.proba:.2f}**")

    # 📊 SHAP Explanation Section
    with st.expander("📈 SHAP Explanation"):
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        shap_html = shap.force_plot(explainer.expected_value, shap_values, input_df)
        shap.save_html("shap_explanation.html", shap_html)

        with open("shap_explanation.html", "rb") as f:
            btn = st.download_button("⬇️ Download SHAP Explanation (HTML)",
                                    data=f,
                                    file_name="shap_explanation.html",
                                    mime="text/html")

    # 🔍 LIME Explanation Section
    with st.expander("🔍 LIME Explanation"):
        X_lime = X_train.copy()
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
        st.image("lime_explanation.png", caption="LIME Explanation", use_column_width=True)

        with open("lime_explanation.png", "rb") as f:
            st.download_button("⬇️ Download LIME Explanation (PNG)", f, "lime_explanation.png", "image/png")
else:
    st.info("Please fill in the patient details and click 'Predict' to see results.")