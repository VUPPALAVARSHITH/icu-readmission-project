import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pickle
from catboost import CatBoostClassifier

# ========== Caching ==========
@st.cache_resource
def load_model():
    with open("catboost_model_smote_tomek.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\varsh\OneDrive\Desktop\minor_project\preprocessed_hospital_readmissions.csv")
    df = df.dropna(subset=["readmitted"])
    X = df.drop("readmitted", axis=1)
    return df, X

# ========== Load Model and Data ==========
model = load_model()
df, X_train = load_data()

categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                    'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')

# ========== Sidebar Input ==========
st.sidebar.header("🧾 Enter Patient Details")

user_input = {
    'age': st.sidebar.selectbox("Age", sorted(X_train['age'].unique())),
    'time_in_hospital': st.sidebar.slider("Time in hospital", 1, 30, 6),
    'n_lab_procedures': st.sidebar.slider("Lab Procedures", 0, 150, 41),
    'n_procedures': st.sidebar.slider("Procedures", 0, 10, 2),
    'n_medications': st.sidebar.slider("Medications", 0, 100, 12),
    'n_outpatient': st.sidebar.slider("Outpatient visits", 0, 10, 0),
    'n_emergency': st.sidebar.slider("Emergency visits", 0, 5, 1),
    'n_inpatient': st.sidebar.slider("Inpatient visits", 0, 10, 0),
    'medical_specialty': st.sidebar.selectbox("Medical Specialty", sorted(X_train['medical_specialty'].unique())),
    'diag_1': st.sidebar.selectbox("Diagnosis 1", sorted(X_train['diag_1'].unique())),
    'diag_2': st.sidebar.selectbox("Diagnosis 2", sorted(X_train['diag_2'].unique())),
    'diag_3': st.sidebar.selectbox("Diagnosis 3", sorted(X_train['diag_3'].unique())),
    'glucose_test': st.sidebar.selectbox("Glucose Test", ["Normal", "Abnormal"]),
    'A1Ctest': st.sidebar.selectbox("A1C Test", ["Yes", "No"]),
    'change': st.sidebar.selectbox("Medication Change", ["Yes", "No"]),
    'diabetes_med': st.sidebar.selectbox("Diabetes Med", ["Yes", "No"])
}

# ========== Format Input ==========
input_df = pd.DataFrame([user_input])
for col in categorical_cols:
    input_df[col] = input_df[col].astype('category')
input_df = input_df[X_train.columns]

# ========== Prediction ==========
st.title("🏥 ICU Readmission Prediction")

with st.expander("🔮 Prediction Result"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.markdown("🔴 **Prediction: Patient is likely to be readmitted.**")
        st.markdown(f"📊 **Risk Score:** {proba:.2f}")
    else:
        st.markdown("🟢 **Prediction: Patient is NOT likely to be readmitted.**")
        st.markdown(f"📊 **Risk Score:** {proba:.2f}")

# ========== SHAP Explanation ==========
with st.expander("📈 SHAP Explanation"):
    try:
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        st.markdown("#### SHAP Force Plot:")
        st_shap = shap.force_plot(explainer.expected_value, shap_values, input_df)
        shap.save_html("shap_explanation.html", st_shap)
        with open("shap_explanation.html", "rb") as f:
            st.download_button("⬇️ Download SHAP (HTML)", data=f, file_name="shap_explanation.html", mime="text/html")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

# ========== LIME Explanation ==========
with st.expander("🔍 LIME Explanation"):
    try:
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
            st.download_button("⬇️ Download LIME (PNG)", f, "lime_explanation.png", "image/png")

    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
