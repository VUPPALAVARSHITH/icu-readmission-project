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

# 🧾 Sidebar Input
st.sidebar.header("Patient Input")
user_input = {
    'age': st.sidebar.selectbox("Age Range", sorted(X_train['age'].unique())),
    'time_in_hospital': st.sidebar.slider("Time in Hospital (days)", 1, 30, 6),
    'n_lab_procedures': 41,
    'n_procedures': 2,
    'n_medications': 12,
    'n_outpatient': 0,
    'n_emergency': 1,
    'n_inpatient': 0,
    'medical_specialty': 'Cardiology',
    'diag_1': '428',
    'diag_2': '250.02',
    'diag_3': '401.9',
    'glucose_test': 'Normal',
    'A1Ctest': 'Yes',
    'change': 'No',
    'diabetes_med': 'Yes'
}

# 🔄 Format input
input_df = pd.DataFrame([user_input])
for col in categorical_cols:
    input_df[col] = input_df[col].astype('category')
input_df = input_df[X_train.columns]

# 🔮 Prediction
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]
label = "✅ Not Readmitted" if prediction == 0 else "🔴 Readmitted"

# 📢 Display prediction result
st.markdown("## 🏥 ICU Readmission Prediction")
st.markdown(f"### Prediction: {label}")
st.markdown(f"### Risk Score: **{proba:.2f}**")

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

