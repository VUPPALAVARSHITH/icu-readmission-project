import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
import pickle
import numpy as np
import streamlit as st

# Load model
with open("catboost_model_smote_tomek.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get column order
df = pd.read_csv("preprocessed_hospital_readmissions.csv")
df = df.dropna(subset=["readmitted"])

# Select the same features used in training
categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                    'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

X = df.drop("readmitted", axis=1)
for col in categorical_cols:
    X[col] = X[col].astype('category')

# Collect user input dynamically
user_input = {
    'age': st.sidebar.selectbox('Age', ['60-70', '70-80', '80-90']),
    'time_in_hospital': st.sidebar.slider('Time in hospital', 1, 100, 10),
    'n_lab_procedures': st.sidebar.slider('Number of lab procedures', 1, 100, 45),
    'n_procedures': st.sidebar.slider('Number of procedures', 1, 20, 6),
    'n_medications': st.sidebar.slider('Number of medications', 1, 50, 18),
    'n_outpatient': st.sidebar.slider('Number of outpatient visits', 1, 10, 3),
    'n_emergency': st.sidebar.slider('Number of emergency visits', 1, 10, 2),
    'n_inpatient': st.sidebar.slider('Number of inpatient visits', 1, 10, 1),
    'medical_specialty': st.sidebar.selectbox('Medical Specialty', ['Cardiology', 'Endocrinology', 'Neurology']),
    'diag_1': st.sidebar.text_input('Diagnosis 1', '428.0'),
    'diag_2': st.sidebar.text_input('Diagnosis 2', '250.00'),
    'diag_3': st.sidebar.text_input('Diagnosis 3', '401.9'),
    'glucose_test': st.sidebar.selectbox('Glucose Test Result', ['Normal', 'Abnormal']),
    'A1Ctest': st.sidebar.selectbox('A1C Test', ['Yes', 'No']),
    'change': st.sidebar.selectbox('Medication Change', ['Yes', 'No']),
    'diabetes_med': st.sidebar.selectbox('Diabetes Med Prescribed?', ['Yes', 'No'])
}

# Format user input as a DataFrame
input_df = pd.DataFrame([user_input])
for col in categorical_cols:
    input_df[col] = input_df[col].astype('category')

# Ensure same order of columns as training data
input_df = input_df[X.columns]

# Create a Pool for prediction (required for categorical features)
pool = Pool(input_df, cat_features=categorical_cols)

# Predict
proba = model.predict_proba(pool)[0][1]
prediction = model.predict(pool)[0]
label = "Readmitted" if prediction == 1 else "Not Readmitted"

st.write(f"🚨 Prediction: {label}")
st.write(f"📊 Risk Score: {proba:.2f}")

# ============== SHAP Explanation (HTML Export) ==============
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

# Save SHAP force plot as HTML
shap_html = shap.force_plot(explainer.expected_value, shap_values, input_df)
shap.save_html("shap_explanation.html", shap_html)
st.write("✅ SHAP force plot saved.")

# ============== LIME Explanation (PNG Export) ==============
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

# Save as PNG
fig = lime_exp.as_pyplot_figure()
fig.savefig("lime_explanation.png", bbox_inches="tight")
plt.close()
st.write("✅ LIME explanation saved.")
