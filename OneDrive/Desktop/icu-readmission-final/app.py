# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ Load model
@st.cache_resource
def load_model():
    with open("catboost_model_smote_tomek.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ✅ Define categorical options (based on your training data)
categorical_features = {
    'age': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'],
    'medical_specialty': ['Cardiology', 'Endocrinology', 'Family/GeneralPractice', 'InternalMedicine', 'Other'],
    'diag_1': ['250.83', '250.13', '414', '428', '401', 'Other'],
    'diag_2': ['250.83', '250.13', '276', '428', 'Other'],
    'diag_3': ['250.83', '250.13', '401', 'Other'],
    'glucose_test': ['None', 'Norm', '>200', '>300'],
    'A1Ctest': ['None', 'Norm', '>7', '>8'],
    'change': ['Ch', 'No'],
    'diabetes_med': ['Yes', 'No']
}

# 🚪 Sidebar Inputs
st.sidebar.header("Enter Patient Info")
user_input = {}
for feature, options in categorical_features.items():
    user_input[feature] = st.sidebar.selectbox(f"{feature.replace('_', ' ').title()}", options)

# Convert input into DataFrame
input_df = pd.DataFrame([user_input])

# Convert to categorical
for col in input_df.columns:
    input_df[col] = input_df[col].astype("category")

# One-hot encode to match model
input_encoded = pd.get_dummies(input_df)

# Add missing columns if any (due to training/test difference)
model_columns = model.feature_names_
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# 🔍 Prediction
if st.sidebar.button("Predict Readmission"):
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High risk of readmission (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Low risk of readmission (Probability: {proba:.2f})")

    # ⬇ Download Result
    result_df = input_df.copy()
    result_df["Readmission Probability"] = proba
    result_df["Predicted Label"] = "Yes" if prediction == 1 else "No"

    csv = result_df.to_csv(index=False)
    st.download_button("📥 Download Result", data=csv, file_name="prediction.csv", mime="text/csv")

# 📂 Collapsible Data Info
with st.expander("🧾 Patient Input Summary"):
    st.dataframe(input_df)
