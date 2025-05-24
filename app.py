import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
import os

st.set_page_config(page_title="ICU Readmission Predictor", layout="centered")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_hospital_readmissions.csv")

# Preprocess data
def preprocess_data(df):
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")  # Use 'sparse=False' if sklearn < 1.2
    X_encoded = encoder.fit_transform(X[categorical_features])
    encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
    X_final = pd.concat([X_numeric.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    return X_final, y, encoder

# Load model
@st.cache_resource
def load_model():
    return joblib.load("catboost_model_smote_tomek.pkl")

# SHAP explanation
def shap_explanation(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    st.subheader("SHAP Explanation")
    st.pyplot(shap.plots.waterfall(shap_values[0], show=False))

# LIME explanation
def lime_explanation(model, X_train, X_sample, feature_names):
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=["Not Readmitted", "Readmitted"],
        mode='classification'
    )
    explanation = lime_explainer.explain_instance(
        data_row=X_sample[0],
        predict_fn=model.predict_proba
    )
    st.subheader("LIME Explanation")
    st.components.v1.html(explanation.as_html(), height=800, scrolling=True)

def main():
    st.title("🏥 ICU Readmission Risk Predictor")

    df = load_data()
    X, y, encoder = preprocess_data(df)
    model = load_model()

    st.write("### Enter Patient Details")
    user_input = {}
    for column in df.drop("readmitted", axis=1).columns:
        if df[column].dtype == 'object':
            user_input[column] = st.selectbox(f"{column}:", df[column].unique())
        else:
            user_input[column] = st.number_input(f"{column}:", float(df[column].min()), float(df[column].max()), float(df[column].mean()))

    user_df = pd.DataFrame([user_input])
    user_categorical = user_df.select_dtypes(include=["object"])
    user_encoded = encoder.transform(user_categorical)
    user_encoded_df = pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(user_categorical.columns))
    user_numeric = user_df.drop(columns=user_categorical.columns)
    user_final = pd.concat([user_numeric.reset_index(drop=True), user_encoded_df.reset_index(drop=True)], axis=1)

    if st.button("Predict"):
        prediction = model.predict(user_final)[0]
        probability = model.predict_proba(user_final)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The patient is likely to be readmitted. (Risk Score: {probability:.2f})")
        else:
            st.success(f"✅ The patient is unlikely to be readmitted. (Risk Score: {probability:.2f})")

        # SHAP and LIME
        shap_explanation(model, user_final)
        lime_explanation(model, X, user_final.to_numpy(), user_final.columns.tolist())

if __name__ == "__main__":
    main()
