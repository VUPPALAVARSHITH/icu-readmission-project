import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_hospital_readmissions.csv")
    return df

@st.cache_data
def preprocess_data(df):
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    # One-hot encode categorical features
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Merge numeric and encoded features
    X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
    X_final = pd.concat([X_numeric, X_encoded_df], axis=1)

    return X_final, y, encoder

def main():
    st.title("ICU Readmission Risk Predictor")

    df = load_data()
    X, y, encoder = preprocess_data(df)

    # Load model
    model = joblib.load("catboost_model_smote_tomek.pkl")

    # User input
    st.sidebar.header("Patient Data Input")
    input_data = {}
    for col in df.columns[:-1]:
        input_data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
    input_df = pd.DataFrame([input_data])

    # Encode input
    input_encoded = encoder.transform(input_df[encoder.feature_names_in_])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())

    input_numeric = input_df.drop(columns=encoder.feature_names_in_).reset_index(drop=True)
    input_full = pd.concat([input_numeric, input_encoded_df], axis=1)

    # Ensure all columns align
    input_full = input_full.reindex(columns=X.columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_full)[0]
    st.subheader("Prediction:")
    st.success("Patient will be readmitted." if prediction == 1 else "Patient will not be readmitted.")

    # SHAP
    st.subheader("SHAP Explanation:")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_full)
    st.pyplot(shap.plots.waterfall(shap_values[0], show=False))

    # LIME
    st.subheader("LIME Explanation:")
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=X.columns.tolist(),
        class_names=["No Readmit", "Readmit"],
        mode="classification"
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=input_full.iloc[0].values,
        predict_fn=model.predict_proba
    )
    st.components.v1.html(lime_exp.as_html(), height=800, scrolling=True)

if __name__ == "__main__":
    main()
