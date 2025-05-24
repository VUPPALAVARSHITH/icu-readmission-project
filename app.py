import streamlit as st
import pandas as pd
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from catboost import CatBoostClassifier

@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_hospital_readmissions.csv")

@st.cache_resource
def load_model():
    return joblib.load("catboost_model.pkl")

def align_input_with_training(input_df, reference_df):
    missing_cols = set(reference_df.columns) - set(input_df.columns)
    extra_cols = set(input_df.columns) - set(reference_df.columns)

    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df.drop(columns=extra_cols, errors='ignore')
    input_df = input_df[reference_df.columns]
    
    return input_df

def main():
    st.title("ICU Readmission Prediction")
    
    df = load_data()
    model = load_model()

    X_train = df.drop("readmitted", axis=1)
    y_train = df["readmitted"]

    user_input = {}
    for col in X_train.columns:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(col, options=df[col].unique())
        else:
            user_input[col] = st.number_input(col, value=float(df[col].mean()))

    input_df = pd.DataFrame([user_input])
    input_df = align_input_with_training(input_df, X_train)

    if st.button("Predict Readmission"):
        prediction = model.predict(input_df)[0]
        st.subheader(f"🔍 Prediction: {'Readmitted' if prediction == 1 else 'Not Readmitted'}")

        st.markdown("---")

        st.subheader("📈 SHAP Explanation")
        explainer_shap = shap.Explainer(model)
        shap_values = explainer_shap(input_df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(bbox_inches='tight')

        st.subheader("🟢 LIME Explanation")
        explainer_lime = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['No', 'Yes'], discretize_continuous=True)
        explanation = explainer_lime.explain_instance(input_df.values[0], model.predict_proba, num_features=10)
        st.pyplot(explanation.as_pyplot_figure())

if __name__ == "__main__":
    main()
