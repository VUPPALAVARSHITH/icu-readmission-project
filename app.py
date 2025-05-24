import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import joblib
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# Load model
@st.cache_resource
def load_model():
    return CatBoostClassifier().load_model("catboost_model.cbm", format="cbm")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_hospital_readmissions.csv")

# Main app
def main():
    st.set_page_config(page_title="ICU Readmission Predictor", layout="wide")
    st.title("🏥 ICU Readmission Risk Predictor")
    st.write("Upload or select values to predict the probability of ICU readmission and explain predictions.")

    model = load_model()
    df = load_data()

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df.select_dtypes(exclude=['object']).drop(columns=['readmitted'], errors='ignore').columns.tolist()
    input_data = {}

    st.subheader("🔧 Input Features")
    with st.form("input_form"):
        for col in numerical_columns:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].median()))
        for col in categorical_columns:
            input_data[col] = st.selectbox(f"{col}", sorted(df[col].dropna().unique()))
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([input_data])

        # Align categories with training data
        for col in categorical_columns:
            input_df[col] = pd.Categorical(input_df[col], categories=df[col].unique())

        # Reorder and fill missing columns if any
        X = df.drop(columns=["readmitted"]) if "readmitted" in df.columns else df
        X = pd.concat([X, input_df], axis=0)
        X = pd.get_dummies(X)
        input_df_encoded = X.tail(1)
        X = X.head(-1)

        # Align model features
        model_features = model.feature_names_
        for feat in model_features:
            if feat not in input_df_encoded.columns:
                input_df_encoded[feat] = 0
        input_df_encoded = input_df_encoded[model_features]

        # Prediction
        prediction = model.predict(input_df_encoded)[0]
        probability = model.predict_proba(input_df_encoded)[0][1]

        st.subheader("🧠 Prediction")
        st.write(f"**Prediction:** {'Readmitted' if prediction == 1 else 'Not Readmitted'}")
        st.write(f"**Probability of Readmission:** {probability:.2f}")

        # SHAP Explanation
        with st.expander("📊 SHAP Explanation"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df_encoded)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_values, input_df_encoded, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')

        # LIME Explanation
        with st.expander("🔍 LIME Explanation"):
            # Prepare data
            X_test = df.drop(columns=["readmitted"]) if "readmitted" in df.columns else df
            X_lime = X_test.copy()
            input_lime = input_df.copy()

            le_dict = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X_lime[col] = le.fit_transform(X_lime[col].astype(str))
                input_lime[col] = le.transform(input_lime[col].astype(str))
                le_dict[col] = le

            lime_explainer = LimeTabularExplainer(
                training_data=np.array(X_lime),
                feature_names=X_lime.columns.tolist(),
                class_names=["Not Readmitted", "Readmitted"],
                categorical_features=[X_lime.columns.get_loc(col) for col in categorical_columns],
                mode='classification'
            )

            lime_exp = lime_explainer.explain_instance(
                data_row=input_lime.iloc[0].values,
                predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=input_lime.columns))
            )
            fig = lime_exp.as_pyplot_figure()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
