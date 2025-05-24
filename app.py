import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_hospital_readmissions.csv")

# Load or train model
@st.cache_resource
def load_model(X_train, y_train, X_test, y_test, categorical_columns):
    try:
        with open("catboost_model.pkl", "rb") as f:
            model = pickle.load(f)
    except:
        train_pool = Pool(X_train, y_train, cat_features=categorical_columns)
        test_pool = Pool(X_test, y_test, cat_features=categorical_columns)
        model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                   eval_metric='AUC', random_seed=42, early_stopping_rounds=50, verbose=False)
        model.fit(train_pool, eval_set=test_pool)
        with open("catboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
    return model

# Main app
def main():
    st.set_page_config(page_title="ICU Readmission Predictor", layout="wide")
    st.title("🏥 ICU Readmission Prediction Dashboard")

    df = load_data()

    # Data prep
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                           'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

    for col in categorical_columns:
        X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = load_model(X_train, y_train, X_test, y_test, categorical_columns)

    st.sidebar.header("📋 Enter Patient Details")

    # Sidebar Inputs
    user_input = {
        'age': st.sidebar.selectbox("Age Range", sorted(df['age'].unique())),
        'time_in_hospital': st.sidebar.slider("Time in hospital (days)", 1, 20, 5),
        'n_lab_procedures': st.sidebar.slider("Number of lab procedures", 0, 100, 40),
        'n_procedures': st.sidebar.slider("Number of procedures", 0, 10, 1),
        'n_medications': st.sidebar.slider("Number of medications", 0, 80, 20),
        'n_outpatient': st.sidebar.slider("Outpatient visits", 0, 20, 0),
        'n_emergency': st.sidebar.slider("Emergency visits", 0, 10, 0),
        'n_inpatient': st.sidebar.slider("Inpatient visits", 0, 20, 0),
        'medical_specialty': st.sidebar.selectbox("Medical Specialty", sorted(df['medical_specialty'].dropna().unique())),
        'diag_1': st.sidebar.selectbox("Diagnosis 1 code", sorted(df['diag_1'].dropna().unique())),
        'diag_2': st.sidebar.selectbox("Diagnosis 2 code", sorted(df['diag_2'].dropna().unique())),
        'diag_3': st.sidebar.selectbox("Diagnosis 3 code", sorted(df['diag_3'].dropna().unique())),
        'glucose_test': st.sidebar.selectbox("Glucose Test", sorted(df['glucose_test'].unique())),
        'A1Ctest': st.sidebar.selectbox("A1C Test", sorted(df['A1Ctest'].unique())),
        'change': st.sidebar.selectbox("Medication Change", sorted(df['change'].unique())),
        'diabetes_med': st.sidebar.selectbox("Diabetes Medication", sorted(df['diabetes_med'].unique()))
    }

    input_df = pd.DataFrame([user_input])
    for col in categorical_columns:
        input_df[col] = input_df[col].astype('category')
    input_df = input_df[X_train.columns]

    # Make prediction
    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("📍 Prediction Result")
    st.write(f"*Prediction:* {'🟥 Readmitted' if prediction == 1 else '🟩 Not Readmitted'}")
    st.write(f"*Risk Score:* {proba:.2f}")

    # SHAP Explanation
    with st.expander("📊 SHAP Explanation"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig)

    # LIME Explanation
    with st.expander("🔍 LIME Explanation"):
        # Use raw numeric/categorical test data
        lime_explainer = LimeTabularExplainer(
            training_data=X_test.values,
            feature_names=X_test.columns.tolist(),
            class_names=["Not Readmitted", "Readmitted"],
            categorical_features=[X_test.columns.get_loc(col) for col in categorical_columns],
            mode='classification'
        )

        lime_input = input_df.copy()
        lime_input = lime_input.astype(X_test.dtypes)  # match dtypes

        lime_exp = lime_explainer.explain_instance(
            data_row=lime_input.iloc[0].values,
            predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=input_df.columns))
        )
        st.pyplot(lime_exp.as_pyplot_figure())

    # Model Evaluation
    with st.expander("📈 Model Evaluation Metrics"):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        st.write("*Classification Report:*")
        st.text(classification_report(y_test, y_pred))
        st.write(f"*ROC AUC Score:* {roc_auc_score(y_test, y_proba):.2f}")
        st.write(f"*Accuracy:* {accuracy_score(y_test, y_pred):.2f}")
        st.write("*Confusion Matrix:*")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

if __name__ == "__main__":
    main()
