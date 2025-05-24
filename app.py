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
    df = pd.read_csv("preprocessed_hospital_readmissions.csv")  # Relative path, no absolute Windows path!
    return df

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

# Main App
def main():
    st.set_page_config(page_title="ICU Readmission Predictor", layout="wide")
    st.title("🏥 ICU Readmission Prediction Dashboard")

    # Removed debug directory prints here

    if not os.path.exists("preprocessed_hospital_readmissions.csv"):
        st.error("❌ Dataset file 'preprocessed_hospital_readmissions.csv' NOT found! Please upload it alongside this script.")
        st.stop()

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
    user_input = {}
    user_input['age'] = st.sidebar.selectbox("Age Range", df['age'].unique())
    user_input['time_in_hospital'] = st.sidebar.slider("Time in hospital (days)", 1, 20, 5)
    user_input['n_lab_procedures'] = st.sidebar.slider("Number of lab procedures", 0, 100, 40)
    user_input['n_procedures'] = st.sidebar.slider("Number of procedures", 0, 10, 1)
    user_input['n_medications'] = st.sidebar.slider("Number of medications", 0, 80, 20)
    user_input['n_outpatient'] = st.sidebar.slider("Outpatient visits", 0, 20, 0)
    user_input['n_emergency'] = st.sidebar.slider("Emergency visits", 0, 10, 0)
    user_input['n_inpatient'] = st.sidebar.slider("Inpatient visits", 0, 20, 0)
    user_input['medical_specialty'] = st.sidebar.selectbox("Medical Specialty", df['medical_specialty'].dropna().unique())
    user_input['diag_1'] = st.sidebar.selectbox("Diagnosis 1", df['diag_1'].dropna().unique())
    user_input['diag_2'] = st.sidebar.selectbox("Diagnosis 2", df['diag_2'].dropna().unique())
    user_input['diag_3'] = st.sidebar.selectbox("Diagnosis 3", df['diag_3'].dropna().unique())
    user_input['glucose_test'] = st.sidebar.selectbox("Glucose Test", df['glucose_test'].unique())
    user_input['A1Ctest'] = st.sidebar.selectbox("A1C Test", df['A1Ctest'].unique())
    user_input['change'] = st.sidebar.selectbox("Medication Change", df['change'].unique())
    user_input['diabetes_med'] = st.sidebar.selectbox("Diabetes Medication", df['diabetes_med'].unique())

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
        try:
            X_lime = X_test.copy()
            for col in categorical_columns:
                X_lime[col] = X_lime[col].astype(str)

            # Sample for speed
            sampled_X_lime = X_lime.sample(n=min(500, len(X_lime)), random_state=42)

            lime_explainer = LimeTabularExplainer(
                training_data=np.array(sampled_X_lime),
                feature_names=sampled_X_lime.columns.tolist(),
                class_names=["Not Readmitted", "Readmitted"],
                categorical_features=[sampled_X_lime.columns.get_loc(col) for col in categorical_columns],
                mode='classification'
            )

            lime_input = input_df.copy()
            for col in categorical_columns:
                lime_input[col] = lime_input[col].astype(str)

            lime_exp = lime_explainer.explain_instance(
                data_row=lime_input.iloc[0].values,
                predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=input_df.columns)),
                num_features=5
            )
            fig = lime_exp.as_pyplot_figure()
            st.pyplot(fig)

        except Exception:
            st.warning("⚠️ LIME explanation is currently unavailable due to an internal error.")

    
if __name__ == "__main__":
    main()
