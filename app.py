import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import os
from lime.lime_tabular import LimeTabularExplainer
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_hospital_readmissions.csv")
    return df

@st.cache_resource
def load_model():
    model_path = "catboost_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        return None

df = load_data()
model = load_model()

# Define categorical columns
categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                      'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

# Split data if model exists
if model:
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Patient Prediction", "Model Analysis", "Data Exploration"])

# Main content
if page == "Patient Prediction":
    st.title("🏥 Hospital Readmission Risk Predictor")
    st.markdown("Predict the likelihood of a patient being readmitted to the hospital.")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.selectbox("Age Range", df['age'].unique())
            time_in_hospital = st.number_input("Time in hospital (days)", min_value=1, max_value=100, value=5)
            n_lab_procedures = st.number_input("Number of lab procedures", min_value=0, max_value=100, value=10)
            n_procedures = st.number_input("Number of procedures", min_value=0, max_value=100, value=5)
            n_medications = st.number_input("Number of medications", min_value=0, max_value=100, value=15)
            
        with col2:
            n_outpatient = st.number_input("Number of outpatient visits", min_value=0, max_value=100, value=0)
            n_emergency = st.number_input("Number of emergency visits", min_value=0, max_value=100, value=0)
            n_inpatient = st.number_input("Number of inpatient visits", min_value=0, max_value=100, value=0)
            medical_specialty = st.selectbox("Medical specialty", df['medical_specialty'].unique())
            glucose_test = st.selectbox("Glucose test result", df['glucose_test'].unique())
            
        diag_1 = st.text_input("Diagnosis 1 code", value="250.00")
        diag_2 = st.text_input("Diagnosis 2 code", value="250.00")
        diag_3 = st.text_input("Diagnosis 3 code", value="250.00")
        
        A1Ctest = st.selectbox("A1C test done?", ["Yes", "No"])
        change = st.selectbox("Change in medications?", ["Yes", "No"])
        diabetes_med = st.selectbox("Diabetes medication prescribed?", ["Yes", "No"])
        
        submitted = st.form_submit_button("Predict Readmission Risk")
    
    if submitted and model:
        user_input = {
            'age': age,
            'time_in_hospital': time_in_hospital,
            'n_lab_procedures': n_lab_procedures,
            'n_procedures': n_procedures,
            'n_medications': n_medications,
            'n_outpatient': n_outpatient,
            'n_emergency': n_emergency,
            'n_inpatient': n_inpatient,
            'medical_specialty': medical_specialty,
            'diag_1': diag_1,
            'diag_2': diag_2,
            'diag_3': diag_3,
            'glucose_test': glucose_test,
            'A1Ctest': A1Ctest,
            'change': change,
            'diabetes_med': diabetes_med
        }
        
        input_df = pd.DataFrame([user_input])
        for col in categorical_columns:
            input_df[col] = input_df[col].astype('category')
        input_df = input_df[X_train.columns]
        
        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error(f"🚨 High Risk of Readmission ({proba:.1%})")
        else:
            st.success(f"✅ Low Risk of Readmission ({proba:.1%})")
        
        # Explanation tabs
        tab1, tab2 = st.tabs(["SHAP Explanation", "LIME Explanation"])
        
        with tab1:
            st.subheader("SHAP Feature Importance")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
            st.pyplot(fig)
        
        with tab2:
            st.subheader("LIME Explanation")
            X_lime = X_test.copy()
            for col in categorical_columns:
                X_lime[col] = X_lime[col].astype(str)
            
            lime_explainer = LimeTabularExplainer(
                training_data=np.array(X_lime),
                feature_names=X_lime.columns.tolist(),
                class_names=["Not Readmitted", "Readmitted"],
                categorical_features=[X_lime.columns.get_loc(col) for col in categorical_columns],
                mode='classification'
            )
            
            lime_input = input_df.copy()
            for col in categorical_columns:
                lime_input[col] = lime_input[col].astype(str)
            
            lime_exp = lime_explainer.explain_instance(
                data_row=lime_input.iloc[0].values,
                predict_fn=lambda x: model.predict_proba(pd.DataFrame(x, columns=input_df.columns))
            )
            
            fig = lime_exp.as_pyplot_figure()
            plt.title("LIME Explanation")
            plt.tight_layout()
            st.pyplot(fig)

elif page == "Model Analysis":
    st.title("Model Performance Analysis")
    
    if model:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.table(pd.DataFrame(report).transpose())
            
            st.subheader("ROC AUC Score")
            st.metric("AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
            
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
        st.subheader("SHAP Summary Plot")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
    else:
        st.warning("Model not found. Please train the model first.")

elif page == "Data Exploration":
    st.title("Data Exploration")
    
    st.subheader("Dataset Overview")
    st.write(f"Total records: {len(df)}")
    st.dataframe(df.head())
    
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature to visualize", df.columns)
    
    fig, ax = plt.subplots()
    if df[feature].dtype == 'object' or df[feature].nunique() < 10:
        df[feature].value_counts().plot(kind='bar', ax=ax)
    else:
        df[feature].plot(kind='hist', bins=30, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    df['readmitted'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)