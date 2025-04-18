import streamlit as st
import pandas as pd
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import os
from catboost import CatBoostClassifier, Pool

# Streamlit page config
st.set_page_config(page_title="ICU Readmission Predictor", layout="wide")

# Load model and data with proper error handling
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("catboost_model_smote_tomek.pkl"):
            st.error("Model file not found. Please ensure 'catboost_model_smote_tomek.pkl' exists.")
            st.stop()
            
        with open("catboost_model_smote_tomek.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

@st.cache_data
def load_columns():
    try:
        if not os.path.exists("preprocessed_hospital_readmissions.csv"):
            st.error("Data file not found. Please ensure 'preprocessed_hospital_readmissions.csv' exists.")
            st.stop()
            
        df = pd.read_csv("preprocessed_hospital_readmissions.csv")
        
        # Define and process categorical columns
        categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                          'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
        
        # Convert to string and handle missing values
        for col in categorical_cols:
            df[col] = df[col].fillna('').astype(str)
            
        return df.drop("readmitted", axis=1), categorical_cols
        
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

# Load model and data
model = load_model()
X_train, categorical_cols = load_columns()

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'refresh' not in st.session_state:
    st.session_state.refresh = False

# Sidebar Input Form
with st.sidebar.form("patient_form"):
    st.header("Patient Input")
    
    # Get unique categories from training data
    def get_categories(col):
        return sorted(X_train[col].astype(str).unique())
    
    user_input = {
        'age': st.selectbox("Age Range", get_categories('age')),
        'time_in_hospital': st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=6),
        'n_lab_procedures': st.number_input("Number of Lab Procedures", min_value=0, value=41),
        'n_procedures': st.number_input("Number of Procedures", min_value=0, value=2),
        'n_medications': st.number_input("Number of Medications", min_value=0, value=12),
        'n_outpatient': st.number_input("Number of Outpatient Visits", min_value=0, value=0),
        'n_emergency': st.number_input("Number of Emergency Visits", min_value=0, value=1),
        'n_inpatient': st.number_input("Number of Inpatient Visits", min_value=0, value=0),
        'medical_specialty': st.selectbox("Medical Specialty", get_categories('medical_specialty')),
        'diag_1': st.text_input("Diagnosis 1 Code", value='428'),
        'diag_2': st.text_input("Diagnosis 2 Code", value='250.02'),
        'diag_3': st.text_input("Diagnosis 3 Code", value='401.9'),
        'glucose_test': st.selectbox("Glucose Test", get_categories('glucose_test')),
        'A1Ctest': st.selectbox("A1C Test", get_categories('A1Ctest')),
        'change': st.selectbox("Change in Medications", get_categories('change')),
        'diabetes_med': st.selectbox("Diabetes Medication", get_categories('diabetes_med'))
    }
    
    col1, col2 = st.columns(2)
    with col1:
        predict_button = st.form_submit_button("Predict")
    with col2:
        refresh_button = st.form_submit_button("Refresh")

# Handle refresh
if refresh_button:
    st.session_state.prediction_made = False
    st.session_state.refresh = True
    st.experimental_rerun()

# Handle prediction
if predict_button and not st.session_state.refresh:
    try:
        # Create input dataframe
        input_df = pd.DataFrame([user_input])
        
        # Ensure all categorical columns are strings
        for col in categorical_cols:
            input_df[col] = input_df[col].astype(str)
        
        # Create CatBoost Pool with proper categorical features
        prediction_pool = Pool(
            data=input_df,
            cat_features=categorical_cols,
            feature_names=list(input_df.columns)
        )
        
        # Make prediction
        prediction = model.predict(prediction_pool)[0]
        proba = model.predict_proba(prediction_pool)[0][1]
        
        # Store results in session state
        st.session_state.update({
            'prediction': prediction,
            'proba': proba,
            'prediction_made': True,
            'input_df': input_df,
            'prediction_pool': prediction_pool
        })
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Display results if prediction was made
if st.session_state.prediction_made:
    st.markdown("## 🏥 ICU Readmission Prediction")
    label = "✅ The patient is not likely to be readmitted" if st.session_state.prediction == 0 else "🔴 The patient is likely to be readmitted"
    st.markdown(f"### {label}")
    st.markdown(f"### Risk Score: **{st.session_state.proba:.2f}**")

    # SHAP Explanation
    with st.expander("📈 SHAP Explanation"):
        try:
            shap.initjs()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(st.session_state.prediction_pool)
            
            # Create force plot
            shap_plot = shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                st.session_state.input_df.iloc[0],
                feature_names=list(st.session_state.input_df.columns)
            )
            
            # Save and display
            shap.save_html("shap_explanation.html", shap_plot)
            with open("shap_explanation.html", "rb") as f:
                st.download_button(
                    "⬇️ Download SHAP Explanation",
                    data=f,
                    file_name="shap_explanation.html",
                    mime="text/html"
                )
            st.components.v1.html(open("shap_explanation.html").read(), height=400)
            
        except Exception as e:
            st.error(f"SHAP explanation failed: {str(e)}")

    # LIME Explanation
    with st.expander("🔍 LIME Explanation"):
        try:
            # Prepare data for LIME
            X_lime = X_train.copy()
            for col in categorical_cols:
                X_lime[col] = X_lime[col].astype(str)
            
            # Create LIME explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_lime),
                feature_names=X_lime.columns.tolist(),
                class_names=["Not Readmitted", "Readmitted"],
                categorical_features=[X_lime.columns.get_loc(c) for c in categorical_cols],
                mode='classification'
            )
            
            # Prepare input for LIME
            input_lime = st.session_state.input_df.copy()
            for col in categorical_cols:
                input_lime[col] = input_lime[col].astype(str)
            
            # Generate explanation
            lime_exp = lime_explainer.explain_instance(
                data_row=input_lime.iloc[0].values,
                predict_fn=lambda x: model.predict_proba(
                    Pool(
                        pd.DataFrame(x, columns=input_lime.columns),
                        cat_features=categorical_cols
                    )
                )
            )
            
            # Display explanation
            fig = lime_exp.as_pyplot_figure()
            st.pyplot(fig)
            
            # Download option
            fig.savefig("lime_explanation.png", bbox_inches="tight")
            with open("lime_explanation.png", "rb") as f:
                st.download_button(
                    "⬇️ Download LIME Explanation",
                    f,
                    "lime_explanation.png",
                    "image/png"
                )
                
        except Exception as e:
            st.error(f"LIME explanation failed: {str(e)}")
else:
    st.info("ℹ️ Please fill in the patient details and click 'Predict' to see results.")