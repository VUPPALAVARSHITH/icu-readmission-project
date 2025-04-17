import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained CatBoost model using caching
@st.cache_resource
def load_model():
    with open("catboost_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Sidebar – User input
st.sidebar.header("📥 Enter Patient Information")

def get_user_input():
    age = st.sidebar.slider("Age", 0, 100, 50)
    time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 5)
    num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 0, 150, 40)
    num_medications = st.sidebar.slider("Number of Medications", 0, 80, 10)
    number_outpatient = st.sidebar.slider("Outpatient Visits", 0, 20, 0)
    number_emergency = st.sidebar.slider("Emergency Visits", 0, 20, 0)
    number_inpatient = st.sidebar.slider("Inpatient Visits", 0, 20, 0)

    data = {
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# Main Area UI
st.title("🏥 ICU Readmission Predictor")
st.markdown("This app predicts whether a patient is at risk of readmission.")

# Tabs
tab1, tab2 = st.tabs(["📊 Prediction", "ℹ️ Info"])

with tab1:
    with st.expander("📋 Patient Data", expanded=True):
        st.dataframe(input_df)

    if st.button("🚀 Predict Readmission"):
        prediction = model.predict(input_df)[0]
        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.markdown("🟥 **High risk of ICU readmission!** ❌")
        else:
            st.markdown("🟩 **Low risk of ICU readmission.** ✅")

with tab2:
    st.markdown("""
    - This dashboard uses a **CatBoost** model.
    - Data is taken from a healthcare dataset.
    - Adjust the sliders to simulate patient conditions.
    """)

# Optional download
st.download_button(
    label="📥 Download Input as CSV",
    data=input_df.to_csv(index=False).encode('utf-8'),
    file_name='patient_data.csv',
    mime='text/csv'
)
