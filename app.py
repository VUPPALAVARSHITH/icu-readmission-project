import streamlit as st
import pandas as pd
import pickle

# --------------------- 🔁 Caching ---------------------
@st.cache_resource
def load_model():
    with open("catboost_model_smote_tomek.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_columns():
    with open("X_train_columns.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
expected_columns = load_columns()

# --------------------- 🖼️ Page Config ---------------------
st.set_page_config(page_title="ICU Readmission Prediction", layout="wide")
st.title("🏥 ICU Readmission Predictor")

# --------------------- 🧾 Sidebar Input ---------------------
with st.sidebar:
    st.header("🧑‍⚕️ Patient Input")
    with st.expander("📋 Fill Patient Details", expanded=True):
        age = st.selectbox("Age Range", ['10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
        time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 6)
        num_lab_procedures = st.slider("Lab procedures", 1, 100, 41)
        num_procedures = st.slider("Number of procedures", 0, 6, 2)
        num_medications = st.slider("Number of medications", 1, 50, 12)
        number_outpatient = st.slider("Outpatient visits", 0, 10, 0)
        number_emergency = st.slider("Emergency visits", 0, 10, 1)
        number_inpatient = st.slider("Inpatient visits", 0, 10, 0)
        medical_specialty = st.selectbox("Medical specialty", ['Cardiology', 'General Practice', 'Nephrology', 'Surgery-General', 'Endocrinology'])
        diag_1 = st.text_input("Diagnosis 1 code", "428")
        diag_2 = st.text_input("Diagnosis 2 code", "250.02")
        diag_3 = st.text_input("Diagnosis 3 code", "401.9")
        glucose_test = st.radio("Glucose test result", ['Normal', 'Abnormal'])
        A1Ctest = st.radio("A1C test done?", ['Yes', 'No'])
        change = st.radio("Change in medications?", ['Yes', 'No'])
        diabetes_med = st.radio("Diabetes medication prescribed?", ['Yes', 'No'])
        submit = st.button("🔍 Predict")

# --------------------- 🔎 On Predict ---------------------
if submit:
    input_data = pd.DataFrame([{
        'age': age,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'medical_specialty': medical_specialty,
        'diag_1': diag_1,
        'diag_2': diag_2,
        'diag_3': diag_3,
        'glucose_test': glucose_test,
        'A1Ctest': A1Ctest,
        'change': change,
        'diabetes_med': diabetes_med
    }])

    # 🔄 One-hot encoding & align with training features
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

    # 🔮 Predict
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    # --------------------- 📊 Prediction Result ---------------------
    tabs = st.tabs(["📈 Prediction Result", "⬇️ Download Data"])

    with tabs[0]:
        st.subheader("Prediction")
        if prediction == 1:
            st.markdown(f"⚠️ **High Risk of ICU Readmission**\n\n🟥 **Confidence**: `{proba:.2f}`")
        else:
            st.markdown(f"✅ **Low Risk of ICU Readmission**\n\n🟩 **Confidence**: `{1 - proba:.2f}`")

    with tabs[1]:
        st.download_button(
            label="📥 Download Input Data as CSV",
            data=input_data.to_csv(index=False),
            file_name="patient_input.csv",
            mime="text/csv"
        )
