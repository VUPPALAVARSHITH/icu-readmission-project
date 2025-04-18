import streamlit as st

# Sidebar input for patient data
st.sidebar.title("Patient Information")

# Dynamic input for hospital days (user can enter any number of days)
time_in_hospital = st.sidebar.number_input(
    label="Time in hospital (days)",
    min_value=1,  # Minimum of 1 day
    max_value=365,  # Optional max value, you can adjust
    value=6,  # Default value
    step=1  # Step of 1 day
)

# Other inputs remain as they were
age = st.sidebar.selectbox("Age Range", ['60-70', '70-80', '50-60'])
number_of_lab_procedures = st.sidebar.number_input("Number of lab procedures", min_value=0, value=41)
number_of_procedures = st.sidebar.number_input("Number of procedures", min_value=0, value=2)
number_of_medications = st.sidebar.number_input("Number of medications", min_value=0, value=12)
number_of_outpatient_visits = st.sidebar.number_input("Number of outpatient visits", min_value=0, value=0)
number_of_emergency_visits = st.sidebar.number_input("Number of emergency visits", min_value=0, value=1)
number_of_inpatient_visits = st.sidebar.number_input("Number of inpatient visits", min_value=0, value=0)
medical_specialty = st.sidebar.selectbox("Medical Specialty", ['Cardiology', 'Endocrinology', 'Neurology'])
diagnosis_1_code = st.sidebar.text_input("Diagnosis 1 code")
diagnosis_2_code = st.sidebar.text_input("Diagnosis 2 code")
diagnosis_3_code = st.sidebar.text_input("Diagnosis 3 code")
glucose_test_result = st.sidebar.selectbox("Glucose test result", ['Normal', 'Abnormal'])
a1c_test_done = st.sidebar.selectbox("A1C test done?", ['Yes', 'No'])
change_in_medications = st.sidebar.selectbox("Change in medications?", ['Yes', 'No'])
is_diabetes_medication_prescribed = st.sidebar.selectbox("Is diabetes medication prescribed?", ['Yes', 'No'])

# Show input values for review
st.sidebar.subheader("Review your inputs")
st.sidebar.write(f"Time in hospital: {time_in_hospital} days")
st.sidebar.write(f"Age: {age}")
st.sidebar.write(f"Number of lab procedures: {number_of_lab_procedures}")
st.sidebar.write(f"Number of procedures: {number_of_procedures}")
st.sidebar.write(f"Number of medications: {number_of_medications}")
st.sidebar.write(f"Medical specialty: {medical_specialty}")
