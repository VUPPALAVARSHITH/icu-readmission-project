**Explainable Machine Learning Model for ICU Readmission Risk Prediction**

**Project Overview**

ICU readmission is a critical hospital care issue often related to premature discharge or avoidable complications. This project proposes an interpretable machine learning model to predict the risk of ICU readmission within 30 days of discharge.

Unlike traditional "black-box" models, this system prioritizes explainability alongside predictability. It utilizes the CatBoost algorithm trained on hospital discharge data, employs SMOTE-Tomek for class imbalance, and integrates SHAP and LIME to provide clinical reasoning behind every prediction.

**Live Demo:** https://icu-readmission-project.streamlit.app/ 

**Key Features**

**High-Performance Classification:** Utilizes CatBoost, a gradient boosting algorithm optimized for categorical features and medical datasets.

**Imbalanced Data Handling:** Implements SMOTE-Tomek (Synthetic Minority Over-sampling Technique + Tomek Links) to handle the scarcity of readmission cases and improve sensitivity.

**Explainable AI (XAI):**
**SHAP (Global):** Visualizes feature importance across the entire dataset.
**LIME (Local):** Explains individual patient predictions to assist clinicians in specific case analysis.

**Real-Time Interface:** A user-friendly interface (CLI/Web) allowing medical staff to input patient data and receive immediate risk assessments.

**Methodology**

The project follows a structured pipeline designed for medical data complexity:

**Data Preprocessing:** Cleaning, encoding categorical variables (diagnosis codes, specialties), and normalizing numeric features.

**Handling Class Imbalance:** Applying SMOTE to generate synthetic minority samples and Tomek Links to remove boundary noise.

**Model Training:** Training a CatBoost Classifier with hyperparameter optimization (Learning rate, Tree depth, L2 regularization).

**Explainability Integration:** Generating SHAP force plots and LIME visualizations for transparency.

**Deployment:** Integration into a real-time input system. 

**Dataset**

The model is trained on a pre-processed version of the **Diabetes 130-US hospitals dataset** (1999–2008).

**Records:** ~25,000 patient encounters (balanced subset).
**Features:** 17 features including demographics, lab tests (HbA1c, Glucose), hospital utilization (time in hospital, procedures), and medication history .

**Feature Category**               **Examples**
**Demographics**                   **Age, Race, Gender** 
**Utilization**                    **Time in hospital, # Lab procedures, # Inpatient visits** **Clinical**                       **Diagnosis codes (Circulatory, Respiratory), HbA1c test** **Medication**                     **Number of medications, Insulin changes** 

**Results & Performance**

The model was evaluated on an 80:20 train-test split. It achieved a competitive **ROC-AUC** **of 0.66** and a **Recall of 0.68**, ensuring that over 60% of actual readmission cases are correctly identified a critical metric for patient safety.

**Performance Metrics** 

**Metric            Score**
Accuracy            0.62
Precision           0.63 
Recall              0.68
F1-Score            0.65 
ROC-AUC             0.66

**Explainability Analysis**
**SHAP Analysis:** Identified Number of Inpatient Visits, HbA1c Results, and Number of Medications as the top features influencing readmission risk.

**LIME Analysis: **Validated local predictions, showing for example that high emergency visits and insulin usage positively contributed to high-risk classification.

**Tech Stack**
**Language:** Python
**Machine Learning:** CatBoost, Scikit-learn
**Data Processing:** Pandas, NumPy, Imbalanced-learn (SMOTE-Tomek)
**Explainability:** SHAP, LIME
**Deployment:** Streamlit 

**Contributors**
This project was submitted for the Bachelor of Technology in CSE (AI & ML) at B V Raju Institute of Technology by Vuppala Varshith under the Supervisors: Dr. P. Srihari (Assistant Professor) Mrs. B. Divya (Assistant Professor) 

**Citation**
If you use this work, please reference the original project report: Varshith V. (2025). Explainable Machine Learning Model for ICU Readmission Risk Prediction. B V Raju Institute of Technology.
