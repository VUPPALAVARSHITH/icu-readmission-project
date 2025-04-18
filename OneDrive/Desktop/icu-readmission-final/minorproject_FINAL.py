import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
import pickle

# 🔹 Load dataset
df = pd.read_csv("C:/Users/varsh/OneDrive/Desktop/hospital_readmissions.csv")

# ✅ Drop rows where target is missing
df = df.dropna(subset=["readmitted"])

# 🔹 Encode target: 'no' = 0, 'yes' = 1
df['readmitted'] = df['readmitted'].map({'no': 0, 'yes': 1})

# 🔹 Define categorical columns
categorical_cols = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
                    'glucose_test', 'A1Ctest', 'change', 'diabetes_med']

# 🔹 Split features/target
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Convert categorical columns
for col in categorical_cols:
    X[col] = X[col].astype('category')

# 🔹 Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 🔹 One-hot encode categorical columns for SMOTE compatibility
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test).reindex(columns=X_train_encoded.columns, fill_value=0)

# 🔹 Apply SMOTE + Tomek Links
print("⚙ Applying SMOTE + Tomek...")
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train_encoded, y_train)

# 🔹 Train CatBoost model
print("🚀 Training CatBoost model on balanced data...")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100
)
model.fit(X_resampled, y_resampled)

# 🔹 Evaluate on test set
y_pred = model.predict(X_test_encoded)
y_proba = model.predict_proba(X_test_encoded)[:, 1]

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 Save model to disk
with open("catboost_model_smote_tomek.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n💾 Model saved as 'catboost_model_smote_tomek.pkl'")
