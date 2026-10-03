"""
train_model.py
---------------
Reproduces the exact preprocessing and model training steps performed in
notebook/medical_disease_pred.ipynb.

Running this script will:
    1. Load the raw dataset from data/original.csv
    2. Apply the same cleaning / encoding steps used in the notebook
    3. Save the cleaned dataset to data/cleaned_data.csv
    4. Train a Logistic Regression model (exactly as in the notebook)
    5. Persist the model and all preprocessing artifacts to the models/ folder
       using joblib, so app.py can load them without hardcoding anything.

This script does NOT change the original ML logic -- it only wraps the
notebook steps into a reusable, production-safe script.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "original.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_DATA_PATH)

# ---------------------------------------------------------------------------
# 2. Preprocessing -- identical mappings to the original notebook
# ---------------------------------------------------------------------------
# NOTE: The notebook confirmed there are no duplicate rows and no missing
# values in the raw dataset, so no dropna()/drop_duplicates() step was
# required there. We keep that behaviour identical here.

# Encoding dictionaries (kept identical to the notebook)
physical_activity_dict = {"Low": 1, "Medium": 2, "High": 3}
gender_dict = {"Male": 1, "Female": 2}
disease_dict = {
    "Healthy": 1,
    "Pre-Diabetes": 2,
    "Hypertension": 3,
    "Heart Disease": 4,
    "Diabetes": 5,
}
family_history_dict = {"No": 0, "Yes": 1}

df["PhysicalActivity"] = df["PhysicalActivity"].map(physical_activity_dict)
df["Gender"] = df["Gender"].map(gender_dict)
df["Disease"] = df["Disease"].map(disease_dict)
df["FamilyHistory"] = df["FamilyHistory"].map(family_history_dict)

# Columns dropped in the notebook (found to be non-predictive / unused)
df.drop(columns=["Smoking", "Alcohol"], inplace=True)

# Save the cleaned dataset for reference / reproducibility
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")

# ---------------------------------------------------------------------------
# 3. Train / test split -- identical to the notebook (test_size=0.2, seed=42)
# ---------------------------------------------------------------------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Model training -- identical algorithm & defaults used in the notebook
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training score: {train_score:.4f}")
print(f"Test score:     {test_score:.4f}")

# ---------------------------------------------------------------------------
# 5. Persist model + preprocessing artifacts (no scaler was used in the
#    original notebook, so none is created here to avoid changing ML logic)
# ---------------------------------------------------------------------------
encoders = {
    "Gender": gender_dict,
    "PhysicalActivity": physical_activity_dict,
    "FamilyHistory": family_history_dict,
    "Disease": disease_dict,
    "Disease_inverse": {v: k for k, v in disease_dict.items()},
}

joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))

print("Model and preprocessing artifacts saved to the models/ directory.")
