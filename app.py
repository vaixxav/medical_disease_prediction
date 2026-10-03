"""
app.py
------
Flask web application that serves the MediPredict multi-page site:

    /           Home page
    /insight    Dataset insight & model statistics page
    /predict    Prediction form + result (the ML-powered page)
    /about      About the project
    /contact    Contact form

The app loads the trained model and preprocessing artifacts once at startup
and never hardcodes feature values, labels, or column order -- everything is
read dynamically from the saved joblib artifacts.
"""

import os
import re
import joblib
import pandas as pd
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")  # not used by this model
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Load model + preprocessing artifacts once, at startup
# ---------------------------------------------------------------------------
def load_artifacts():
    """Load the trained model and all preprocessing objects from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python train_model.py` first to train and save the model."
        )

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    # Scaler is optional -- the original notebook did not use one.
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    return model, encoders, feature_columns, scaler


model, encoders, feature_columns, scaler = load_artifacts()

# Human-friendly field metadata used to render the prediction form.
NUMERIC_FIELDS = {
    "Age": {"label": "Age", "placeholder": "e.g. 45", "min": 18, "max": 100},
    "BMI": {"label": "BMI", "placeholder": "e.g. 24.5", "min": 10, "max": 60},
    "BloodPressure": {"label": "Blood Pressure (mmHg)", "placeholder": "e.g. 120", "min": 60, "max": 220},
    "GlucoseLevel": {"label": "Glucose Level (mg/dL)", "placeholder": "e.g. 100", "min": 50, "max": 400},
    "Cholesterol": {"label": "Cholesterol (mg/dL)", "placeholder": "e.g. 200", "min": 100, "max": 400},
    "HeartRate": {"label": "Heart Rate (bpm)", "placeholder": "e.g. 75", "min": 40, "max": 200},
}

CATEGORICAL_FIELDS = {
    "Gender": {"label": "Gender", "options": list(encoders["Gender"].keys())},
    "PhysicalActivity": {
        "label": "Physical Activity Level",
        "options": list(encoders["PhysicalActivity"].keys()),
    },
    "FamilyHistory": {
        "label": "Family History of Disease",
        "options": list(encoders["FamilyHistory"].keys()),
    },
}


def validate_and_preprocess(form_data):
    """
    Validate raw prediction-form input and convert it into a model-ready
    feature vector.

    Returns:
        (feature_vector, errors) tuple where:
            - feature_vector is a list of floats in `feature_columns` order,
              or None if validation failed.
            - errors is a list of human-readable error strings (empty if
              validation succeeded).
    """
    errors = []
    processed_values = {}

    # --- Validate numeric fields -----------------------------------------
    for field in NUMERIC_FIELDS:
        raw_value = form_data.get(field, "").strip()
        if not raw_value:
            errors.append(f"{NUMERIC_FIELDS[field]['label']} is required.")
            continue
        try:
            processed_values[field] = float(raw_value)
        except ValueError:
            errors.append(f"{NUMERIC_FIELDS[field]['label']} must be a number.")

    # --- Validate categorical fields ---------------------------------------
    for field in CATEGORICAL_FIELDS:
        raw_value = form_data.get(field, "").strip()
        valid_options = encoders[field]
        if not raw_value:
            errors.append(f"{CATEGORICAL_FIELDS[field]['label']} is required.")
            continue
        if raw_value not in valid_options:
            errors.append(f"Invalid value for {CATEGORICAL_FIELDS[field]['label']}.")
            continue
        processed_values[field] = valid_options[raw_value]

    if errors:
        return None, errors

    # --- Build the feature vector in the exact training column order ------
    try:
        feature_vector = [processed_values[col] for col in feature_columns]
    except KeyError as missing_col:
        errors.append(f"Missing required field: {missing_col}")
        return None, errors

    return feature_vector, errors


def validate_contact_form(form_data):
    """Validate the contact page form. Returns a list of error strings."""
    errors = []

    name = form_data.get("name", "").strip()
    email = form_data.get("email", "").strip()
    subject = form_data.get("subject", "").strip()
    message = form_data.get("message", "").strip()

    if not name:
        errors.append("Full name is required.")
    if not email:
        errors.append("Email address is required.")
    elif not EMAIL_REGEX.match(email):
        errors.append("Please enter a valid email address.")
    if not subject:
        errors.append("Subject is required.")
    if not message:
        errors.append("Message is required.")

    return errors


def get_dataset_insights():
    """
    Compute summary statistics from the cleaned dataset for the Insights
    page. Falls back gracefully if the cleaned dataset is not present.
    """
    if not os.path.exists(CLEANED_DATA_PATH):
        return {
            "total_records": 0,
            "disease_distribution": [],
            "condition_averages": [],
        }

    df = pd.read_csv(CLEANED_DATA_PATH)
    disease_inverse = encoders["Disease_inverse"]

    total_records = len(df)

    # Disease distribution (count + percentage), ordered by encoded value
    counts = df["Disease"].value_counts().sort_index()
    disease_distribution = [
        {
            "label": disease_inverse.get(int(code), "Unknown"),
            "count": int(count),
            "percent": round(float(count) / total_records * 100, 1),
        }
        for code, count in counts.items()
    ]

    # Average health metrics grouped by condition
    metric_cols = [
        "Age", "BMI", "GlucoseLevel", "Cholesterol", "HeartRate", "BloodPressure"
    ]
    grouped = df.groupby("Disease")[metric_cols].mean().round(1)

    condition_averages = [
        {
            "label": disease_inverse.get(int(code), "Unknown"),
            "age": row["Age"],
            "bmi": row["BMI"],
            "glucose": row["GlucoseLevel"],
            "cholesterol": row["Cholesterol"],
            "heart_rate": row["HeartRate"],
            "blood_pressure": row["BloodPressure"],
        }
        for code, row in grouped.iterrows()
    ]

    return {
        "total_records": total_records,
        "disease_distribution": disease_distribution,
        "condition_averages": condition_averages,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Landing page."""
    return render_template("home.html", active_page="home")


@app.route("/insight")
def insight():
    """Dataset insight & model statistics page."""
    stats = get_dataset_insights()

    return render_template(
        "insight.html",
        active_page="insight",
        total_records=stats["total_records"],
        disease_distribution=stats["disease_distribution"],
        condition_averages=stats["condition_averages"],
        model_accuracy="65.6",
        feature_count=len(feature_columns),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form (GET) and prediction handling (POST)."""
    result = None
    confidence = None
    errors = []

    if request.method == "POST":
        feature_vector, errors = validate_and_preprocess(request.form)

        if not errors:
            try:
                # Build a DataFrame with the exact training column names/order
                # so scikit-learn does not warn about missing feature names.
                X = pd.DataFrame([feature_vector], columns=feature_columns)

                # Apply scaler only if one was actually used during training
                if scaler is not None:
                    X = scaler.transform(X)

                prediction = model.predict(X)[0]
                result = encoders["Disease_inverse"].get(int(prediction), "Unknown")

                # Confidence score, if the model supports probability output
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)[0]
                    confidence = round(float(max(probabilities)) * 100, 2)
            except Exception as exc:  # noqa: BLE001 - surface a friendly error
                errors.append(f"An error occurred during prediction: {exc}")

    return render_template(
        "predict.html",
        active_page="predict",
        numeric_fields=NUMERIC_FIELDS,
        categorical_fields=CATEGORICAL_FIELDS,
        form_values=request.form,
        result=result,
        confidence=confidence,
        errors=errors,
    )


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html", active_page="about")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    """Contact page (GET) and contact form handling (POST)."""
    errors = []
    success = False

    if request.method == "POST":
        errors = validate_contact_form(request.form)
        if not errors:
            # No email/database backend is configured in this template --
            # the message is simply acknowledged. Wire this up to an email
            # service or database as needed for production use.
            success = True

    return render_template(
        "contact.html",
        active_page="contact",
        form_values=request.form if not success else {},
        errors=errors,
        success=success,
    )


if __name__ == "__main__":
    app.run(debug=True)
