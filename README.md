# MediPredict — Medical Disease Prediction Web App

A production-ready Flask web application that predicts a likely medical
condition (Healthy, Pre-Diabetes, Hypertension, Heart Disease, or Diabetes)
from basic patient health metrics, using a Logistic Regression model trained
on a 10,000-record medical dataset.

---

## 1. Project Overview

This project takes an exploratory Jupyter notebook (`notebook/medical_disease_pred.ipynb`)
and turns it into a complete, deployable web application:

- The exact preprocessing and encoding steps from the notebook are reproduced
  in `train_model.py`.
- The trained model and all preprocessing artifacts (label encoders, feature
  column order) are persisted with `joblib` so the Flask app never hardcodes
  any values.
- A clean, responsive, medical-themed UI (white + blue, card layout) lets a
  user enter their health data and instantly see a predicted condition along
  with the model's confidence score.

---

## 2. Features

- **End-to-end ML pipeline**: raw CSV → cleaned dataset → trained model → web app.
- **No hardcoded values**: the Flask backend loads the model, encoders, and
  feature column order dynamically from disk.
- **Form validation**: server-side validation with clear, actionable error
  messages for missing or invalid fields.
- **Confidence score**: displays the model's prediction confidence as a
  percentage with a visual progress bar.
- **Responsive design**: fully usable on desktop, tablet, and mobile.
- **Clean, professional UI**: medical white + blue theme, card-based layout,
  no external CSS frameworks (pure HTML + CSS).
- **PEP8-compliant, commented code** with proper error handling throughout.

---

## 3. Folder Structure

```
medical_disease_prediction/
│
├── app.py                     # Flask application (backend + routing)
├── train_model.py             # Reproduces notebook preprocessing & training
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
├── .gitignore                 # Files/folders excluded from version control
│
├── models/                    # Saved model + preprocessing artifacts
│   ├── model.pkl               # Trained Logistic Regression model
│   ├── encoders.pkl            # Label encoding dictionaries
│   └── feature_columns.pkl     # Exact feature column order used at training
│
├── data/
│   ├── original.csv            # Raw source dataset
│   └── cleaned_data.csv        # Cleaned & encoded dataset (generated)
│
├── static/
│   ├── css/
│   │   └── style.css           # Application styling (medical white/blue theme)
│   └── images/                 # Static image assets (optional)
│
├── templates/
│   └── index.html              # Jinja2 template (form + result UI)
│
└── notebook/
    └── medical_disease_pred.ipynb   # Original exploratory notebook
```

> **Note:** `scaler.pkl` is intentionally not present because the original
> notebook trained Logistic Regression directly on the raw numeric features
> without feature scaling. `app.py` is written to automatically use a scaler
> if one is later added to `models/scaler.pkl`, without any code changes.

---

## 4. Technologies Used

| Layer            | Technology            |
|-------------------|------------------------|
| Backend            | Python, Flask          |
| Machine Learning   | scikit-learn (Logistic Regression), pandas, numpy |
| Model Persistence  | joblib                 |
| Frontend           | HTML5, Jinja2, CSS3 (no frameworks) |
| Data               | CSV (10,000 patient records) |

---

## 5. Installation

### Prerequisites
- Python 3.9+
- pip

### Step 1 — Clone or download the project
```bash
git clone <your-repo-url>
cd medical_disease_prediction
```

### Step 2 — Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Training the Model

The trained model artifacts in `models/` are excluded from version control
(see `.gitignore`). Before running the app for the first time, generate them
by running:

```bash
python train_model.py
```

This will:
1. Read `data/original.csv`.
2. Apply the same cleaning/encoding steps used in the original notebook.
3. Save the cleaned dataset to `data/cleaned_data.csv`.
4. Train the Logistic Regression model.
5. Save `model.pkl`, `encoders.pkl`, and `feature_columns.pkl` into `models/`.

---

## 7. Running the App Locally

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

Fill in the health details form and click **Predict Disease** to view the
result and confidence score.

---

## 8. Screenshots

> _Add screenshots of the running application here._

- Home page / form: `static/images/screenshot-home.png`
- Prediction result: `static/images/screenshot-result.png`

---

## 9. Requirements

See `requirements.txt`:

```
Flask==3.0.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
```

---

## 11. Troubleshooting

**`InconsistentVersionWarning` or `'LogisticRegression' object has no attribute ...`**
This means the installed `scikit-learn` version differs from the version
used to create `models/model.pkl`. Pickled scikit-learn models are tied to
the exact library version that trained them. Fix it by doing **one** of the
following:

- Recommended: delete your virtual environment, recreate it, run
  `pip install -r requirements.txt` (which pins the exact versions used to
  train the shipped model), then run `python train_model.py` again before
  starting the app. This guarantees your training and serving environments
  match.
- Alternative: check your installed version with `pip show scikit-learn`,
  update `requirements.txt` to that version, then re-run `train_model.py`
  to regenerate `model.pkl` against your installed version.

---

## 12. Disclaimer

This application is built for educational and demonstrative purposes only.
Predictions generated by the model should **not** be used as a substitute
for professional medical advice, diagnosis, or treatment.
