"""
Breast Cancer Prediction Web Application
College Mini Project: AIML / Cloud Computing

This Flask web application serves a trained Machine Learning model (Random Forest)
to classify breast mass fine needle aspirate (FNA) cell nuclei as Benign or Malignant.

Pipeline:
1. User provides 30 numeric cell measurements.
2. Inputs are standardized using StandardScaler (fitted during training).
3. Random Forest Classifier predicts classification (0 = Benign, 1 = Malignant)
   and computes prediction confidence via predict_proba.
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

# Suppress benign sklearn warnings regarding unpickling across minor versions
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Application Initialization & Path Configuration
# ---------------------------------------------------------
# We use absolute directory paths so that the app works seamlessly
# both in local environments (python app.py) and on serverless platforms like Vercel.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
    static_url_path='/static'
)

@app.context_processor
def inject_site_assets():
    """Injects CSS directly into templates to guarantee styling on serverless CDNs"""
    css_content = ""
    try:
        css_path = os.path.join(BASE_DIR, 'static', 'css', 'style.css')
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
    except Exception as e:
        print(f"Could not read style.css for inline injection: {e}")
    return dict(inline_css=css_content)

# ---------------------------------------------------------
# Load Machine Learning Model, Scaler, and Metadata
# ---------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
INFO_PATH = os.path.join(BASE_DIR, 'model_info.json')

# Load the trained scikit-learn classifier
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

# Load the fitted StandardScaler
try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading scaler from {SCALER_PATH}: {e}")
    scaler = None

# Load model metadata (ordered feature list, test accuracy, model name)
try:
    with open(INFO_PATH, 'r') as f:
        model_info = json.load(f)
except Exception as e:
    print(f"Error loading model_info from {INFO_PATH}: {e}")
    model_info = {
        "features": [],
        "test_accuracy": 0.9737,
        "model_name": "Random Forest"
    }

# Load authentic pre-extracted test cases from the Wisconsin dataset
TEST_SAMPLES_PATH = os.path.join(BASE_DIR, 'test_samples.json')
try:
    with open(TEST_SAMPLES_PATH, 'r') as f:
        test_samples = json.load(f)
except Exception as e:
    print(f"Error loading test_samples from {TEST_SAMPLES_PATH}: {e}")
    test_samples = []

# Human-readable labels and descriptions for all 30 features
# Grouped into: Mean, Standard Error (SE), and Worst (largest values)
FEATURE_METADATA = {
    # Mean features (mean of cell nuclei measurements)
    "radius_mean": {"label": "Mean Radius", "group": "Mean Values", "desc": "Mean distance from center to perimeter points", "unit": "μm"},
    "texture_mean": {"label": "Mean Texture", "group": "Mean Values", "desc": "Standard deviation of gray-scale values", "unit": ""},
    "perimeter_mean": {"label": "Mean Perimeter", "group": "Mean Values", "desc": "Mean perimeter of cell nuclei", "unit": "μm"},
    "area_mean": {"label": "Mean Area", "group": "Mean Values", "desc": "Mean surface area of cell nuclei", "unit": "μm²"},
    "smoothness_mean": {"label": "Mean Smoothness", "group": "Mean Values", "desc": "Local variation in radius lengths", "unit": ""},
    "compactness_mean": {"label": "Mean Compactness", "group": "Mean Values", "desc": "Perimeter² / area - 1.0", "unit": ""},
    "concavity_mean": {"label": "Mean Concavity", "group": "Mean Values", "desc": "Severity of concave portions of contour", "unit": ""},
    "concave_points_mean": {"label": "Mean Concave Points", "group": "Mean Values", "desc": "Number of concave portions of contour", "unit": ""},
    "symmetry_mean": {"label": "Mean Symmetry", "group": "Mean Values", "desc": "Nuclear symmetry score", "unit": ""},
    "fractal_dimension_mean": {"label": "Mean Fractal Dim", "group": "Mean Values", "desc": "Coastline approximation - 1", "unit": ""},

    # Standard Error (SE) features
    "radius_se": {"label": "Radius (SE)", "group": "Standard Error (SE)", "desc": "Standard error of radius measurements", "unit": "μm"},
    "texture_se": {"label": "Texture (SE)", "group": "Standard Error (SE)", "desc": "Standard error of texture", "unit": ""},
    "perimeter_se": {"label": "Perimeter (SE)", "group": "Standard Error (SE)", "desc": "Standard error of perimeter", "unit": "μm"},
    "area_se": {"label": "Area (SE)", "group": "Standard Error (SE)", "desc": "Standard error of area", "unit": "μm²"},
    "smoothness_se": {"label": "Smoothness (SE)", "group": "Standard Error (SE)", "desc": "Standard error of smoothness", "unit": ""},
    "compactness_se": {"label": "Compactness (SE)", "group": "Standard Error (SE)", "desc": "Standard error of compactness", "unit": ""},
    "concavity_se": {"label": "Concavity (SE)", "group": "Standard Error (SE)", "desc": "Standard error of concavity", "unit": ""},
    "concave_points_se": {"label": "Concave Points (SE)", "group": "Standard Error (SE)", "desc": "Standard error of concave points", "unit": ""},
    "symmetry_se": {"label": "Symmetry (SE)", "group": "Standard Error (SE)", "desc": "Standard error of symmetry", "unit": ""},
    "fractal_dimension_se": {"label": "Fractal Dim (SE)", "group": "Standard Error (SE)", "desc": "Standard error of fractal dimension", "unit": ""},

    # Worst features (mean of the three largest values)
    "radius_worst": {"label": "Worst Radius", "group": "Extreme / Worst Values", "desc": "Largest radius value measured", "unit": "μm"},
    "texture_worst": {"label": "Worst Texture", "group": "Extreme / Worst Values", "desc": "Largest texture value measured", "unit": ""},
    "perimeter_worst": {"label": "Worst Perimeter", "group": "Extreme / Worst Values", "desc": "Largest perimeter measured", "unit": "μm"},
    "area_worst": {"label": "Worst Area", "group": "Extreme / Worst Values", "desc": "Largest area measured", "unit": "μm²"},
    "smoothness_worst": {"label": "Worst Smoothness", "group": "Extreme / Worst Values", "desc": "Largest smoothness measured", "unit": ""},
    "compactness_worst": {"label": "Worst Compactness", "group": "Extreme / Worst Values", "desc": "Largest compactness measured", "unit": ""},
    "concavity_worst": {"label": "Worst Concavity", "group": "Extreme / Worst Values", "desc": "Largest concavity measured", "unit": ""},
    "concave_points_worst": {"label": "Worst Concave Points", "group": "Extreme / Worst Values", "desc": "Largest concave points measured", "unit": ""},
    "symmetry_worst": {"label": "Worst Symmetry", "group": "Extreme / Worst Values", "desc": "Largest symmetry measured", "unit": ""},
    "fractal_dimension_worst": {"label": "Worst Fractal Dim", "group": "Extreme / Worst Values", "desc": "Largest fractal dimension measured", "unit": ""}
}

# Verified sample cases from the Wisconsin dataset for quick one-click testing
SAMPLE_CASES = {
    "benign": {
        "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, "area_mean": 566.3,
        "smoothness_mean": 0.0978, "compactness_mean": 0.0813, "concavity_mean": 0.0666, "concave_points_mean": 0.0478,
        "symmetry_mean": 0.1885, "fractal_dimension_mean": 0.0577, "radius_se": 0.2699, "texture_se": 0.7886,
        "perimeter_se": 2.058, "area_se": 23.56, "smoothness_se": 0.0085, "compactness_se": 0.0146,
        "concavity_se": 0.0239, "concave_points_se": 0.0132, "symmetry_se": 0.0198, "fractal_dimension_se": 0.0023,
        "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.7, "area_worst": 711.2,
        "smoothness_worst": 0.144, "compactness_worst": 0.1773, "concavity_worst": 0.239,
        "concave_points_worst": 0.1288, "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.0726
    },
    "malignant": {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0,
        "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001, "concave_points_mean": 0.1471,
        "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.0787, "radius_se": 1.095, "texture_se": 0.9053,
        "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.0064, "compactness_se": 0.049,
        "concavity_se": 0.0537, "concave_points_se": 0.0159, "symmetry_se": 0.03, "fractal_dimension_se": 0.0062,
        "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
        "concave_points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
    }
}


# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------

@app.route("/")
def home():
    """
    1. Home Page:
    Displays overview of the Breast Cancer Prediction System,
    project stats, model accuracy, and link to start prediction.
    """
    accuracy_pct = round(model_info.get("test_accuracy", 0.9737) * 100, 2)
    return render_template(
        "index.html",
        accuracy=accuracy_pct,
        model_name=model_info.get("model_name", "Random Forest"),
        features_count=len(model_info.get("features", []))
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    2. Prediction Page & 3. Result Section:
    - GET: Renders the prediction form dynamically from model_info.json features.
    - POST:
      1. Reads form inputs in the exact order of `features`.
      2. Validates numerical integrity.
      3. Standardizes input via scaler.transform().
      4. Generates prediction (0: Benign, 1: Malignant) and confidence via model.predict_proba().
      5. Returns the template populated with the result section.
    """
    features = model_info.get("features", [])
    accuracy_pct = round(model_info.get("test_accuracy", 0.9737) * 100, 2)
    model_name = model_info.get("model_name", "Random Forest")

    # Group features by category for organized, clean UI tabs
    categorized_features = []
    for feat_name in features:
        meta = FEATURE_METADATA.get(feat_name, {
            "label": feat_name.replace("_", " ").title(),
            "group": "General",
            "desc": "Cellular feature measurement",
            "unit": ""
        })
        categorized_features.append({
            "name": feat_name,
            "label": meta["label"],
            "group": meta["group"],
            "desc": meta["desc"],
            "unit": meta["unit"]
        })

    # When accessing page via GET, display empty form
    if request.method == "GET":
        return render_template(
            "predict.html",
            features=categorized_features,
            feature_names=features,
            accuracy=accuracy_pct,
            model_name=model_name,
            sample_cases=SAMPLE_CASES,
            test_samples=test_samples,
            result=None,
            form_values={},
            error=None
        )

    # When user submits the form via POST
    form_values = {}
    ordered_values = []
    missing_or_invalid = []

    # Step 1: Extract and validate each feature in exact order
    for feat in features:
        raw_val = request.form.get(feat, "").strip()
        form_values[feat] = raw_val
        try:
            val_float = float(raw_val)
            ordered_values.append(val_float)
        except (ValueError, TypeError):
            missing_or_invalid.append(feat)

    # Gracefully handle missing or invalid non-numeric inputs
    if missing_or_invalid:
        error_msg = f"Please provide valid numeric values for all features. Invalid or missing: {', '.join(missing_or_invalid[:3])}"
        if len(missing_or_invalid) > 3:
            error_msg += f" and {len(missing_or_invalid) - 3} others."

        return render_template(
            "predict.html",
            features=categorized_features,
            feature_names=features,
            accuracy=accuracy_pct,
            model_name=model_name,
            sample_cases=SAMPLE_CASES,
            test_samples=test_samples,
            result=None,
            form_values=form_values,
            error=error_msg
        )

    try:
        # Step 2: Reshape input for 2D array [1, n_features]
        raw_array = np.array(ordered_values).reshape(1, -1)

        # Step 3: Standardize the input features using the loaded StandardScaler
        # In scikit-learn, the model was trained on scaled features.
        scaled_features = scaler.transform(raw_array)

        # Step 4: Model Prediction
        # Classes: 0 = Benign, 1 = Malignant
        raw_pred = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        # In this dataset/model:
        # 1 = Malignant, 0 = Benign
        if raw_pred == 1:
            pred_label = "Malignant (Cancer detected)"
            status = "malignant"
            confidence = round(float(probabilities[1]) * 100, 2)
            risk_level = "High Risk"
            recommendation = "Immediate biopsy confirmation and clinical oncology consultation are advised."
        else:
            pred_label = "Benign (No cancer detected)"
            status = "benign"
            confidence = round(float(probabilities[0]) * 100, 2)
            risk_level = "Normal / Low Risk"
            recommendation = "Measurements are consistent with healthy tissue. Routine screening recommended."

        result_data = {
            "label": pred_label,
            "status": status,
            "confidence": confidence,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "proba_benign": round(float(probabilities[0]) * 100, 2),
            "proba_malignant": round(float(probabilities[1]) * 100, 2),
            "model_name": model_name,
            "test_accuracy": accuracy_pct
        }

        return render_template(
            "predict.html",
            features=categorized_features,
            feature_names=features,
            accuracy=accuracy_pct,
            model_name=model_name,
            sample_cases=SAMPLE_CASES,
            test_samples=test_samples,
            result=result_data,
            form_values=form_values,
            error=None
        )

    except Exception as e:
        return render_template(
            "predict.html",
            features=categorized_features,
            feature_names=features,
            accuracy=accuracy_pct,
            model_name=model_name,
            sample_cases=SAMPLE_CASES,
            test_samples=test_samples,
            result=None,
            form_values=form_values,
            error=f"Prediction error occurred: {str(e)}"
        )


@app.route("/about")
def about():
    """
    4. About Page:
    Comprehensive background on the Wisconsin Breast Cancer Dataset,
    feature extraction via Fine Needle Aspiration (FNA),
    comparison of ML algorithms (Logistic Regression, SVM, Random Forest),
    and justification for choosing Random Forest.
    """
    accuracy_pct = round(model_info.get("test_accuracy", 0.9737) * 100, 2)
    return render_template(
        "about.html",
        accuracy=accuracy_pct,
        model_name=model_info.get("model_name", "Random Forest")
    )


@app.route("/api/samples")
def api_samples():
    """Helper endpoint to supply sample benign and malignant data to JavaScript"""
    return jsonify(SAMPLE_CASES)


@app.route("/api/test-samples")
def api_test_samples():
    """Returns the full collection of 30 authentic test cases from the Wisconsin dataset"""
    return jsonify(test_samples)


@app.route("/api/random-sample")
def api_random_sample():
    """Returns a single random test case for instant demoing"""
    import random
    if test_samples:
        return jsonify(random.choice(test_samples))
    return jsonify({"error": "No test samples available"}), 404


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Fallback static asset handler for serverless environments"""
    static_dir = os.path.join(BASE_DIR, "static")
    return send_from_directory(static_dir, filename)


# ---------------------------------------------------------
# Local Server Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Use PORT environment variable or default to 5001 (5000 is used by AirPlay on macOS)
    port = int(os.environ.get("PORT", 5001))
    print(f"\n🩺 Breast Cancer Prediction System running at: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)

