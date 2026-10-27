# 🩺 Breast Cancer Prediction System
### College Mini Project — AIML & Cloud Computing

An end-to-end, clinical-grade Machine Learning web application designed to predict whether a breast mass is **Benign (Non-cancerous)** or **Malignant (Cancerous)** based on 30 cell nucleus characteristics extracted from Fine Needle Aspiration (FNA) biopsy images.

The system is built with **Python Flask**, powered by a **Random Forest Classifier** (achieving **97.37% test accuracy** on the Wisconsin Diagnostic Breast Cancer dataset), and styled with a modern, responsive, light-mode medical user interface ready for deployment on **Vercel**.

---

## 📸 Screenshots & Key Highlights

- **Modern Light Theme Medical UI**: Crisp white and slate background with healthcare teal and sky blue accents.
- **Dynamic Feature Generation**: All 30 biomarker input fields are generated dynamically in the exact order expected by `model_info.json`.
- **Instant 1-Click Demo Buttons**: Quick-fill buttons for both **Benign** and **Malignant** cases allow instant demonstration during project presentations.
- **Transparent Output Gauge**: Displays the predicted diagnosis, confidence percentage (`predict_proba`), model name, and test accuracy with a visual confidence bar.
- **Educational Disclaimer**: Clearly marked academic notice distinguishing research demonstration from certified medical diagnosis.

---

## 🗂️ Project Directory Structure

```text
AIML mini project/
├── model.pkl               # Trained scikit-learn Random Forest model
├── scaler.pkl              # Fitted StandardScaler for 30 features
├── model_info.json         # Feature list, test accuracy (97.37%), model name
├── test_samples.json       # 30 authentic Wisconsin patient test cases
├── app.py                  # Core Flask backend with routes and ML inference pipeline
├── requirements.txt        # Python package dependencies
├── vercel.json             # Vercel serverless deployment routing
├── api/
│   └── index.py            # Vercel serverless WSGI entrypoint
├── templates/
│   ├── base.html           # Base layout, navbar, disclaimer banner, and footer
│   ├── index.html          # Home page with hero, stats, and workflow steps
│   ├── predict.html        # Prediction form, quick-fill buttons, and result card
│   └── about.html          # Detailed dataset, algorithm comparison, and methodology
├── static/
│   ├── css/
│   │   └── style.css       # Clean medical light-mode styles, responsive layout
│   ├── js/
│   │   └── main.js         # Client-side autofill, input validation, and category tabs
│   └── images/
│       ├── hero.jpg        # Clinical AI oncology hero illustration
│       └── cell_analysis.jpg # Microscopic FNA cell segmentation diagram
└── README.md               # Complete setup, run, and viva presentation guide
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.9+ (Python 3.10, 3.11, or 3.12 recommended)
- `pip` (Python package manager)

### 2. Clone or Open the Project Directory
```bash
cd "AIML mini project"
```

### 3. Create and Activate a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate.bat

# On Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
python app.py
```

### 6. View in Browser
Open your browser and navigate to:
```
http://127.0.0.1:5001
```

---

## ☁️ How to Deploy on Vercel

The project is already pre-configured for **Vercel** serverless deployment with `vercel.json` and `api/index.py`.

### Method A: Deploy via GitHub (Recommended)
1. Push this project repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Breast Cancer Prediction Web App"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Log into [Vercel](https://vercel.com).
3. Click **"Add New..."** &rarr; **"Project"**.
4. Import your GitHub repository.
5. Vercel automatically detects `vercel.json` and `requirements.txt`.
6. Click **Deploy**. Your application will be live on a public `.vercel.app` URL in ~1 minute!

### Method B: Deploy via Vercel CLI
```bash
npm install -g vercel
vercel login
vercel
```
Follow the interactive prompts and select default settings.

---

## 🧠 Machine Learning Details (For Project Viva / Presentation)

### 1. Dataset
- **Name:** Wisconsin Diagnostic Breast Cancer (WDBC) Dataset.
- **Source:** University of Wisconsin Hospitals (Dr. William H. Wolberg).
- **Instances:** 569 patient biopsy samples (357 Benign, 212 Malignant).
- **Features:** 30 continuous real-valued cellular nuclei measurements computed from digitized images of Fine Needle Aspiration (FNA) slides.

### 2. The 30 Biomarkers
- **10 Core Characteristics:** Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension.
- **3 Variants Each:**
  - `*_mean`: Average value across all cells in the patient's biopsy.
  - `*_se`: Standard error of measurement (variation/uncertainty).
  - `*_worst`: Average of the 3 largest/most severe cell nuclei.

### 3. Why Feature Scaling (`StandardScaler`) is Necessary
Features have wildly differing numerical scales (e.g., `area_mean` can exceed $1000\,\text{μm}^2$, whereas `smoothness_mean` is $< 0.15$). Standardizing via:
$$z = \frac{x - \mu}{\sigma}$$
ensures all features are centered around mean 0 with standard deviation 1, preventing high-magnitude features from skewing distance and boundary thresholds.

### 4. Why Random Forest Outperformed Other Models
| Model | Test Accuracy | Strengths & Weaknesses |
| :--- | :---: | :--- |
| **Logistic Regression** | ~95.6% | Simple baseline, but assumes linear decision boundary |
| **K-Nearest Neighbors (KNN)** | ~94.7% | Suffers from the curse of dimensionality across 30 features |
| **Support Vector Machine (SVM)**| ~96.5% | Strong boundary, but computationally slow probability calibration |
| **Random Forest (Selected)** | **97.37%** | **Ensemble bagging resists overfitting, naturally handles collinear features, provides calibrated probabilities, and maximizes recall (low false negatives).** |

---

## 🛡️ Medical Disclaimer
This software application was developed strictly for academic, research, and educational mini-project purposes. It is **not** a certified medical diagnosis tool and should never be used as a substitute for professional clinical biopsy, diagnostic pathology, or physician consultation.
