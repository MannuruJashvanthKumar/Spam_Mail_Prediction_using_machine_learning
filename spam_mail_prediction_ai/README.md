# 🛡️ MailGuard AI — Spam Mail Prediction System

A complete machine-learning project for classifying messages as **Spam** or **Ham (Not Spam)** using the supplied `mail_data.csv` dataset and the workflow represented in the provided spam-mail prediction notebook.

The package includes the trained model, TF-IDF vectorizer, reproducible training code, an upgraded Streamlit web application, documentation, charts, examples, and the original/cleaned notebooks.

---

## ✨ What is included

- **Modern Streamlit web app** with:
  - prediction workspace
  - probability cards
  - quick example messages
  - model-feature explanation
  - analytics dashboard
  - project/about page
- **Trained Logistic Regression classifier**
- **Trained TF-IDF vectorizer**
- **Reproducible training script**
- **Command-line prediction utility**
- **Evaluation metrics and confusion matrix**
- **Dataset documentation**
- **Model card / limitations**
- **Architecture and project resources**
- **Clean training notebook**
- **Original supplied notebook preserved for reference**
- Windows and Linux/macOS launch scripts

---

## 📊 Dataset and model summary

The supplied dataset contains **5,572 messages** with the columns:

- `Category` — class label (`spam` or `ham`)
- `Message` — message text

The trained pipeline follows this sequence:

```text
Message text
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Spam / Ham prediction
    ↓
Class probability estimates
```

Training uses an **80/20 stratified train/test split** with `random_state=3`, matching the supplied project workflow.

### Held-out test results

| Metric | Result |
|---|---:|
| Test accuracy | **97.13%** |
| Spam precision | **99.16%** |
| Spam recall | **79.19%** |
| Spam F1-score | **88.06%** |

See `docs/PROJECT_DOCUMENTATION.md` for the full methodology and `resources/` for supporting charts.

---

## 🚀 Quick start

### 1. Create a virtual environment (recommended)

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the web application

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

### Windows shortcut

Double-click `run_app.bat`.

### Linux/macOS shortcut

```bash
./run_app.sh
```

---

## 🧠 Re-train the model

The repository contains the source CSV and all training code needed to reproduce the model artifacts:

```bash
python train_model.py
```

This regenerates:

- `spam_mail_model.joblib`
- `tfidf_vectorizer.joblib`
- `metrics.json`

---

## 🔎 Use the command-line predictor

```bash
python predict.py "Congratulations! You have won a free prize."
```

Example output:

```text
Prediction: SPAM
Spam probability: 98.00%
Ham probability: 2.00%
```

The exact probability depends on the trained artifacts in the package.

---

## 📁 Project structure

```text
spam_mail_prediction_ai/
│
├── app.py
├── train_model.py
├── predict.py
├── mail_data.csv
├── spam_mail_model.joblib
├── tfidf_vectorizer.joblib
├── metrics.json
├── sample_predictions.csv
├── requirements.txt
├── run_app.bat
├── run_app.sh
├── .gitignore
│
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   ├── DATASET_REFERENCE.md
│   ├── MODEL_CARD.md
│   └── TROUBLESHOOTING.md
│
├── resources/
│   ├── architecture.svg
│   ├── class_distribution.png
│   └── confusion_matrix.png
│
├── tests/
│   └── smoke_test.py
│
├── Spam_Mail_Prediction_Training.ipynb
└── Spam_Mail_Prediction_using_machine_learning_original.ipynb
```

---

## 📚 Documentation map

| Document | Purpose |
|---|---|
| `docs/PROJECT_DOCUMENTATION.md` | End-to-end project documentation |
| `docs/DATASET_REFERENCE.md` | Dataset fields, labels, preprocessing, and counts |
| `docs/MODEL_CARD.md` | Model purpose, metrics, limitations, and responsible use |
| `docs/TROUBLESHOOTING.md` | Common installation and runtime issues |
| `resources/architecture.svg` | Visual pipeline diagram |
| `resources/class_distribution.png` | Dataset class distribution |
| `resources/confusion_matrix.png` | Test-set confusion matrix |

---

## ⚠️ Important limitation

This classifier is a project/educational system. A prediction is an automated estimate based on message text and the trained dataset; it is not proof that a message is safe, malicious, authentic, or fraudulent. Important messages should be reviewed independently.

---

## 🧪 Smoke test

After installing dependencies, run:

```bash
python tests/smoke_test.py
```

This checks that the trained model and vectorizer can be loaded and that a prediction can be produced.
