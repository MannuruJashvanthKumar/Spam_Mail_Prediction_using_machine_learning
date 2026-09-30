# Troubleshooting

## `streamlit` is not recognized

Use the module form instead of the executable form:

```bash
python -m streamlit run app.py
```

Also make sure the virtual environment is activated.

## Missing model artifacts

Run:

```bash
python train_model.py
```

This regenerates the Joblib model and TF-IDF vectorizer.

## Dataset column error

The training file must contain:

```text
Category
Message
```

The category values should be `spam` and `ham`.

## Empty prediction result

The application requires a non-empty message. Paste a full message or use one of the example buttons.

## Retraining after changing the dataset

After editing `mail_data.csv`, retrain:

```bash
python train_model.py
```

Then restart Streamlit so the cached model artifacts are reloaded.

## Dependency problems

Create a fresh environment and install the project requirements:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
