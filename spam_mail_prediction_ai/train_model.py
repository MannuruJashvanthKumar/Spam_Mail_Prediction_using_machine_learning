"""Train the spam-mail classifier from mail_data.csv."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mail_data.csv"
MODEL_PATH = BASE_DIR / "spam_mail_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"


def main() -> None:
    mail_data = pd.read_csv(DATA_PATH).where(lambda df: pd.notnull(df), "")
    required = {"Category", "Message"}
    missing = required - set(mail_data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    mail_data["Category"] = mail_data["Category"].str.lower().map({"spam": 0, "ham": 1})
    if mail_data["Category"].isna().any():
        raise ValueError("Category must contain only 'spam' or 'ham'.")
    mail_data["Category"] = mail_data["Category"].astype(int)

    X = mail_data["Message"].astype(str)
    y = mail_data["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3, stratify=y
    )

    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_features, y_train)

    train_pred = model.predict(X_train_features)
    test_pred = model.predict(X_test_features)

    metrics = {
        "dataset_rows": int(len(mail_data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "spam_count": int((y == 0).sum()),
        "ham_count": int((y == 1).sum()),
        "training_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "precision_spam": float(precision_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "recall_spam": float(recall_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "f1_spam": float(f1_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "label_mapping": {"spam": 0, "ham": 1},
        "random_state": 3,
        "test_size": 0.2,
        "model": "LogisticRegression(max_iter=1000)",
        "vectorizer": "TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)",
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
