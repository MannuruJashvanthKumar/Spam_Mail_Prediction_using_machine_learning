from pathlib import Path
import sys
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_mail_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"


def predict_message(message: str) -> dict:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    features = vectorizer.transform([message])
    prediction = int(model.predict(features)[0])
    probs = {int(c): float(p) for c, p in zip(model.classes_, model.predict_proba(features)[0])}
    return {
        "label": "spam" if prediction == 0 else "ham",
        "spam_probability": probs.get(0, 0.0),
        "ham_probability": probs.get(1, 0.0),
    }


if __name__ == "__main__":
    message = " ".join(sys.argv[1:]).strip()
    if not message:
        message = input("Paste a message: ").strip()
    result = predict_message(message)
    print(f"Prediction: {result['label'].upper()}")
    print(f"Spam probability: {result['spam_probability']:.2%}")
    print(f"Ham probability: {result['ham_probability']:.2%}")
