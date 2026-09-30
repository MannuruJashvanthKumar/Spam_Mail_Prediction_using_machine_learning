"""Small end-to-end smoke test for packaged artifacts."""

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
model = joblib.load(BASE_DIR / "spam_mail_model.joblib")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")

sample = "Congratulations! You have won a free prize. Call now to claim your reward!"
features = vectorizer.transform([sample])
prediction = int(model.predict(features)[0])
probabilities = model.predict_proba(features)[0]

assert prediction in (0, 1)
assert len(probabilities) == 2
assert abs(float(probabilities.sum()) - 1.0) < 1e-6

print("SMOKE TEST PASSED")
print(f"Prediction: {'SPAM' if prediction == 0 else 'HAM'}")
print(f"Probabilities: {probabilities}")
