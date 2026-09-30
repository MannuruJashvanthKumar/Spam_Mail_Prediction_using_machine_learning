# Project Documentation — MailGuard AI

## 1. Project overview

**MailGuard AI** is a natural-language processing (NLP) classification project that predicts whether a supplied message belongs to the `spam` or `ham` category.

The implementation is based on the supplied dataset and the main workflow shown in the supplied spam-mail prediction notebook. The upgraded package keeps that modelling approach while adding a production-style project structure and a redesigned Streamlit interface.

## 2. Problem statement

Email and message streams can contain unwanted promotional or fraudulent-looking content. A supervised classifier can learn patterns from previously labelled examples and provide a first-pass classification for new text.

This project focuses on a binary classification task:

- **Spam** → unwanted/spam message class
- **Ham** → normal/non-spam message class

## 3. Dataset

The project uses `mail_data.csv`.

### Dataset shape

- Total rows: **5,572**
- Spam: **747**
- Ham: **4,825**
- Columns: `Category`, `Message`

### Label encoding

The training script uses:

```text
spam → 0
ham  → 1
```

This mapping is preserved in the trained model and application.

## 4. Training workflow

The training pipeline is:

1. Load the CSV dataset.
2. Replace missing values with empty strings.
3. Validate the required columns.
4. Convert class labels to numeric values.
5. Split data into 80% training and 20% test sets using stratification and `random_state=3`.
6. Convert message text into TF-IDF feature vectors.
7. Train Logistic Regression with `max_iter=1000`.
8. Evaluate the model on the held-out test set.
9. Save the model and vectorizer with Joblib.
10. Save evaluation metrics to `metrics.json`.

## 5. Why TF-IDF?

TF-IDF converts raw text into numerical features based on word importance within a message relative to the training corpus. It provides a compact representation that can be used by a conventional machine-learning classifier.

The supplied workflow uses:

```python
TfidfVectorizer(
    min_df=1,
    stop_words="english",
    lowercase=True,
)
```

## 6. Why Logistic Regression?

Logistic Regression is used as the classifier in the supplied workflow. It is a standard linear method for binary classification and works naturally with sparse TF-IDF text features.

The implementation uses:

```python
LogisticRegression(max_iter=1000)
```

## 7. Evaluation

The package evaluates the model on the 20% held-out test data.

### Results

| Metric | Value |
|---|---:|
| Training accuracy | 96.68% |
| Test accuracy | 97.13% |
| Spam precision | 99.16% |
| Spam recall | 79.19% |
| Spam F1-score | 88.06% |

### Confusion matrix

The test confusion matrix, using the training label order `[spam, ham]`, is:

```text
                 Predicted
               Spam     Ham
Actual Spam     118      31
Actual Ham        1     965
```

A visual copy is available at `resources/confusion_matrix.png`.

## 8. Application design

The redesigned Streamlit application contains three main workspaces:

### Predict

- message input area
- one-click example messages
- classification result card
- spam/ham probability indicators
- model confidence display
- approximate feature-level explanation using Logistic Regression contributions

### Analytics

- accuracy, precision, recall, and F1 cards
- dataset composition chart
- confusion matrix table
- model/training details

### About

- project objective
- source workflow description
- project resource overview
- educational-use notice

## 9. Model explanation feature

When a prediction is made, the application can display terms from the input message that contribute most strongly toward each class under the trained Logistic Regression coefficients.

This is an approximate feature-level explanation. It is not a causal explanation and should not be interpreted as a guarantee that a specific word caused the classification.

## 10. How to run the project

### Installation

```bash
python -m pip install -r requirements.txt
```

### Run the application

```bash
python -m streamlit run app.py
```

### Retrain

```bash
python train_model.py
```

### CLI prediction

```bash
python predict.py "Your message goes here"
```

## 11. Important files

| File | Description |
|---|---|
| `app.py` | Redesigned Streamlit application |
| `train_model.py` | Reproducible training pipeline |
| `predict.py` | Command-line predictor |
| `mail_data.csv` | Training dataset |
| `spam_mail_model.joblib` | Trained Logistic Regression model |
| `tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer |
| `metrics.json` | Evaluation and training metadata |
| `requirements.txt` | Python dependencies |
| `Spam_Mail_Prediction_Training.ipynb` | Clean training notebook |
| `Spam_Mail_Prediction_using_machine_learning_original.ipynb` | Original supplied notebook |

## 12. Reproducibility notes

The training split is deterministic because `random_state=3` is fixed. The same `mail_data.csv` and dependency family should therefore reproduce the same project workflow, although exact floating-point behaviour can vary slightly across library/platform versions.

## 13. Limitations

- The classifier only learns from the supplied labelled dataset.
- It can make false positives and false negatives.
- The model is text-based and does not independently verify a sender, domain, attachment, URL destination, or authentication headers.
- The displayed class probabilities are model estimates, not guarantees.
- Distribution changes in future messages can reduce performance.

## 14. Responsible use

Use the classifier as a decision-support or demonstration tool. Do not treat a single automated prediction as definitive evidence when reviewing sensitive or high-impact communications.

## 15. Project resources

See the `resources/` folder for:

- architecture diagram
- class distribution chart
- confusion matrix visualization

The `docs/` folder also contains the dataset reference, model card, and troubleshooting guide.
