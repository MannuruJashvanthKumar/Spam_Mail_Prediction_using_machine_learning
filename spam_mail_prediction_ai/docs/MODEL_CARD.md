# Model Card — MailGuard AI

## Model purpose

Binary text classification of messages into the `spam` or `ham` classes contained in the supplied training dataset.

## Model type

**Logistic Regression** trained on **TF-IDF** text features.

## Training data

- File: `mail_data.csv`
- Total messages: 5,572
- Spam: 747
- Ham: 4,825

## Training configuration

```text
Train/test split: 80/20
Stratified split: yes
Random state: 3
Vectorizer: TfidfVectorizer
Stop words: English
Classifier: LogisticRegression(max_iter=1000)
```

## Evaluation

| Metric | Test value |
|---|---:|
| Accuracy | 97.13% |
| Precision (spam) | 99.16% |
| Recall (spam) | 79.19% |
| F1 (spam) | 88.06% |

## Intended use

- Student/portfolio demonstration
- NLP and machine-learning learning project
- First-pass message classification
- Local demonstration of text classification

## Out-of-scope use

The model should not be used as a standalone security decision, an automated enforcement mechanism, or a guarantee that a message is malicious/safe.

## Known limitations

Performance depends on how similar new messages are to the training data. The model may misclassify messages containing patterns that differ from the historical dataset. It does not inspect mail headers, attachments, sender reputation, authentication records, or live URLs.

## Human review

Important communications should be independently verified. The model output is an estimate and should be treated as decision-support rather than absolute truth.
