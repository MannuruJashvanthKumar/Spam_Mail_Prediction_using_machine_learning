# Dataset Reference

## Source file

`mail_data.csv`

## Columns

| Column | Type | Description |
|---|---|---|
| `Category` | text | Target class: `spam` or `ham` |
| `Message` | text | Raw message content used as the input feature |

## Dataset size

- Rows: **5,572**
- Spam: **747**
- Ham: **4,825**

## Class mapping used by the model

```text
spam = 0
ham  = 1
```

## Missing values

The training code replaces null values with an empty string before processing.

## Text processing

The project uses `TfidfVectorizer` with:

- `min_df=1`
- `stop_words="english"`
- `lowercase=True`

The project does not add a custom spelling correction, stemming stage, or external text source.
