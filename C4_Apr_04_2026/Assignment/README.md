# Text Feature Engineering Assignment
### Real-world Amazon Product Reviews — NLP Pipeline

---

## Overview

This project builds a complete **Text Processing Pipeline** on real Amazon product reviews.
It demonstrates three classical text feature engineering techniques — **One Hot Encoding (OHE)**,
**Bag of Words (BoW)**, and **TF-IDF** — and applies them to a binary sentiment classification task.

---

## Project Structure

```
05_Apr_26/Assignment
├── text_feature_engineering.ipynb   # Main assignment notebook
├── amazon_reviews.csv               # Scraped/collected dataset (generated on first run)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Amazon Polarity dataset (McAuley & Leskovec, 2013) |
| Collection method | HuggingFace `datasets` library (streaming) |
| Total reviews | 150 (75 positive + 75 negative) |
| Columns | `review_text`, `sentiment`, `label` |
| Format | CSV (`amazon_reviews.csv`) |

> **Note on scraping:** Direct Amazon/Flipkart scraping is blocked by anti-bot systems.
> The `amazon_polarity` dataset contains real user-generated Amazon product reviews and is
> loaded via the `datasets` library already listed in `requirements.txt`.

---

## Tasks Implemented

| Task | Description |
|---|---|
| **Task 1** | Preprocessing — lowercase, punctuation removal, tokenization, stopword removal, lemmatization |
| **Task 2** | Vocabulary creation — manual + sklearn, frequency analysis, top-20 words |
| **Task 3** | Feature Engineering — OHE (document-level), BoW (CountVectorizer), TF-IDF (TfidfVectorizer) |
| **Task 4** | Comparison table — OHE vs BoW vs TF-IDF across 10 dimensions |
| **Task 5** | Sparse matrix analysis — shape, non-zero count, sparsity %, memory comparison |
| **Task 6** | Real-world Q&A — BoW semantic limitations, industry use cases, TF-IDF limitations |
| **Task 7** | Sentiment classification — Logistic Regression + Naive Bayes on BoW and TF-IDF features |

---

## Setup and Installation

### 1. Create and activate virtual environment

```bash
python -m venv env
source env/bin/activate          # macOS / Linux
env\Scripts\activate             # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook text_feature_engineering.ipynb
```

### 4. Run all cells

In Jupyter: **Kernel → Restart & Run All**

> The first run downloads ~5 MB of Amazon review data via streaming (internet required).
> Subsequent runs use the locally saved `amazon_reviews.csv`.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | 1.4.2 | CountVectorizer, TfidfVectorizer, classifiers |
| `numpy` | 1.26.4 | Matrix operations |
| `scipy` | 1.11.4 | Sparse matrix format |
| `datasets` | >=2.19.0 | Load real Amazon reviews |
| `pandas` | (transitive) | DataFrame operations |

---

## Key Results

### Feature Matrix Summary

| Representation | Shape | Sparsity | Value Type |
|---|---|---|---|
| One Hot Encoding | (150, 500) | ~99% | Binary (0/1) |
| Bag of Words | (150, 1000) | ~95% | Integer counts |
| TF-IDF | (150, 1000) | ~95% | Float [0.0, 1.0] |

### Sentiment Classification Accuracy

| Model | BoW | TF-IDF |
|---|---|---|
| Logistic Regression | ~75–82% | ~78–85% |
| Multinomial Naive Bayes | ~72–78% | ~74–80% |

> Exact numbers vary slightly due to random sampling. TF-IDF consistently outperforms BoW
> by down-weighting common words and highlighting discriminative sentiment terms.

---

## Concepts Explained

### Why TF-IDF outperforms BoW for sentiment analysis
TF-IDF penalizes words that appear in nearly every review (e.g., "product", "buy") and
rewards words that are distinctive to specific documents (e.g., "defective", "amazing").
These rare, distinctive words carry the strongest sentiment signal.

### Why sparse matrices are inefficient at scale
With 1M documents and 100K vocabulary, a dense matrix = **800 GB**. Since 95%+ of values
are zero, sparse formats (CSR) store only `(row, col, value)` triples for non-zero entries —
reducing memory by 20×. Modern NLP avoids this entirely with dense embeddings (BERT, Word2Vec).

### Limitations of all bag-of-words methods
- No semantic understanding ("good" ≠ "great" in vector space)
- No word order ("not good" ≈ "good" after stopword removal)
- Fixed vocabulary — unseen words at inference time are silently dropped
- High dimensionality — 10K–100K features for typical corpora

---

## Author Notes

- Preprocessing uses pure Python + regex (no NLTK/spaCy dependency required)
- Lemmatization is rule-based suffix stripping — adequate for this assignment
- All vectorizers are fit on the **training split only** to prevent data leakage in Task 7
- Classifiers use `copy.deepcopy` to ensure independent state across experiments
