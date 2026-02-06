# NewspaperScrape

A focused NLP tool that takes **one article URL**, parses the article text, and determines whether the writing appears **biased or balanced**.

## What it does

1. Extracts article text from a URL using `newspaper3k`.
2. Cleans text (whitespace, links, noisy characters).
3. Computes sentiment (`VADER` + `TextBlob`).
4. Uses an **unsupervised clustering method** (2-cluster k-means over sentence-level linguistic features) to estimate bias.

No hardcoded left/right keyword lists are used.

## Bias method (unsupervised)

- The article is split into sentences.
- Each sentence is converted into features:
  - polarity
  - subjectivity
  - normalized sentence length
- Sentences are grouped with 2-cluster k-means.
- A bias score is derived from:
  - **polarization gap** between cluster polarities
  - **one-sidedness** (whether non-neutral sentences lean mostly in one direction)
- Labels:
  - `Likely Biased`
  - `Potential Bias`
  - `Likely Balanced`
  - `Insufficient Evidence` (for very short texts)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python pipeline.py "https://example.com/news/article"
```

Optional output path:

```bash
python pipeline.py "https://example.com/news/article" --output article_analysis.json
```

## Output

A JSON file containing:

- `url`
- `title`
- `published_at`
- `cleaned_text`
- `sentiment_compound`
- `sentiment_label`
- `subjectivity`
- `bias_score`
- `bias_label`
- `extracted_at`

## Test

```bash
pytest -q
```
