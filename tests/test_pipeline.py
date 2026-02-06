from pipeline import NewsPipeline


def test_clean_text_removes_noise():
    raw = "Hello\nworld! Visit http://example.com ###"
    cleaned = NewsPipeline.clean_text(raw)
    assert "http" not in cleaned
    assert "#" not in cleaned
    assert "Hello world!" in cleaned


def test_unsupervised_bias_for_one_sided_text():
    pipeline = NewsPipeline()
    text = (
        "The policy is terrible and destructive. "
        "It is a disaster for everyone involved. "
        "Leaders failed badly and ignored obvious risks. "
        "The outcome remains deeply harmful and unacceptable."
    )
    result = pipeline.analyze_bias_unsupervised(text)
    assert result.label in {"Likely Biased", "Potential Bias"}
    assert 0.0 <= result.score <= 1.0


def test_unsupervised_bias_insufficient_evidence_short_text():
    pipeline = NewsPipeline()
    result = pipeline.analyze_bias_unsupervised("This is a short note. It is brief.")
    assert result.label == "Insufficient Evidence"
    assert result.score == 0.0


def test_sentiment_output_shape():
    pipeline = NewsPipeline()
    out = pipeline.analyze_sentiment("The meeting started at noon and ended at two.")
    assert out["label"] in {"Neutral", "Positive", "Negative"}
    assert isinstance(out["compound"], float)
