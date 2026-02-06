from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

import nltk
from newspaper import Article
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)

LOG = logging.getLogger("news_pipeline")


@dataclass
class BiasAnalysis:
    score: float
    label: str
    dominant_cluster_share: float
    cluster_polarities: List[float]


@dataclass
class ArticleRecord:
    url: str
    title: str
    published_at: str
    cleaned_text: str
    sentiment_compound: float
    sentiment_label: str
    subjectivity: float
    bias_score: float
    bias_label: str
    extracted_at: str


class NewsPipeline:
    """Single-URL news parser with sentiment + unsupervised bias detection."""

    def __init__(self) -> None:
        self.sia = SentimentIntensityAnalyzer()

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s\.,;:!?\-']", "", text)
        return text.strip()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def analyze_sentiment(self, text: str) -> Dict[str, float | str]:
        vader = self.sia.polarity_scores(text)
        subjectivity = TextBlob(text).sentiment.subjectivity
        compound = vader["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return {
            "compound": round(compound, 4),
            "subjectivity": round(subjectivity, 4),
            "label": label,
        }

    @staticmethod
    def _sentence_features(sentences: Sequence[str]) -> List[List[float]]:
        features: List[List[float]] = []
        for sentence in sentences:
            blob = TextBlob(sentence).sentiment
            polarity = float(blob.polarity)
            subjectivity = float(blob.subjectivity)
            length_norm = min(len(sentence.split()) / 40.0, 1.0)
            features.append([polarity, subjectivity, length_norm])
        return features

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _kmeans_two_clusters(self, points: Sequence[Sequence[float]], max_iter: int = 20) -> Tuple[List[int], List[List[float]]]:
        if len(points) < 2:
            return [0 for _ in points], [list(points[0]) if points else [0.0, 0.0, 0.0]]

        centroids = [list(points[0]), list(points[-1])]
        assignments = [0] * len(points)

        for _ in range(max_iter):
            changed = False
            for i, point in enumerate(points):
                d0 = self._distance(point, centroids[0])
                d1 = self._distance(point, centroids[1])
                new_cluster = 0 if d0 <= d1 else 1
                if assignments[i] != new_cluster:
                    assignments[i] = new_cluster
                    changed = True

            for cluster_id in (0, 1):
                cluster_points = [p for p, a in zip(points, assignments) if a == cluster_id]
                if not cluster_points:
                    continue
                centroids[cluster_id] = [
                    sum(values) / len(values) for values in zip(*cluster_points)
                ]

            if not changed:
                break

        return assignments, centroids

    def analyze_bias_unsupervised(self, text: str) -> BiasAnalysis:
        sentences = self.split_sentences(text)
        if len(sentences) < 4:
            return BiasAnalysis(0.0, "Insufficient Evidence", 0.0, [0.0, 0.0])

        features = self._sentence_features(sentences)
        assignments, _ = self._kmeans_two_clusters(features)

        clusters = {0: [], 1: []}
        for sentence, feature, cluster in zip(sentences, features, assignments):
            clusters[cluster].append((sentence, feature))

        sizes = [len(clusters[0]), len(clusters[1])]
        dominant_share = max(sizes) / len(sentences)

        cluster_polarities: List[float] = []
        for cluster_id in (0, 1):
            if not clusters[cluster_id]:
                cluster_polarities.append(0.0)
                continue
            mean_polarity = sum(item[1][0] for item in clusters[cluster_id]) / len(clusters[cluster_id])
            cluster_polarities.append(mean_polarity)

        polarization_gap = abs(cluster_polarities[0] - cluster_polarities[1])

        non_neutral = [f[0] for f in features if abs(f[0]) >= 0.1]
        if non_neutral:
            same_direction = max(
                sum(1 for p in non_neutral if p > 0),
                sum(1 for p in non_neutral if p < 0),
            ) / len(non_neutral)
        else:
            same_direction = 0.5

        raw_score = (0.55 * polarization_gap) + (0.45 * max(0.0, same_direction - 0.5) * 2)
        score = min(max(raw_score, 0.0), 1.0)

        if score >= 0.45 and dominant_share >= 0.6:
            label = "Likely Biased"
        elif score >= 0.3:
            label = "Potential Bias"
        else:
            label = "Likely Balanced"

        return BiasAnalysis(
            score=round(score, 4),
            label=label,
            dominant_cluster_share=round(dominant_share, 4),
            cluster_polarities=[round(p, 4) for p in cluster_polarities],
        )

    def analyze_url(self, url: str) -> ArticleRecord:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text.strip()
        if not text:
            raise ValueError("No article text could be extracted from this URL.")

        cleaned_text = self.clean_text(text)
        sentiment = self.analyze_sentiment(cleaned_text)
        bias = self.analyze_bias_unsupervised(cleaned_text)

        published_at = (
            article.publish_date.astimezone(timezone.utc).isoformat()
            if article.publish_date
            else "unknown"
        )

        return ArticleRecord(
            url=url,
            title=article.title or "Untitled",
            published_at=published_at,
            cleaned_text=cleaned_text,
            sentiment_compound=sentiment["compound"],
            sentiment_label=sentiment["label"],
            subjectivity=sentiment["subjectivity"],
            bias_score=bias.score,
            bias_label=bias.label,
            extracted_at=datetime.now(tz=timezone.utc).isoformat(),
        )


def write_json(record: ArticleRecord, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(record), fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a single article URL for sentiment and unsupervised bias."
    )
    parser.add_argument("url", type=str, help="Article URL")
    parser.add_argument("--output", type=str, default="article_analysis.json")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    pipeline = NewsPipeline()
    record = pipeline.analyze_url(args.url)
    write_json(record, args.output)

    LOG.info("Saved analysis for %s", args.url)
    LOG.info("Sentiment: %s (compound=%.3f)", record.sentiment_label, record.sentiment_compound)
    LOG.info("Bias: %s (score=%.3f)", record.bias_label, record.bias_score)


if __name__ == "__main__":
    main()
