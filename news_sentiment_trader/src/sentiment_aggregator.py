from typing import List, Dict, Optional
from .config import SENTIMENT_THRESHOLDS


def aggregate_article_sentiments(article_results: List[Dict]) -> Optional[Dict]:
    """
    Aggregate FinBERT sentiment results for multiple articles of a single stock.

    article_results: list of dicts with keys:
        - score
        - label
        - positive
        - negative
        - neutral
        (plus any extra metadata)

    Returns a dict with:
        - final_score      (float)
        - avg_positive
        - avg_negative
        - avg_neutral
        - article_count
        - bullish_ratio    (#positive / total)
        - bearish_ratio    (#negative / total)
        - neutral_ratio    (#neutral / total)
        - sentiment_view   ("bullish" / "bearish" / "mixed/neutral")
    """
    if not article_results:
        return None

    n = len(article_results)

    sum_score = 0.0
    sum_pos = 0.0
    sum_neg = 0.0
    sum_neu = 0.0

    pos_count = 0
    neg_count = 0
    neu_count = 0

    for res in article_results:
        score = res.get("score", 0.0)
        sum_score += score
        sum_pos += res.get("positive", 0.0)
        sum_neg += res.get("negative", 0.0)
        sum_neu += res.get("neutral", 0.0)

        label = res.get("label", "").lower()
        if label == "positive":
            pos_count += 1
        elif label == "negative":
            neg_count += 1
        else:
            neu_count += 1

    final_score = sum_score / n
    avg_positive = sum_pos / n
    avg_negative = sum_neg / n
    avg_neutral = sum_neu / n

    bullish_ratio = pos_count / n
    bearish_ratio = neg_count / n
    neutral_ratio = neu_count / n

    buy_thr = SENTIMENT_THRESHOLDS["buy"]
    sell_thr = SENTIMENT_THRESHOLDS["sell"]

    # Simple interpretation
    if final_score > buy_thr:
        sentiment_view = "bullish"
    elif final_score < sell_thr:
        sentiment_view = "bearish"
    else:
        sentiment_view = "mixed/neutral"

    return {
        "final_score": final_score,
        "avg_positive": avg_positive,
        "avg_negative": avg_negative,
        "avg_neutral": avg_neutral,
        "article_count": n,
        "bullish_ratio": bullish_ratio,
        "bearish_ratio": bearish_ratio,
        "neutral_ratio": neutral_ratio,
        "sentiment_view": sentiment_view,
    }
