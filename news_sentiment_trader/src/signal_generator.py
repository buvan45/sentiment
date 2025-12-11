from typing import Dict, Optional
from .config import SENTIMENT_THRESHOLDS


def generate_trading_signal(
    symbol: str,
    summary: Dict,
    buy_thr: Optional[float] = None,
    sell_thr: Optional[float] = None,
) -> Dict:
    """
    Convert aggregated sentiment summary into a trading signal.

    Tunable logic:
      - If final_score >= buy_thr -> BUY
      - If final_score <= sell_thr -> SELL
      - Else -> HOLD

    buy_thr / sell_thr:
      - If not provided, values from SENTIMENT_THRESHOLDS in config.py are used.

    Expected keys in `summary`:
        - final_score
        - article_count
        - bullish_ratio
        - bearish_ratio
        - neutral_ratio
        - sentiment_view
    """
    # Use defaults from config if not overridden
    if buy_thr is None:
        buy_thr = SENTIMENT_THRESHOLDS["buy"]
    if sell_thr is None:
        sell_thr = SENTIMENT_THRESHOLDS["sell"]

    final_score = summary.get("final_score", 0.0)
    article_count = summary.get("article_count", 0)
    bullish_ratio = summary.get("bullish_ratio", 0.0)
    bearish_ratio = summary.get("bearish_ratio", 0.0)
    sentiment_view = summary.get("sentiment_view", "mixed/neutral")

    signal = "HOLD"
    confidence = "low"
    reason_parts = []

    if article_count == 0:
        reason_parts.append("No recent news articles available.")
    else:
        reason_parts.append(
            f"{article_count} news articles analyzed. "
            f"Overall sentiment is {sentiment_view.upper()} with score {final_score:.2f}."
        )
        reason_parts.append(
            f"Bullish articles: {bullish_ratio * 100:.0f}%, "
            f"Bearish articles: {bearish_ratio * 100:.0f}%."
        )

        # Aggressive but tunable rule:
        if final_score >= buy_thr:
            signal = "BUY"
            # higher confidence if strongly positive
            if final_score >= buy_thr + 0.15:
                confidence = "high"
            else:
                confidence = "medium"
            reason_parts.append(
                "Overall news tone is positive, suggesting upside potential."
            )

        elif final_score <= sell_thr:
            signal = "SELL"
            # higher confidence if strongly negative
            if final_score <= sell_thr - 0.15:
                confidence = "high"
            else:
                confidence = "medium"
            reason_parts.append(
                "Overall news tone is negative, indicating downside risk."
            )

        else:
            signal = "HOLD"
            confidence = "medium"
            reason_parts.append(
                "Sentiment is close to neutral. Waiting for clearer news direction."
            )

    reason = " ".join(reason_parts)

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "reason": reason,
        "final_score": final_score,
        "sentiment_view": sentiment_view,
        "article_count": article_count,
        "bullish_ratio": bullish_ratio,
        "bearish_ratio": bearish_ratio,
    }

