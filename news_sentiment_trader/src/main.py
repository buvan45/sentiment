from typing import List, Dict

from .config import WATCHLIST, PROJECT_NAME
from .news_fetcher import get_news_for_symbol
from .sentiment_model import FinBertSentimentAnalyzer
from .sentiment_aggregator import aggregate_article_sentiments
from .signal_generator import generate_trading_signal


def build_text_from_article(article: Dict) -> str:
    """
    Build a single text string from news article fields that will be
    passed into FinBERT.

    For now, we simply combine title + description.
    """
    title = article.get("title", "") or ""
    desc = article.get("description", "") or ""
    text = title.strip()
    if desc:
        text += ". " + desc.strip()
    return text


def analyze_symbol_news(analyzer: FinBertSentimentAnalyzer, symbol: str) -> Dict:
    """
    Fetch news for a given stock symbol, run FinBERT on each article,
    aggregate sentiment, generate a trading signal, and print a summary.

    Returns:
        signal_info dict from generate_trading_signal(...)
    """
    articles = get_news_for_symbol(symbol)

    print(f"\n=== {symbol} – News & Sentiment ===")
    if not articles:
        print("No articles found.")
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": "low",
            "reason": "No recent news articles available.",
            "final_score": 0.0,
            "sentiment_view": "mixed/neutral",
            "article_count": 0,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
        }

    # Prepare texts for FinBERT
    texts: List[str] = [build_text_from_article(a) for a in articles]

    # Run FinBERT
    results = analyzer.predict(texts)

    # Attach basic metadata (title, source, url) to each result for nicer printing
    enriched_results: List[Dict] = []
    for res, article in zip(results, articles):
        enriched = dict(res)  # copy FinBERT result
        enriched["title"] = article.get("title", "").strip()
        enriched["source"] = article.get("source", "Unknown")
        enriched["published_at"] = article.get("published_at", "N/A")
        enriched["url"] = article.get("url", "")
        enriched_results.append(enriched)

    # Print per-article sentiment
    for i, r in enumerate(enriched_results, start=1):
        print(f"\nArticle {i}: {r['title']}")
        print(f"  Source      : {r['source']} | Published: {r['published_at']}")
        print(f"  Label       : {r['label']}")
        print(f"  Score       : {r['score']:.4f}")
        print(f"  Probabilities -> pos={r['positive']:.4f}, neg={r['negative']:.4f}, neu={r['neutral']:.4f}")
        if r["url"]:
            print(f"  URL         : {r['url']}")

    # Aggregate to get overall stock sentiment
    summary = aggregate_article_sentiments(enriched_results)
    if summary:
        print("\n--- Aggregated Sentiment Summary ---")
        print(f"Articles           : {summary['article_count']}")
        print(f"Final score        : {summary['final_score']:.4f}")
        print(f"Avg positive       : {summary['avg_positive']:.4f}")
        print(f"Avg negative       : {summary['avg_negative']:.4f}")
        print(f"Avg neutral        : {summary['avg_neutral']:.4f}")
        print(f"Bullish ratio      : {summary['bullish_ratio']:.2f}")
        print(f"Bearish ratio      : {summary['bearish_ratio']:.2f}")
        print(f"Neutral ratio      : {summary['neutral_ratio']:.2f}")
        print(f"Overall view       : {summary['sentiment_view'].upper()}")

        # 🔥 Generate trading signal here
        signal_info = generate_trading_signal(symbol, summary)

        print("\n*** Trading Signal ***")
        print(f"Signal       : {signal_info['signal']}")
        print(f"Confidence   : {signal_info['confidence']}")
        print(f"Reason       : {signal_info['reason']}")

        print("\n" + "-" * 90)
        return signal_info

    # Fallback (should not normally happen)
    print("\nCould not compute sentiment summary.")
    print("\n" + "-" * 90)
    return {
        "symbol": symbol,
        "signal": "HOLD",
        "confidence": "low",
        "reason": "Could not compute sentiment summary.",
        "final_score": 0.0,
        "sentiment_view": "mixed/neutral",
        "article_count": len(articles),
        "bullish_ratio": 0.0,
        "bearish_ratio": 0.0,
    }


def main():
    print(f"=== {PROJECT_NAME} – Full Sentiment Pipeline with Trading Signals ===\n")
    analyzer = FinBertSentimentAnalyzer()

    all_signals: List[Dict] = []

    for symbol in WATCHLIST:
        signal_info = analyze_symbol_news(analyzer, symbol)
        all_signals.append(signal_info)

    # Print a compact summary at the end
    print("\n\n=== Summary of Signals ===")
    for s in all_signals:
        print(
            f"{s['symbol']}: Signal={s['signal']}, "
            f"Score={s['final_score']:.2f}, "
            f"View={s['sentiment_view']}, "
            f"Confidence={s['confidence']}"
        )


if __name__ == "__main__":
    main()
