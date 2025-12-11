import json
from typing import List, Dict
import requests
from requests.exceptions import ReadTimeout, RequestException

from .config import (
    SAMPLE_NEWS_FILE,
    NEWS_SOURCE_MODE,
    WATCHLIST,
    PROJECT_NAME,
    NEWSAPI_KEY,
    NEWSAPI_ENDPOINT,
    NEWSAPI_LANGUAGE,
    NEWSAPI_PAGE_SIZE,
    SYMBOL_TO_QUERY,
)


# -----------------------------
# Local JSON sample loader
# -----------------------------
def load_local_sample_news() -> Dict[str, List[Dict]]:
    """
    Load sample news data from the JSON file.

    Returns:
        dict mapping stock symbol -> list of news articles
    """
    if not SAMPLE_NEWS_FILE.exists():
        raise FileNotFoundError(f"Sample news file not found at: {SAMPLE_NEWS_FILE}")

    with open(SAMPLE_NEWS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid sample news format: expected a dict at top level")

    return data


# -----------------------------
# NewsAPI integration
# -----------------------------
def fetch_news_from_newsapi(symbol: str) -> List[Dict]:
    """
    Fetch recent news articles for a given stock symbol using NewsAPI.

    Returns a list of dicts with keys:
        - title
        - description
        - source
        - published_at
        - url

    On timeout or any network error, returns an empty list instead of crashing.
    """
    if not NEWSAPI_KEY:
        raise RuntimeError(
            "NEWSAPI_KEY is not set. Please add it to your .env file as NEWSAPI_KEY=<your_key>."
        )

    query = SYMBOL_TO_QUERY.get(symbol, symbol)

    params = {
        "q": query,
        "language": NEWSAPI_LANGUAGE,
        "sortBy": "publishedAt",
        "pageSize": NEWSAPI_PAGE_SIZE,
        "apiKey": NEWSAPI_KEY,
    }

    try:
        # Increased timeout from 10s -> 20s to be more tolerant
        resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=20)
    except ReadTimeout:
        # Log in console, but don't crash the app
        print(
            f"[{PROJECT_NAME}] ReadTimeout while calling NewsAPI for symbol {symbol}. "
            f"Returning no articles."
        )
        return []
    except RequestException as e:
        # Any other network-related error
        print(
            f"[{PROJECT_NAME}] Error calling NewsAPI for symbol {symbol}: {e}. "
            f"Returning no articles."
        )
        return []

    if resp.status_code != 200:
        print(
            f"[{PROJECT_NAME}] NewsAPI request failed for symbol {symbol} "
            f"with status code {resp.status_code}: {resp.text}"
        )
        return []

    data = resp.json()
    articles_raw = data.get("articles", [])

    articles: List[Dict] = []
    for a in articles_raw:
        articles.append(
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": (a.get("source") or {}).get("name", "Unknown"),
                "published_at": a.get("publishedAt", "N/A"),
                "url": a.get("url", ""),
            }
        )

    return articles


# -----------------------------
# Public function
# -----------------------------
def get_news_for_symbol(symbol: str) -> List[Dict]:
    """
    Get list of news articles for a given stock symbol.

    Mode:
        - "local": load from sample JSON
        - "api"  : fetch from NewsAPI
    """
    if NEWS_SOURCE_MODE == "local":
        all_data = load_local_sample_news()
        return all_data.get(symbol, [])

    elif NEWS_SOURCE_MODE == "api":
        return fetch_news_from_newsapi(symbol)

    else:
        raise ValueError(f"Unsupported NEWS_SOURCE_MODE: {NEWS_SOURCE_MODE}")


# -----------------------------
# Demo
# -----------------------------
def demo():
    """
    Small demo to check that news loading works properly.
    """
    print(f"=== {PROJECT_NAME} – News Fetcher Demo ===\n")
    print(f"Mode: {NEWS_SOURCE_MODE}")
    if NEWS_SOURCE_MODE == "local":
        print(f"Using sample file: {SAMPLE_NEWS_FILE}")
    else:
        print("Using NewsAPI as live data source.")
    print(f"Watchlist: {WATCHLIST}\n")

    for symbol in WATCHLIST:
        try:
            articles = get_news_for_symbol(symbol)
        except Exception as e:
            print(f"Symbol: {symbol} | ERROR: {e}")
            print("-" * 80)
            continue

        print(f"Symbol: {symbol} | Articles found: {len(articles)}")
        for i, a in enumerate(articles, start=1):
            title = a.get("title", "").strip()
            source = a.get("source", "Unknown")
            published_at = a.get("published_at", "N/A")
            print(f"  [{i}] {title}")
            print(f"       Source: {source} | Published: {published_at}")
        print("-" * 80)


if __name__ == "__main__":
    demo()
