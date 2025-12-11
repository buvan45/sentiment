import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # load variables from .env

# -----------------------------
# General project settings
# -----------------------------
PROJECT_NAME = "Stock Market News Sentiment Analyzer"

# Stocks we care about
WATCHLIST = ["TSLA", "INFY", "AAPL", "HDFCBANK"]

# -----------------------------
# News source configuration
# -----------------------------
# Options: "local" (JSON file), "api"
NEWS_SOURCE_MODE = "api"   # <--- switch to "api" to use NewsAPI

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_NEWS_FILE = DATA_DIR / "sample_news.json"

# NEW: file to store sentiment history
SENTIMENT_HISTORY_FILE = DATA_DIR / "sentiment_history.csv"


# ---- Portfolio simulator settings ----
PORTFOLIO_TRADES_FILE = DATA_DIR / "portfolio_trades.csv"
PORTFOLIO_HISTORY_FILE = DATA_DIR / "portfolio_history.csv"

INITIAL_CAPITAL = 100000.0      # starting virtual capital
MAX_TRADE_FRACTION = 0.25       # max 25% of equity per new BUY trade



# NewsAPI configuration
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # from .env
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
NEWSAPI_LANGUAGE = "en"
NEWSAPI_PAGE_SIZE = 5  # number of articles per stock

# ----------------------------- whatsapp alert settings -----------------------------
# ------------- WhatsApp Alert Settings (via Twilio) -------------

# These are read from environment variables / .env
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")  # e.g. 'whatsapp:+14155238886'
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "")      # e.g. 'whatsapp:+91XXXXXXXXXX'
ENABLE_WHATSAPP_ALERTS = os.getenv("ENABLE_WHATSAPP_ALERTS", "false").lower() == "true"


# Map symbols -> query text for NewsAPI
SYMBOL_TO_QUERY = {
    "TSLA": "Tesla",
    "INFY": "Infosys",
    "AAPL": "Apple Inc",
    "HDFCBANK": "HDFC Bank",
}

# -----------------------------
# Sentiment / signal settings
# -----------------------------
SENTIMENT_THRESHOLDS = {
    "buy": 0.02,   # very sensitive
    "sell": -0.02  # very sensitive
}

