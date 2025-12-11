import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from src.sentiment_model import FinBertSentimentAnalyzer
from src.news_fetcher import get_news_for_symbol
from src.sentiment_aggregator import aggregate_article_sentiments
from src.signal_generator import generate_trading_signal
from src.portfolio import apply_signals_and_update_portfolio
from src.config import (
    WATCHLIST,
    PROJECT_NAME,
    SENTIMENT_HISTORY_FILE,
    SENTIMENT_THRESHOLDS,
    PORTFOLIO_HISTORY_FILE,
)
from src.whatsapp_notifier import send_whatsapp_alerts_for_run


def build_text_from_article(article):
    title = article.get("title", "") or ""
    desc = article.get("description", "") or ""
    text = title.strip()
    if desc:
        text += ". " + desc.strip()
    return text


def analyze_symbol(analyzer, symbol: str, buy_thr: float, sell_thr: float):
    """
    Analyze a single symbol: fetch news, run FinBERT, aggregate sentiment,
    and generate a trading signal using the given thresholds.
    """
    articles = get_news_for_symbol(symbol)

    if not articles:
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
            "articles": [],
        }

    texts = [build_text_from_article(a) for a in articles]
    results = analyzer.predict(texts)

    enriched_results = []
    for res, article in zip(results, articles):
        enriched = dict(res)
        enriched["title"] = article.get("title", "").strip()
        enriched["source"] = article.get("source", "Unknown")
        enriched["published_at"] = article.get("published_at", "N/A")
        enriched["url"] = article.get("url", "")
        enriched_results.append(enriched)

    summary = aggregate_article_sentiments(enriched_results)
    if summary is None:
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
            "articles": enriched_results,
        }

    # use UI-provided thresholds
    signal_info = generate_trading_signal(
        symbol, summary, buy_thr=buy_thr, sell_thr=sell_thr
    )
    signal_info["articles"] = enriched_results  # attach article-level details

    return signal_info


# ----------------------------- logging details -----------------------------
def log_signals_to_csv(all_signals):
    """
    Append current run's signals to a CSV file for history/trend analysis.
    One row per stock per run.
    """
    rows = []
    run_ts = datetime.utcnow().isoformat()  # UTC timestamp

    for s in all_signals:
        rows.append(
            {
                "timestamp": run_ts,
                "symbol": s["symbol"],
                "final_score": s["final_score"],
                "signal": s["signal"],
                "sentiment_view": s["sentiment_view"],
                "confidence": s["confidence"],
                "article_count": s["article_count"],
                "bullish_ratio": s["bullish_ratio"],
                "bearish_ratio": s["bearish_ratio"],
            }
        )

    df_run = pd.DataFrame(rows)

    # Create file if not exists, otherwise append
    if SENTIMENT_HISTORY_FILE.exists():
        df_run.to_csv(SENTIMENT_HISTORY_FILE, mode="a", index=False, header=False)
    else:
        SENTIMENT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_run.to_csv(SENTIMENT_HISTORY_FILE, mode="w", index=False, header=True)


# ------------------------------ styling helpers -----------------------------
def color_signal_row(row):
    """
    Return a list of CSS styles for a row based on the Signal value.
    BUY  -> light green
    SELL -> light red
    HOLD -> light yellow
    """
    signal = row.get("Signal", "")
    base = [""] * len(row)

    if signal == "BUY":
        return ["background-color: rgba(0, 200, 0, 0.25)"] * len(row)
    elif signal == "SELL":
        return ["background-color: rgba(255, 0, 0, 0.25)"] * len(row)
    elif signal == "HOLD":
        return ["background-color: rgba(255, 215, 0, 0.20)"] * len(row)

    return base


# ------------------------- report generation helper ------------------------
def build_text_report(
    all_signals,
    df_summary: pd.DataFrame,
    portfolio_snapshot: dict,
    current_watchlist,
    buy_thr: float,
    sell_thr: float,
) -> str:
    """Build a plain-text report for download."""
    lines = []
    lines.append("Stock Market News Sentiment Analyzer – Report")
    lines.append("=" * 60)
    lines.append(f"Generated at (UTC): {datetime.utcnow().isoformat()}")
    lines.append("")

    # Settings
    lines.append("Settings")
    lines.append("-" * 60)
    lines.append(f"Watchlist: {', '.join(current_watchlist)}")
    lines.append(f"Buy threshold:  {buy_thr:.2f}")
    lines.append(f"Sell threshold: {sell_thr:.2f}")
    lines.append("")

    # Portfolio snapshot
    lines.append("Portfolio Snapshot (Paper Trading)")
    lines.append("-" * 60)
    lines.append(f"Equity        : {portfolio_snapshot['equity']:.2f}")
    lines.append(f"Cash          : {portfolio_snapshot['cash']:.2f}")
    lines.append(f"Realized PnL  : {portfolio_snapshot['realized_pnl']:.2f}")
    lines.append(f"Unrealized PnL: {portfolio_snapshot['unrealized_pnl']:.2f}")
    lines.append("")

    positions = portfolio_snapshot.get("positions", [])
    if positions:
        lines.append("Open Positions:")
        for pos in positions:
            lines.append(
                f"  - {pos['symbol']}: qty={pos['qty']}, "
                f"avg_cost={pos['avg_cost']:.2f}, "
                f"last_price={pos['last_price']:.2f}, "
                f"market_value={pos['market_value']:.2f}, "
                f"unrealized_pnl={pos['unrealized_pnl']:.2f}"
            )
    else:
        lines.append("Open Positions: none")
    lines.append("")

    # Summary table
    lines.append("Per-Symbol Sentiment & Signals")
    lines.append("-" * 60)
    for _, row in df_summary.iterrows():
        lines.append(
            f"{row['Symbol']}: Signal={row['Signal']}, "
            f"Score={row['Final Score']:.3f}, View={row['View']}, "
            f"Articles={row['Articles']}, "
            f"Bullish%={row['Bullish %']:.1f}, Bearish%={row['Bearish %']:.1f}"
        )
    lines.append("")

    # Signal explanations
    lines.append("Signal Explanations")
    lines.append("-" * 60)
    for s in all_signals:
        lines.append(f"{s['symbol']} – {s['signal']}")
        lines.append(f"  Score      : {s['final_score']:.3f}")
        lines.append(f"  View       : {s['sentiment_view']}")
        lines.append(f"  Confidence : {s['confidence']}")
        lines.append(f"  Reason     : {s['reason']}")
        lines.append("")

    return "\n".join(lines)


# ------------------------------ Main App -----------------------------
def main():
    st.set_page_config(page_title=PROJECT_NAME, layout="wide")

    st.title(PROJECT_NAME)
    st.caption(
        "News-driven sentiment analysis with FinBERT, trade signals, "
        "and a paper-trading portfolio"
    )

    # ---- Sidebar ----
    st.sidebar.header("Settings")

    # Customisable watchlist
    default_watchlist_str = ", ".join(WATCHLIST)
    watchlist_str = st.sidebar.text_input(
        "Watchlist (comma-separated symbols):",
        value=default_watchlist_str,
        help="Example: TSLA, INFY, AAPL, HDFCBANK",
    )

    current_watchlist = [
        s.strip().upper() for s in watchlist_str.split(",") if s.strip()
    ]

    st.sidebar.write("Active watchlist:")
    if current_watchlist:
        for s in current_watchlist:
            st.sidebar.write(f"- {s}")
    else:
        st.sidebar.write("_(no symbols entered)_")

    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
    refresh_minutes = st.sidebar.slider(
        "Refresh interval (minutes)", min_value=1, max_value=30, value=5
    )

    # ---- Signal Sensitivity (threshold sliders) ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Signal Sensitivity")

    buy_thr = st.sidebar.slider(
        "Buy threshold (sentiment score)",
        min_value=0.0,
        max_value=1.0,
        value=float(SENTIMENT_THRESHOLDS["buy"]),
        step=0.01,
        help="Higher = more conservative (fewer BUY signals).",
    )

    sell_thr = st.sidebar.slider(
        "Sell threshold (sentiment score)",
        min_value=-1.0,
        max_value=0.0,
        value=float(SENTIMENT_THRESHOLDS["sell"]),
        step=0.01,
        help="Lower = more conservative (fewer SELL signals).",
    )

    # If auto-refresh is enabled, rerun app every X minutes
    if auto_refresh:
        st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")
        run_clicked = True  # always run when page refreshes
    else:
        run_clicked = st.sidebar.button("Run Analysis")

    if run_clicked:
        if not current_watchlist:
            st.warning("Please enter at least one symbol in the watchlist.")
            return

        # Run analysis
        with st.spinner("Running FinBERT sentiment analysis..."):
            analyzer = FinBertSentimentAnalyzer()
            all_signals = [
                analyze_symbol(analyzer, sym, buy_thr, sell_thr)
                for sym in current_watchlist
            ]

        # Log this run into history CSV
        log_signals_to_csv(all_signals)

        # Create summary table
        df = pd.DataFrame(
            [
                {
                    "Symbol": s["symbol"],
                    "Signal": s["signal"],
                    "Confidence": s["confidence"],
                    "Final Score": round(s["final_score"], 3),
                    "View": s["sentiment_view"],
                    "Articles": s["article_count"],
                    "Bullish %": round(s["bullish_ratio"] * 100, 1),
                    "Bearish %": round(s["bearish_ratio"] * 100, 1),
                }
                for s in all_signals
            ]
        )

        # ----------------- Portfolio Simulator (Paper Trading) -----------------
        st.subheader("Portfolio Simulator (Paper Trading)")

        portfolio_snapshot = apply_signals_and_update_portfolio(all_signals)

        # ----------------- WhatsApp Alerts -----------------
        # (Actual sending is controlled by ENABLE_WHATSAPP_ALERTS in config)
        send_whatsapp_alerts_for_run(all_signals, portfolio_snapshot)

        # ----------------- Generate Text Report -----------------
        report_text = build_text_report(
            all_signals=all_signals,
            df_summary=df,
            portfolio_snapshot=portfolio_snapshot,
            current_watchlist=current_watchlist,
            buy_thr=buy_thr,
            sell_thr=sell_thr,
        )

        st.subheader("Export Report")
        st.download_button(
            label="Download text report",
            data=report_text,
            file_name=f"sentiment_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Portfolio Equity", f"{portfolio_snapshot['equity']:.2f}")
        col_p2.metric("Cash Balance", f"{portfolio_snapshot['cash']:.2f}")
        col_p3.metric("Realized PnL", f"{portfolio_snapshot['realized_pnl']:.2f}")
        col_p4.metric("Unrealized PnL", f"{portfolio_snapshot['unrealized_pnl']:.2f}")

        positions = portfolio_snapshot.get("positions", [])
        if positions:
            st.markdown("**Open Positions**")
            pos_df = pd.DataFrame(positions)
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.markdown("_No open positions currently._")

        # ----------------- Signal Summary KPIs -----------------
        st.subheader("Signal Summary")

        buy_count = (df["Signal"] == "BUY").sum()
        sell_count = (df["Signal"] == "SELL").sum()
        hold_count = (df["Signal"] == "HOLD").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("BUY signals", buy_count)
        col2.metric("SELL signals", sell_count)
        col3.metric("HOLD signals", hold_count)

        # ----------------- Summary table with color coding -----------------
        st.subheader("Summary of Trade Signals")

        styled_df = df.style.apply(color_signal_row, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        # ----------------- Sentiment Charts -----------------
        st.subheader("Sentiment Charts")

        # 1) Bar chart: Final sentiment score per stock
        score_chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Symbol:N", title="Stock"),
                y=alt.Y("Final Score:Q", title="Sentiment Score"),
                color=alt.Color("Signal:N", title="Signal"),
                tooltip=[
                    "Symbol",
                    "Signal",
                    "Final Score",
                    "View",
                    "Bullish %",
                    "Bearish %",
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(score_chart, use_container_width=True)

        # 2) Stacked bar: Bullish vs Bearish % per stock
        stacked_source = df.melt(
            id_vars=["Symbol"],
            value_vars=["Bullish %", "Bearish %"],
            var_name="Sentiment",
            value_name="Percentage",
        )

        stacked_chart = (
            alt.Chart(stacked_source)
            .mark_bar()
            .encode(
                x=alt.X("Symbol:N", title="Stock"),
                y=alt.Y(
                    "Percentage:Q",
                    title="Percentage (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color("Sentiment:N", title="Sentiment Type"),
                tooltip=["Symbol", "Sentiment", "Percentage"],
            )
            .properties(height=300)
        )
        st.altair_chart(stacked_chart, use_container_width=True)

        # ----------------- Sentiment Trend Over Time -----------------
        st.subheader("Sentiment Trend Over Time")

        if SENTIMENT_HISTORY_FILE.exists():
            df_hist = pd.read_csv(SENTIMENT_HISTORY_FILE)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

            symbols_available = sorted(df_hist["symbol"].unique().tolist())
            selected_symbols = st.multiselect(
                "Select stocks to show in trend chart:",
                options=symbols_available,
                default=symbols_available,
            )

            df_hist_filtered = df_hist[df_hist["symbol"].isin(selected_symbols)]

            if not df_hist_filtered.empty:
                trend_chart = (
                    alt.Chart(df_hist_filtered)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("timestamp:T", title="Time"),
                        y=alt.Y("final_score:Q", title="Sentiment Score"),
                        color=alt.Color("symbol:N", title="Stock"),
                        tooltip=[
                            "timestamp:T",
                            "symbol:N",
                            "final_score:Q",
                            "signal:N",
                            "sentiment_view:N",
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(trend_chart, use_container_width=True)
            else:
                st.info("No historical data available for selected stocks yet.")
        else:
            st.info("Run the analysis a few times to build sentiment history.")

        # ----------------- Portfolio Equity Curve -----------------
        st.subheader("Portfolio Equity Curve")

        if PORTFOLIO_HISTORY_FILE.exists():
            df_port = pd.read_csv(PORTFOLIO_HISTORY_FILE)
            df_port["timestamp"] = pd.to_datetime(df_port["timestamp"])

            equity_chart = (
                alt.Chart(df_port)
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("equity:Q", title="Portfolio Equity"),
                    tooltip=[
                        "timestamp:T",
                        "equity:Q",
                        "cash:Q",
                        "realized_pnl:Q",
                        "unrealized_pnl:Q",
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(equity_chart, use_container_width=True)
        else:
            st.info("No portfolio history yet. Run analysis a few times to build it.")

        # ----------------- Per-stock details -----------------
        st.subheader("Per-Stock Details")

        for s in all_signals:
            with st.expander(
                f"{s['symbol']} – {s['signal']} "
                f"(score={s['final_score']:.2f}, view={s['sentiment_view']})"
            ):
                # Color-coded signal text
                sig_color = (
                    "green" if s["signal"] == "BUY"
                    else "red" if s["signal"] == "SELL"
                    else "gold"
                )
                st.markdown(
                    f"**Signal:** <span style='color:{sig_color}; "
                    f"font-weight:bold'>{s['signal']}</span>  \n"
                    f"**Confidence:** {s['confidence']}  \n"
                    f"**Reason:** {s['reason']}",
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.markdown("**Underlying News Articles:**")
                for art in s.get("articles", []):
                    st.markdown(
                        f"- **{art['title']}**  \n"
                        f"  Source: `{art['source']}` | Published: `{art['published_at']}`  \n"
                        f"  Sentiment: **{art['label']}** (score={art['score']:.3f})"
                    )
                    if art["url"]:
                        st.markdown(f"  [Read more]({art['url']})")
                    st.markdown("---")

    else:
        st.info(
            "Enter a watchlist in the sidebar and click **Run Analysis**, "
            "or enable auto-refresh to analyze continuously."
        )


if __name__ == "__main__":
    main()
