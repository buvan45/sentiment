from __future__ import annotations

from typing import List, Dict
from twilio.rest import Client

from .config import (
    PROJECT_NAME,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_FROM,
    TWILIO_WHATSAPP_TO,
    ENABLE_WHATSAPP_ALERTS,
)


def build_alert_message(all_signals: List[Dict], portfolio_snapshot: Dict) -> str:
    """
    Build a concise alert message summarizing BUY/SELL signals and portfolio status.
    """
    lines = []
    lines.append(f"{PROJECT_NAME} – WhatsApp Alert")
    lines.append("")

    # Only alert on actionable signals (BUY/SELL)
    actionable = [s for s in all_signals if s["signal"] in ("BUY", "SELL")]

    if not actionable:
        lines.append("No actionable signals (all HOLD).")
    else:
        lines.append("Signals:")
        for s in actionable:
            lines.append(
                f"- {s['symbol']}: {s['signal']} | "
                f"score={s['final_score']:.2f}, view={s['sentiment_view']}, "
                f"conf={s['confidence']}"
            )

    lines.append("")
    lines.append("Portfolio (paper trading):")
    lines.append(f"  Equity        : {portfolio_snapshot['equity']:.2f}")
    lines.append(f"  Cash          : {portfolio_snapshot['cash']:.2f}")
    lines.append(f"  Realized PnL  : {portfolio_snapshot['realized_pnl']:.2f}")
    lines.append(f"  Unrealized PnL: {portfolio_snapshot['unrealized_pnl']:.2f}")

    return "\n".join(lines)


def send_whatsapp_message(text: str) -> None:
    """
    Send a WhatsApp message using Twilio API.
    Safe: logs to console if config is missing.
    """
    if not ENABLE_WHATSAPP_ALERTS:
        return

    if (
        not TWILIO_ACCOUNT_SID
        or not TWILIO_AUTH_TOKEN
        or not TWILIO_WHATSAPP_FROM
        or not TWILIO_WHATSAPP_TO
    ):
        print("[WhatsApp Alerts] Enabled, but Twilio credentials/numbers not set.")
        return

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=text,
            from_=TWILIO_WHATSAPP_FROM,
            to=TWILIO_WHATSAPP_TO,
        )
        print(f"[WhatsApp Alerts] Message SID: {msg.sid}")
    except Exception as e:
        print(f"[WhatsApp Alerts] Error sending message: {e}")


def send_whatsapp_alerts_for_run(all_signals: List[Dict], portfolio_snapshot: Dict) -> None:
    """
    Public entrypoint: build message + send it via WhatsApp.
    Never raises; only logs errors to console.
    """
    try:
        text = build_alert_message(all_signals, portfolio_snapshot)
        send_whatsapp_message(text)
    except Exception as e:
        print(f"[WhatsApp Alerts] Unexpected error: {e}")
