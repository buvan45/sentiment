from __future__ import annotations

from typing import Dict, List, Tuple
from datetime import datetime

import pandas as pd
import yfinance as yf

from .config import (
    PORTFOLIO_TRADES_FILE,
    PORTFOLIO_HISTORY_FILE,
    INITIAL_CAPITAL,
    MAX_TRADE_FRACTION,
)

# ============================================================
# INTERNAL HELPERS
# ============================================================

def _load_trades_df() -> pd.DataFrame:
    """
    Load existing trades from CSV. If none, return an empty DataFrame.
    Always ensures timestamp column is datetime.
    """
    if PORTFOLIO_TRADES_FILE.exists():
        df = pd.read_csv(PORTFOLIO_TRADES_FILE)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    PORTFOLIO_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    return pd.DataFrame(
        columns=["timestamp", "symbol", "side", "qty", "price", "value"]
    )


def _get_latest_price(symbol: str) -> float:
    """
    Fetch latest closing price from yfinance.
    If price cannot be fetched, return 0.0.
    """
    try:
        data = yf.download(symbol, period="1d", interval="1d", progress=False)
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        print(f"[Portfolio] Error fetching price for {symbol}: {e}")
    return 0.0


def _replay_trades(trades_df: pd.DataFrame) -> Tuple[float, Dict[str, Dict], float]:
    """
    Replay trades in chronological order to compute:
    - cash
    - open positions
    - realized PnL
    """

    cash = INITIAL_CAPITAL
    positions: Dict[str, Dict] = {}
    realized_pnl = 0.0

    if trades_df.empty:
        return cash, positions, realized_pnl

    # FIX: Ensure timestamps are datetime everywhere
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce")

    # FIX: Sort cleanly after fixing types
    trades_sorted = trades_df.sort_values("timestamp").reset_index(drop=True)

    for _, row in trades_sorted.iterrows():
        sym = str(row["symbol"])
        side = str(row["side"]).upper()
        qty = float(row["qty"])
        price = float(row["price"])

        if side == "BUY":
            cost = qty * price
            cash -= cost
            positions[sym] = {"qty": qty, "avg_cost": price}

        elif side == "SELL":
            if sym in positions:
                pos = positions.pop(sym)
                sell_value = qty * price
                buy_value = pos["qty"] * pos["avg_cost"]

                cash += sell_value
                realized_pnl += (price - pos["avg_cost"]) * qty

    return cash, positions, realized_pnl


def _compute_snapshot(trades_df: pd.DataFrame) -> Dict:
    """
    Compute current portfolio snapshot using latest prices.
    """

    cash, positions, realized_pnl = _replay_trades(trades_df)

    unrealized_pnl = 0.0
    equity = cash
    market_positions = []

    for sym, pos in positions.items():
        last_price = _get_latest_price(sym)

        if last_price <= 0:
            last_price = pos["avg_cost"]

        market_value = pos["qty"] * last_price
        u_pnl = pos["qty"] * (last_price - pos["avg_cost"])
        unrealized_pnl += u_pnl
        equity += market_value

        market_positions.append(
            {
                "symbol": sym,
                "qty": pos["qty"],
                "avg_cost": pos["avg_cost"],
                "last_price": last_price,
                "market_value": market_value,
                "unrealized_pnl": u_pnl,
            }
        )

    return {
        "cash": cash,
        "equity": equity,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "positions": market_positions,
    }


# ============================================================
# PUBLIC MAIN FUNCTION
# ============================================================

def apply_signals_and_update_portfolio(all_signals: List[Dict]) -> Dict:
    """
    Apply BUY/SELL signals to the virtual portfolio.
    """

    trades_df = _load_trades_df()
    cash, positions, realized_pnl = _replay_trades(trades_df)

    approx_equity = cash + sum(pos["qty"] * pos["avg_cost"] for pos in positions.values())

    new_trades: List[Dict] = []
    now_iso = datetime.utcnow().isoformat()

    for sig in all_signals:
        sym = sig["symbol"]
        action = sig["signal"].upper()

        # ---------------- BUY ------------------
        if action == "BUY" and sym not in positions:
            price = _get_latest_price(sym)
            if price <= 0:
                print(f"[Portfolio] Skipping BUY for {sym}: no price data.")
                continue

            trade_budget = min(MAX_TRADE_FRACTION * approx_equity, cash)
            qty = int(trade_budget // price)

            if qty <= 0:
                print(f"[Portfolio] Not enough cash to buy {sym}.")
                continue

            value = qty * price
            cash -= value

            positions[sym] = {"qty": qty, "avg_cost": price}

            new_trades.append(
                {
                    "timestamp": now_iso,
                    "symbol": sym,
                    "side": "BUY",
                    "qty": qty,
                    "price": price,
                    "value": value,
                }
            )

        # ---------------- SELL ------------------
        elif action == "SELL" and sym in positions:
            price = _get_latest_price(sym)
            if price <= 0:
                print(f"[Portfolio] Skipping SELL for {sym}: no price data.")
                continue

            qty = positions[sym]["qty"]
            value = qty * price
            cash += value

            realized_pnl += (price - positions[sym]["avg_cost"]) * qty

            new_trades.append(
                {
                    "timestamp": now_iso,
                    "symbol": sym,
                    "side": "SELL",
                    "qty": qty,
                    "price": price,
                    "value": value,
                }
            )

            del positions[sym]

    # Save new trades
    if new_trades:
        df_new = pd.DataFrame(new_trades)
        trades_df = pd.concat([trades_df, df_new], ignore_index=True)
        trades_df.to_csv(PORTFOLIO_TRADES_FILE, index=False)

    # Compute final snapshot
    snapshot = _compute_snapshot(trades_df)

    # Log portfolio history (equity curve)
    hist_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "equity": snapshot["equity"],
        "cash": snapshot["cash"],
        "realized_pnl": snapshot["realized_pnl"],
        "unrealized_pnl": snapshot["unrealized_pnl"],
    }

    if PORTFOLIO_HISTORY_FILE.exists():
        pd.DataFrame([hist_row]).to_csv(
            PORTFOLIO_HISTORY_FILE, mode="a", header=False, index=False
        )
    else:
        PORTFOLIO_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([hist_row]).to_csv(
            PORTFOLIO_HISTORY_FILE, mode="w", header=True, index=False
        )

    return snapshot
