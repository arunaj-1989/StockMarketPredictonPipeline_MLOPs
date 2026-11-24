"""NSE equity holidays and the dashboard's manual data refresh."""

import logging
import subprocess
import sys
import threading
from datetime import date, datetime

import pandas as pd
import requests

from data_prep import NSE_HEADERS, PROCESSED_DIR, ROOT

LOGGER = logging.getLogger("market_data")
SYNC_LOCK = threading.Lock()
HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"


class MarketDataError(RuntimeError):
    pass


def fetch_trading_holidays() -> dict[date, str]:
    """Use the cash-market calendar, not bank or commodity holidays."""
    try:
        with requests.Session() as session:
            session.headers.update({**NSE_HEADERS, "Accept": "application/json"})
            bootstrap = session.get("https://www.nseindia.com/", timeout=10)
            if bootstrap.status_code != 200:
                LOGGER.warning("NSE cookie bootstrap returned %s; trying the calendar endpoint", bootstrap.status_code)
            response = session.get(HOLIDAY_URL, timeout=15)
            response.raise_for_status()
            records = response.json()["CM"]
        holidays = {}
        for record in records:
            day = datetime.strptime(record["tradingDate"], "%d-%b-%Y").date()
            name = record["description"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("holiday description is missing")
            holidays[day] = name.strip()
        if not holidays:
            raise ValueError("cash-market calendar is empty")
        return dict(sorted(holidays.items()))
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("NSE holiday calendar unavailable: %s", exc)
        raise MarketDataError(f"NSE holiday calendar unavailable: {exc}") from exc


def sync_market_data() -> date:
    """Run the existing refresh-only pipeline; retain all training artifacts."""
    if not SYNC_LOCK.acquire(blocking=False):
        raise MarketDataError("A manual market-data sync is already running. Please try again later.")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "data_prep.py"), "--refresh-only"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=900,
        )
        if result.returncode:
            detail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-12:])
            raise MarketDataError(f"Market-data sync failed (exit {result.returncode}).\n{detail}")
        features = pd.read_parquet(PROCESSED_DIR / "latest_features.parquet", columns=["date"])
        latest = features["date"].max()
        if pd.isna(latest):
            raise MarketDataError("Market-data sync produced no dated features.")
        return pd.Timestamp(latest).date()
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        LOGGER.exception("Market-data sync failed")
        raise MarketDataError(f"Market-data sync failed: {exc}") from exc
    except MarketDataError:
        LOGGER.exception("Market-data sync failed")
        raise
    finally:
        SYNC_LOCK.release()
