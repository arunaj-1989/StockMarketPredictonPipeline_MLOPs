"""Data preparation for the Nifty 50 multi-horizon forecasting pipeline.

Stages: NSE constituents -> adjusted OHLCV -> RSS news + FinBERT sentiment (aligned to
the 3:30 PM IST close) -> stationary features/targets -> chronological split with a
10-trading-day embargo -> train-only preprocessor fit.

Note: RSS feeds only expose recent headlines. Raw news is appended to
``data/raw/news.parquet`` on every run so history accumulates; days without scored news
get neutral (0) sentiment features.
"""

from __future__ import annotations

import argparse
import gc
import io
import logging
import re
from datetime import time as dtime
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import pytz
import requests

LOGGER = logging.getLogger("data_prep")

RANDOM_STATE = 42
SUPPORTED_HORIZONS = [1, 2, 3, 5, 7, 10]
EMBARGO_DAYS = 10  # == max(SUPPORTED_HORIZONS)
HISTORY_YEARS = 6
IST = pytz.timezone("Asia/Kolkata")
MARKET_CLOSE_IST = dtime(15, 30)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

NSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
FALLBACK_NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT",
    "ULTRACEMCO", "WIPRO",
]
NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

RSS_FEEDS = {
    "economictimes": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "economictimes_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
}
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32

TECH_FEATURES = [
    "ret_1d", "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "rsi_14",
    "macd_norm", "atr_14_norm", "ema20_dist", "volume_z20", "rel_strength_5d",
    "india_vix",
]
NLP_FEATURES = ["stock_sentiment_1d", "stock_news_volume", "sentiment_momentum_5d"]
NUMERIC_FEATURES = TECH_FEATURES + NLP_FEATURES
CATEGORICAL_FEATURES = ["symbol"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMNS = [f"target_{k}" for k in SUPPORTED_HORIZONS]


# --------------------------------------------------------------------------- ingestion
def fetch_nifty50_symbols(timeout: int = 15) -> list[str]:
    """Return current Nifty 50 NSE symbols; fall back to a static list on any failure."""
    try:
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com/", timeout=timeout)  # seed cookies
        resp = session.get(NSE_URL, timeout=timeout)
        resp.raise_for_status()
        symbols = pd.read_csv(io.StringIO(resp.text))["Symbol"].str.strip().tolist()
        if len(symbols) != 50:
            raise ValueError(f"expected 50 constituents, got {len(symbols)}")
        return symbols
    except (requests.RequestException, ValueError, KeyError, pd.errors.ParserError) as exc:
        # network/format problems must not break the pipeline
        LOGGER.warning("NSE constituents fetch failed (%s); using fallback list", exc)
        return list(FALLBACK_NIFTY50)


def download_prices(
    symbols: list[str], years: int = HISTORY_YEARS, period: str | None = None
) -> pd.DataFrame:
    """Download split/dividend-adjusted daily OHLCV for the symbols (long format)."""
    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols]
    data = yf.download(
        tickers, period=period or f"{years}y", interval="1d", auto_adjust=True,
        group_by="ticker", threads=True, progress=False,
    )
    frames = []
    for symbol, ticker in zip(symbols, tickers):
        try:
            sub = data[ticker][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        except KeyError:
            LOGGER.warning("No price data for %s; skipping", ticker)
            continue
        if sub.empty:
            LOGGER.warning("Empty price data for %s; skipping", ticker)
            continue
        sub = sub.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
        sub["symbol"] = symbol
        frames.append(sub)
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def download_index_series(years: int = HISTORY_YEARS) -> tuple[pd.Series, pd.Series]:
    """Return (nifty_close, india_vix_close) indexed by normalized date."""
    import yfinance as yf

    out = []
    for ticker in (NIFTY_TICKER, VIX_TICKER):
        df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out.append(s.rename(ticker))
    return out[0], out[1]


def valid_article_url(value: object) -> str | None:
    """Return a safe HTTP(S) article URL, or None if it is missing or malformed."""
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url or re.search(r'[\s\x00-\x1f\x7f<>"\\]', url):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if parts.scheme not in {"http", "https"} or not host or parts.username is not None:
            return None
        _ = parts.port
        if ":" in host:
            ip_address(host)
        else:
            domain = host.rstrip(".").encode("idna").decode("ascii")
            labels = domain.split(".")
            if len(domain) > 253 or any(
                len(label) > 63 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
                for label in labels
            ):
                return None
    except (UnicodeError, ValueError):
        return None
    return url


def normalize_article_urls(values: pd.Series) -> pd.Series:
    """Validate article links and log nonempty links that cannot safely be used."""
    urls = values.map(valid_article_url).astype("object")
    present = values.notna() & values.map(lambda value: not isinstance(value, str) or bool(value.strip()))
    invalid = present & urls.isna()
    if invalid.any():
        LOGGER.warning("Ignoring %d malformed or unsafe article URLs", invalid.sum())
    return urls


def _fill_missing_article_urls(news: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Fill missing links only from a single unambiguous exact-identity match."""
    news = news.copy()
    if "url" not in news:
        news["url"] = None
    news["url"] = normalize_article_urls(news["url"])
    candidates = news.dropna(subset=[*keys, "url"])
    missing = news["url"].isna() & news[keys].notna().all(axis=1)
    if candidates.empty or not missing.any():
        return news
    groups = candidates.groupby(keys, sort=False)["url"]
    counts = groups.nunique()
    identities = pd.MultiIndex.from_frame(news.loc[missing, keys])
    ambiguous = identities[identities.isin(counts[counts > 1].index)].unique()
    if len(ambiguous):
        LOGGER.warning("Conflicting article URLs for %d headline identities; links were not backfilled",
                       len(ambiguous))
    links = groups.first()[counts == 1]
    news.loc[missing, "url"] = links.reindex(identities).to_numpy()
    return news


def fetch_news(timeout: int = 20) -> pd.DataFrame:
    """Fetch RSS headlines with their UTC publication date, source, and article URL."""
    import feedparser

    rows = []
    for source, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, headers=NSE_HEADERS, timeout=timeout)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("RSS fetch failed for %s: %s", source, exc)
            continue
        for entry in feed.entries:
            parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            title = (entry.get("title") or "").strip()
            if not parsed or not title:
                continue
            rows.append({
                "published_at": pd.Timestamp(*parsed[:6], tz="UTC"),
                "title": title,
                "source": source,
                "url": entry.get("link"),
            })
    news = pd.DataFrame(rows, columns=["published_at", "title", "source", "url"])
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True)
    news["url"] = normalize_article_urls(news["url"])
    return news


def merge_news_history(new: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Append headlines and backfill URLs without changing title/timestamp deduplication."""
    path = path or RAW_DIR / "news.parquet"
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    else:
        new = new.copy()
    if new.empty:
        if "url" not in new:
            new["url"] = None
        return new
    new["published_at"] = pd.to_datetime(new["published_at"], utc=True)
    new = _fill_missing_article_urls(new, ["title", "published_at", "source"])
    return (
        new.drop_duplicates(subset=["title", "published_at"], keep="first")
        .sort_values("published_at").reset_index(drop=True)
    )


# --------------------------------------------------------------------------- sentiment
def score_sentiment(
    titles: list[str], batch_size: int = FINBERT_BATCH_SIZE, max_length: int = 128
) -> np.ndarray:
    """Score headlines with FinBERT: net_sentiment = P_pos - P_neg.

    Uses FP16 on CUDA and always frees the model and VRAM afterwards.
    """
    if not titles:
        return np.array([], dtype="float32")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        LOGGER.warning("CUDA unavailable; scoring FinBERT on CPU in float32")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL, dtype=dtype)
    model.to(device).eval()
    id2label = {int(i): lbl.lower() for i, lbl in model.config.id2label.items()}
    pos_idx = next(i for i, lbl in id2label.items() if lbl == "positive")
    neg_idx = next(i for i, lbl in id2label.items() if lbl == "negative")

    scores: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(titles), batch_size):
                batch = titles[start:start + batch_size]
                enc = tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
                scores.append(probs[:, pos_idx] - probs[:, neg_idx])
        if device == "cuda":
            LOGGER.info("Peak VRAM: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(scores).astype("float32")


# --------------------------------------------------------------- point-in-time alignment
def assign_effective_trading_date(
    published_at: pd.Series, trading_days: pd.DatetimeIndex
) -> pd.Series:
    """Map publication timestamps to the trading session in which the news is usable.

    News published after 3:30 PM IST is shifted to the next calendar day; the result is
    then rolled forward to the first trading session on/after that day. News that falls
    beyond the last known trading day gets NaT (its session has not happened yet).
    """
    ist = pd.to_datetime(published_at, utc=True).dt.tz_convert(IST)
    close_td = pd.Timedelta(hours=MARKET_CLOSE_IST.hour, minutes=MARKET_CLOSE_IST.minute)
    after_close = (ist - ist.dt.normalize()) > close_td
    candidate = ist.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")

    days = pd.DatetimeIndex(trading_days).sort_values().normalize()
    pos = days.searchsorted(candidate.to_numpy(), side="left")
    valid = pos < len(days)
    out = np.full(len(candidate), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[valid] = days[pos[valid]].to_numpy()
    return pd.Series(out, index=published_at.index)


def _stock_patterns(symbols: list[str]) -> dict[str, re.Pattern]:
    patterns = {}
    for s in symbols:
        body = re.escape(s.replace("-", " "))
        # Short tickers (LT, ITC, TCS...) must match in upper case to avoid common words.
        flags = 0 if len(s) <= 3 else re.IGNORECASE
        patterns[s] = re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", flags)
    return patterns


def tag_headlines(news: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Expand headlines to one row per mentioned Nifty 50 symbol."""
    patterns = _stock_patterns(symbols)
    rows = []
    for idx, title in news["title"].items():
        norm = title.replace("-", " ")
        rows.extend((idx, sym) for sym, pat in patterns.items() if pat.search(norm))
    if not rows:
        return pd.DataFrame(columns=[*news.columns, "symbol"])
    tagged = pd.DataFrame(rows, columns=["idx", "symbol"]).merge(
        news, left_on="idx", right_index=True
    )
    return tagged.drop(columns="idx").reset_index(drop=True)


def aggregate_news_features(scored: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Daily per-stock NLP features aligned on ``effective_trading_date``.

    ``scored`` needs [symbol, effective_trading_date, net_sentiment]; ``calendar`` is the
    full (symbol, date) grid. Missing days get neutral values. Only news with an effective
    date <= t is used for day t, so no future information leaks into the features.
    """
    if scored.empty:
        daily = pd.DataFrame({
            "symbol": pd.Series(dtype="object"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "stock_sentiment_1d": pd.Series(dtype="float64"),
            "stock_news_volume": pd.Series(dtype="float64"),
        })
    else:
        daily = (
            scored.dropna(subset=["effective_trading_date"])
            .groupby(["symbol", "effective_trading_date"])["net_sentiment"]
            .agg(stock_sentiment_1d="mean", stock_news_volume="size")
            .reset_index().rename(columns={"effective_trading_date": "date"})
        )
    out = calendar[["symbol", "date"]].merge(daily, on=["symbol", "date"], how="left")
    out["stock_sentiment_1d"] = out["stock_sentiment_1d"].fillna(0.0)
    out["stock_news_volume"] = out["stock_news_volume"].fillna(0.0)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    roll = out.groupby("symbol")["stock_sentiment_1d"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    out["sentiment_momentum_5d"] = out["stock_sentiment_1d"] - roll
    return out


# --------------------------------------------------------------------------- features
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return rsi.where(loss != 0, 100.0).where(loss.notna())


def _symbol_features(g: pd.DataFrame, nifty_close: pd.Series, vix: pd.Series) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    close = g["close"]
    log_ret = np.log(close / close.shift(1))
    g["ret_1d"] = log_ret
    for lag in (1, 2, 3, 5):
        g[f"ret_lag_{lag}"] = log_ret.shift(lag)
    g["rsi_14"] = _rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    g["macd_norm"] = (ema12 - ema26) / close
    prev_close = close.shift(1)
    true_range = pd.concat(
        [g["high"] - g["low"], (g["high"] - prev_close).abs(), (g["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    g["atr_14_norm"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    g["ema20_dist"] = close / close.ewm(span=20, adjust=False).mean() - 1
    vol = g["volume"].astype("float64")
    g["volume_z20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std().replace(0, np.nan)

    nifty = pd.Series(nifty_close.reindex(g["date"]).ffill().to_numpy(), index=g.index)
    g["rel_strength_5d"] = np.log(close / close.shift(5)) - np.log(nifty / nifty.shift(5))
    g["india_vix"] = vix.reindex(g["date"]).ffill().to_numpy()

    # Forward log returns y_{t+k} = ln(Close_{t+k} / Close_t); NaN when not yet realised.
    for k in SUPPORTED_HORIZONS:
        g[f"target_{k}"] = np.log(close.shift(-k) / close)
    return g


def build_feature_frame(
    prices: pd.DataFrame,
    nifty_close: pd.Series,
    vix: pd.Series,
    news_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Technical + NLP features and forward-return targets, one row per (symbol, date)."""
    parts = [_symbol_features(g, nifty_close, vix) for _, g in prices.groupby("symbol", sort=True)]
    df = pd.concat(parts, ignore_index=True)
    if news_features is None:
        news_features = aggregate_news_features(pd.DataFrame(), df)
    df = df.drop(columns=[c for c in NLP_FEATURES if c in df.columns]).merge(
        news_features[["symbol", "date", *NLP_FEATURES]], on=["symbol", "date"], how="left"
    )
    df[NLP_FEATURES] = df[NLP_FEATURES].fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=TECH_FEATURES).reset_index(drop=True)
    return df[["date", "symbol", "close", *NUMERIC_FEATURES, *TARGET_COLUMNS]]


# ------------------------------------------------------------- cross-sectional ranking data
SECTOR_MAP = {
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "KOTAKBANK": "Financials", "AXISBANK": "Financials",
    "SBIN": "Financials", "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "HDFCLIFE": "Financials",
    "SBILIFE": "Financials", "SHRIRAMFIN": "Financials", "JIOFIN": "Financials", "BSE": "Financials",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "RELIANCE": "Energy", "ONGC": "Energy", "COALINDIA": "Energy", "NTPC": "Utilities", "POWERGRID": "Utilities",
    "MARUTI": "Auto", "TMPV": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto", "EICHERMOT": "Auto", "BAJAJ-AUTO": "Auto",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "TATACONSUM": "FMCG",
    "SUNPHARMA": "Healthcare", "CIPLA": "Healthcare", "DRREDDY": "Healthcare", "APOLLOHOSP": "Healthcare",
    "MAXHEALTH": "Healthcare",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "ULTRACEMCO": "Materials", "GRASIM": "Materials", "ASIANPAINT": "Materials",
    "TITAN": "Consumer", "TRENT": "Consumer", "ETERNAL": "Consumer",
    "BHARTIARTL": "Telecom", "LT": "Industrials", "BEL": "Industrials", "ADANIENT": "Industrials",
    "ADANIPORTS": "Industrials", "INDIGO": "Industrials",
}
SECTORS = sorted(set(SECTOR_MAP.values()) | {"Other"})
RANK_FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_21", "ret_63", "ret_126", "mom_12_1", "vol_21", "vol_63",
    "downside_vol_63", "idio_vol_63", "beta_126", "max_ret_21", "dist_high_252", "dist_low_252",
    "rsi_14", "ema20_dist", "atr_14_norm", "volume_z20", "volume_trend", "illiquidity_21", "gap_1d",
    "gap_5d", "sec_rel_5", "sec_rel_21",
]
CS_FEATURES = [f"cs_{c}" for c in RANK_FEATURES]
RANK_MODEL_FEATURES = [*CS_FEATURES, "sector_id", "india_vix", "mkt_ret_21"]
REL_TARGETS = [f"rel_{k}" for k in SUPPORTED_HORIZONS]
RANK_LEVELS = 5  # relevance labels 0..4 (within-day quintiles) for lambdarank
LONG_HORIZONS = [21, 42, 63]  # research-only: where momentum-type effects are expected
LONG_HISTORY_YEARS = 15
MIN_CS_STOCKS = 30  # a cross-section needs enough stocks to rank


def _ranking_symbol_features(
    g: pd.DataFrame, nifty_close: pd.Series, horizons: list[int] = SUPPORTED_HORIZONS
) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    close, vol = g["close"], g["volume"].astype("float64")
    ret = np.log(close / close.shift(1))
    nifty = pd.Series(nifty_close.reindex(g["date"]).ffill().to_numpy(), index=g.index)
    mkt = np.log(nifty / nifty.shift(1))
    out = g[["date", "symbol", "close"]].copy()
    for n in (1, 5, 10, 21, 63, 126):
        out[f"ret_{n}"] = np.log(close / close.shift(n))
    out["mom_12_1"] = np.log(close.shift(21) / close.shift(252))  # 12-month momentum skipping the last month
    out["vol_21"], out["vol_63"] = ret.rolling(21).std(), ret.rolling(63).std()
    out["downside_vol_63"] = ret.clip(upper=0).rolling(63).std()
    beta = ret.rolling(126).cov(mkt) / mkt.rolling(126).var()
    out["beta_126"] = beta
    out["idio_vol_63"] = (ret - beta * mkt).rolling(63).std()
    out["max_ret_21"] = ret.rolling(21).max()
    out["dist_high_252"] = close / close.rolling(252).max() - 1
    out["dist_low_252"] = close / close.rolling(252).min() - 1
    out["rsi_14"] = _rsi(close)
    out["ema20_dist"] = close / close.ewm(span=20, adjust=False).mean() - 1
    prev = close.shift(1)
    true_range = pd.concat([g["high"] - g["low"], (g["high"] - prev).abs(), (g["low"] - prev).abs()], axis=1).max(axis=1)
    out["atr_14_norm"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    out["volume_z20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std().replace(0, np.nan)
    out["volume_trend"] = np.log(vol.rolling(20).mean() / vol.rolling(60).mean().replace(0, np.nan))
    out["illiquidity_21"] = (ret.abs() / (vol * close).replace(0, np.nan)).rolling(21).mean()
    gap = g["open"] / prev - 1
    out["gap_1d"], out["gap_5d"] = gap, gap.rolling(5).mean()
    out["mkt_ret_21"] = np.log(nifty / nifty.shift(21))
    for k in horizons:
        out[f"target_{k}"] = np.log(close.shift(-k) / close)
    return out


def build_ranking_frame(
    prices: pd.DataFrame, nifty_close: pd.Series, vix: pd.Series, horizons: list[int] = SUPPORTED_HORIZONS
) -> pd.DataFrame:
    """Market-relative, cross-sectionally ranked features and targets, one row per (date, symbol).

    Every ``cs_*`` feature is the stock's percentile rank among all stocks on that date (centred on 0),
    so the model only learns "who is stronger than whom today". Targets ``rel_k`` are forward log returns
    minus that day's cross-sectional mean; ``label_k`` are within-day quintiles (0..4) for lambdarank.
    """
    parts = [_ranking_symbol_features(g, nifty_close, horizons) for _, g in prices.groupby("symbol", sort=True)]
    df = pd.concat(parts, ignore_index=True)
    df["sector"] = df["symbol"].map(SECTOR_MAP).fillna("Other")
    df["sector_id"] = df["sector"].map({s: i for i, s in enumerate(SECTORS)}).astype("int16")
    for n in (5, 21):
        df[f"sec_rel_{n}"] = df[f"ret_{n}"] - df.groupby(["date", "sector"])[f"ret_{n}"].transform("mean")
    df["india_vix"] = vix.reindex(df["date"]).ffill().to_numpy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[*RANK_FEATURES, "mkt_ret_21"])
    df = df[df.groupby("date")["symbol"].transform("size") >= MIN_CS_STOCKS]  # thin days (partial sessions)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    by_date = df.groupby("date")
    for col in RANK_FEATURES:
        df[f"cs_{col}"] = by_date[col].rank(pct=True) - 0.5
    for k in horizons:
        target = df[f"target_{k}"]
        df[f"rel_{k}"] = target - df.groupby("date")[f"target_{k}"].transform("mean")
        ranks = df.groupby("date")[f"rel_{k}"].rank(method="first", pct=True)
        df[f"label_{k}"] = np.where(target.notna(), np.minimum((ranks * RANK_LEVELS).apply(np.ceil) - 1, RANK_LEVELS - 1), np.nan)
    return df


# ------------------------------------------------------------------------------ splitting
def assign_split_labels(
    dates: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo: int = EMBARGO_DAYS,
) -> pd.Series:
    """Chronological labels over unique trading dates: train | embargo | val | embargo | test.

    The embargo is ``embargo`` whole trading dates, matching the longest forecast horizon,
    so no train label window overlaps a validation/test feature or label window.
    """
    dates = pd.to_datetime(dates)
    unique = np.sort(pd.unique(dates))
    n = len(unique)
    train_end = int(n * train_frac)
    val_start = train_end + embargo
    val_end = val_start + int(n * val_frac)
    test_start = val_end + embargo
    if test_start >= n:
        raise ValueError("Not enough history for the requested split and embargo")
    labels = np.full(n, "test", dtype=object)
    labels[:train_end] = "train"
    labels[train_end:val_start] = "embargo"
    labels[val_start:val_end] = "val"
    labels[val_end:test_start] = "embargo"
    mapping = pd.Series(labels, index=pd.DatetimeIndex(unique))
    return pd.Series(mapping.reindex(pd.DatetimeIndex(dates)).to_numpy(), index=dates.index)


def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split rows with fully realised targets into train/val/test; embargo rows dropped."""
    labelled = df.dropna(subset=TARGET_COLUMNS).copy()
    labelled["split"] = assign_split_labels(labelled["date"])
    return {
        name: labelled[labelled["split"] == name].drop(columns="split").reset_index(drop=True)
        for name in ("train", "val", "test")
    }


def fit_preprocessor(train: pd.DataFrame):
    """Fit scaler/encoder on the training split ONLY."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return pre.fit(train[FEATURE_COLUMNS])


# ------------------------------------------------------------------------------ pipeline
def score_new_headlines(tagged: pd.DataFrame, path: Path) -> pd.DataFrame:
    """FinBERT-score only (symbol, headline, timestamp) rows not scored before; cached scores are reused.

    Keeps the daily GPU cost flat while the news archive grows.
    """
    def key(frame: pd.DataFrame) -> pd.Series:
        return frame["symbol"] + "|" + frame["published_at"].astype(str) + "|" + frame["title"]

    cached = pd.read_parquet(path) if path.exists() else None
    if cached is not None and not cached.empty:
        scores = dict(zip(key(cached), cached["net_sentiment"]))
        tagged = tagged.assign(net_sentiment=key(tagged).map(scores))
        if {"source", "url"} <= set(cached.columns) and "source" in tagged:
            link_keys = ["symbol", "published_at", "title", "source"]
            links = pd.concat(
                [tagged.reindex(columns=[*link_keys, "url"]), cached[[*link_keys, "url"]]],
                ignore_index=True,
            )
            links["published_at"] = pd.to_datetime(links["published_at"], utc=True)
            links = _fill_missing_article_urls(links, link_keys)
            tagged = tagged.assign(url=links["url"].iloc[:len(tagged)].to_numpy())
    else:
        tagged = tagged.assign(net_sentiment=np.nan)
    missing = tagged["net_sentiment"].isna()
    if missing.any():
        LOGGER.info("Scoring %d new headline rows with FinBERT (%d cached)", missing.sum(), (~missing).sum())
        tagged.loc[missing, "net_sentiment"] = score_sentiment(tagged.loc[missing, "title"].tolist())
    return tagged


def run_pipeline(
    skip_sentiment: bool = False, years: int = HISTORY_YEARS, refit: bool = True
) -> dict[str, pd.DataFrame]:
    """Refresh raw data and features. ``refit=False`` leaves the splits, preprocessor and drift reference
    untouched (the daily refresh); only retraining should replace those."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    symbols = fetch_nifty50_symbols()
    prices = download_prices(symbols, years)
    nifty, vix = download_index_series(years)
    prices.to_parquet(RAW_DIR / "prices.parquet", compression="zstd", index=False)
    nifty.to_frame().join(vix, how="outer").to_parquet(RAW_DIR / "indices.parquet", compression="zstd")
    symbols = sorted(prices["symbol"].unique())

    news_features = None
    if not skip_sentiment:
        news = merge_news_history(fetch_news())
        if not news.empty:
            news.to_parquet(RAW_DIR / "news.parquet", compression="zstd", index=False)
            tagged = tag_headlines(news, symbols)
            if not tagged.empty:
                tagged = score_new_headlines(tagged, RAW_DIR / "news_scored.parquet")
                tagged["effective_trading_date"] = assign_effective_trading_date(
                    tagged["published_at"], pd.DatetimeIndex(sorted(prices["date"].unique()))
                )
                tagged.to_parquet(RAW_DIR / "news_scored.parquet", compression="zstd", index=False)
                news_features = aggregate_news_features(tagged, prices)

    features = build_feature_frame(prices, nifty, vix, news_features)
    features.to_parquet(PROCESSED_DIR / "features.parquet", compression="zstd", index=False)
    build_ranking_frame(prices, nifty, vix).to_parquet(
        PROCESSED_DIR / "rank_features.parquet", compression="zstd", index=False
    )
    # Rows on the latest date have unrealised targets; the serving layer scores them.
    features[features["date"] == features["date"].max()].to_parquet(
        PROCESSED_DIR / "latest_features.parquet", compression="zstd", index=False
    )
    if not refit:
        return {}

    splits = split_dataset(features)
    joblib.dump(fit_preprocessor(splits["train"]), PROCESSED_DIR / "preprocessor.joblib")
    for name, part in splits.items():
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", compression="zstd", index=False)
    splits["train"][["date", "symbol", *NUMERIC_FEATURES]].sample(
        n=min(5000, len(splits["train"])), random_state=RANDOM_STATE
    ).to_parquet(PROCESSED_DIR / "reference_sample.parquet", compression="zstd", index=False)
    return splits


def summarize_processed() -> pd.DataFrame:
    """Query the processed Parquet files with DuckDB."""
    import duckdb

    path = (PROCESSED_DIR / "features.parquet").as_posix()
    return duckdb.sql(
        "SELECT COUNT(*) AS n_rows, COUNT(DISTINCT symbol) AS n_symbols, MIN(date) AS first_date, "
        f"MAX(date) AS last_date FROM read_parquet('{path}')"
    ).df()


def download_long_history(years: int = LONG_HISTORY_YEARS) -> pd.DataFrame:
    """Research data: long adjusted history for the current constituents, kept apart from the live data.

    Written to ``prices_long.parquet`` / ``indices_long.parquet`` so the production features, splits and
    drift reference are untouched. NOTE: current constituents only, so results carry survivorship bias.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    symbols = fetch_nifty50_symbols()
    prices = download_prices(symbols, years)
    nifty, vix = download_index_series(years)
    prices.to_parquet(RAW_DIR / "prices_long.parquet", compression="zstd", index=False)
    nifty.to_frame().join(vix, how="outer").to_parquet(RAW_DIR / "indices_long.parquet", compression="zstd")
    return prices


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build Nifty 50 features, targets and splits")
    parser.add_argument("--skip-sentiment", action="store_true", help="skip RSS/FinBERT stage")
    parser.add_argument("--years", type=int, default=HISTORY_YEARS)
    parser.add_argument(
        "--refresh-only", action="store_true",
        help="daily refresh: update prices/news/features but keep splits, preprocessor and drift reference",
    )
    parser.add_argument(
        "--long-history", type=int, nargs="?", const=LONG_HISTORY_YEARS, metavar="YEARS",
        help="only download long research history into prices_long.parquet and exit",
    )
    args = parser.parse_args()
    if args.long_history:
        prices = download_long_history(args.long_history)
        per_symbol = prices.groupby("symbol")["date"].agg(["min", "count"])
        print(f"{len(prices):,} rows, {prices['date'].min().date()} -> {prices['date'].max().date()}")
        print(per_symbol.sort_values("min").tail(8).to_string())
        return
    splits = run_pipeline(skip_sentiment=args.skip_sentiment, years=args.years, refit=not args.refresh_only)
    for name, part in splits.items():
        LOGGER.info("%s: %d rows, %s -> %s", name, len(part), part["date"].min().date(), part["date"].max().date())
    print(summarize_processed().to_string(index=False))


if __name__ == "__main__":
    main()
