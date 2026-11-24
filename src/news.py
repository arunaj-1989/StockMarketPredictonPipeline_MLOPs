"""Display stored stock-tagged news and observed daily price movements.

The session move compares the effective trading session's adjusted close with
the preceding recorded session. The following move compares the next recorded
session with the effective session. Neither measure attributes causation.
"""

import string

import numpy as np
import pandas as pd
import streamlit as st

from data_prep import IST, RAW_DIR, normalize_article_urls

NEWS_COLUMNS = ["symbol", "published_at", "title", "source", "net_sentiment", "effective_trading_date"]
PRICE_COLUMNS = ["symbol", "date", "close"]
TABLE_COLUMNS = {
    "symbol": "Stock",
    "published_ist": "Published (IST)",
    "title": "Headline",
    "sentiment_tone": "Headline tone",
    "net_sentiment": "FinBERT",
    "session_move_pct": "Session %",
    "following_move_pct": "Next session %",
    "effective_trading_date": "Effective session",
    "following_session": "Following session",
    "source": "Source",
}
PAGE_SIZE = 100
_MARKDOWN_ESCAPES = str.maketrans({char: "\\" + char for char in string.punctuation})


def news_price_frame(news: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Match each headline to exact market sessions, leaving missing prices unavailable."""
    frame = news[NEWS_COLUMNS].copy()
    frame["url"] = normalize_article_urls(news["url"]) if "url" in news else None
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    if frame["published_at"].isna().any() or frame["symbol"].isna().any():
        raise ValueError("News contains a missing publication date or stock symbol.")
    if not frame["title"].map(lambda title: isinstance(title, str) and bool(title.strip())).all():
        raise ValueError("News contains a missing or invalid headline.")
    frame["published_ist"] = frame["published_at"].dt.tz_convert(IST).dt.strftime("%d %b %Y %H:%M")
    frame["effective_trading_date"] = pd.to_datetime(frame["effective_trading_date"]).dt.normalize()
    frame["net_sentiment"] = pd.to_numeric(frame["net_sentiment"])
    score = frame["net_sentiment"]
    if (score.notna() & (~np.isfinite(score) | (score.abs() > 1))).any():
        raise ValueError("News contains an invalid FinBERT sentiment score.")
    frame["sentiment_tone"] = np.select(
        [score > 0, score < 0, score == 0],
        ["Positive leaning", "Negative leaning", "Balanced"],
        default="Unavailable",
    )

    if prices.empty:
        frame["following_session"] = pd.NaT
        frame["session_move_pct"] = np.nan
        frame["following_move_pct"] = np.nan
    else:
        daily = prices[PRICE_COLUMNS].copy()
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        daily["close"] = pd.to_numeric(daily["close"])
        if daily["date"].isna().any() or daily["symbol"].isna().any():
            raise ValueError("Price history contains a missing session date or stock symbol.")
        if daily.duplicated(["symbol", "date"]).any():
            raise ValueError("Price history contains duplicate stock/session prices.")

        calendar = pd.DataFrame({"effective_trading_date": sorted(daily["date"].unique())})
        calendar["previous_session"] = calendar["effective_trading_date"].shift(1)
        calendar["following_session"] = calendar["effective_trading_date"].shift(-1)
        frame = frame.merge(calendar, on="effective_trading_date", how="left", validate="many_to_one")

        for date_column, close_column in (
            ("previous_session", "previous_close"),
            ("effective_trading_date", "session_close"),
            ("following_session", "following_close"),
        ):
            frame = frame.merge(
                daily.rename(columns={"date": date_column, "close": close_column}),
                on=["symbol", date_column], how="left", validate="many_to_one",
            )

        for result, start, end in (
            ("session_move_pct", "previous_close", "session_close"),
            ("following_move_pct", "session_close", "following_close"),
        ):
            valid = (frame[start] > 0) & (frame[end] > 0) & np.isfinite(frame[start]) & np.isfinite(frame[end])
            frame[result] = ((frame[end] / frame[start] - 1) * 100).where(valid)

    return frame.sort_values(["published_at", "symbol"], ascending=[False, True], kind="stable").reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def load_news() -> tuple[pd.DataFrame | None, bool]:
    path = RAW_DIR / "news_scored.parquet"
    if not path.exists():
        return None, False
    news = pd.read_parquet(path)
    if news.empty:
        return news, False
    price_path = RAW_DIR / "prices.parquet"
    prices = pd.read_parquet(price_path, columns=PRICE_COLUMNS) if price_path.exists() else pd.DataFrame()
    return news_price_frame(news, prices), not prices.empty


def _news_table(view: pd.DataFrame) -> pd.DataFrame:
    display = view[list(TABLE_COLUMNS)].copy()
    titles = [" ".join(title.split()).translate(_MARKDOWN_ESCAPES) for title in view["title"]]
    display["title"] = [
        f"[{title}](<{url.replace('&', '&amp;')}>)" if url else f"{title} (link unavailable)"
        for title, url in zip(titles, view["url"])
    ]
    for column, precision in (("net_sentiment", 3), ("session_move_pct", 2), ("following_move_pct", 2)):
        suffix = "%" if column != "net_sentiment" else ""
        display[column] = display[column].map(
            lambda value, precision=precision, suffix=suffix:
                f"{value:+.{precision}f}{suffix}" if pd.notna(value) else "None"
        )
    for column in ("effective_trading_date", "following_session"):
        display[column] = display[column].dt.strftime("%d %b %Y").fillna("None")
    sources = view["source"].fillna("Unknown source").str.translate(_MARKDOWN_ESCAPES)
    display["published_ist"] = display["published_ist"] + "  \n" + sources
    display["sentiment_tone"] = display["sentiment_tone"] + "  \n" + display["net_sentiment"]
    display["session_move_pct"] = display["session_move_pct"] + "  \n" + display["effective_trading_date"]
    display["following_move_pct"] = display["following_move_pct"] + "  \n" + display["following_session"]
    columns = ["symbol", "published_ist", "title", "sentiment_tone", "session_move_pct", "following_move_pct"]
    return display[columns].rename(columns={**TABLE_COLUMNS, "sentiment_tone": "Sentiment"})


def news_view() -> None:
    st.subheader("Nifty 50 news and sentiment", anchor=False)
    st.caption(
        "Ticker-matched RSS headlines; coverage is incomplete. Click a headline to open its article. "
        "Older links may be unavailable; Sync market data can recover links still in the feeds."
    )
    st.info(
        "Price moves are observed changes, not proof that the headline caused them. "
        "The session move can include trading before an intraday headline."
    )
    try:
        frame, prices_available = load_news()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load news and price data: {exc}")
        return
    if frame is None or frame.empty:
        st.info("No scored stock headlines are available yet. Use Sync market data to collect news.")
        return
    if not prices_available:
        st.warning("Price history is unavailable, so price moves cannot be calculated.")

    stock_col, tone_col, search_col = st.columns([1, 1, 2])
    stocks = ["All stocks", *sorted(frame["symbol"].unique().tolist())]
    stock = stock_col.selectbox("Stock", stocks, key="news_stock", persist_state="session")
    tone = tone_col.selectbox(
        "Headline tone", ["All tones", "Positive leaning", "Negative leaning", "Balanced", "Unavailable"],
        key="news_tone", persist_state="session",
    )
    query = search_col.text_input("Search headlines", key="news_search", persist_state="session")

    view = frame
    if stock != "All stocks":
        view = view[view["symbol"] == stock]
    if tone != "All tones":
        view = view[view["sentiment_tone"] == tone]
    if query.strip():
        view = view[view["title"].str.contains(query.strip(), case=False, regex=False, na=False)]
    st.caption(
        f"Showing {len(view):,} of {len(frame):,} matched stock-headline rows. "
        "None means a required session or stock price is unavailable; latest prices may be provisional."
    )
    if view.empty:
        st.info("No headlines match these filters.")
        return

    sort_options = {
        **{label: column for column, label in TABLE_COLUMNS.items() if column != "published_ist"},
        "Published (IST)": "published_at",
    }
    sort_col, direction_col, download_col = st.columns([2, 1, 1], vertical_alignment="bottom")
    sort_by = sort_col.selectbox(
        "Sort by", ["Published (IST)", *[name for name in sort_options if name != "Published (IST)"]],
        key="news_sort", persist_state="session",
    )
    descending = direction_col.checkbox("Descending", value=True, key="news_descending", persist_state="session")
    view = view.sort_values(sort_options[sort_by], ascending=not descending, kind="stable", na_position="last")
    download_col.download_button(
        "Download CSV", view[[*TABLE_COLUMNS, "url"]].to_csv(index=False),
        file_name="nifty50_news_sentiment.csv", mime="text/csv", on_click="ignore",
        help="Download all filtered rows, including article URLs.",
    )
    if len(view) > PAGE_SIZE:
        pages = (len(view) + PAGE_SIZE - 1) // PAGE_SIZE
        page = st.selectbox("Page", range(1, pages + 1), key="news_page", persist_state="session")
        start = (page - 1) * PAGE_SIZE
        st.caption(f"Page {page} of {pages}; CSV includes all {len(view):,} filtered rows.")
        view = view.iloc[start:start + PAGE_SIZE]
    st.table(
        _news_table(view),
        hide_index=True,
        width="stretch",
        height=600 if len(view) > 12 else "content",
        alt="Stock headlines, FinBERT sentiment, and observed session and following-session price changes",
    )
    st.caption(
        "FinBERT score = P(positive) minus P(negative), from -1 to +1. Tone reflects the score's sign; "
        "scores near zero are weak or balanced and do not necessarily mean a neutral classification. "
        "The source appears below the publication time, and each session date appears below its price move."
    )
    st.caption(
        "Session move: previous recorded session's adjusted close to the effective session's adjusted close. "
        "Next session: effective session's adjusted close to the next recorded session's adjusted close."
    )
