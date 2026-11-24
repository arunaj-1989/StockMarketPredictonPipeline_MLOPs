"""Minimal pastel Streamlit trader UI.

Run:  streamlit run src/dashboard.py
Env:  API_URL (default http://127.0.0.1:8000)

Sidebar: official NSE cash-market holidays (IST; cached for six hours) and a bottom-pinned manual
price/index/news/feature sync using the refresh-only pipeline. Sync keeps training
splits, preprocessing, drift reference and the production model unchanged. The API
must share this dashboard's data directory to serve the refreshed features.
The news page shows stored ticker-matched headlines, their FinBERT scores, and observed
session and following-session adjusted close changes, not causal price effects.
"""

import json
import os
from datetime import date, datetime

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from data_prep import IST, PROCESSED_DIR, RAW_DIR, SUPPORTED_HORIZONS
from market_data import MarketDataError, fetch_trading_holidays, sync_market_data
from news import load_news, news_view

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
RECON_PATH = PROCESSED_DIR / "reconciliation.parquet"
DRIFT_PATH = PROCESSED_DIR / "drift_status.json"

BG, CARD = "#FAF9F6", "#FFFFFF"
SAGE_BG, SAGE_FG = "#E6F4EA", "#1E7E34"
CORAL_BG, CORAL_FG = "#FCE8E6", "#C5221F"
LAV_BG, LAV_FG = "#F3E8FD", "#6A1B9A"
NEUTRAL_BG, NEUTRAL_FG = "#F1EFEA", "#5F5B52"

CSS = f"""
<style>
[class*="st-key-card"] {{
  background: {CARD}; border: 1px solid #EFEDE8; border-radius: 16px;
  padding: 0.9rem 1.1rem; box-shadow: 0 1px 3px rgba(60, 50, 30, 0.05);
}}
[class*="st-key-card-sage"] {{ background: {SAGE_BG}; border-color: {SAGE_BG}; }}
[class*="st-key-card-coral"] {{ background: {CORAL_BG}; border-color: {CORAL_BG}; }}
[class*="st-key-card-lav"] {{ background: {LAV_BG}; border-color: {LAV_BG}; }}
[class*="st-key-card-sage"] * {{ color: {SAGE_FG}; }}
[class*="st-key-card-coral"] * {{ color: {CORAL_FG}; }}
[class*="st-key-card-lav"] * {{ color: {LAV_FG}; }}
.pill {{ display: inline-block; padding: 0.35rem 1.1rem; border-radius: 999px;
  font-weight: 700; letter-spacing: 0.04em; font-size: 1.25rem; }}
.pill-buy {{ background: {SAGE_BG}; color: {SAGE_FG}; }}
.pill-hold {{ background: {NEUTRAL_BG}; color: {NEUTRAL_FG}; }}
.price-label {{ font-size: 0.85rem; opacity: 0.85; }}
.price-value {{ font-size: 1.6rem; font-weight: 650; }}
.st-key-sidebar-market-pane {{
  container-type: inline-size;
  padding-bottom: 5rem;
}}
.st-key-sync_market_data {{
  position: fixed;
  bottom: 1rem;
  width: 100cqw;
  z-index: 10;
  background: {CARD};
}}
</style>
"""


# ------------------------------------------------------------------------------ data
class ApiError(RuntimeError):
    pass


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, params: tuple[tuple[str, object], ...] = ()) -> dict:
    try:
        resp = requests.get(f"{API_URL}{path}", params=dict(params), timeout=30)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {API_URL} ({type(exc).__name__}).") from exc
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith(
            "application/json"
        ) else resp.text
        raise ApiError(f"API returned {resp.status_code}: {detail}")
    return resp.json()


@st.cache_data(ttl=600, show_spinner=False)
def load_history(symbol: str, days: int = 90) -> pd.DataFrame:
    path = RAW_DIR / "prices.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    return duckdb.execute(
        "SELECT date, open, high, low, close FROM read_parquet(?) WHERE symbol = ? "
        "ORDER BY date DESC LIMIT ?",
        [path.as_posix(), symbol, days],
    ).df().sort_values("date")


@st.cache_data(ttl=60, show_spinner=False)
def load_scorecard() -> tuple[pd.DataFrame | None, dict | None]:
    recon = pd.read_parquet(RECON_PATH) if RECON_PATH.exists() else None
    drift = json.loads(DRIFT_PATH.read_text(encoding="utf-8")) if DRIFT_PATH.exists() else None
    return recon, drift


def pct(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}%"


@st.cache_data(ttl=21600, show_spinner=False)
def load_holidays() -> dict[date, str]:
    return fetch_trading_holidays()


def market_sidebar() -> None:
    today = datetime.now(IST).date()
    with st.sidebar, st.container(key="sidebar-market-pane"):
        st.caption(f"Today: {today:%d %b %Y} (IST)")
        st.subheader("NSE holidays")
        try:
            holidays = load_holidays()
            if not any(day.year == today.year for day in holidays):
                st.warning(f"NSE has not supplied a calendar for {today.year}; today's holiday status is unknown.")
            elif today in holidays:
                st.info(f"Holiday today: {holidays[today]}")
            elif today.weekday() >= 5:
                st.info("Weekend: no regular NSE trading session today.")
            else:
                st.caption("No NSE trading holiday today.")
            st.markdown("**Upcoming holidays**")
            upcoming = [(day, name) for day, name in sorted(holidays.items()) if day > today][:5]
            for day, name in upcoming:
                st.caption(f"{day:%a, %d %b %Y} - {name}")
            if not upcoming:
                st.caption("No upcoming holidays in the published NSE calendar.")
            st.caption("Source: NSE cash-market calendar. Special sessions may be announced separately.")
        except MarketDataError as exc:
            st.warning(str(exc))

        if st.button("Sync market data", key="sync_market_data", icon=":material/sync:", width="stretch"):
            st.session_state.pop("market_sync_result", None)
            try:
                with st.spinner("Syncing market data..."):
                    as_of = sync_market_data()
                api_get.clear()
                load_history.clear()
                load_scorecard.clear()
                load_news.clear()
                st.session_state["market_sync_result"] = (
                    f"Sync completed at {datetime.now(IST):%d %b %Y %H:%M} IST. "
                    f"Latest data: {as_of:%d %b %Y}. Daily bars may not advance on holidays or before close."
                )
            except MarketDataError as exc:
                st.error(str(exc))
        if result := st.session_state.get("market_sync_result"):
            st.success(result)


# ----------------------------------------------------------------------------- charts
def forecast_path(pred: dict) -> pd.DataFrame:
    """Piecewise-linear [q10, q50, q90] log-return path through the model's horizon points."""
    points = {0: (0.0, 0.0, 0.0)} | {p["day"]: (p["q10"], p["q50"], p["q90"]) for p in pred["forecast"]}
    known = sorted(points)
    days = list(range(pred["horizon"] + 1))
    rows = []
    for d in days:
        lo_d = max(k for k in known if k <= d)
        hi_d = min(k for k in known if k >= d)
        w = 0.0 if hi_d == lo_d else (d - lo_d) / (hi_d - lo_d)
        rows.append([a + w * (b - a) for a, b in zip(points[lo_d], points[hi_d])])
    frame = pd.DataFrame(rows, columns=["q10", "q50", "q90"], index=days)
    close = pred["close"]
    for col in ("q10", "q50", "q90"):
        frame[f"p_{col}"] = close * np.exp(frame[col])
    return frame


def build_chart(
    history: pd.DataFrame, pred: dict, show_cone: bool = True, show_median: bool = True,
    show_levels: bool = True,
) -> go.Figure:
    path = forecast_path(pred)
    last = pd.Timestamp(pred["as_of_date"])
    future = pd.bdate_range(last, periods=pred["horizon"] + 1)
    path.index = future
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=history["date"], open=history["open"], high=history["high"], low=history["low"],
        close=history["close"], name="Price",
        increasing_line_color=SAGE_FG, increasing_fillcolor=SAGE_BG,
        decreasing_line_color=CORAL_FG, decreasing_fillcolor=CORAL_BG,
    ))
    if show_cone:
        fig.add_trace(go.Scatter(x=path.index, y=path["p_q90"], mode="lines", line={"width": 0},
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=path.index, y=path["p_q10"], mode="lines", line={"width": 0},
                                 fill="tonexty", fillcolor="rgba(106, 27, 154, 0.12)", name="q10–q90 band",
                                 hovertemplate="q10 %{y:,.2f}<extra></extra>"))
    if show_median:
        model_days = [0] + [p["day"] for p in pred["forecast"]]
        fig.add_trace(go.Scatter(x=path.index, y=path["p_q50"], mode="lines",
                                 line={"color": LAV_FG, "width": 2, "dash": "dot"}, name="q50 median",
                                 hovertemplate="q50 %{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=path.index[model_days], y=path["p_q50"].iloc[model_days], mode="markers",
                                 marker={"color": LAV_FG, "size": 7}, name="Model horizons", hoverinfo="skip"))
    if show_levels:
        for level, color, label in (
            (pred["take_profit"], SAGE_FG, "Take-profit"), (pred["stop_loss"], CORAL_FG, "Stop-loss"),
        ):
            fig.add_hline(y=level, line={"color": color, "width": 1, "dash": "dash"},
                          annotation_text=f"{label} {level:,.2f}", annotation_position="top left",
                          annotation_font_color=color, annotation_bgcolor="rgba(255, 255, 255, 0.85)")
    axis_style = {
        "gridcolor": "#F0EEE8", "showline": True, "linecolor": "#CFCBC0", "ticks": "outside",
        "tickcolor": "#CFCBC0", "tickfont": {"size": 12, "color": "#2F2D2A"}, "automargin": True,
        "title_font": {"size": 13, "color": "#5F5B52"},
    }
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD, height=480, margin={"l": 70, "r": 25, "t": 45, "b": 65},
        xaxis={**axis_style, "rangeslider": {"visible": False}, "title": {"text": "Date"},
               "tickformat": "%d %b"},
        yaxis={**axis_style, "title": {"text": "Price (₹)"}, "tickformat": ",.0f"},
        legend={"orientation": "h", "y": 1.1, "x": 0}, font={"color": "#2F2D2A"},
    )
    return fig


# ------------------------------------------------------------------------------- views
def model_banner(model_version: str, override: bool, as_of: str) -> None:
    with st.container(key="card-lav-model"):
        st.markdown(
            f"**Model** nifty50_multi_horizon v{model_version} · features as of {as_of} · "
            f"quantile LightGBM, horizons {SUPPORTED_HORIZONS}"
        )
        if override:
            st.caption(
                "This model was promoted by manual override: it does not beat a naive no-change "
                "baseline on median return. Treat BUY signals as a drift-based screen, not an edge."
            )


SORT_OPTIONS = {
    "Expected gain (high to low)": ("expected_return_pct", False),
    "Sentiment (high to low)": ("stock_sentiment_1d", False),
    "Reward : risk (high to low)": ("reward_risk", False),
    "Symbol (A to Z)": ("symbol", True),
}
HISTORY_WINDOWS = [30, 60, 90, 180]
CHART_LAYERS = ["Forecast cone", "Median line", "Take-profit / stop-loss"]
FILTER_DEFAULTS = {
    "flt_signal": "All",
    "flt_min_gain": -5.0,
    "flt_sentiment": (-1.0, 1.0),
    "flt_max_band": 40.0,
    "flt_min_rr": 0.0,
    "flt_sort": next(iter(SORT_OPTIONS)),
}
CHART_DEFAULTS = {"chart_history": 90, "chart_layers": list(CHART_LAYERS)}


def universe_frame(items: list[dict]) -> pd.DataFrame:
    """Watchlist items plus the derived columns the trade filters need."""
    frame = pd.DataFrame(items)
    risk = frame["buy_entry"] - frame["stop_loss"]
    reward = frame["take_profit"] - frame["buy_entry"]
    frame["band_pct"] = (frame["q90"] - frame["q10"]) * 100
    frame["reward_risk"] = (reward / risk).where((risk > 0) & (reward > 0))
    return frame


def apply_filters(
    frame: pd.DataFrame, signal: str, min_gain: float, sentiment: tuple[float, float],
    max_band: float, min_rr: float, sort_by: str,
) -> pd.DataFrame:
    mask = (
        (frame["expected_return_pct"] >= min_gain)
        & frame["stock_sentiment_1d"].between(sentiment[0], sentiment[1])
        & (frame["band_pct"] <= max_band)
    )
    if signal != "All":
        mask &= frame["signal"] == signal
    if min_rr > 0:
        mask &= frame["reward_risk"] >= min_rr  # stocks without a valid ratio never pass
    column, ascending = SORT_OPTIONS[sort_by]
    return frame[mask].sort_values(column, ascending=ascending, na_position="last", kind="stable")


def reset_filters() -> None:
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


def filter_bar(frame: pd.DataFrame) -> dict | None:
    """Trade-desk filters. Returns the chosen symbol and chart options, or None if nothing matches."""
    for key, value in (FILTER_DEFAULTS | CHART_DEFAULTS).items():
        st.session_state.setdefault(key, value)

    with st.container(key="card-filters"):
        a, b, c, d = st.columns([1.1, 1.2, 1.2, 1.2])
        a.segmented_control("Signal", ["All", "BUY", "HOLD"], key="flt_signal", required=True)
        b.slider("Min expected gain (%)", -5.0, 5.0, step=0.1, key="flt_min_gain",
                 help="-5.0 means no minimum.")
        c.slider("FinBERT sentiment range", -1.0, 1.0, step=0.05, key="flt_sentiment")
        d.slider("Max forecast band width (%)", 0.0, 40.0, step=0.5, key="flt_max_band",
                 help="q90 minus q10 as a percentage of price: a volatility proxy. 40 means no maximum.")
        e, f, h = st.columns([1.2, 1.2, 1.1], vertical_alignment="bottom")
        e.slider("Min reward : risk", 0.0, 5.0, step=0.1, key="flt_min_rr",
                 help="Take-profit gain over buy limit divided by buy limit loss to stop-loss.")
        f.selectbox("Sort by", list(SORT_OPTIONS), key="flt_sort")
        h.button("Reset filters", on_click=reset_filters, icon=":material/restart_alt:", width="stretch")

    matches = apply_filters(
        frame, st.session_state["flt_signal"], st.session_state["flt_min_gain"],
        st.session_state["flt_sentiment"], st.session_state["flt_max_band"],
        st.session_state["flt_min_rr"], st.session_state["flt_sort"],
    )
    st.caption(f"{len(matches)} of {len(frame)} stocks match the filters.")

    options = matches["symbol"].tolist()
    # Widgets that are not drawn lose their state, so chart controls render even with no matches
    # and the last pick is mirrored in a plain session key.
    preferred = st.session_state.get("trade_symbol") or st.session_state.get("last_trade_symbol")
    if options and st.session_state.get("trade_symbol") not in options:
        fallback = "RELIANCE" if "RELIANCE" in options else options[0]
        st.session_state["trade_symbol"] = preferred if preferred in options else fallback
    labels = {
        r.symbol: f"{r.symbol} · {r.signal} · {r.expected_return_pct:+.2f}%" for r in matches.itertuples()
    }
    s1, s2, s3 = st.columns([1.4, 1, 1.6], vertical_alignment="bottom")
    symbol = None
    if options:
        symbol = s1.selectbox("Stock", options, format_func=labels.get, key="trade_symbol")
        st.session_state["last_trade_symbol"] = symbol
    else:
        s1.caption("No stock matches the filters.")
    s2.segmented_control("Chart history", HISTORY_WINDOWS, format_func=lambda d: f"{d}D",
                         key="chart_history", required=True)
    layers = s3.pills("Chart layers", CHART_LAYERS, selection_mode="multi", key="chart_layers") or []
    if symbol is None:
        st.info("No stock matches these filters. Loosen them or use Reset filters.")
        return None
    return {
        "symbol": symbol, "days": st.session_state["chart_history"],
        "cone": "Forecast cone" in layers, "median": "Median line" in layers,
        "levels": "Take-profit / stop-loss" in layers,
    }


def signal_warning(payload: dict) -> None:
    """Warn when the forecast is not stock-specific; .get so an older API without the fields still works."""
    if payload.get("warning"):
        title = ("Drift-based signals (not stock-specific)" if payload.get("drift_override")
                 else "No stock-specific signal at this horizon")
        st.warning(payload["warning"], title=title, icon=":material/info:")


def trade_desk(symbol: str, horizon: int, min_ret: float, floor: float, allow_drift: bool, chart: dict) -> None:
    pred = api_get(
        f"/predict/{symbol}",
        (("horizon", horizon), ("min_return", min_ret), ("sentiment_floor", floor),
         ("allow_uninformative", allow_drift)),
    )
    is_buy = pred["signal"] == "BUY"
    informative = pred.get("forecast_informative", True)
    drift_only = pred.get("drift_override", False)
    signal_warning(pred)

    cols = st.columns(4)
    with cols[0], st.container(key="card-close"):
        st.metric("Current close", f"₹{pred['close']:,.2f}")
    with cols[1], st.container(key="card-gain"):
        st.metric(
            f"Exp. {horizon}D gain", pct(pred["expected_return_pct"]),
            help=None if informative else "Identical for every stock: historical drift, not a forecast.",
        )
    with cols[2], st.container(key="card-sentiment"):
        st.metric("FinBERT sentiment", f"{pred['stock_sentiment_1d']:+.2f}")
    with cols[3], st.container(key="card-signal-sage" if is_buy else "card-signal"):
        st.caption("Signal")
        label = f"{pred['signal']} (drift)" if is_buy and drift_only else pred["signal"]
        st.html(f'<span class="pill {"pill-buy" if is_buy else "pill-hold"}">{label}</span>')

    if pred["signal"] == "HOLD":
        reasons = []
        if not informative and not drift_only:
            reasons.append("the median forecast is not stock-specific")
        if pred["expected_return"] <= min_ret:
            reasons.append(f"median {horizon}D return {pct(pred['expected_return'] * 100)} ≤ hurdle {pct(min_ret * 100)}")
        if pred["stock_sentiment_1d"] <= floor:
            reasons.append(f"sentiment {pred['stock_sentiment_1d']:+.2f} ≤ circuit breaker {floor:+.2f}")
        st.caption("Holding because " + " and ".join(reasons) + ".")

    history = load_history(symbol, chart["days"])
    with st.container(key="card-chart"):
        if history.empty:
            st.info("Price history is unavailable. Run `python src/data_prep.py` to download it.")
        else:
            st.plotly_chart(
                build_chart(history, pred, chart["cone"], chart["median"], chart["levels"]),
                theme=None, width="stretch",
                alt=f"Candlestick chart of {symbol} over {chart['days']} days with a {horizon}-day forecast",
            )

    st.subheader("Execution card", anchor=False)
    risk = pred["buy_entry"] - pred["stop_loss"]
    reward = pred["take_profit"] - pred["buy_entry"]
    c1, c2, c3, c4 = st.columns(4)
    for col, key, label, value in (
        (c1, "card-sage-entry", "Buy limit (q10, t+1)", pred["buy_entry"]),
        (c2, "card-sage-tp", f"Take-profit (q50, t+{horizon})", pred["take_profit"]),
        (c3, "card-coral-sl", f"Stop-loss (q10, t+{horizon})", pred["stop_loss"]),
    ):
        with col, st.container(key=key):
            st.html(f'<div class="price-label">{label}</div><div class="price-value">₹{value:,.2f}</div>')
    with c4, st.container(key="card-lav-rr"):
        ratio = f"{reward / risk:.2f}" if risk > 0 and reward > 0 else "n/a"
        st.html(f'<div class="price-label">Reward : risk</div><div class="price-value">{ratio}</div>')
    st.caption(
        "Cone dates use weekdays and ignore NSE holidays. Levels are model quantiles, not guarantees; "
        "min-return hurdle and circuit breaker come from the controls above."
    )


def watchlist_view(horizon: int, min_ret: float, floor: float, allow_drift: bool) -> None:
    data = api_get(
        "/watchlist",
        (("horizon", horizon), ("min_return", min_ret), ("sentiment_floor", floor), ("only_buy", True),
         ("allow_uninformative", allow_drift)),
    )
    st.markdown(f"**{data['count']} BUY signals** out of 50 for the {horizon}-day horizon (as of {data['as_of_date']}).")
    signal_warning(data)
    if not data["items"]:
        st.info("No stock clears the current return hurdle and sentiment circuit breaker.")
        return
    frame = pd.DataFrame(data["items"])
    frame.insert(0, "rank", range(1, len(frame) + 1))
    st.dataframe(
        frame[["rank", "symbol", "close", "expected_return_pct", "q10", "q90", "stock_sentiment_1d", "take_profit", "stop_loss"]],
        hide_index=True, width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "symbol": "Symbol",
            "close": st.column_config.NumberColumn("Close", format="₹%.2f"),
            "expected_return_pct": st.column_config.NumberColumn(f"Exp. {horizon}D gain", format="%+.2f%%"),
            "q10": st.column_config.NumberColumn("q10", format="%.3f"),
            "q90": st.column_config.NumberColumn("q90", format="%.3f"),
            "stock_sentiment_1d": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
            "take_profit": st.column_config.NumberColumn("Take-profit", format="₹%.2f"),
            "stop_loss": st.column_config.NumberColumn("Stop-loss", format="₹%.2f"),
        },
        alt=f"Nifty 50 stocks with a BUY signal for the {horizon}-day horizon, ranked by expected gain",
    )


def scorecard_view(horizon: int) -> None:
    recon, drift = load_scorecard()
    st.subheader("Drift health", anchor=False)
    if drift is None:
        st.info("No drift report yet. Run `python src/monitoring.py` after predictions have been logged.")
    else:
        status = drift.get("status", "UNKNOWN")
        psi, share = drift.get("prediction_psi"), drift.get("drifted_feature_share")
        cols = st.columns(4)
        cols[0].metric("Status", status)
        cols[1].metric("Prediction PSI", "n/a" if psi is None else f"{psi:.3f}", help="Alert above 0.20")
        cols[2].metric("Drifted feature share", "n/a" if share is None else f"{share:.0%}", help="Alert above 30%")
        cols[3].metric("Perf. degraded", "Yes" if drift.get("perf_degraded") else "No")

    st.subheader("Delayed ground-truth reconciliation", anchor=False)
    if recon is None or recon.empty:
        st.info(
            "No graded predictions yet. Forecasts are graded once the actual NSE close for t+N "
            "has arrived, so the first results appear after the first horizon elapses."
        )
        return
    view = recon[recon["horizon"] == horizon]
    if view.empty:
        st.info(f"No graded {horizon}-day forecasts yet.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Graded forecasts", f"{len(view):,}")
    c2.metric("Directional accuracy", f"{view['direction_correct'].mean():.1%}")
    c3.metric("q10–q90 coverage", f"{view['in_interval'].mean():.1%}", help="Target 80%")
    st.dataframe(
        view.sort_values("as_of_date", ascending=False).head(200), hide_index=True, width="stretch",
        alt=f"Graded {horizon}-day forecasts versus actual NSE closes",
    )


# --------------------------------------------------------------------------------- app
# Preserve widget values when their page is not drawn.
PAGE_STATE_KEYS = [*FILTER_DEFAULTS, *CHART_DEFAULTS, "trade_symbol", "last_trade_symbol"]


def keep_state(keys: list[str]) -> None:
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def controls() -> None:
    """Global settings, drawn above every view."""
    with st.container(key="card-controls"):
        a, b, c, d = st.columns([1.2, 1, 1, 0.9], vertical_alignment="bottom")
        a.select_slider("Forecast horizon (trading days)", options=SUPPORTED_HORIZONS, value=3, key="ctl_horizon",
                        persist_state="session")
        b.slider("Minimum return hurdle (%)", 0.0, 3.0, 0.5, 0.05, key="ctl_min_ret_pct",
                 help="Expected N-day median return must exceed this to signal BUY.", persist_state="session")
        c.slider("FinBERT circuit breaker", -1.0, 0.5, -0.40, 0.05, key="ctl_floor",
                 help="No BUY when the stock's 1-day sentiment is at or below this value.", persist_state="session")
        d.toggle("Show drift-based signals", value=False, key="ctl_allow_drift",
                 help="Off by default. When the model's median forecast is the same for every stock, BUY signals "
                      "only reflect historical market drift, not a view on any stock. Turn on for demos only.",
                 persist_state="session")


def settings() -> tuple[int, float, float, bool]:
    state = st.session_state
    return state["ctl_horizon"], state["ctl_min_ret_pct"] / 100, state["ctl_floor"], state["ctl_allow_drift"]


def page_trade_desk() -> None:
    horizon, min_ret, floor, allow = settings()
    try:
        board = api_get(
            "/watchlist",
            (("horizon", horizon), ("min_return", min_ret), ("sentiment_floor", floor),
             ("allow_uninformative", allow)),
        )
        choice = filter_bar(universe_frame(board["items"]))
        if choice:
            trade_desk(choice["symbol"], horizon, min_ret, floor, allow, choice)
    except ApiError as exc:
        st.error(str(exc))


def page_watchlist() -> None:
    horizon, min_ret, floor, allow = settings()
    try:
        watchlist_view(horizon, min_ret, floor, allow)
    except ApiError as exc:
        st.error(str(exc))


def page_scorecard() -> None:
    scorecard_view(settings()[0])


st.set_page_config(page_title="Nifty 50 trader desk", page_icon=":material/show_chart:", layout="wide")
st.html(CSS)
st.title("Nifty 50 trader desk", anchor=False)
market_sidebar()

keep_state(PAGE_STATE_KEYS)

navigation = st.navigation(
    [
        st.Page(page_trade_desk, title="Trade desk", icon=":material/candlestick_chart:",
                url_path="trade-desk", default=True),
        st.Page(page_watchlist, title="Nifty 50 watchlist", icon=":material/format_list_numbered:",
                url_path="watchlist"),
        st.Page(news_view, title="News & sentiment", icon=":material/newspaper:", url_path="news"),
        st.Page(page_scorecard, title="Scorecard & drift", icon=":material/fact_check:",
                url_path="scorecard"),
    ],
    position="sidebar",
)
if navigation.url_path != "news":
    try:
        universe = api_get("/watchlist", (("horizon", 3),))
    except ApiError as exc:
        st.error(str(exc), icon=":material/error:")
        st.caption("Start the service with `uvicorn app:app --app-dir src --port 8000`.")
        st.stop()
    model_banner(universe["model_version"], universe["gate_override"], universe["as_of_date"])
    controls()
navigation.run()
