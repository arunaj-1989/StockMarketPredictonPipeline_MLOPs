import itertools
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import app as api
import data_prep as dp
import evaluate as ev
import market_data as md
import monitoring as mon
import news as nw
import retrain as rt
import train


def _prices(n_days: int = 400, symbols=("AAA", "BBB"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for sym in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        frames.append(pd.DataFrame({
            "date": dates, "symbol": sym, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": rng.integers(1_000, 5_000, n_days),
        }))
    return pd.concat(frames, ignore_index=True)


def _index_series(prices: pd.DataFrame):
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    nifty = pd.Series(np.linspace(15000, 18000, len(dates)), index=dates)
    vix = pd.Series(np.linspace(12, 20, len(dates)), index=dates)
    return nifty, vix


# ----------------------------------------------------------------------- embargo
def test_embargo_gap_is_ten_trading_days_between_splits():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=1000))
    labels = dp.assign_split_labels(dates)
    unique = pd.DatetimeIndex(dates)
    pos = {d: i for i, d in enumerate(unique)}

    def bounds(name):
        d = dates[labels == name]
        return pos[d.min()], pos[d.max()]

    train_lo, train_hi = bounds("train")
    val_lo, val_hi = bounds("val")
    test_lo, _ = bounds("test")
    assert (labels == "embargo").sum() == 2 * dp.EMBARGO_DAYS
    assert val_lo - train_hi - 1 == dp.EMBARGO_DAYS
    assert test_lo - val_hi - 1 == dp.EMBARGO_DAYS
    assert train_lo == 0


def test_no_label_window_overlap_across_splits():
    df = dp.build_feature_frame(_prices(), *_index_series(_prices()))
    splits = dp.split_dataset(df)
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    pos = {d: i for i, d in enumerate(all_dates)}
    max_h = max(dp.SUPPORTED_HORIZONS)
    train_end = pos[splits["train"]["date"].max()]
    val_start = pos[splits["val"]["date"].min()]
    val_end = pos[splits["val"]["date"].max()]
    test_start = pos[splits["test"]["date"].min()]
    # The longest train label window (t, t+10] must close before validation begins.
    assert train_end + max_h < val_start
    assert val_end + max_h < test_start
    assert splits["train"]["date"].max() < splits["val"]["date"].min()
    assert splits["val"]["date"].max() < splits["test"]["date"].min()
    assert splits["train"][dp.TARGET_COLUMNS].notna().all().all()


def test_preprocessor_fit_on_train_only():
    df = dp.build_feature_frame(_prices(), *_index_series(_prices()))
    splits = dp.split_dataset(df)
    pre = dp.fit_preprocessor(splits["train"])
    scaler = pre.named_transformers_["num"]
    np.testing.assert_allclose(
        scaler.mean_, splits["train"][dp.NUMERIC_FEATURES].mean().to_numpy(), rtol=1e-9
    )
    out = pre.transform(splits["test"][dp.FEATURE_COLUMNS])
    assert out.shape[0] == len(splits["test"])


# -------------------------------------------------------------- 3:30 PM IST shift
TRADING_DAYS = pd.DatetimeIndex(pd.bdate_range("2024-03-01", "2024-03-29")).drop(pd.Timestamp("2024-03-25"))


def _effective(*ist_stamps: str) -> list:
    ts = pd.Series(
        pd.to_datetime(list(ist_stamps), format="mixed").tz_localize("Asia/Kolkata").tz_convert("UTC")
    )
    return dp.assign_effective_trading_date(ts, TRADING_DAYS).tolist()


def test_news_before_and_at_close_keeps_same_session():
    assert _effective("2024-03-05 09:00", "2024-03-05 15:30:00") == [pd.Timestamp("2024-03-05")] * 2


def test_news_after_close_shifts_to_next_session():
    assert _effective("2024-03-05 15:30:01", "2024-03-05 20:00") == [pd.Timestamp("2024-03-06")] * 2


def test_news_after_friday_close_and_weekend_roll_to_monday():
    assert _effective("2024-03-08 15:31", "2024-03-09 11:00", "2024-03-10 23:00") == [
        pd.Timestamp("2024-03-11")
    ] * 3


def test_news_on_holiday_rolls_to_next_trading_day():
    # 2024-03-25 (Mon) is removed from the calendar as a holiday.
    assert _effective("2024-03-25 10:00") == [pd.Timestamp("2024-03-26")]


def test_utc_timestamps_are_converted_to_ist_before_cutoff():
    # 10:00 UTC == 15:30 IST (still same session); 10:01 UTC == 15:31 IST (next session).
    ts = pd.Series(pd.to_datetime(["2024-03-05 10:00:00", "2024-03-05 10:01:00"], utc=True))
    out = dp.assign_effective_trading_date(ts, TRADING_DAYS).tolist()
    assert out == [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")]


def test_news_beyond_last_known_session_is_unassigned():
    assert pd.isna(_effective("2024-03-29 16:00")[0])


def test_news_features_never_appear_before_effective_date():
    prices = _prices(60, ("AAA",))
    effective = prices["date"].iloc[30]
    scored = pd.DataFrame({
        "symbol": ["AAA"], "effective_trading_date": [effective], "net_sentiment": [0.8],
    })
    feats = dp.aggregate_news_features(scored, prices)
    assert (feats.loc[feats["date"] < effective, "stock_sentiment_1d"] == 0).all()
    assert feats.loc[feats["date"] == effective, "stock_sentiment_1d"].iloc[0] == pytest.approx(0.8)
    assert feats.loc[feats["date"] == effective, "stock_news_volume"].iloc[0] == 1


def test_headline_tagging_matches_symbols_not_substrings():
    news = pd.DataFrame({
        "title": ["Reliance Industries hits record", "TCS wins deal", "Results of the quarter", "Bajaj-Auto sales up"],
        "published_at": pd.Timestamp("2024-03-05", tz="UTC"), "source": "x",
    })
    tagged = dp.tag_headlines(news, ["RELIANCE", "TCS", "LT", "BAJAJ-AUTO"])
    assert set(zip(tagged["title"], tagged["symbol"])) == {
        ("Reliance Industries hits record", "RELIANCE"),
        ("TCS wins deal", "TCS"),
        ("Bajaj-Auto sales up", "BAJAJ-AUTO"),
    }


@pytest.mark.parametrize("url", [
    "javascript:alert(1)",
    "//example.com/article",
    "https://example.com@other.example/article",
    "https://example.com:bad/article",
    "https://bad^host/article",
    "https://example.com/article with spaces",
    "https://example.com/article>](https://other.example)",
    "https://example.com\\@other.example/article",
])
def test_article_url_rejects_unsafe_or_malformed_links(url):
    assert dp.valid_article_url(url) is None


def test_fetch_news_keeps_article_urls_and_logs_invalid_links(monkeypatch, caplog):
    stamp = (2024, 3, 26, 4, 30, 0, 0, 0, 0)
    entries = [
        {"title": "TCS update", "published_parsed": stamp, "link": "https://news.example/a?x=(1)&y=2"},
        {"title": "TCS unsafe", "published_parsed": stamp, "link": "javascript:alert(1)"},
        {"title": "TCS no link", "published_parsed": stamp},
    ]
    monkeypatch.setattr(dp, "RSS_FEEDS", {"test_feed": "https://news.example/feed"})
    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        content=b"feed", raise_for_status=lambda: None,
    ))
    monkeypatch.setattr("feedparser.parse", lambda content: SimpleNamespace(entries=entries))

    news = dp.fetch_news()
    assert news["url"].tolist() == ["https://news.example/a?x=(1)&y=2", None, None]
    assert news["source"].tolist() == ["test_feed"] * 3
    assert "Ignoring 1 malformed or unsafe article URLs" in caplog.text


def test_news_history_backfills_only_unambiguous_matching_source(tmp_path, caplog):
    stamp = pd.Timestamp("2024-03-26 04:30", tz="UTC")
    old = pd.DataFrame({
        "published_at": stamp,
        "title": ["Exact", "Other source", "Conflicting", "Old only"],
        "source": "feed_a",
    })
    path = tmp_path / "news.parquet"
    old.to_parquet(path, index=False)
    new = pd.DataFrame({
        "published_at": stamp,
        "title": ["Exact", "Other source", "Conflicting", "Conflicting", "New"],
        "source": ["feed_a", "feed_b", "feed_a", "feed_a", "feed_a"],
        "url": [
            "https://news.example/exact", "https://other.example/article",
            "https://news.example/first", "https://news.example/second", "https://news.example/new",
        ],
    })

    merged = dp.merge_news_history(new, path).set_index("title")
    assert len(merged) == 5
    assert merged.loc["Exact", "url"] == "https://news.example/exact"
    assert pd.isna(merged.loc["Other source", "url"])
    assert merged.loc["Other source", "source"] == "feed_a"
    assert pd.isna(merged.loc["Conflicting", "url"])
    assert pd.isna(merged.loc["Old only", "url"])
    assert merged.loc["New", "url"] == "https://news.example/new"
    assert "Conflicting article URLs" in caplog.text


def test_news_history_preserves_existing_article_url(tmp_path):
    stamp = pd.Timestamp("2024-03-26 04:30", tz="UTC")
    old = pd.DataFrame({
        "published_at": [stamp], "title": ["TCS update"], "source": ["feed"],
        "url": ["https://news.example/original"],
    })
    path = tmp_path / "news.parquet"
    old.to_parquet(path, index=False)
    new = old.assign(url="https://news.example/different")
    assert dp.merge_news_history(new, path)["url"].tolist() == ["https://news.example/original"]


def test_news_history_does_not_backfill_a_different_publication_time(tmp_path):
    stamp = pd.Timestamp("2024-03-26 04:30", tz="UTC")
    old = pd.DataFrame({"published_at": [stamp], "title": ["TCS update"], "source": ["feed"]})
    path = tmp_path / "news.parquet"
    old.to_parquet(path, index=False)
    new = old.assign(published_at=stamp + pd.Timedelta(hours=1), url="https://news.example/new")
    merged = dp.merge_news_history(new, path).set_index("published_at")
    assert pd.isna(merged.loc[stamp, "url"])
    assert merged.loc[stamp + pd.Timedelta(hours=1), "url"] == "https://news.example/new"


# ------------------------------------------------------------ targets / feature leakage
def test_targets_are_forward_log_returns():
    prices = _prices(120, ("AAA",))
    df = dp.build_feature_frame(prices, *_index_series(prices))
    merged = df.merge(prices[["date", "close"]].rename(columns={"close": "c"}), on="date")
    close = prices.set_index("date")["close"]
    for k in dp.SUPPORTED_HORIZONS:
        expected = np.log(close.shift(-k) / close).reindex(merged["date"]).to_numpy()
        np.testing.assert_allclose(merged[f"target_{k}"].to_numpy(), expected, equal_nan=True)
    assert df[dp.TARGET_COLUMNS].isna().sum().max() <= max(dp.SUPPORTED_HORIZONS)


def test_features_do_not_depend_on_future_prices():
    prices = _prices(150, ("AAA",))
    nifty, vix = _index_series(prices)
    cutoff = prices["date"].iloc[100]
    base = dp.build_feature_frame(prices, nifty, vix)

    shocked = prices.copy()
    shocked.loc[shocked["date"] > cutoff, ["open", "high", "low", "close"]] *= 3
    shocked.loc[shocked["date"] > cutoff, "volume"] *= 10
    alt = dp.build_feature_frame(shocked, nifty * 1.0, vix)

    a = base[base["date"] <= cutoff].reset_index(drop=True)
    b = alt[alt["date"] <= cutoff].reset_index(drop=True)
    np.testing.assert_allclose(a[dp.NUMERIC_FEATURES].to_numpy(), b[dp.NUMERIC_FEATURES].to_numpy())


def test_horizon_constants_are_consistent():
    assert dp.SUPPORTED_HORIZONS == [1, 2, 3, 5, 7, 10]
    assert dp.EMBARGO_DAYS == max(dp.SUPPORTED_HORIZONS)
    assert dp.RANDOM_STATE == 42
    assert len(dp.FALLBACK_NIFTY50) == 50


# ------------------------------------------------------------------ evaluation / gates


def test_pinball_loss_known_values():
    y = np.array([1.0, -1.0])
    assert ev.pinball_loss(y, np.zeros(2), 0.5) == pytest.approx(0.5)
    # Under-prediction is penalised by alpha, over-prediction by (1 - alpha).
    assert ev.pinball_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert ev.pinball_loss(np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(0.1)


def test_directional_accuracy_ignores_zero_predictions():
    y = np.array([1.0, -1.0, 1.0, -1.0])
    assert ev.directional_accuracy(y, np.array([2.0, -2.0, -2.0, 0.0])) == pytest.approx(2 / 3)


def test_quantile_ordering_q10_le_q50_le_q90():
    crossed = np.array([[0.03, 0.01, -0.02], [-0.01, 0.0, 0.02]])
    fixed = ev.enforce_quantile_order(crossed)
    assert (fixed[:, 0] <= fixed[:, 1]).all() and (fixed[:, 1] <= fixed[:, 2]).all()
    np.testing.assert_array_equal(fixed[1], crossed[1])


def test_baseline_gate_requires_beating_v1_and_persistence():
    v1 = {"pinball_q50": 0.010}
    good = {"pinball_q50": 0.009, "persistence_pinball_q50": 0.010}
    assert ev.baseline_gate(good, v1)["passed"]
    assert not ev.baseline_gate({**good, "pinball_q50": 0.0105}, v1)["passed"]
    assert not ev.baseline_gate({**good, "persistence_pinball_q50": 0.0085}, v1)["passed"]


def test_challenger_gate_needs_lower_loss_and_no_direction_regression():
    champ = {"pinball_mean": 0.010, "directional_accuracy": 0.52}
    assert ev.challenger_vs_champion_gate({"pinball_mean": 0.009, "directional_accuracy": 0.52}, champ)["promote"]
    assert not ev.challenger_vs_champion_gate({"pinball_mean": 0.011, "directional_accuracy": 0.55}, champ)["promote"]
    assert not ev.challenger_vs_champion_gate({"pinball_mean": 0.009, "directional_accuracy": 0.50}, champ)["promote"]
    assert ev.challenger_vs_champion_gate(champ, None)["promote"]


def test_persistence_lag_check_flags_lag_echo():
    rng = np.random.default_rng(1)
    lagged = rng.normal(0, 0.01, 2000)
    y = rng.normal(0, 0.01, 2000)  # target unrelated to the lagged return
    echo = ev.persistence_lag_check(y, lagged, lagged)
    assert not echo["beats_persistence"] and not echo["not_a_lag_echo"]
    informed = ev.persistence_lag_check(y, y * 0.5, lagged)
    assert informed["beats_persistence"] and informed["not_a_lag_echo"]


# ------------------------------------------------------------------------------ API
SYMS = ["AAA", "BBB", "CCC", "DDD"]


def _fake_features() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Timestamp("2026-11-20"),
        "symbol": SYMS,
        "close": [100.0, 200.0, 50.0, 400.0],
        "stock_sentiment_1d": [0.10, 0.20, -0.60, 0.0],
    })


def _fake_predict(feats: pd.DataFrame) -> pd.DataFrame:
    # Per-symbol drift: AAA +0.4%/day, BBB +0.1%/day, CCC +0.5%/day, DDD -0.2%/day.
    drift = dict(zip(SYMS, [0.004, 0.001, 0.005, -0.002]))
    out = {}
    for h in dp.SUPPORTED_HORIZONS:
        mid = feats["symbol"].map(drift).to_numpy() * h
        out[f"q10_t{h}"] = mid - 0.02 * np.sqrt(h)
        out[f"q50_t{h}"] = mid
        out[f"q90_t{h}"] = mid + 0.02 * np.sqrt(h)
    return pd.DataFrame(out, index=feats.index)


@pytest.fixture
def client(tmp_path):
    handle = api.ModelHandle(predict=_fake_predict, name="fake", version="7", gate_override=True)
    with TestClient(api.create_app(lambda: handle, _fake_features, tmp_path)) as c:
        c.log_dir = tmp_path
        yield c


def _flat_predict(feats: pd.DataFrame) -> pd.DataFrame:
    """Median forecast identical for every stock (the market-drift-only failure mode); bands still vary."""
    out = {}
    width = 1 + 0.1 * np.arange(len(feats))
    for h in dp.SUPPORTED_HORIZONS:
        out[f"q10_t{h}"] = 0.01 * h - 0.02 * np.sqrt(h) * width
        out[f"q50_t{h}"] = np.full(len(feats), 0.01 * h)
        out[f"q90_t{h}"] = 0.01 * h + 0.02 * np.sqrt(h) * width
    return pd.DataFrame(out, index=feats.index)


@pytest.fixture
def flat_client(tmp_path):
    handle = api.ModelHandle(predict=_flat_predict, name="flat", version="3")
    with TestClient(api.create_app(lambda: handle, _fake_features, tmp_path)) as c:
        c.log_dir = tmp_path
        yield c


def test_health_ok_and_degraded_without_model(client, tmp_path):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == "7" and body["n_symbols"] == 4

    def broken():
        raise RuntimeError("no production alias")

    with TestClient(api.create_app(broken, _fake_features, tmp_path)) as bad:
        resp = bad.get("/health")
        assert resp.status_code == 503 and resp.json()["status"] == "degraded"
        assert bad.get("/predict/AAA").status_code == 503


def test_predict_default_horizon_and_price_reconstruction(client):
    body = client.get("/predict/AAA").json()
    assert body["horizon"] == 3
    assert body["q10"] <= body["q50"] <= body["q90"]
    close = 100.0
    assert body["buy_entry"] == pytest.approx(close * np.exp(_fake_predict(_fake_features())["q10_t1"][0]))
    assert body["take_profit"] == pytest.approx(close * np.exp(0.004 * 3))
    assert body["stop_loss"] == pytest.approx(close * np.exp(0.004 * 3 - 0.02 * np.sqrt(3)))
    assert [p["day"] for p in body["forecast"]] == [1, 2, 3]
    assert all(p["price_q10"] <= p["price_q50"] <= p["price_q90"] for p in body["forecast"])
    assert body["gate_override"] is True


@pytest.mark.parametrize("horizon", dp.SUPPORTED_HORIZONS)
def test_predict_accepts_every_supported_horizon(client, horizon):
    resp = client.get(f"/predict/aaa.ns?horizon={horizon}")
    assert resp.status_code == 200
    assert resp.json()["forecast"][-1]["day"] == horizon


@pytest.mark.parametrize("horizon", [0, 4, 6, 11, -1])
def test_predict_rejects_unsupported_horizon(client, horizon):
    assert client.get(f"/predict/AAA?horizon={horizon}").status_code == 422
    assert client.get(f"/watchlist?horizon={horizon}").status_code == 422


def test_root_redirects_to_swagger_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307) and resp.headers["location"] == "/docs"
    assert client.get("/docs").status_code == 200


def test_constant_median_forecast_forces_hold_and_warns(flat_client, client):
    body = flat_client.get("/predict/AAA?horizon=3").json()
    assert body["expected_return"] == pytest.approx(0.03) and body["expected_return"] > body["min_return"]
    assert body["stock_sentiment_1d"] > body["sentiment_floor"]  # would be BUY under the old rule
    assert body["signal"] == "HOLD"
    assert body["forecast_informative"] is False and body["median_dispersion"] == pytest.approx(0)
    assert "not a view" in body["warning"]

    board = flat_client.get("/watchlist?horizon=10").json()
    assert board["forecast_informative"] is False and board["warning"]
    assert all(i["signal"] == "HOLD" for i in board["items"])
    assert flat_client.get("/watchlist?horizon=10&only_buy=true").json()["count"] == 0

    # A model whose median differs between stocks keeps the normal rule and carries no warning.
    ok = client.get("/predict/AAA?horizon=3").json()
    assert ok["forecast_informative"] is True and ok["warning"] is None and ok["signal"] == "BUY"


def test_drift_override_shows_buy_with_an_explicit_warning(flat_client, client):
    default = flat_client.get("/predict/AAA?horizon=3").json()
    assert default["signal"] == "HOLD" and default["drift_override"] is False

    allowed = flat_client.get("/predict/AAA", params={"horizon": 3, "allow_uninformative": True}).json()
    assert allowed["signal"] == "BUY" and allowed["drift_override"] is True
    assert allowed["forecast_informative"] is False and "demos only" in allowed["warning"]

    board = flat_client.get("/watchlist", params={"horizon": 3, "allow_uninformative": True, "only_buy": True}).json()
    assert board["drift_override"] is True
    assert {i["symbol"] for i in board["items"]} == {"AAA", "BBB", "DDD"}  # CCC still trips the sentiment breaker

    # The switch changes nothing for a model whose median really differs between stocks.
    normal = client.get("/predict/AAA", params={"horizon": 3, "allow_uninformative": True}).json()
    assert normal["drift_override"] is False and normal["warning"] is None


def test_signal_quality_flag_is_audited_in_the_log(flat_client):
    flat_client.get("/predict/BBB?horizon=5")
    table = pd.read_parquet(flat_client.log_dir / "prediction_logs.parquet")
    assert table["forecast_informative"].eq(False).all() and table["signal"].eq("HOLD").all()


def test_predict_validates_inputs(client):
    assert client.get("/predict/NOPE").status_code == 404
    assert client.get("/predict/AAA?horizon=abc").status_code == 422
    assert client.get("/predict/AAA?sentiment_floor=5").status_code == 422


def test_signal_rule_hurdle_and_sentiment_circuit_breaker(client):
    def signal(sym, **q):
        return client.get(f"/predict/{sym}", params={"horizon": 3, **q}).json()["signal"]

    assert signal("AAA") == "BUY"  # +1.2% > 0.5% and sentiment 0.10 > -0.40
    assert signal("BBB") == "HOLD"  # +0.3% is below the 0.5% friction hurdle ...
    assert signal("BBB", min_return=0.0) == "BUY"  # ... but clears a lower one
    assert signal("DDD") == "HOLD"  # negative expected return
    assert signal("CCC") == "HOLD"  # strong return but sentiment -0.60 trips the breaker
    assert signal("CCC", sentiment_floor=-0.9) == "BUY"
    assert signal("AAA", min_return=0.02) == "HOLD"


def test_watchlist_ranked_and_filtered(client):
    body = client.get("/watchlist?horizon=7").json()
    returns = [i["expected_return"] for i in body["items"]]
    assert body["count"] == 4 and returns == sorted(returns, reverse=True)
    buys = client.get("/watchlist?horizon=7&only_buy=true").json()
    assert {i["symbol"] for i in buys["items"]} == {"AAA", "BBB"}
    assert all(i["signal"] == "BUY" for i in buys["items"])


def test_predictions_are_logged_to_jsonl_and_parquet(client):
    client.get("/predict/AAA?horizon=5")
    client.get("/predict/AAA?horizon=5")  # duplicate payload is audited but de-duplicated in Parquet
    client.get("/watchlist?horizon=2")
    lines = (client.log_dir / "prediction_logs.jsonl").read_text().splitlines()
    assert len(lines) == 2 + 4
    table = pd.read_parquet(client.log_dir / "prediction_logs.parquet")
    assert len(table) == 5  # AAA@5 once, plus the four symbols at horizon 2
    assert {"q10_t1", "q50_t10", "q90_t7", "model_version", "signal"} <= set(table.columns)
    assert table["as_of_date"].dtype.kind == "M"
    assert not table.duplicated(["symbol", "as_of_date", "horizon", "model_version"]).any()


# ------------------------------------------------------------------------ dashboard filters
DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"


@pytest.fixture(autouse=True)
def offline_holidays(monkeypatch):
    today = datetime.now(dp.IST).date()
    fetch = md.fetch_trading_holidays
    monkeypatch.setattr(md, "fetch_trading_holidays", lambda: {
        today - timedelta(days=1): "Past holiday",
        today + timedelta(days=3): "Future holiday",
    })
    return fetch


@pytest.fixture
def desk(client, monkeypatch):
    """Dashboard script running in-process against the fake-model API."""
    return _launch_dashboard(client, monkeypatch)


@pytest.fixture
def flat_desk(flat_client, monkeypatch):
    return _launch_dashboard(flat_client, monkeypatch)


def _launch_dashboard(api_client, monkeypatch, *, auth: dict | None = None, logged_in: bool = True):
    st.cache_data.clear()
    monkeypatch.setattr(
        requests, "get",
        lambda url, params=None, timeout=None: api_client.get(url.removeprefix(dash_api_url()), params=params),
    )
    monkeypatch.setattr(
        st, "user",
        SimpleNamespace(is_logged_in=logged_in, name="Demo Trader", email="trader@tradeassist.local"),
    )
    at = AppTest.from_file(str(DASHBOARD), default_timeout=30)
    if auth is not None:
        at.secrets["auth"] = auth
    return at.run()


def dash_api_url() -> str:
    return os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


def _picker(at) -> list[str]:
    return [label.split(" · ")[0] for label in at.selectbox(key="trade_symbol").options]


def test_dashboard_displays_tradeassist_logo(client, monkeypatch):
    calls = []
    render_logo = st.logo

    def record_logo(image, **kwargs):
        calls.append((Path(image), kwargs))
        return render_logo(image, **kwargs)

    monkeypatch.setattr(st, "logo", record_logo)
    dashboard = _launch_dashboard(client, monkeypatch)
    assert not dashboard.exception
    assert len(calls) == 1
    wordmark, kwargs = calls[0]
    assert wordmark.is_file() and "wordmark" in wordmark.name
    assert Path(kwargs["icon_image"]).is_file()
    assert dashboard.title[0].value == "TradeAssist"


def test_dashboard_shows_login_screen_when_logged_out(client, monkeypatch):
    at = _launch_dashboard(client, monkeypatch, auth={"client_id": "tradeassist-dashboard"}, logged_in=False)
    assert not at.exception
    assert at.subheader[0].value == "Sign in to continue"
    assert at.button(key="login")
    assert not at.title and not at.sidebar.button


def test_sidebar_has_account_controls_when_authenticated(client, monkeypatch):
    at = _launch_dashboard(client, monkeypatch, auth={"client_id": "tradeassist-dashboard"}, logged_in=True)
    assert not at.exception
    assert at.title[0].value == "TradeAssist"
    assert at.sidebar.button(key="logout")


def test_controls_live_in_the_main_pane_and_drive_the_picker(desk):
    assert not desk.sidebar.slider and not desk.sidebar.select_slider
    assert desk.select_slider(key="ctl_horizon").value == 3
    assert {"ctl_min_ret_pct", "ctl_floor"} <= {s.key for s in desk.slider}
    desk.select_slider(key="ctl_horizon").set_value(10).run()
    assert desk.selectbox(key="trade_symbol").options[0] == "CCC · HOLD · +5.13%"  # +0.5%/day over 10 days


def test_dashboard_warns_and_holds_when_forecast_is_not_stock_specific(flat_desk, desk):
    assert not flat_desk.exception
    assert any("not a view" in w.value for w in flat_desk.warning)
    assert any("not stock-specific" in c.value for c in flat_desk.caption)
    assert not desk.warning  # a model with real cross-stock spread shows no warning


def test_drift_switch_is_off_by_default_and_reveals_labelled_buy_candidates(flat_desk):
    assert flat_desk.toggle(key="ctl_allow_drift").value is False
    assert not any("BUY" in label for label in flat_desk.selectbox(key="trade_symbol").options)
    flat_desk.toggle(key="ctl_allow_drift").set_value(True).run()
    assert not flat_desk.exception
    labels = flat_desk.selectbox(key="trade_symbol").options
    assert any("· BUY ·" in label for label in labels)
    assert any("demos only" in w.value for w in flat_desk.warning)  # still clearly labelled


def test_dashboard_tolerates_an_older_api_without_signal_quality_fields(client, monkeypatch):
    """A dashboard reloaded against an API started before these fields existed must not crash."""
    st.cache_data.clear()
    new_fields = {"warning", "forecast_informative", "median_dispersion"}

    class OldApiResponse:
        def __init__(self, resp):
            self.status_code, self.headers, self.text = resp.status_code, resp.headers, resp.text
            self._body = {k: v for k, v in resp.json().items() if k not in new_fields}

        def json(self):
            return self._body

    monkeypatch.setattr(
        requests, "get",
        lambda url, params=None, timeout=None: OldApiResponse(
            client.get(url.removeprefix(dash_api_url()), params=params)
        ),
    )
    monkeypatch.setattr(st, "user", SimpleNamespace(is_logged_in=True, name="Demo Trader", email="t@t.local"))
    old = AppTest.from_file(str(DASHBOARD), default_timeout=30).run()
    assert not old.exception
    assert not old.warning and _picker(old)


def test_desk_lists_all_stocks_ranked_by_gain_by_default(desk):
    assert not desk.exception
    assert _picker(desk) == ["CCC", "AAA", "BBB", "DDD"]  # +1.5%, +1.2%, +0.3%, -0.6%
    assert desk.selectbox(key="trade_symbol").value == "CCC"  # RELIANCE is not in the fake universe


def test_signal_filter_narrows_the_stock_picker(desk):
    desk.session_state["flt_signal"] = "BUY"
    desk.run()
    assert _picker(desk) == ["AAA"]
    desk.session_state["flt_signal"] = "HOLD"
    desk.run()
    assert _picker(desk) == ["CCC", "BBB", "DDD"]


def test_sentiment_gain_band_and_ratio_filters(desk):
    desk.session_state["flt_sentiment"] = (-0.5, 1.0)  # drops CCC (-0.60)
    desk.run()
    assert "CCC" not in _picker(desk)
    desk.session_state["flt_min_gain"] = 1.0
    desk.run()
    assert _picker(desk) == ["AAA"]
    desk.session_state["flt_min_gain"] = -5.0
    desk.session_state["flt_max_band"] = 4.0  # all bands are ~7% wide at 3 days
    desk.run()
    assert any("No stock matches" in i.value for i in desk.info)


def test_sort_order_and_selection_survive_filter_changes(desk):
    desk.session_state["flt_sort"] = "Symbol (A to Z)"
    desk.run()
    assert _picker(desk) == ["AAA", "BBB", "CCC", "DDD"]
    desk.selectbox(key="trade_symbol").select("BBB").run()
    desk.session_state["flt_signal"] = "BUY"  # BBB no longer matches -> falls back to a valid pick
    desk.run()
    assert not desk.exception and desk.session_state["trade_symbol"] == "AAA"


def test_reset_filters_restores_defaults(desk):
    desk.session_state["flt_signal"] = "BUY"
    desk.session_state["flt_min_gain"] = 2.0
    desk.run()
    next(button for button in desk.button if button.label == "Reset filters").click().run()
    assert desk.session_state["flt_signal"] == "All" and desk.session_state["flt_min_gain"] == -5.0
    assert len(_picker(desk)) == 4


def test_chart_options_and_pick_survive_an_empty_result(desk):
    desk.selectbox(key="trade_symbol").select("BBB").run()
    desk.session_state["chart_history"] = 30
    desk.session_state["flt_min_gain"] = 5.0  # nothing matches
    desk.run()
    assert any("No stock matches" in i.value for i in desk.info)
    desk.session_state["flt_min_gain"] = -5.0
    desk.run()
    assert not desk.exception
    assert desk.session_state["chart_history"] == 30
    assert desk.session_state["trade_symbol"] == "BBB"


def test_sidebar_shows_today_and_only_upcoming_holidays(client, monkeypatch):
    today = datetime.now(dp.IST).date()
    monkeypatch.setattr(md, "fetch_trading_holidays", lambda: {
        today + timedelta(days=3): "Future holiday",
        today - timedelta(days=1): "Past holiday",
        today: "Today's holiday",
    })
    desk = _launch_dashboard(client, monkeypatch)
    assert not desk.exception
    assert any("Holiday today: Today's holiday" in item.value for item in desk.sidebar.info)
    captions = "\n".join(item.value for item in desk.sidebar.caption)
    assert "Future holiday" in captions and "Past holiday" not in captions
    assert "Today's holiday" not in captions


def test_sidebar_calendar_failure_is_visible_and_sync_remains_available(client, monkeypatch):
    def unavailable():
        raise md.MarketDataError("NSE holiday calendar unavailable")

    monkeypatch.setattr(md, "fetch_trading_holidays", unavailable)
    desk = _launch_dashboard(client, monkeypatch)
    assert not desk.exception
    assert any("calendar unavailable" in item.value for item in desk.sidebar.warning)
    assert desk.sidebar.button(key="sync_market_data")


def test_sidebar_does_not_claim_nonholiday_for_unpublished_year(client, monkeypatch):
    today = datetime.now(dp.IST).date()
    monkeypatch.setattr(md, "fetch_trading_holidays", lambda: {
        today.replace(year=today.year - 1, month=1, day=1): "Old holiday",
    })
    desk = _launch_dashboard(client, monkeypatch)
    assert any("holiday status is unknown" in item.value for item in desk.sidebar.warning)
    assert not any("No NSE trading holiday today" in item.value for item in desk.sidebar.caption)


@pytest.mark.parametrize("malformed", [False, True])
def test_holiday_fetch_uses_equity_calendar_and_validates_records(offline_holidays, monkeypatch, malformed):
    body = {
        "CM": [{"tradingDate": "24-Nov-2026", "description": "Guru Nanak Jayanti"}],
        "CBM": [{"tradingDate": "01-Apr-2026", "description": "Bank closing"}],
    }
    if malformed:
        body["CM"][0]["tradingDate"] = "invalid"

    class Session:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, timeout):
            if url == "https://www.nseindia.com/":
                return SimpleNamespace(status_code=403)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr(md.requests, "Session", Session)
    if malformed:
        with pytest.raises(md.MarketDataError, match="calendar unavailable"):
            offline_holidays()
    else:
        assert offline_holidays() == {pd.Timestamp("2026-11-24").date(): "Guru Nanak Jayanti"}


def test_manual_sync_updates_cached_api_data_and_keeps_selection(desk, client, monkeypatch):
    fresh = _fake_features().assign(date=pd.Timestamp("2026-11-23"))

    def sync():
        client.app.state.engine.features_loader = lambda: fresh
        return fresh["date"].max().date()

    monkeypatch.setattr(md, "sync_market_data", sync)
    desk.selectbox(key="trade_symbol").select("BBB").run()
    desk.sidebar.button(key="sync_market_data").click().run()
    assert not desk.exception
    assert any("Latest data: 23 Nov 2026" in item.value for item in desk.sidebar.success)
    assert any("features as of 2026-11-23" in item.value for item in desk.markdown)
    assert desk.selectbox(key="trade_symbol").value == "BBB"


def test_manual_sync_failure_is_not_reported_as_success(desk, monkeypatch):
    def failed():
        raise md.MarketDataError("Market-data sync failed: network down")

    monkeypatch.setattr(md, "sync_market_data", failed)
    desk.sidebar.button(key="sync_market_data").click().run()
    assert not desk.exception and not desk.sidebar.success
    assert any("network down" in item.value for item in desk.sidebar.error)


# ------------------------------------------------------------------------ news and sentiment
def _news_inputs():
    sessions = pd.DatetimeIndex(["2024-03-22", "2024-03-26", "2024-03-27"])
    published = pd.Series(pd.to_datetime([
        "2024-03-22 10:01Z",  # after Friday's close; Monday is a holiday
        "2024-03-26 04:30Z",  # during Tuesday's session
        "2024-03-26 05:00Z",
        "2024-03-27 04:30Z",
        "2024-03-28 04:30Z",  # beyond the last known session
    ], utc=True))
    news = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "AAA", "AAA"],
        "published_at": published,
        "title": ["After-close news", "Intraday news", "Missing price", "Latest session", "Pending news"],
        "source": "feed",
        "net_sentiment": [0.8, -0.6, 0.0, 0.1, np.nan],
        "effective_trading_date": dp.assign_effective_trading_date(published, sessions),
    })
    prices = pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        "date": pd.to_datetime(["2024-03-22", "2024-03-26", "2024-03-27", "2024-03-22", "2024-03-27"]),
        "close": [100.0, 110.0, 99.0, 200.0, 220.0],
    })
    return news, prices


def test_news_price_moves_use_effective_sessions_and_do_not_skip_missing_prices():
    news, prices = _news_inputs()
    view = nw.news_price_frame(news, prices).set_index("title")

    after_close = view.loc["After-close news"]
    assert after_close["published_ist"] == "22 Mar 2024 15:31"
    assert after_close["effective_trading_date"] == pd.Timestamp("2024-03-26")
    assert after_close["following_session"] == pd.Timestamp("2024-03-27")
    assert after_close["session_move_pct"] == pytest.approx(10)
    assert after_close["following_move_pct"] == pytest.approx(-10)
    assert after_close["sentiment_tone"] == "Positive leaning"

    assert view.loc["Intraday news", "sentiment_tone"] == "Negative leaning"
    assert view.loc["Missing price", "sentiment_tone"] == "Balanced"
    assert pd.isna(view.loc["Missing price", "session_move_pct"])
    assert pd.isna(view.loc["Missing price", "following_move_pct"])
    assert view.loc["Latest session", "session_move_pct"] == pytest.approx(-10)
    assert pd.isna(view.loc["Latest session", "following_move_pct"])
    assert pd.isna(view.loc["Pending news", "effective_trading_date"])
    assert pd.isna(view.loc["Pending news", "session_move_pct"])
    assert view.loc["Pending news", "sentiment_tone"] == "Unavailable"


def test_news_price_moves_do_not_use_invalid_or_duplicate_prices():
    news, prices = _news_inputs()
    prices.loc[(prices["symbol"] == "AAA") & (prices["date"] == "2024-03-26"), "close"] = 0
    result = nw.news_price_frame(news, prices).set_index("title")
    assert pd.isna(result.loc["After-close news", "session_move_pct"])
    assert pd.isna(result.loc["After-close news", "following_move_pct"])
    with pytest.raises(ValueError, match="duplicate stock/session"):
        nw.news_price_frame(news, pd.concat([prices, prices.iloc[[0]]], ignore_index=True))


def test_news_rejects_missing_headlines():
    news, prices = _news_inputs()
    news.loc[0, "title"] = None
    with pytest.raises(ValueError, match="missing or invalid headline"):
        nw.news_price_frame(news, prices)


def _launch_news(tmp_path, monkeypatch, with_news=True, with_prices=True):
    news, prices = _news_inputs()
    monkeypatch.setattr(nw, "RAW_DIR", tmp_path)
    nw.load_news.clear()
    if with_news:
        news.to_parquet(tmp_path / "news_scored.parquet", index=False)
    if with_prices:
        prices.to_parquet(tmp_path / "prices.parquet", index=False)
    return AppTest.from_string("from news import news_view\nnews_view()", default_timeout=30).run()


def test_news_page_shows_both_moves_and_filters_stock_and_tone(tmp_path, monkeypatch):
    page = _launch_news(tmp_path, monkeypatch)
    assert not page.exception
    table = page.table[0].value
    assert len(table) == 5
    assert {"Session %", "Next session %", "Sentiment"} <= set(table.columns)
    assert any("not proof" in item.value for item in page.info)
    assert any("before an intraday headline" in item.value for item in page.info)
    assert any("coverage is incomplete" in item.value for item in page.caption)
    page.selectbox(key="news_stock").select("BBB").run()
    assert page.table[0].value["Headline"].tolist() == ["Missing price (link unavailable)"]
    page.selectbox(key="news_tone").select("Negative leaning").run()
    assert not page.table
    assert any("No headlines match" in item.value for item in page.info)


def test_news_page_handles_missing_data_and_invalid_files(tmp_path, monkeypatch):
    page = _launch_news(tmp_path, monkeypatch, with_news=False, with_prices=False)
    assert any("No scored stock headlines" in item.value for item in page.info)

    page = _launch_news(tmp_path, monkeypatch, with_prices=False)
    assert not page.exception and len(page.table[0].value) == 5
    assert any("Price history is unavailable" in item.value for item in page.warning)
    assert page.table[0].value["Session %"].str.startswith("None").all()

    (tmp_path / "news_scored.parquet").write_bytes(b"not parquet")
    nw.load_news.clear()
    page.run()
    assert not page.exception
    assert any("Could not load news and price data" in item.value for item in page.error)


def test_news_page_renders_safe_links_and_keeps_sorting(tmp_path, monkeypatch):
    news, prices = _news_inputs()
    news["title"] = ["TCS [results]", "No link", "Unsafe link", "Duplicate URL", "Pending news"]
    news["url"] = [
        "https://news.example/article?value=(two)&x=1", None, "javascript:alert(1)",
        "https://news.example/article?value=(two)&x=1", None,
    ]
    monkeypatch.setattr(nw, "RAW_DIR", tmp_path)
    news.to_parquet(tmp_path / "news_scored.parquet", index=False)
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    nw.load_news.clear()
    page = AppTest.from_string("from news import news_view\nnews_view()", default_timeout=30).run()
    assert not page.exception
    headlines = page.table[0].value["Headline"].tolist()
    assert r"[TCS \[results\]](<https://news.example/article?value=(two)&amp;x=1>)" in headlines
    assert "[Duplicate URL](<https://news.example/article?value=(two)&amp;x=1>)" in headlines
    assert "Unsafe link (link unavailable)" in headlines
    assert "No link (link unavailable)" in headlines
    page.selectbox(key="news_sort").select("Stock").run()
    assert page.table[0].value["Stock"].tolist() == ["BBB", "AAA", "AAA", "AAA", "AAA"]
    page.checkbox(key="news_descending").uncheck().run()
    assert page.table[0].value["Stock"].tolist() == ["AAA", "AAA", "AAA", "AAA", "BBB"]


def test_news_page_paginates_large_filtered_results(tmp_path, monkeypatch):
    base, prices = _news_inputs()
    news = pd.concat([base.iloc[[0]]] * 301, ignore_index=True)
    news["title"] = [
        f"{'Keep' if index < 150 else 'Other'} headline {index:03d}" for index in range(len(news))
    ]
    monkeypatch.setattr(nw, "RAW_DIR", tmp_path)
    news.to_parquet(tmp_path / "news_scored.parquet", index=False)
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    nw.load_news.clear()
    page = AppTest.from_string("from news import news_view\nnews_view()", default_timeout=30).run()
    assert not page.exception and len(page.table[0].value) == 100
    page.selectbox(key="news_page").select(4).run()
    assert not page.exception and len(page.table[0].value) == 1
    page.text_input(key="news_search").set_value("Keep").run()
    assert not page.exception and page.selectbox(key="news_page").value in (1, 2)
    assert len(page.table[0].value) in (50, 100)
    page.text_input(key="news_search").set_value("Keep headline 000").run()
    assert not page.exception and len(page.table[0].value) == 1


def test_news_stock_filter_recovers_when_refreshed_news_drops_a_stock(tmp_path, monkeypatch):
    page = _launch_news(tmp_path, monkeypatch)
    page.selectbox(key="news_stock").select("BBB").run()
    news, _ = _news_inputs()
    news[news["symbol"] == "AAA"].to_parquet(tmp_path / "news_scored.parquet", index=False)
    nw.load_news.clear()
    page.run()
    assert not page.exception
    assert page.selectbox(key="news_stock").value == "All stocks"


def test_manual_sync_invalidates_cached_news(desk, tmp_path, monkeypatch):
    news, prices = _news_inputs()
    monkeypatch.setattr(nw, "RAW_DIR", tmp_path)
    news.iloc[:1].to_parquet(tmp_path / "news_scored.parquet", index=False)
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    nw.load_news.clear()
    assert len(nw.load_news()[0]) == 1

    def sync():
        news.to_parquet(tmp_path / "news_scored.parquet", index=False)
        return pd.Timestamp("2024-03-27").date()

    monkeypatch.setattr(md, "sync_market_data", sync)
    desk.sidebar.button(key="sync_market_data").click().run()
    assert not desk.exception
    assert len(nw.load_news()[0]) == 5


def test_sync_refreshes_without_refitting(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "PROCESSED_DIR", tmp_path)

    def run(command, **kwargs):
        assert command[-1] == "--refresh-only"
        assert "--skip-sentiment" not in command
        pd.DataFrame({"date": [pd.Timestamp("2026-11-23")]}).to_parquet(tmp_path / "latest_features.parquet")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(md.subprocess, "run", run)
    assert md.sync_market_data() == pd.Timestamp("2026-11-23").date()
    assert not md.SYNC_LOCK.locked()


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_sync_subprocess_failure_releases_lock(monkeypatch, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=1, stdout="", stderr="download failed")

    monkeypatch.setattr(md.subprocess, "run", run)
    with pytest.raises(md.MarketDataError, match="sync failed"):
        md.sync_market_data()
    assert not md.SYNC_LOCK.locked()


def test_sync_rejects_concurrent_manual_runs():
    with md.SYNC_LOCK, pytest.raises(md.MarketDataError, match="already running"):
        md.sync_market_data()


# ----------------------------------------------------------------- reconciliation / drift
def _recon_inputs(n_days: int = 30, forecast_idx: int = 5):
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    closes = 100 * np.exp(np.cumsum(np.full(n_days, 0.01)))  # +1% log return per day
    prices = pd.DataFrame({"symbol": "AAA", "date": dates, "close": closes})
    row = {"symbol": "AAA", "as_of_date": dates[forecast_idx], "model_version": "1", "horizon": 3}
    for h in dp.SUPPORTED_HORIZONS:
        row.update({f"q10_t{h}": -0.005 * h, f"q50_t{h}": 0.01 * h, f"q90_t{h}": 0.02 * h})
    return dates, prices, pd.DataFrame([row])


def test_reconcile_grades_each_horizon_once_close_arrives():
    dates, prices, logs = _recon_inputs()
    recon = mon.reconcile(logs, prices).set_index("horizon")
    assert list(recon.index) == dp.SUPPORTED_HORIZONS
    first = recon.loc[1]
    assert first["target_date"] == dates[6]
    assert first["actual_return"] == pytest.approx(0.01)
    assert recon.loc[10, "target_date"] == dates[15]
    assert recon["direction_correct"].all() and recon["in_interval"].all()


def test_reconcile_waits_for_delayed_ground_truth():
    _, prices, logs = _recon_inputs(n_days=30, forecast_idx=26)  # only t+1..t+3 exist
    assert list(mon.reconcile(logs, prices)["horizon"]) == [1, 2, 3]


def test_reconcile_flags_misses_and_skips_unknown_forecast_dates():
    _, prices, logs = _recon_inputs()
    logs["q50_t1"], logs["q10_t1"], logs["q90_t1"] = -0.01, -0.02, 0.005
    miss = mon.reconcile(logs, prices).set_index("horizon").loc[1]
    assert not miss["direction_correct"] and not miss["in_interval"]
    logs["as_of_date"] = pd.Timestamp("2026-01-04")  # a Sunday: not in the price history
    assert mon.reconcile(logs, prices).empty


def _recon_frame(correct_by_day: list[float], per_day: int = 10) -> pd.DataFrame:
    rows = []
    for i, share in enumerate(correct_by_day):
        for j in range(per_day):
            rows.append({
                "horizon": 1, "target_date": pd.Timestamp("2026-02-02") + pd.offsets.BDay(i),
                "direction_correct": j < round(share * per_day), "in_interval": True,
            })
    return pd.DataFrame(rows)


def test_rolling_window_needs_ten_days_and_pools_all_stocks():
    short = mon.daily_performance(_recon_frame([0.4] * 9))
    assert short["da_roll"].isna().all()
    full = mon.daily_performance(_recon_frame([0.4] * 10))
    assert full["da_roll"].iloc[-1] == pytest.approx(0.4)
    assert full["n"].iloc[-1] == 10


def test_perf_degraded_requires_three_consecutive_low_days():
    good, bad = 0.6, 0.3
    # Rolling accuracy only falls below 50% on the final two days -> not yet degraded.
    assert not mon.performance_signal(mon.daily_performance(_recon_frame([good] * 10 + [bad, bad])))["perf_degraded"]
    # After a third day of drag the 10-day pooled accuracy has been below 50% three days running.
    sig = mon.performance_signal(mon.daily_performance(_recon_frame([good] * 10 + [bad] * 6)))
    assert sig["perf_degraded"] and sig["consecutive_days_below"] >= 3
    assert not mon.performance_signal(mon.daily_performance(_recon_frame([0.55] * 20)))["perf_degraded"]
    assert not mon.performance_signal(pd.DataFrame(columns=["da_roll"]))["perf_degraded"]


def test_population_stability_index():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert mon.population_stability_index(ref, rng.normal(0, 1, 500)) < 0.10
    assert mon.population_stability_index(ref, rng.normal(1.5, 1, 500)) > mon.PSI_THRESHOLD
    const = np.full(1000, 0.002)
    assert mon.population_stability_index(const, np.full(50, 0.002)) == pytest.approx(0.0)
    assert mon.population_stability_index(const, np.full(50, 0.009)) > mon.PSI_THRESHOLD


def test_status_classification():
    assert mon.classify_status(False, 0.05, 0.10) == "OK"
    assert mon.classify_status(False, 0.15, 0.10) == "WARN"
    assert mon.classify_status(False, 0.25, 0.10) == "ALERT"
    assert mon.classify_status(False, None, 0.40) == "ALERT"
    assert mon.classify_status(True, 0.0, 0.0) == "ALERT"
    assert mon.classify_status(False, None, None) == "UNKNOWN"


# ------------------------------------------------------------------------------ retraining
CAL = pd.bdate_range("2026-03-02", periods=40)
QUIET = {"perf_degraded": False, "prediction_psi": 0.05, "drifted_feature_share": 0.10}


def _decide(status=None, last=None, today=CAL[-1], force=False):
    return rt.evaluate_guardrails({**QUIET, **(status or {})}, last, today.date(), CAL, force)


def test_no_retrain_without_a_trigger():
    assert _decide()["action"] == "skip_no_trigger"


@pytest.mark.parametrize("status", [
    {"perf_degraded": True}, {"prediction_psi": 0.21}, {"drifted_feature_share": 0.31},
])
def test_any_single_signal_triggers_retrain(status):
    decision = _decide(status)
    assert decision["action"] == "retrain" and decision["reasons"]


def test_signal_thresholds_are_strict():
    assert _decide({"prediction_psi": 0.20, "drifted_feature_share": 0.30})["action"] == "skip_no_trigger"
    assert _decide({"prediction_psi": None, "drifted_feature_share": None})["action"] == "skip_no_trigger"


def test_cooldown_blocks_retrain_for_seven_trading_days():
    trigger = {"perf_degraded": True}
    assert _decide(trigger, last=CAL[-4].date())["action"] == "skip_cooldown"  # 3 days ago
    assert _decide(trigger, last=CAL[-7].date())["action"] == "skip_cooldown"  # 6 days ago
    assert _decide(trigger, last=CAL[-8].date())["action"] == "retrain"  # 7 days ago
    assert rt.trading_days_since(CAL[-8].date(), CAL[-1].date(), CAL) == 7
    assert rt.trading_days_since(None, CAL[-1].date(), CAL) is None


def test_manual_force_overrides_trigger_and_cooldown():
    decision = _decide(last=CAL[-2].date(), force=True)
    assert decision["action"] == "retrain" and decision["reasons"][0] == "manual_force"


def test_holdout_split_is_chronological_with_embargo():
    prices = _prices(700, ("AAA", "BBB"))
    feats = dp.build_feature_frame(prices, *_index_series(prices))
    splits = rt.make_holdout_splits(feats)
    all_dates = pd.DatetimeIndex(sorted(feats["date"].unique()))
    pos = {d: i for i, d in enumerate(all_dates)}

    def span(name):
        d = splits[name]["date"]
        return pos[d.min()], pos[d.max()]

    (_, tr_hi), (va_lo, va_hi), (ho_lo, ho_hi) = span("train"), span("val"), span("holdout")
    assert va_lo - tr_hi - 1 >= dp.EMBARGO_DAYS
    assert ho_lo - va_hi - 1 >= dp.EMBARGO_DAYS
    assert splits["holdout"]["date"].nunique() == rt.HOLDOUT_DAYS
    assert splits["holdout"][dp.TARGET_COLUMNS].notna().all().all()
    assert ho_hi == len(all_dates) - 1 - max(dp.SUPPORTED_HORIZONS)  # last fully realised dates


# --------------------------------------------------------------------------- daily run pieces
def test_batch_logging_covers_every_horizon_and_is_idempotent(tmp_path):
    handle = api.ModelHandle(predict=_fake_predict, name="fake", version="7")
    logger = api.PredictionLogger(tmp_path)
    assert api.log_batch_forecasts(handle, _fake_features(), logger) == len(SYMS) * len(dp.SUPPORTED_HORIZONS)
    table = pd.read_parquet(tmp_path / "prediction_logs.parquet")
    assert len(table) == 24 and set(table["horizon"]) == set(dp.SUPPORTED_HORIZONS)
    assert {"q10_t1", "q50_t10", "q90_t5"} <= set(table.columns)  # everything reconciliation needs

    api.log_batch_forecasts(handle, _fake_features(), logger)  # a second run the same day
    assert len(pd.read_parquet(tmp_path / "prediction_logs.parquet")) == 24
    assert len((tmp_path / "prediction_logs.jsonl").read_text().splitlines()) == 48  # audit trail keeps both


def test_only_new_headlines_are_scored_and_cached_scores_are_reused(tmp_path, monkeypatch):
    calls = []

    def fake_score(titles, **_):
        calls.append(list(titles))
        return np.full(len(titles), 0.5, dtype="float32")

    monkeypatch.setattr(dp, "score_sentiment", fake_score)
    stamp = pd.Timestamp("2026-11-20 05:00", tz="UTC")

    def tagged(*titles):
        return pd.DataFrame({"symbol": "TCS", "published_at": stamp, "title": list(titles), "source": "x"})

    path = tmp_path / "news_scored.parquet"
    first = dp.score_new_headlines(tagged("A up", "B down"), path)
    first.to_parquet(path)
    second = dp.score_new_headlines(tagged("A up", "B down", "C new"), path)
    assert calls == [["A up", "B down"], ["C new"]]
    assert second["net_sentiment"].notna().all() and len(second) == 3


def test_cached_scores_preserve_matching_article_urls(tmp_path, monkeypatch):
    stamp = pd.Timestamp("2024-03-26 04:30", tz="UTC")
    cached = pd.DataFrame({
        "symbol": "TCS", "published_at": stamp, "title": ["Exact", "Other source"],
        "source": ["feed_a", "feed_b"], "url": ["https://news.example/exact", "https://other.example/story"],
        "net_sentiment": [0.4, -0.3],
    })
    path = tmp_path / "news_scored.parquet"
    cached.to_parquet(path, index=False)
    tagged = cached.drop(columns=["url", "net_sentiment"]).assign(source="feed_a")
    monkeypatch.setattr(dp, "score_sentiment", lambda titles: pytest.fail("Cached scores should be reused"))
    result = dp.score_new_headlines(tagged, path).set_index("title")
    assert result.loc["Exact", "url"] == "https://news.example/exact"
    assert pd.isna(result.loc["Other source", "url"])
    assert result.loc["Exact", "net_sentiment"] == pytest.approx(0.4)
    assert result.loc["Other source", "net_sentiment"] == pytest.approx(-0.3)


def test_refresh_backfills_links_without_rescoring_or_changing_sentiment(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    monkeypatch.setattr(dp, "RAW_DIR", raw)
    monkeypatch.setattr(dp, "PROCESSED_DIR", processed)
    stamp = pd.Timestamp("2024-03-26 04:30", tz="UTC")
    old = pd.DataFrame({"published_at": [stamp], "title": ["TCS update"], "source": ["feed"]})
    old.to_parquet(raw / "news.parquet", index=False)
    old.assign(symbol="TCS", net_sentiment=0.4, effective_trading_date=pd.Timestamp("2024-03-26")).to_parquet(
        raw / "news_scored.parquet", index=False,
    )
    dates = pd.to_datetime(["2024-03-26", "2024-03-27"])
    prices = pd.DataFrame({"date": dates, "symbol": "TCS", "close": [100.0, 101.0]})
    captured = {}
    monkeypatch.setattr(dp, "fetch_nifty50_symbols", lambda: ["TCS"])
    monkeypatch.setattr(dp, "download_prices", lambda symbols, years: prices)
    monkeypatch.setattr(dp, "download_index_series", lambda years: (
        pd.Series([1.0, 2.0], index=dates, name="nifty"),
        pd.Series([1.0, 2.0], index=dates, name="vix"),
    ))
    monkeypatch.setattr(dp, "fetch_news", lambda: old.assign(url="https://news.example/tcs"))
    monkeypatch.setattr(dp, "score_sentiment", lambda titles: pytest.fail("Cached scores should be reused"))

    def build_features(prices, nifty, vix, news_features):
        captured["news_features"] = news_features
        return prices[["date", "symbol"]]

    monkeypatch.setattr(dp, "build_feature_frame", build_features)
    monkeypatch.setattr(dp, "build_ranking_frame", lambda *args: prices[["date", "symbol"]])
    dp.run_pipeline(refit=False)
    assert pd.read_parquet(raw / "news.parquet")["url"].tolist() == ["https://news.example/tcs"]
    scored = pd.read_parquet(raw / "news_scored.parquet")
    assert scored["url"].tolist() == ["https://news.example/tcs"]
    assert scored["net_sentiment"].tolist() == [0.4]
    assert captured["news_features"].loc[0, "stock_sentiment_1d"] == pytest.approx(0.4)
    assert captured["news_features"].loc[0, "stock_news_volume"] == 1


# ------------------------------------------------------------------------- ranking models
def _ranking_inputs(n_days: int = 320, n_symbols: int = 34):
    prices = _prices(n_days, tuple(f"S{i:02d}" for i in range(n_symbols)), seed=3)
    return prices, *_index_series(prices)


def test_ranking_frame_is_market_relative_and_balanced():
    df = dp.build_ranking_frame(*_ranking_inputs())
    assert df[dp.CS_FEATURES].abs().max().max() <= 0.5 + 1e-9
    assert df.groupby("date")[dp.CS_FEATURES].mean().abs().max().max() < 0.02  # ranks are centred each day
    assert df.groupby("date").size().min() >= dp.MIN_CS_STOCKS
    labelled = df.dropna(subset=["rel_5"])
    assert labelled.groupby("date")["rel_5"].mean().abs().max() < 1e-12  # market return removed
    counts = labelled.groupby("date")["label_5"].value_counts().unstack()
    assert set(counts.columns) == set(range(dp.RANK_LEVELS))
    assert (counts.max(axis=1) - counts.min(axis=1)).max() <= 2  # quintiles stay balanced every day
    assert df.loc[df["date"] == df["date"].max(), "label_10"].isna().all()  # future not yet realised


def test_ranking_frame_supports_custom_horizons():
    prices, nifty, vix = _ranking_inputs()
    df = dp.build_ranking_frame(prices, nifty, vix, horizons=[21])
    assert {"target_21", "rel_21", "label_21"} <= set(df.columns)
    assert "rel_5" not in df.columns
    assert df["rel_21"].notna().any() and df["label_21"].isna().sum() >= 21


def test_ranking_features_do_not_depend_on_future_prices():
    prices, nifty, vix = _ranking_inputs()
    cutoff = sorted(prices["date"].unique())[290]
    shocked = prices.copy()
    later = shocked["date"] > cutoff
    shocked.loc[later, ["open", "high", "low", "close"]] *= 3
    shocked.loc[later, "volume"] *= 10
    cols = [*dp.CS_FEATURES, "india_vix", "mkt_ret_21"]
    base = dp.build_ranking_frame(prices, nifty, vix)
    alt = dp.build_ranking_frame(shocked, nifty * 1.0, vix)
    pd.testing.assert_frame_equal(
        base[base["date"] <= cutoff].set_index(["date", "symbol"])[cols],
        alt[alt["date"] <= cutoff].set_index(["date", "symbol"])[cols],
    )


def test_daily_rank_ic_and_t_statistic_use_non_overlapping_dates():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=40)
    df = pd.DataFrame({"date": np.repeat(dates, 20), "target": rng.normal(size=800)})
    df["same"], df["inverse"] = df["target"], -df["target"]
    assert ev.daily_rank_ic(df, "same", "target").eq(1).all()
    assert ev.daily_rank_ic(df, "inverse", "target").eq(-1).all()
    ic = pd.Series(rng.normal(0.05, 0.1, 200), index=pd.bdate_range("2024-01-01", periods=200))
    summary = ev.ic_summary(ic, horizon=10)
    sparse = ic.iloc[::10]
    assert summary["n_days"] == 200
    assert summary["t_stat"] == pytest.approx(sparse.mean() / sparse.std() * np.sqrt(len(sparse)))
    assert summary["t_stat"] < ic.mean() / ic.std() * np.sqrt(len(ic))  # overlap would inflate the naive t


def test_walk_forward_windows_are_embargoed_and_contiguous():
    folds = ev.walk_forward_windows(1000, n_folds=5, first_test_frac=0.55, embargo=10)
    assert folds[0]["test_start"] == 550 and folds[-1]["test_end"] == 1000
    assert all(f["train_end"] == f["test_start"] - 10 for f in folds)
    assert all(a["test_end"] == b["test_start"] for a, b in itertools.pairwise(folds))
    assert all(f["train_end"] > 0 and f["test_end"] > f["test_start"] for f in folds)


def test_ranking_gate_needs_large_significant_and_consistent_ic():
    good = {"mean_ic": 0.04, "t_stat": 3.5}
    assert ev.ranking_gate(good, [0.05, 0.03, 0.04, 0.02, -0.01])["passed"]
    assert not ev.ranking_gate({**good, "mean_ic": 0.02}, [0.05, 0.03, 0.04, 0.02, 0.01])["passed"]
    assert not ev.ranking_gate({**good, "t_stat": 2.0}, [0.05, 0.03, 0.04, 0.02, 0.01])["passed"]
    assert not ev.ranking_gate(good, [0.05, -0.01, -0.02, 0.01, -0.03])["passed"]  # unstable across folds


def _synthetic_rank_frame(signal: float, n_dates: int = 420, n_stocks: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    df = pd.DataFrame({"date": np.repeat(dates, n_stocks), "symbol": np.tile(np.arange(n_stocks), n_dates)})
    for col in dp.CS_FEATURES:
        df[col] = rng.uniform(-0.5, 0.5, len(df))
    df["sector_id"] = rng.integers(0, 11, len(df)).astype("int16")
    df["india_vix"], df["mkt_ret_21"] = 15.0, 0.0
    rel = signal * df["cs_ret_21"] + rng.normal(0, 0.02, len(df))
    df["rel_5"] = rel - rel.groupby(df["date"]).transform("mean")
    pct = df.groupby("date")["rel_5"].rank(method="first", pct=True)
    df["label_5"] = np.ceil(pct * dp.RANK_LEVELS) - 1
    return df


@pytest.mark.parametrize("kind", train.RANKER_KINDS)
def test_ranker_finds_a_planted_signal_and_none_in_noise(kind):
    planted = train.rank_report(_synthetic_rank_frame(0.05), horizon=5, kind=kind)
    noise = train.rank_report(_synthetic_rank_frame(0.0), horizon=5, kind=kind)
    assert planted["mean_ic"] > 0.15 and planted["t_stat"] > 3 and planted["gate"]["passed"]
    assert abs(noise["mean_ic"]) < 0.05 and not noise["gate"]["passed"]
