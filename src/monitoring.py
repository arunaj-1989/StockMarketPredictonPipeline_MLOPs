"""Delayed ground-truth reconciliation, drift detection and prediction PSI.

Run:  python src/monitoring.py [--refresh-prices]

Outputs (data/processed/):
  reconciliation.parquet   graded forecasts (one row per symbol, forecast date, horizon)
  drift_status.json        headline health read by the dashboard and retrain.py
  prediction_psi.json      PSI of predicted q50 per horizon vs. the training reference
  evidently_drift.html     Evidently DataDriftPreset report
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import NUMERIC_FEATURES, PROCESSED_DIR, RAW_DIR, SUPPORTED_HORIZONS

LOGGER = logging.getLogger("monitoring")

LOG_PATH = PROCESSED_DIR / "prediction_logs.parquet"
PRICES_PATH = RAW_DIR / "prices.parquet"
RECON_PATH = PROCESSED_DIR / "reconciliation.parquet"
DRIFT_PATH = PROCESSED_DIR / "drift_status.json"
PSI_PATH = PROCESSED_DIR / "prediction_psi.json"
REPORT_PATH = PROCESSED_DIR / "evidently_drift.html"

ROLLING_WINDOW_DAYS = 10  # pooled over all stocks
PERSISTENCE_DAYS = 3  # consecutive days the rolling accuracy must stay low
DA_THRESHOLD = 0.50
PSI_THRESHOLD = 0.20
DRIFT_SHARE_THRESHOLD = 0.30
PSI_WARN, DRIFT_SHARE_WARN = 0.10, 0.15
PERF_HORIZON = 1  # fastest-maturing horizon drives the retraining signal
CURRENT_WINDOW_DAYS = 20  # trading days of features compared with the reference
PSI_PREDICTION_DATES = 10  # most recent forecast dates used for prediction PSI
PSI_MIN_ROWS = 30
PSI_BINS = 10


# ----------------------------------------------------------------------- reconciliation
def reconcile(logs: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Grade logged forecasts against realised closes once t+k has arrived.

    Every logged forecast stores q10/q50/q90 for all supported horizons, so each one is
    graded for every k whose k-th trading day after the forecast date exists in ``prices``.
    """
    columns = [
        "symbol", "as_of_date", "horizon", "target_date", "close", "q10", "q50", "q90",
        "actual_return", "direction_correct", "in_interval", "model_version",
    ]
    if logs.empty or prices.empty:
        return pd.DataFrame(columns=columns)
    forecasts = logs.drop_duplicates(["symbol", "as_of_date", "model_version"], keep="first")
    rows = []
    for symbol, fc in forecasts.groupby("symbol"):
        px = prices[prices["symbol"] == symbol].sort_values("date")
        dates, closes = px["date"].to_numpy(), px["close"].to_numpy(dtype="float64")
        as_of = fc["as_of_date"].to_numpy()
        pos = np.searchsorted(dates, as_of)
        for i, (_, forecast) in enumerate(fc.iterrows()):
            p = pos[i]
            if p >= len(dates) or dates[p] != as_of[i]:
                continue  # forecast date missing from the price history
            for k in SUPPORTED_HORIZONS:
                if p + k >= len(dates):
                    continue  # t+k close has not arrived yet
                actual = float(np.log(closes[p + k] / closes[p]))
                q10, q50, q90 = (float(forecast[f"{q}_t{k}"]) for q in ("q10", "q50", "q90"))
                rows.append({
                    "symbol": symbol, "as_of_date": forecast["as_of_date"], "horizon": k,
                    "target_date": pd.Timestamp(dates[p + k]), "close": float(closes[p]),
                    "q10": q10, "q50": q50, "q90": q90, "actual_return": actual,
                    "direction_correct": bool(np.sign(q50) == np.sign(actual)),
                    "in_interval": bool(q10 <= actual <= q90),
                    "model_version": forecast["model_version"],
                })
    return pd.DataFrame(rows, columns=columns)


def daily_performance(recon: pd.DataFrame, horizon: int = PERF_HORIZON,
                      window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Per-target-date pooled accuracy plus the rolling ``window``-trading-day accuracy."""
    view = recon[recon["horizon"] == horizon]
    if view.empty:
        return pd.DataFrame(columns=["n", "correct", "da", "covered", "da_roll", "coverage_roll"])
    daily = view.groupby("target_date").agg(
        n=("direction_correct", "size"), correct=("direction_correct", "sum"),
        covered=("in_interval", "sum"),
    ).sort_index()
    daily["da"] = daily["correct"] / daily["n"]
    roll = daily[["n", "correct", "covered"]].rolling(window, min_periods=window).sum()
    daily["da_roll"] = roll["correct"] / roll["n"]
    daily["coverage_roll"] = roll["covered"] / roll["n"]
    return daily


def performance_signal(
    daily: pd.DataFrame, threshold: float = DA_THRESHOLD, persistence: int = PERSISTENCE_DAYS
) -> dict:
    """perf_degraded: rolling accuracy below ``threshold`` for ``persistence`` consecutive days."""
    if daily.empty or daily["da_roll"].isna().all():
        return {"perf_degraded": False, "rolling_da": None, "consecutive_days_below": 0}
    below = (daily["da_roll"] < threshold) & daily["da_roll"].notna()
    streak = 0
    for flag in reversed(below.tolist()):
        if not flag:
            break
        streak += 1
    last = daily["da_roll"].dropna()
    return {
        "perf_degraded": streak >= persistence,
        "rolling_da": float(last.iloc[-1]),
        "consecutive_days_below": streak,
    }


# ------------------------------------------------------------------------------- drift
def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """PSI with reference-quantile bins; a constant reference collapses to <=c / >c."""
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    ref_idx = np.searchsorted(edges, reference, side="left")
    cur_idx = np.searchsorted(edges, current, side="left")
    size = len(edges) + 1
    eps = 1e-4
    ref_pct = np.maximum(np.bincount(ref_idx, minlength=size) / len(reference), eps)
    cur_pct = np.maximum(np.bincount(cur_idx, minlength=size) / len(current), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def prediction_psi(reference_preds: dict[int, np.ndarray], logs: pd.DataFrame) -> dict:
    """PSI of predicted q50 per horizon: reference-sample predictions vs recent logged forecasts."""
    if logs.empty:
        return {"prediction_psi": None, "psi_by_horizon": {}, "n_current": 0}
    forecasts = logs.drop_duplicates(["symbol", "as_of_date", "model_version"], keep="first")
    recent_dates = sorted(forecasts["as_of_date"].unique())[-PSI_PREDICTION_DATES:]
    current = forecasts[forecasts["as_of_date"].isin(recent_dates)]
    if len(current) < PSI_MIN_ROWS:
        return {"prediction_psi": None, "psi_by_horizon": {}, "n_current": len(current)}
    by_horizon = {
        h: population_stability_index(reference_preds[h], current[f"q50_t{h}"].to_numpy())
        for h in SUPPORTED_HORIZONS if h in reference_preds
    }
    return {
        "prediction_psi": max(by_horizon.values()) if by_horizon else None,
        "psi_by_horizon": {str(h): v for h, v in by_horizon.items()},
        "n_current": len(current),
    }


def feature_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, html_path: Path = REPORT_PATH
) -> dict:
    """Evidently DataDriftPreset; saves the HTML report and returns the drifted-column share."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(current[NUMERIC_FEATURES], reference[NUMERIC_FEATURES])
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    for metric in snapshot.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return {
                "drifted_feature_share": float(metric["value"]["share"]),
                "drifted_feature_count": int(metric["value"]["count"]),
            }
    raise RuntimeError("Evidently report has no DriftedColumnsCount metric")


def classify_status(perf_degraded: bool, psi: float | None, share: float | None) -> str:
    psi_v, share_v = psi or 0.0, share or 0.0
    if perf_degraded or psi_v > PSI_THRESHOLD or share_v > DRIFT_SHARE_THRESHOLD:
        return "ALERT"
    if psi_v > PSI_WARN or share_v > DRIFT_SHARE_WARN:
        return "WARN"
    return "OK" if (psi is not None or share is not None) else "UNKNOWN"


# ------------------------------------------------------------------------------ driver
def refresh_prices(path: Path = PRICES_PATH, period: str = "3mo") -> pd.DataFrame:
    """Fetch recent closes and upsert them into the raw price store."""
    from data_prep import download_prices

    existing = pd.read_parquet(path)
    fresh = download_prices(sorted(existing["symbol"].unique()), period=period)
    merged = (
        pd.concat([existing, fresh], ignore_index=True)
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"]).reset_index(drop=True)
    )
    merged.to_parquet(path, compression="zstd", index=False)
    return merged


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def run_monitoring(refresh: bool = False, production_bundle: dict | None = None) -> dict:
    """Reconcile, compute drift signals, write all artefacts and return the status dict."""
    prices = refresh_prices() if refresh else pd.read_parquet(PRICES_PATH)
    logs = pd.read_parquet(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()

    recon = reconcile(logs, prices)
    recon.to_parquet(RECON_PATH, compression="zstd", index=False)
    perf = performance_signal(daily_performance(recon))

    reference = pd.read_parquet(PROCESSED_DIR / "reference_sample.parquet")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    last_dates = sorted(features["date"].unique())[-CURRENT_WINDOW_DAYS:]
    drift = feature_drift_report(reference, features[features["date"].isin(last_dates)])

    psi = {"prediction_psi": None, "psi_by_horizon": {}, "n_current": 0}
    if production_bundle is not None and not logs.empty:
        from train import predict_bundle

        ref_preds = {h: arr[:, 1] for h, arr in predict_bundle(production_bundle, reference).items()}
        psi = prediction_psi(ref_preds, logs)
    PSI_PATH.write_text(json.dumps(_json_safe(psi), indent=2), encoding="utf-8")

    status = {
        "status": classify_status(perf["perf_degraded"], psi["prediction_psi"], drift["drifted_feature_share"]),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "as_of": str(prices["date"].max().date()),
        "n_graded": len(recon),
        "n_forecast_rows": len(logs),
        **perf, **drift, **psi,
        "thresholds": {
            "da": DA_THRESHOLD, "psi": PSI_THRESHOLD, "drifted_feature_share": DRIFT_SHARE_THRESHOLD,
            "rolling_window_days": ROLLING_WINDOW_DAYS, "persistence_days": PERSISTENCE_DAYS,
        },
    }
    DRIFT_PATH.write_text(json.dumps(_json_safe(status), indent=2), encoding="utf-8")
    return status


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Reconcile forecasts and check drift")
    parser.add_argument("--refresh-prices", action="store_true", help="fetch recent NSE closes first")
    args = parser.parse_args()

    import train

    bundle = None
    try:
        client = train.setup_mlflow()
        _, bundle = train.load_production_bundle(client)
    except Exception as exc:  # noqa: BLE001 - monitoring still reports without PSI
        LOGGER.warning("Production model unavailable, skipping prediction PSI: %s", exc)
    status = run_monitoring(refresh=args.refresh_prices, production_bundle=bundle)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
