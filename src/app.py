"""FastAPI inference service for the multi-horizon Nifty 50 forecaster.

Run:  uvicorn app:app --app-dir src --port 8000
Env:  MLFLOW_TRACKING_URI, FEATURES_PATH, LOG_DIR
"""

import json
import logging
import math
import os
import threading
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar, Literal, NamedTuple

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from data_prep import PROCESSED_DIR, SUPPORTED_HORIZONS

LOGGER = logging.getLogger("app")

DEFAULT_HORIZON = 3
DEFAULT_MIN_RETURN = 0.005  # clears NSE STT + brokerage friction
DEFAULT_SENTIMENT_FLOOR = -0.40  # FinBERT circuit breaker
# Below this cross-stock spread (std of q50) the median forecast is market-wide drift, not a stock view.
MIN_SIGNAL_SPREAD = float(os.environ.get("MIN_SIGNAL_SPREAD", "0.001"))
NO_SIGNAL_WARNING = (
    "The median forecast is almost identical for every stock at this horizon (spread across stocks "
    "{bps:.1f} bps). It reflects historical market drift, not a view on this stock, so every signal is HOLD."
)
DRIFT_OVERRIDE_WARNING = (
    "Drift-based signals are switched on. The median forecast is almost identical for every stock "
    "(spread across stocks {bps:.1f} bps), so BUY here reflects historical market drift, not a view on "
    "this stock. Use for demos only."
)
QUANTILE_COLS = ("q10", "q50", "q90")
Signal = Literal["BUY", "HOLD"]


class SignalQuality(NamedTuple):
    informative: bool  # the median forecast differs between stocks
    dispersion: float  # std of q50 across the universe
    warning: str | None
    drift_override: bool  # uninformative forecast, but the caller asked for signals anyway


INFORMATIVE = SignalQuality(True, 0.0, None, False)


# ------------------------------------------------------------------------------ schemas
class ForecastPoint(BaseModel):
    day: int = Field(description="Trading days ahead")
    q10: float
    q50: float
    q90: float
    price_q10: float
    price_q50: float
    price_q90: float


class PredictionResponse(BaseModel):
    symbol: str
    as_of_date: date
    horizon: int
    close: float
    expected_return: float = Field(description="Median N-day log return (q50)")
    expected_return_pct: float = Field(description="Median N-day return in percent")
    q10: float
    q50: float
    q90: float
    buy_entry: float = Field(description="Close * exp(q10 at t+1)")
    take_profit: float = Field(description="Close * exp(q50 at t+N)")
    stop_loss: float = Field(description="Close * exp(q10 at t+N)")
    stock_sentiment_1d: float
    signal: Signal
    forecast_informative: bool = Field(
        description="False when the median forecast is (nearly) the same for every stock"
    )
    median_dispersion: float = Field(description="Std of q50 across all stocks at this horizon")
    warning: str | None = None
    drift_override: bool = Field(False, description="BUY shown from a drift-only forecast because the caller allowed it")
    min_return: float
    sentiment_floor: float
    forecast: list[ForecastPoint]
    model_version: str
    gate_override: bool
    generated_at: datetime


class WatchlistItem(BaseModel):
    symbol: str
    close: float
    expected_return: float
    expected_return_pct: float
    q10: float
    q90: float
    buy_entry: float
    take_profit: float
    stop_loss: float
    stock_sentiment_1d: float
    signal: Signal


class WatchlistResponse(BaseModel):
    as_of_date: date
    horizon: int
    model_version: str
    gate_override: bool
    forecast_informative: bool
    median_dispersion: float
    warning: str | None = None
    drift_override: bool = False
    min_return: float
    sentiment_floor: float
    count: int
    items: list[WatchlistItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    gate_override: bool = False
    features_as_of: date | None = None
    n_symbols: int = 0
    detail: str | None = None


# -------------------------------------------------------------------------- model + data
@dataclass
class ModelHandle:
    predict: Callable[[pd.DataFrame], pd.DataFrame]
    name: str
    version: str
    gate_override: bool = False


def load_production_model() -> ModelHandle:
    """Load the MLflow registry's ``@production`` multi-horizon bundle."""
    import mlflow
    from mlflow.tracking import MlflowClient

    import train

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", train.TRACKING_URI))
    version = MlflowClient().get_model_version_by_alias(train.MODEL_NAME, train.PRODUCTION_ALIAS)
    model = mlflow.pyfunc.load_model(f"models:/{train.MODEL_NAME}@{train.PRODUCTION_ALIAS}")
    return ModelHandle(
        predict=model.predict,
        name=train.MODEL_NAME,
        version=str(version.version),
        gate_override=version.tags.get("promotion_override") == "true",
    )


def load_latest_features() -> pd.DataFrame:
    path = Path(os.environ.get("FEATURES_PATH", PROCESSED_DIR / "latest_features.parquet"))
    return pd.read_parquet(path)


# ----------------------------------------------------------------------- prediction log
class PredictionLogger:
    """Append-only JSONL audit trail plus a de-duplicated Parquet table.

    JSONL keeps every payload. ``prediction_logs.parquet`` keeps the first row per
    (symbol, as_of_date, horizon, model_version) so reconciliation grades each forecast once.
    """

    DEDUPE_KEY: ClassVar[list[str]] = ["symbol", "as_of_date", "horizon", "model_version"]

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.jsonl_path = self.directory / "prediction_logs.jsonl"
        self.parquet_path = self.directory / "prediction_logs.parquet"
        self._lock = threading.Lock()

    def append(self, records: list[dict]) -> None:
        if not records:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock, self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)

    def compact(self) -> None:
        with self._lock:
            if not self.jsonl_path.exists():
                return
            frame = pd.read_json(self.jsonl_path, lines=True, dtype=False)
            frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
            frame["logged_at"] = pd.to_datetime(frame["logged_at"], utc=True)
            frame = frame.drop_duplicates(self.DEDUPE_KEY, keep="first").reset_index(drop=True)
            tmp = self.parquet_path.with_suffix(".tmp")
            frame.to_parquet(tmp, compression="zstd", index=False)
            tmp.replace(self.parquet_path)


# ---------------------------------------------------------------------------- core logic
def _q(row: pd.Series, horizon: int) -> tuple[float, float, float]:
    return tuple(float(row[f"{c}_t{horizon}"]) for c in QUANTILE_COLS)


def decide_signal(
    median_return: float, sentiment: float, min_return: float, floor: float, signals_allowed: bool = True
) -> Signal:
    """BUY only if signals are allowed, the median clears the friction hurdle and sentiment is not alarming."""
    return "BUY" if signals_allowed and median_return > min_return and sentiment > floor else "HOLD"


def signal_quality(scored: pd.DataFrame, horizon: int, allow_uninformative: bool = False) -> SignalQuality:
    """Whether the median forecast differs between stocks (std of q50 across the universe)."""
    dispersion = float(scored[f"q50_t{horizon}"].std(ddof=0))
    if dispersion >= MIN_SIGNAL_SPREAD:
        return SignalQuality(True, dispersion, None, False)
    template = DRIFT_OVERRIDE_WARNING if allow_uninformative else NO_SIGNAL_WARNING
    return SignalQuality(False, dispersion, template.format(bps=dispersion * 1e4), allow_uninformative)


def build_prediction(
    row: pd.Series, horizon: int, min_return: float, sentiment_floor: float, model: ModelHandle,
    quality: SignalQuality = INFORMATIVE,
) -> PredictionResponse:
    informative, dispersion, warning, drift_override = quality
    close = float(row["close"])
    sentiment = float(row["stock_sentiment_1d"])
    q10_1, _, _ = _q(row, 1)
    q10_n, q50_n, q90_n = _q(row, horizon)
    forecast = []
    for day in (h for h in SUPPORTED_HORIZONS if h <= horizon):
        lo, mid, hi = _q(row, day)
        forecast.append(ForecastPoint(
            day=day, q10=lo, q50=mid, q90=hi,
            price_q10=close * math.exp(lo), price_q50=close * math.exp(mid), price_q90=close * math.exp(hi),
        ))
    return PredictionResponse(
        symbol=str(row["symbol"]),
        as_of_date=pd.Timestamp(row["date"]).date(),
        horizon=horizon,
        close=close,
        expected_return=q50_n,
        expected_return_pct=(math.exp(q50_n) - 1) * 100,
        q10=q10_n, q50=q50_n, q90=q90_n,
        buy_entry=close * math.exp(q10_1),
        take_profit=close * math.exp(q50_n),
        stop_loss=close * math.exp(q10_n),
        stock_sentiment_1d=sentiment,
        signal=decide_signal(q50_n, sentiment, min_return, sentiment_floor, informative or drift_override),
        forecast_informative=informative,
        median_dispersion=dispersion,
        warning=warning,
        drift_override=drift_override,
        min_return=min_return,
        sentiment_floor=sentiment_floor,
        forecast=forecast,
        model_version=model.version,
        gate_override=model.gate_override,
        generated_at=datetime.now(UTC),
    )


def to_log_record(pred: PredictionResponse, row: pd.Series) -> dict:
    """Flat payload incl. every horizon's quantiles so t+1..t+N can be reconciled later."""
    record = {
        "log_id": uuid.uuid4().hex,
        "logged_at": pred.generated_at.isoformat(),
        "as_of_date": pred.as_of_date.isoformat(),
        "symbol": pred.symbol,
        "horizon": pred.horizon,
        "close": pred.close,
        "expected_return": pred.expected_return,
        "buy_entry": pred.buy_entry,
        "take_profit": pred.take_profit,
        "stop_loss": pred.stop_loss,
        "stock_sentiment_1d": pred.stock_sentiment_1d,
        "signal": pred.signal,
        "forecast_informative": pred.forecast_informative,
        "drift_override": pred.drift_override,
        "median_dispersion": pred.median_dispersion,
        "min_return": pred.min_return,
        "sentiment_floor": pred.sentiment_floor,
        "model_version": pred.model_version,
    }
    for h in SUPPORTED_HORIZONS:
        for c, value in zip(QUANTILE_COLS, _q(row, h)):
            record[f"{c}_t{h}"] = value
    return record


def log_batch_forecasts(
    model: ModelHandle, features: pd.DataFrame, logger: PredictionLogger,
    min_return: float = DEFAULT_MIN_RETURN, sentiment_floor: float = DEFAULT_SENTIMENT_FLOOR,
) -> int:
    """Score every stock for every supported horizon and log the payloads (the daily batch run).

    Uses the same ``build_prediction``/``to_log_record`` path as the API, so batch and on-demand
    forecasts are identical records. Returns the number of payloads logged.
    """
    feats = features.reset_index(drop=True)
    scored = pd.concat([feats, model.predict(feats).reset_index(drop=True)], axis=1)
    records = []
    for horizon in SUPPORTED_HORIZONS:
        quality = signal_quality(scored, horizon)
        for _, row in scored.iterrows():
            pred = build_prediction(row, horizon, min_return, sentiment_floor, model, quality)
            records.append(to_log_record(pred, row))
    logger.append(records)
    logger.compact()
    return len(records)


class Engine:
    def __init__(self, model_loader, features_loader, logger: PredictionLogger):
        self.model_loader = model_loader
        self.features_loader = features_loader
        self.logger = logger
        self.model: ModelHandle | None = None
        self.load_error: str | None = None

    def load(self) -> None:
        try:
            self.model = self.model_loader()
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 - the service must start and report "degraded"
            self.model, self.load_error = None, f"{type(exc).__name__}: {exc}"
            LOGGER.error("Could not load @production model: %s", self.load_error)

    def scored(self) -> tuple[pd.DataFrame, ModelHandle]:
        if self.model is None:
            self.load()  # picks up a model trained/promoted after the service started
        if self.model is None:
            raise HTTPException(503, f"Production model unavailable: {self.load_error}")
        feats = self.features_loader().reset_index(drop=True)
        preds = self.model.predict(feats).reset_index(drop=True)
        return pd.concat([feats, preds], axis=1), self.model


def valid_horizon(horizon: int = Query(DEFAULT_HORIZON, description="Forecast horizon in trading days")) -> int:
    if horizon not in SUPPORTED_HORIZONS:
        raise HTTPException(422, f"horizon must be one of {SUPPORTED_HORIZONS}, got {horizon}")
    return horizon


def create_app(
    model_loader: Callable[[], ModelHandle] = load_production_model,
    features_loader: Callable[[], pd.DataFrame] = load_latest_features,
    log_dir: Path | None = None,
) -> FastAPI:
    engine = Engine(
        model_loader, features_loader, PredictionLogger(log_dir or os.environ.get("LOG_DIR") or PROCESSED_DIR)
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.load()
        yield

    app = FastAPI(title="Nifty 50 multi-horizon forecaster", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health():
        if engine.model is None:
            engine.load()
        if engine.model is None:
            body = HealthResponse(status="degraded", model_loaded=False, detail=engine.load_error)
            return JSONResponse(status_code=503, content=json.loads(body.model_dump_json()))
        try:
            feats = engine.features_loader()
            as_of, n = pd.Timestamp(feats["date"].max()).date(), int(feats["symbol"].nunique())
        except (OSError, KeyError, ValueError) as exc:
            return JSONResponse(
                status_code=503,
                content=json.loads(HealthResponse(
                    status="degraded", model_loaded=True, model_name=engine.model.name,
                    model_version=engine.model.version, detail=f"features unavailable: {exc}",
                ).model_dump_json()),
            )
        return HealthResponse(
            status="ok", model_loaded=True, model_name=engine.model.name,
            model_version=engine.model.version, gate_override=engine.model.gate_override,
            features_as_of=as_of, n_symbols=n,
        )

    @app.get("/predict/{symbol}", response_model=PredictionResponse, tags=["forecast"])
    def predict(
        symbol: str,
        background: BackgroundTasks,
        horizon: int = Depends(valid_horizon),
        min_return: float = Query(DEFAULT_MIN_RETURN, ge=-0.5, le=0.5),
        sentiment_floor: float = Query(DEFAULT_SENTIMENT_FLOOR, ge=-1.0, le=1.0),
        allow_uninformative: bool = Query(
            False, description="Show BUY signals even when the median forecast is drift-only (demo use)"
        ),
    ):
        scored, model = engine.scored()
        key = symbol.strip().upper().removesuffix(".NS")
        match = scored[scored["symbol"] == key]
        if match.empty:
            raise HTTPException(404, f"Unknown Nifty 50 symbol: {symbol}")
        row = match.iloc[0]
        quality = signal_quality(scored, horizon, allow_uninformative)
        pred = build_prediction(row, horizon, min_return, sentiment_floor, model, quality)
        engine.logger.append([to_log_record(pred, row)])
        background.add_task(engine.logger.compact)
        return pred

    @app.get("/watchlist", response_model=WatchlistResponse, tags=["forecast"])
    def watchlist(
        background: BackgroundTasks,
        horizon: int = Depends(valid_horizon),
        min_return: float = Query(DEFAULT_MIN_RETURN, ge=-0.5, le=0.5),
        sentiment_floor: float = Query(DEFAULT_SENTIMENT_FLOOR, ge=-1.0, le=1.0),
        only_buy: bool = Query(False, description="Return only BUY signals"),
        allow_uninformative: bool = Query(
            False, description="Show BUY signals even when the median forecast is drift-only (demo use)"
        ),
    ):
        scored, model = engine.scored()
        quality = signal_quality(scored, horizon, allow_uninformative)
        preds, records = [], []
        for _, row in scored.iterrows():
            pred = build_prediction(row, horizon, min_return, sentiment_floor, model, quality)
            preds.append(pred)
            records.append(to_log_record(pred, row))
        engine.logger.append(records)
        background.add_task(engine.logger.compact)

        preds.sort(key=lambda p: (-p.expected_return, p.q90 - p.q10, p.symbol))
        if only_buy:
            preds = [p for p in preds if p.signal == "BUY"]
        items = [
            WatchlistItem(**p.model_dump(include=set(WatchlistItem.model_fields))) for p in preds
        ]
        as_of = pd.Timestamp(scored["date"].max()).date()
        return WatchlistResponse(
            as_of_date=as_of, horizon=horizon, model_version=model.version,
            gate_override=model.gate_override, forecast_informative=quality.informative,
            median_dispersion=quality.dispersion, warning=quality.warning,
            drift_override=quality.drift_override, min_return=min_return,
            sentiment_floor=sentiment_floor, count=len(items), items=items,
        )

    return app


app = create_app()
