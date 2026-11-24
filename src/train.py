"""Train V1 (Ridge) and V2 (multi-horizon quantile LightGBM), track in MLflow, promote.

Run from anywhere:  python src/train.py [--no-promote]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import evaluate as ev
from data_prep import (
    CATEGORICAL_FEATURES,
    EMBARGO_DAYS,
    FEATURE_COLUMNS,
    LONG_HORIZONS,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RANDOM_STATE,
    RANK_LEVELS,
    RANK_MODEL_FEATURES,
    RAW_DIR,
    ROOT,
    SUPPORTED_HORIZONS,
    build_ranking_frame,
)

LOGGER = logging.getLogger("train")

N_JOBS = 8
MODEL_NAME = "nifty50_multi_horizon"
EXPERIMENT = "nifty50-multi-horizon"
PRODUCTION_ALIAS = "production"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT") or (ROOT / "mlruns").as_uri()

QUANTILES = ev.QUANTILES
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
LGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "n_jobs": N_JOBS,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 50
WINSOR_PCT = (0.5, 99.5)


# ------------------------------------------------------------------------------ bundle
def predict_bundle(bundle: dict, df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Quantile forecasts per horizon as (n, 3) arrays [q10, q50, q90] with q10<=q50<=q90."""
    X = bundle["preprocessor"].transform(df[bundle["feature_columns"]])
    out = {}
    for h in bundle["horizons"]:
        raw = np.column_stack([bundle["models"][h][q].predict(X) for q in bundle["quantiles"]])
        out[h] = ev.enforce_quantile_order(raw)
    return out


class ForecastModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper: DataFrame of features -> columns q{10,50,90}_t{h}."""

    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        preds = predict_bundle(self.bundle, model_input)
        cols = {}
        for h, arr in preds.items():
            for i, q in enumerate(self.bundle["quantiles"]):
                cols[f"q{int(q * 100)}_t{h}"] = arr[:, i]
        return pd.DataFrame(cols, index=model_input.index)


# --------------------------------------------------------------------------- V1 ridge
def winsor_bounds(train: pd.DataFrame) -> pd.DataFrame:
    lo, hi = np.percentile(train[NUMERIC_FEATURES], WINSOR_PCT, axis=0)
    return pd.DataFrame({"lo": lo, "hi": hi}, index=NUMERIC_FEATURES)


def winsorize(df: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLUMNS].copy()
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].clip(bounds["lo"], bounds["hi"], axis=1)
    return out


def _ridge_pipeline(alpha: float) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE))])


def train_v1(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict[int, Pipeline], pd.DataFrame, dict]:
    """One Ridge Pipeline per horizon; alpha chosen on validation q50 pinball loss."""
    bounds = winsor_bounds(train)
    X_tr, X_va = winsorize(train, bounds), winsorize(val, bounds)
    models, chosen = {}, {}
    for h in SUPPORTED_HORIZONS:
        best = None
        for alpha in RIDGE_ALPHAS:
            pipe = _ridge_pipeline(alpha).fit(X_tr, train[f"target_{h}"])
            loss = ev.pinball_loss(val[f"target_{h}"], pipe.predict(X_va), 0.5)
            if best is None or loss < best[0]:
                best = (loss, alpha, pipe)
        models[h], chosen[h] = best[2], best[1]
    return models, bounds, chosen


# ------------------------------------------------------------------------------ V2 lgbm
def train_v2(train: pd.DataFrame, val: pd.DataFrame, preprocessor) -> dict:
    """Quantile LightGBM for every (horizon, quantile); early stopping on validation."""
    X_tr = preprocessor.transform(train[FEATURE_COLUMNS])
    X_va = preprocessor.transform(val[FEATURE_COLUMNS])
    models: dict[int, dict[float, lgb.LGBMRegressor]] = {}
    for h in SUPPORTED_HORIZONS:
        models[h] = {}
        for q in QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGBM_PARAMS)
            model.fit(
                X_tr, train[f"target_{h}"],
                eval_set=[(X_va, val[f"target_{h}"])], eval_metric="quantile",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            models[h][q] = model
            LOGGER.info("h=%-2d q=%.2f best_iteration=%d", h, q, model.best_iteration_)
    return {
        "preprocessor": preprocessor,
        "models": models,
        "horizons": list(SUPPORTED_HORIZONS),
        "quantiles": list(QUANTILES),
        "feature_columns": list(FEATURE_COLUMNS),
    }


# --------------------------------------------------------------------------- evaluation
def evaluate_v2(bundle: dict, df: pd.DataFrame) -> dict[int, dict[str, float]]:
    preds = predict_bundle(bundle, df)
    sign = np.sign(df["ret_1d"].to_numpy())
    return {h: ev.horizon_metrics(df[f"target_{h}"].to_numpy(), preds[h], sign) for h in preds}


def evaluate_v1(models: dict, bounds: pd.DataFrame, df: pd.DataFrame) -> dict[int, dict[str, float]]:
    X = winsorize(df, bounds)
    sign = np.sign(df["ret_1d"].to_numpy())
    return {h: ev.point_metrics(df[f"target_{h}"].to_numpy(), models[h].predict(X), sign) for h in models}


def lag_check(bundle: dict, df: pd.DataFrame) -> dict[int, dict]:
    preds = predict_bundle(bundle, df)
    return {
        h: ev.persistence_lag_check(df[f"target_{h}"].to_numpy(), preds[h][:, 1], df["ret_1d"].to_numpy())
        for h in preds
    }


# ------------------------------------------------------------------------ ranking models
RANK_PARAMS = {
    "n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 300,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 20.0,
    "n_jobs": N_JOBS, "random_state": RANDOM_STATE, "verbose": -1,
}
RANKER_KINDS = ("lambdarank", "rank_regression")


def fit_ranker(train: pd.DataFrame, horizon: int, kind: str = "lambdarank"):
    """Fit a market-relative ranker for one horizon on the ranking frame (see data_prep.build_ranking_frame)."""
    d = train.dropna(subset=[f"label_{horizon}"]).sort_values(["date", "symbol"])
    X = d[RANK_MODEL_FEATURES]
    if kind == "lambdarank":
        model = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(RANK_LEVELS)), **RANK_PARAMS)
        groups = d.groupby("date", sort=True).size().to_numpy()
        return model.fit(X, d[f"label_{horizon}"].astype(int), group=groups, categorical_feature=["sector_id"])
    if kind == "rank_regression":
        y = d.groupby("date")[f"rel_{horizon}"].rank(pct=True) - 0.5
        model = lgb.LGBMRegressor(objective="regression", **RANK_PARAMS)
        return model.fit(X, y, categorical_feature=["sector_id"])
    raise ValueError(f"unknown ranker kind {kind!r}")


def walk_forward_rank(frame: pd.DataFrame, horizon: int, kind: str, n_folds: int = 5):
    """Out-of-sample scores from expanding-window folds (10-day embargo); returns (frame, fold mean ICs)."""
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    parts, fold_ics = [], []
    for fold in ev.walk_forward_windows(len(dates), n_folds, embargo=max(EMBARGO_DAYS, horizon)):
        train = frame[frame["date"].isin(dates[: fold["train_end"]])]
        test = frame[frame["date"].isin(dates[fold["test_start"]: fold["test_end"]])]
        model = fit_ranker(train, horizon, kind)
        test = test.assign(score=model.predict(test[RANK_MODEL_FEATURES]))
        fold_ics.append(float(ev.daily_rank_ic(test, "score", f"rel_{horizon}").mean()))
        parts.append(test)
    return pd.concat(parts, ignore_index=True), fold_ics


def rank_report(frame: pd.DataFrame, horizon: int, kind: str, n_folds: int = 5) -> dict:
    """Walk-forward IC, quintile alpha and gate verdict for one horizon and ranker kind."""
    oos, fold_ics = walk_forward_rank(frame, horizon, kind, n_folds)
    target = f"rel_{horizon}"
    summary = ev.ic_summary(ev.daily_rank_ic(oos, "score", target), horizon)
    alpha = ev.quintile_alpha(oos, "score", target)
    gate = ev.ranking_gate(summary, fold_ics)
    return {"horizon": horizon, "kind": kind, "fold_ics": fold_ics, **summary, **alpha, "gate": gate}


def _log_split_metrics(prefix: str, per_horizon: dict[int, dict[str, float]]) -> None:
    flat = {f"{prefix}_h{h}_{name}": value for h, m in per_horizon.items() for name, value in m.items()}
    flat.update({f"{prefix}_mean_{name}": v for name, v in ev.aggregate(per_horizon).items()})
    mlflow.log_metrics({k: v for k, v in flat.items() if np.isfinite(v)})


# ------------------------------------------------------------------------------ registry
def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def load_production_bundle(client: MlflowClient):
    """Return (version, ForecastModel python model) for @production, or (None, None)."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, PRODUCTION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None, None
    loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}")
    return version, loaded.unwrap_python_model().bundle


def model_params() -> dict:
    params = {f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()}
    params.update(horizons=SUPPORTED_HORIZONS, quantiles=list(QUANTILES),
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    return params


def log_and_register_v2(bundle: dict, metadata: dict, params: dict) -> str:
    """Log the bundle as a pyfunc model and register it; returns the new model version."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = Path(tmp) / "bundle.joblib"
        joblib.dump(bundle, bundle_path, compress=3)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ForecastModel(),
            artifacts={"bundle": str(bundle_path)},
            code_paths=[str(Path(__file__).parent / f) for f in ("train.py", "evaluate.py", "data_prep.py")],
            registered_model_name=MODEL_NAME,
            pip_requirements=str(ROOT / "requirements-serve.txt"),
        )
    client = MlflowClient()
    version = info.registered_model_version
    for key, value in metadata.items():
        client.set_model_version_tag(MODEL_NAME, str(version), key, value)
    mlflow.log_params(params)
    return str(version)


GOVERNANCE_LOG = ROOT / "docs" / "GOVERNANCE_LOG.md"


def append_governance_log(
    event: str, version: str | None, promoted: bool, override: bool, detail: str
) -> None:
    """Append one decision record to docs/GOVERNANCE_LOG.md (created on first use)."""
    GOVERNANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not GOVERNANCE_LOG.exists():
        GOVERNANCE_LOG.write_text(
            "# Governance Log\n\nEvery promotion decision, including blocked and overridden ones.\n\n"
            "| Timestamp (UTC) | Event | Model version | Promoted | Gate override | Detail |\n"
            "|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    detail = detail.replace("|", "/").replace("\n", " ")
    version_cell = f"v{version}" if version else "-"
    with GOVERNANCE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"| {stamp} | {event} | {version_cell} | {promoted} | {override} | {detail} |\n")


def run(promote: bool = True, force_promote: bool = False) -> dict:
    client = setup_mlflow()
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    preprocessor = joblib.load(PROCESSED_DIR / "preprocessor.joblib")
    data_tags = {
        "train_rows": str(len(train)), "val_rows": str(len(val)), "test_rows": str(len(test)),
        "train_end": str(train["date"].max().date()), "test_end": str(test["date"].max().date()),
    }

    # ---- V1 baseline
    with mlflow.start_run(run_name="v1_ridge_baseline"):
        mlflow.set_tags({"model_family": "ridge", **data_tags})
        v1_models, bounds, alphas = train_v1(train, val)
        mlflow.log_params({f"ridge_alpha_h{h}": a for h, a in alphas.items()})
        mlflow.log_param("horizons", SUPPORTED_HORIZONS)
        v1_val, v1_test = evaluate_v1(v1_models, bounds, val), evaluate_v1(v1_models, bounds, test)
        _log_split_metrics("val", v1_val)
        _log_split_metrics("test", v1_test)

    # ---- V2 advanced
    with mlflow.start_run(run_name="v2_lgbm_quantile") as run_v2:
        mlflow.set_tags({"model_family": "lightgbm_quantile", **data_tags})
        bundle = train_v2(train, val, preprocessor)
        v2_val, v2_test = evaluate_v2(bundle, val), evaluate_v2(bundle, test)
        _log_split_metrics("val", v2_val)
        _log_split_metrics("test", v2_test)

        v1_mean, v2_mean = ev.aggregate(v1_test), ev.aggregate(v2_test)
        baseline = ev.baseline_gate(v2_mean, v1_mean)
        lags = lag_check(bundle, test)
        lag_ok = all(r["beats_persistence"] for r in lags.values())
        mlflow.log_dict(
            {"baseline_gate": baseline, "persistence_lag_check": {str(h): r for h, r in lags.items()}},
            "gates.json",
        )

        version_prod, champion_bundle = load_production_bundle(client)
        champion_metrics = ev.aggregate(evaluate_v2(champion_bundle, test)) if champion_bundle else None
        champion_gate = ev.challenger_vs_champion_gate(v2_mean, champion_metrics)

        passed = bool(baseline["passed"] and champion_gate["promote"])
        params = model_params()
        metadata = {
            "test_pinball_mean": f"{v2_mean['pinball_mean']:.6f}",
            "test_directional_accuracy": f"{v2_mean['directional_accuracy']:.4f}",
            "test_coverage_q10_q90": f"{v2_mean['coverage_q10_q90']:.4f}",
            "gates_passed": str(passed), **data_tags,
        }
        version = log_and_register_v2(bundle, metadata, params)
        mlflow.set_tag("registered_version", version)
        mlflow.log_metric("gates_passed", float(passed))

        promoted, override = False, False
        if promote and (passed or force_promote):
            override = not passed
            client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
            promoted = True
            if override:
                client.set_model_version_tag(MODEL_NAME, version, "promotion_override", "true")
                mlflow.set_tag("promotion_override", "true")
        append_governance_log(
            event="initial_training" if version_prod is None else "manual_training",
            version=version, promoted=promoted, override=override,
            detail=f"baseline_gate={json.dumps(baseline)}; champion_gate={champion_gate['reason']}",
        )

    return {
        "run_id": run_v2.info.run_id, "version": version, "promoted": promoted,
        "baseline_gate": baseline, "champion_gate": champion_gate, "lag_check_ok": lag_ok,
        "v1_test": v1_test, "v2_test": v2_test, "v2_val": v2_val,
        "previous_production": getattr(version_prod, "version", None),
        "override": override,
    }


def _summary_table(result: dict) -> pd.DataFrame:
    rows = []
    for h in SUPPORTED_HORIZONS:
        v1, v2 = result["v1_test"][h], result["v2_test"][h]
        rows.append({
            "h": h,
            "persist_q50": v2["persistence_pinball_q50"], "v1_q50": v1["pinball_q50"],
            "v2_q50": v2["pinball_q50"], "v2_mean_pinball": v2["pinball_mean"],
            "v1_DA": v1["directional_accuracy"], "v2_DA": v2["directional_accuracy"],
            "naive_DA": v2["persistence_directional_accuracy"],
            "cov_10_90": v2["coverage_q10_q90"],
        })
    return pd.DataFrame(rows).set_index("h")


def ranking_report_cli(long_history: bool = False) -> None:
    """Walk-forward test of the market-relative rankers, plus a governance entry.

    Default: the supported 1-10 day horizons on the live 6-year data. ``long_history``: also the research
    horizons (21/42/63 days) on the ~15-year file, whose longer span gives far more independent observations.
    """
    if long_history:
        prices = pd.read_parquet(RAW_DIR / "prices_long.parquet")
        indices = pd.read_parquet(RAW_DIR / "indices_long.parquet")
        horizons = [*SUPPORTED_HORIZONS, *LONG_HORIZONS]
        frame = build_ranking_frame(prices, indices["^NSEI"].dropna(), indices["^INDIAVIX"].dropna(), horizons)
        csv_name, scope = "rank_walkforward_long.csv", "15y history"
    else:
        horizons = SUPPORTED_HORIZONS
        frame = pd.read_parquet(PROCESSED_DIR / "rank_features.parquet")
        csv_name, scope = "rank_walkforward.csv", "6y history"
    print(f"{scope}: {len(frame):,} rows, {frame['date'].min().date()} -> {frame['date'].max().date()}, "
          f"{frame['date'].nunique()} dates")
    rows = [rank_report(frame, h, kind) for h in horizons for kind in RANKER_KINDS]
    table = pd.DataFrame(rows)
    table["passed"] = table["gate"].map(lambda g: g["passed"])
    table["positive_fold_share"] = table["gate"].map(lambda g: g["positive_fold_share"])
    show = table[["horizon", "kind", "mean_ic", "t_stat", "positive_fold_share", "long_short_spread", "passed"]]
    print(show.round(4).to_string(index=False))
    print("\nper-fold mean IC:")
    for r in rows:
        print(f"  h={r['horizon']:>2} {r['kind']:<16} {[round(x, 3) for x in r['fold_ics']]}")
    table.drop(columns=["gate", "fold_ics"]).to_csv(PROCESSED_DIR / csv_name, index=False)
    passing = sorted(set(table.loc[table["passed"], "horizon"]))
    thresholds = f"IC>={ev.IC_MIN}, t>={ev.IC_T_MIN}, >={ev.IC_POSITIVE_FOLDS_MIN:.0%} positive folds"
    append_governance_log(
        "ranking_gate", None, False, False,
        f"walk-forward on {scope} ({thresholds}); passing horizons: {passing or 'none'}; "
        f"best mean IC {table['mean_ic'].max():+.4f}. App signals {'use' if passing else 'do not use'} the ranker",
    )
    print(f"\nHorizons passing the ranking gate: {passing or 'none'}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train and register the multi-horizon forecaster")
    parser.add_argument("--no-promote", action="store_true", help="register but do not set @production")
    parser.add_argument(
        "--force-promote", action="store_true",
        help="promote even if the baseline gate fails (logged as an override)",
    )
    parser.add_argument(
        "--ranking-report", action="store_true",
        help="run the walk-forward rank-correlation gate on the market-relative rankers and exit",
    )
    parser.add_argument(
        "--long", action="store_true",
        help="with --ranking-report: use the 15-year file and add 21/42/63-day research horizons",
    )
    args = parser.parse_args()
    if args.ranking_report:
        ranking_report_cli(long_history=args.long)
        return
    result = run(promote=not args.no_promote, force_promote=args.force_promote)
    print("\nTest-set metrics by horizon (lower pinball is better; coverage target 0.80)")
    print(_summary_table(result).round(5).to_string())
    print("\nBaseline gate:", json.dumps(result["baseline_gate"], indent=2))
    print("Champion gate:", result["champion_gate"])
    print("Persistence-lag check passed on all horizons:", result["lag_check_ok"])
    print(
        f"Registered {MODEL_NAME} v{result['version']}; promoted to @{PRODUCTION_ALIAS}: "
        f"{result['promoted']} (gate override: {result['override']})"
    )


if __name__ == "__main__":
    main()
