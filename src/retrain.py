"""Anti-noise retraining: multi-signal trigger, challenger-vs-champion gate, governance log.

Run:  python src/retrain.py [--dry-run] [--force] [--refresh-data]

Guardrails (all must allow a retrain):
  (a) performance is judged on a rolling 10-trading-day window pooled over all stocks,
  (b) it must stay below 50% directional accuracy for 3 consecutive days,
  (c) no retrain within 7 trading days of the previous one,
  (d) trigger = perf_degraded OR prediction_psi > 0.20 OR drifted_feature_share > 0.30.
Even when triggered, the challenger must beat @production on a 20-day holdout to be promoted.
"""

import argparse
import json
import logging
from datetime import date

import mlflow
import pandas as pd

import evaluate as ev
import monitoring as mon
import train
from data_prep import (
    EMBARGO_DAYS,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET_COLUMNS,
    fit_preprocessor,
)

LOGGER = logging.getLogger("retrain")

STATE_PATH = PROCESSED_DIR / "retrain_state.json"
COOLDOWN_TRADING_DAYS = 7
HOLDOUT_DAYS = 20
VAL_FRACTION = 0.15
MIN_TRAIN_DATES = 250


# ----------------------------------------------------------------------------- guardrails
def trading_days_since(last: date | None, today: date, calendar: pd.DatetimeIndex) -> int | None:
    if last is None:
        return None
    cal = pd.DatetimeIndex(calendar)
    return int(((cal > pd.Timestamp(last)) & (cal <= pd.Timestamp(today))).sum())


def evaluate_guardrails(
    status: dict, last_retrain: date | None, today: date, calendar: pd.DatetimeIndex, force: bool = False
) -> dict:
    """Decide whether to retrain. ``perf_degraded`` already encodes guardrails (a) and (b)."""
    psi, share = status.get("prediction_psi"), status.get("drifted_feature_share")
    signals = {
        "perf_degraded": bool(status.get("perf_degraded")),
        "prediction_psi": psi is not None and psi > mon.PSI_THRESHOLD,
        "drifted_feature_share": share is not None and share > mon.DRIFT_SHARE_THRESHOLD,
    }
    reasons = [name for name, hit in signals.items() if hit]
    elapsed = trading_days_since(last_retrain, today, calendar)
    cooldown_active = elapsed is not None and elapsed < COOLDOWN_TRADING_DAYS
    if force:
        action, reasons = "retrain", ["manual_force", *reasons]
    elif not reasons:
        action = "skip_no_trigger"
    elif cooldown_active:
        action = "skip_cooldown"
    else:
        action = "retrain"
    return {
        "action": action, "reasons": reasons, "signals": signals, "forced": force,
        "trading_days_since_retrain": elapsed, "cooldown_active": bool(cooldown_active),
        "cooldown_trading_days": COOLDOWN_TRADING_DAYS,
    }


# ----------------------------------------------------------------------------- splitting
def make_holdout_splits(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """train | embargo | val | embargo | 20-day holdout, all chronological.

    The embargo equals the longest horizon, so label windows never cross a boundary. The
    holdout is the most recent HOLDOUT_DAYS dates whose targets are fully realised.
    """
    labelled = features.dropna(subset=TARGET_COLUMNS)
    dates = pd.DatetimeIndex(sorted(labelled["date"].unique()))
    fit_dates = dates[: -(HOLDOUT_DAYS + EMBARGO_DAYS)]
    if len(fit_dates) < MIN_TRAIN_DATES:
        raise ValueError(f"Need at least {MIN_TRAIN_DATES} fit dates, have {len(fit_dates)}")
    val_start = int(len(fit_dates) * (1 - VAL_FRACTION))
    parts = {
        "train": fit_dates[: val_start - EMBARGO_DAYS],
        "val": fit_dates[val_start:],
        "holdout": dates[-HOLDOUT_DAYS:],
    }
    return {name: labelled[labelled["date"].isin(d)].reset_index(drop=True) for name, d in parts.items()}


# --------------------------------------------------------------------------------- state
def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"last_retrain_date": None, "history": []}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------------- driver
def _summary(m: dict[str, float]) -> str:
    return f"pinball_mean={m['pinball_mean']:.6f}, DA={m['directional_accuracy']:.4f}"


def run_retrain(force: bool = False, dry_run: bool = False, refresh_data: bool = False) -> dict:
    client = train.setup_mlflow()
    champion_version, champion_bundle = train.load_production_bundle(client)
    champion_label = getattr(champion_version, "version", None)

    if refresh_data:
        import data_prep

        data_prep.run_pipeline()
    status = mon.run_monitoring(refresh=False, production_bundle=champion_bundle)

    prices = pd.read_parquet(mon.PRICES_PATH, columns=["date"])
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    today = calendar.max().date()
    state = load_state()
    last = date.fromisoformat(state["last_retrain_date"]) if state["last_retrain_date"] else None
    decision = evaluate_guardrails(status, last, today, calendar, force=force)
    result = {"decision": decision, "status": status["status"], "promoted": False, "dry_run": dry_run}

    detail_base = f"signals={decision['signals']}; days_since_retrain={decision['trading_days_since_retrain']}"
    if decision["action"] != "retrain":
        result["version"] = None
        if not dry_run:
            train.append_governance_log(decision["action"], champion_label, False, False, detail_base)
        return result
    if dry_run:
        return result

    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    splits = make_holdout_splits(features)
    preprocessor = fit_preprocessor(splits["train"])
    with mlflow.start_run(run_name="retrain_challenger"):
        mlflow.set_tags({
            "trigger": ",".join(decision["reasons"]), "champion_version": str(champion_label),
            "holdout_start": str(splits["holdout"]["date"].min().date()),
            "holdout_end": str(splits["holdout"]["date"].max().date()),
        })
        challenger = train.train_v2(splits["train"], splits["val"], preprocessor)
        ch_metrics = ev.aggregate(train.evaluate_v2(challenger, splits["holdout"]))
        champ_metrics = (
            ev.aggregate(train.evaluate_v2(champion_bundle, splits["holdout"])) if champion_bundle else None
        )
        gate = ev.challenger_vs_champion_gate(ch_metrics, champ_metrics)
        mlflow.log_metrics({f"holdout_{k}": v for k, v in ch_metrics.items() if pd.notna(v)})
        if champ_metrics:
            mlflow.log_metrics({f"champion_holdout_{k}": v for k, v in champ_metrics.items() if pd.notna(v)})
        mlflow.log_dict(gate, "challenger_gate.json")

        version = None
        if gate["promote"]:
            metadata = {
                "retrain_trigger": ",".join(decision["reasons"]), "previous_production": str(champion_label),
                "holdout_pinball_mean": f"{ch_metrics['pinball_mean']:.6f}",
                "holdout_directional_accuracy": f"{ch_metrics['directional_accuracy']:.4f}",
            }
            version = train.log_and_register_v2(challenger, metadata, train.model_params())
            client.set_registered_model_alias(train.MODEL_NAME, train.PRODUCTION_ALIAS, version)
            result["promoted"] = True
            # Future drift checks must compare against what the new champion was trained on.
            splits["train"][["date", "symbol", *NUMERIC_FEATURES]].sample(
                n=min(5000, len(splits["train"])), random_state=RANDOM_STATE
            ).to_parquet(PROCESSED_DIR / "reference_sample.parquet", compression="zstd", index=False)
    result["version"] = version

    detail = (
        f"trigger={decision['reasons']}; challenger {_summary(ch_metrics)}"
        + (f" vs champion v{champion_label} {_summary(champ_metrics)}" if champ_metrics else "")
        + f"; gate: {gate['reason']}"
    )
    train.append_governance_log(
        "retrain_promoted" if result["promoted"] else "retrain_rejected",
        version or champion_label, result["promoted"], False, detail,
    )
    state["last_retrain_date"] = today.isoformat()  # cooldown applies after any retrain attempt
    state["history"].append({
        "date": today.isoformat(), "reasons": decision["reasons"], "promoted": result["promoted"],
        "challenger_version": version, "champion_version": champion_label,
    })
    save_state(state)
    result["gate"] = gate
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate retraining guardrails and retrain if triggered")
    parser.add_argument("--dry-run", action="store_true", help="evaluate guardrails only; no retrain, no log")
    parser.add_argument("--force", action="store_true", help="manual override of trigger and cooldown")
    parser.add_argument("--refresh-data", action="store_true", help="rebuild features (data_prep) first")
    args = parser.parse_args()
    result = run_retrain(force=args.force, dry_run=args.dry_run, refresh_data=args.refresh_data)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
