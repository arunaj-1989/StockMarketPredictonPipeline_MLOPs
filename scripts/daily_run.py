"""Daily cycle: refresh data -> log forecasts -> grade past forecasts -> drift + retraining guardrails.

Usage:
  python scripts/daily_run.py                  # full cycle (news + FinBERT included)
  python scripts/daily_run.py --skip-news      # skip the RSS/FinBERT step
  python scripts/daily_run.py --no-retrain     # monitoring only; never retrain
  python scripts/daily_run.py --skip-refresh   # reuse the data already on disk

Scheduled by scripts/register_daily_task.ps1 (weekdays 4:15 PM IST). Output is also written to logs/daily_run.log.
Exit code is non-zero if any step failed; later steps still run so one failure does not hide the rest.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOG_FILE = ROOT / "logs" / "daily_run.log"
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)

sys.path.insert(0, str(SRC))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

LOGGER = logging.getLogger("daily_run")


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def refresh_data(skip_news: bool) -> None:
    """Separate process: FinBERT frees its GPU memory when it exits, and failures stay isolated."""
    cmd = [PYTHON, str(SRC / "data_prep.py"), "--refresh-only"]
    if skip_news:
        cmd.append("--skip-sentiment")
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONWARNINGS": "ignore", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    LOGGER.info("data_prep tail:\n%s", "\n".join((result.stdout + result.stderr).strip().splitlines()[-6:]))
    if result.returncode != 0:
        raise RuntimeError(f"data_prep exited with code {result.returncode}")


def log_forecasts() -> dict:
    import app
    import pandas as pd

    model = app.load_production_model()
    features = app.load_latest_features()
    logger = app.PredictionLogger(app.PROCESSED_DIR)
    n = app.log_batch_forecasts(model, features, logger)
    return {"payloads": n, "as_of": str(pd.Timestamp(features["date"].max()).date()), "model_version": model.version}


def grading_summary(as_of: str) -> list[str]:
    """Forecasts whose target date is the latest session: what was graded by today's close."""
    import pandas as pd

    import monitoring as mon

    if not mon.RECON_PATH.exists():
        return ["no reconciliation file yet"]
    recon = pd.read_parquet(mon.RECON_PATH)
    today = recon[recon["target_date"] == pd.Timestamp(as_of)]
    if today.empty:
        return [f"no forecasts matured on {as_of}"]
    lines = []
    for horizon, g in today.groupby("horizon"):
        lines.append(
            f"h={horizon:>2}: {len(g)} graded | direction {g['direction_correct'].mean():.1%} | "
            f"q10-q90 coverage {g['in_interval'].mean():.1%} | mean actual {g['actual_return'].mean():+.2%}"
        )
    return lines


def run_monitoring_and_guardrails(retrain: bool) -> dict:
    import retrain as rt
    import train

    if retrain:
        result = rt.run_retrain(force=False, dry_run=False, refresh_data=False)
        return {"decision": result["decision"]["action"], "reasons": result["decision"]["reasons"],
                "promoted": result["promoted"], "status": result["status"]}
    import monitoring as mon

    client = train.setup_mlflow()
    _, bundle = train.load_production_bundle(client)
    status = mon.run_monitoring(refresh=False, production_bundle=bundle)
    return {"decision": "retraining disabled", "reasons": [], "promoted": False, "status": status["status"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the daily forecasting cycle")
    parser.add_argument("--skip-news", action="store_true", help="skip RSS headlines and FinBERT scoring")
    parser.add_argument("--skip-refresh", action="store_true", help="do not download new data")
    parser.add_argument("--no-retrain", action="store_true", help="monitor only; never retrain")
    args = parser.parse_args()
    setup_logging()
    LOGGER.info("=== daily run started ===")
    failures: list[str] = []
    summary: dict = {}
    started = time.time()

    def step(name: str, fn):
        t0 = time.time()
        try:
            out = fn()
            LOGGER.info("[ok]   %s (%.0fs)", name, time.time() - t0)
            return out
        except Exception as exc:  # keep going so one failed step does not hide the others
            LOGGER.exception("[fail] %s: %s", name, exc)
            failures.append(name)
            return None

    if not args.skip_refresh:
        step("refresh data", lambda: refresh_data(args.skip_news))
    logged = step("log forecasts", log_forecasts)
    if logged:
        summary["forecasts"] = logged
        LOGGER.info("logged %d payloads for %s (model v%s)", logged["payloads"], logged["as_of"], logged["model_version"])
    guard = step("monitor + guardrails", lambda: run_monitoring_and_guardrails(not args.no_retrain))
    if guard:
        summary["guardrails"] = guard
        LOGGER.info("drift status: %s | retrain decision: %s %s | promoted: %s",
                    guard["status"], guard["decision"], guard["reasons"], guard["promoted"])
    if logged:
        for line in step("grading summary", lambda: grading_summary(logged["as_of"])) or []:
            LOGGER.info("graded today -> %s", line)

    LOGGER.info("=== daily run finished in %.0fs; failed steps: %s ===", time.time() - started, failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
