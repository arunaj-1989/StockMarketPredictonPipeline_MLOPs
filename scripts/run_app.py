"""Launch the Nifty 50 forecaster: FastAPI + Streamlit (+ optional MLflow UI).

Usage:
  python scripts/run_app.py                 # API :8000 + dashboard :8501, opens the browser
  python scripts/run_app.py --mlflow        # also serve the MLflow UI on :5000
  python scripts/run_app.py --setup         # build data and train first if they are missing
  python scripts/run_app.py --no-browser

Ctrl+C stops every service. Service logs go to logs/.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOG_DIR = ROOT / "logs"
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)

REQUIRED_DATA = [
    ROOT / "data" / "processed" / "latest_features.parquet",
    ROOT / "data" / "processed" / "features.parquet",
    ROOT / "data" / "raw" / "prices.parquet",
]


def run_step(args: list[str], env: dict) -> None:
    print(f"  $ {' '.join(args)}")
    subprocess.run([PYTHON, *args], cwd=ROOT, env=env, check=True)


def production_model_ready(env: dict) -> bool:
    code = (
        "import sys; sys.path.insert(0, 'src'); import train;"
        "c = train.setup_mlflow();"
        "c.get_model_version_by_alias(train.MODEL_NAME, train.PRODUCTION_ALIAS)"
    )
    result = subprocess.run([PYTHON, "-c", code], cwd=ROOT, env=env, capture_output=True)
    return result.returncode == 0


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for(url: str, timeout: float, proc: subprocess.Popen, label: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"  [x] {label} exited early (code {proc.returncode}); see logs/")
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    print(f"  [x] {label} did not become healthy within {timeout:.0f}s; see logs/")
    return False


def start(name: str, args: list[str], env: dict) -> subprocess.Popen:
    LOG_DIR.mkdir(exist_ok=True)
    log = open(LOG_DIR / f"{name}.log", "w", encoding="utf-8")  # stays open for the child's lifetime
    return subprocess.Popen([PYTHON, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def stop_all(procs: dict[str, subprocess.Popen]) -> None:
    for proc in procs.values():
        if proc.poll() is None:
            if os.name == "nt":
                # The venv python.exe is a shim that spawns the real interpreter: kill the whole tree.
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.terminate()
    for name, proc in procs.items():
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"  stopped {name}")


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when piped or redirected
    parser = argparse.ArgumentParser(description="Run the Nifty 50 forecaster locally")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--mlflow-port", type=int, default=5000)
    parser.add_argument("--mlflow", action="store_true", help="also serve the MLflow UI")
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    parser.add_argument("--setup", action="store_true",
                        help="run data_prep and train (with --force-promote) if data/model are missing")
    args = parser.parse_args()

    env = {**os.environ, "PYTHONPATH": str(SRC), "MLFLOW_DISABLE_AGENT_HINT": "1",
           "API_URL": f"http://127.0.0.1:{args.api_port}"}

    print("Checking prerequisites")
    if any(not p.exists() for p in REQUIRED_DATA):
        if not args.setup:
            print("  [x] Processed data is missing. Run:  python src/data_prep.py   (or use --setup)")
            return 1
        print("  Building data (downloads prices and scores news; takes a few minutes)")
        run_step(["src/data_prep.py"], env)
    if not production_model_ready(env):
        if not args.setup:
            print("  [x] No @production model in the MLflow registry. Run:  python src/train.py --force-promote")
            print("    (the strict gate currently fails, so --force-promote is needed; see docs/GOVERNANCE_LOG.md)")
            return 1
        print("  Training and promoting with --force-promote (gate override is logged)")
        run_step(["src/train.py", "--force-promote"], env)

    wanted = {"api": args.api_port, "dashboard": args.ui_port}
    if args.mlflow:
        wanted["mlflow"] = args.mlflow_port
    busy = [f"{n} (:{p})" for n, p in wanted.items() if port_in_use(p)]
    if busy:
        print(f"  [x] Port already in use for: {', '.join(busy)}. Stop the other process or pick another port.")
        return 1
    print("  [ok] data and production model found")

    procs: dict[str, subprocess.Popen] = {}
    try:
        print("Starting services")
        procs["api"] = start("api", ["-m", "uvicorn", "app:app", "--app-dir", "src",
                                      "--host", "127.0.0.1", "--port", str(args.api_port)], env)
        if not wait_for(f"http://127.0.0.1:{args.api_port}/health", 120, procs["api"], "API"):
            return 1
        print(f"  [ok] API        http://127.0.0.1:{args.api_port}/docs")

        procs["dashboard"] = start("dashboard", ["-m", "streamlit", "run", "src/dashboard.py",
                                                  "--server.port", str(args.ui_port),
                                                  "--server.address", "127.0.0.1",
                                                  "--server.headless", "true"], env)
        if not wait_for(f"http://127.0.0.1:{args.ui_port}/_stcore/health", 60, procs["dashboard"], "dashboard"):
            return 1
        dashboard_url = f"http://localhost:{args.ui_port}"
        print(f"  [ok] Dashboard  {dashboard_url}")

        if args.mlflow:
            code = "import sys; sys.path.insert(0, 'src'); import train; print(train.TRACKING_URI)"
            uri = subprocess.run([PYTHON, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                 text=True, check=True).stdout.strip()
            procs["mlflow"] = start("mlflow", ["-m", "mlflow", "ui", "--backend-store-uri", uri,
                                                "--port", str(args.mlflow_port)], env)
            if wait_for(f"http://127.0.0.1:{args.mlflow_port}/health", 60, procs["mlflow"], "MLflow UI"):
                print(f"  [ok] MLflow UI  http://localhost:{args.mlflow_port}")

        if not args.no_browser:
            webbrowser.open(dashboard_url)
        print(f"\nRunning. Press Ctrl+C to stop. Logs: {LOG_DIR}")

        while True:
            for name, proc in procs.items():
                if proc.poll() is not None:
                    print(f"\n{name} stopped unexpectedly (code {proc.returncode}); see logs/{name}.log")
                    return 1
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nShutting down")
        return 0
    finally:
        stop_all(procs)


if __name__ == "__main__":
    sys.exit(main())
