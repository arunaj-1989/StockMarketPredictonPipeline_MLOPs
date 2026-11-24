# Windows launcher: uses the project's virtualenv. Args pass through, e.g.  .\scripts\run_app.ps1 --mlflow
$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtualenv not found. Create it with: py -3.11 -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}
& $venvPython (Join-Path $PSScriptRoot "run_app.py") @args
exit $LASTEXITCODE
