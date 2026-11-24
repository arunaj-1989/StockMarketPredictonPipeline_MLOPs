# Registers (or removes) the weekday 4:15 PM daily run in Windows Task Scheduler.
#   .\scripts\register_daily_task.ps1              install / update
#   .\scripts\register_daily_task.ps1 -Remove      uninstall
#   .\scripts\register_daily_task.ps1 -Time 17:00  different time (local machine time; IST on your PC)
param(
    [string]$Time = "16:15",
    [string]$TaskName = "Nifty50-DailyRun",
    [switch]$Remove
)

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task '$TaskName'."
    exit 0
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { Write-Error "Virtualenv not found at $python"; exit 1 }

$action = New-ScheduledTaskAction -Execute $python -Argument "`"$(Join-Path $root 'scripts\daily_run.py')`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
# StartWhenAvailable: if the PC was off at 4:15 PM the run happens as soon as it is back on.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Refresh Nifty 50 data, log forecasts, grade past forecasts, check drift and retraining guardrails" -Force | Out-Null

$info = Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
Write-Host "Registered '$TaskName': weekdays at $Time. Next run: $($info.NextRunTime)"
Write-Host "Logs: $(Join-Path $root 'logs\daily_run.log')"
