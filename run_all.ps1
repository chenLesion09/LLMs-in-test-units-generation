# One-click: run experiment -> analyze -> charts. Default uses offline Mock (no API cost).
# Examples (run in this folder):
#   .\run_all.ps1
#   .\run_all.ps1 -Limit 30
#   .\run_all.ps1 -Backend bailian -Source all -Limit 0 -Rounds 3   # FULL dataset (humaneval+mbpp) on real API
#   .\run_all.ps1 -Conditions C0,C2 -Models glm

param(
    [ValidateSet("mock", "bailian")] [string]$Backend = "mock",
    [string]$Models = "glm,qwen,deepseek",
    [string]$Conditions = "C0,C1,C2,C3",
    [string]$Source = "samples",
    [int]$Limit = 6,
    [int]$Rounds = 3,
    [int]$Parallel = 1,
    [double]$Temperature = 0.0,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Host "venv python not found: $Py" -ForegroundColor Red; exit 1 }

Write-Host "==== [1/3] Run experiment (backend=$Backend models=$Models conditions=$Conditions source=$Source limit=$Limit R=$Rounds parallel=$Parallel) ====" -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$pyArgs = @("main.py", "--backend", $Backend, "--models", $Models, "--conditions", $Conditions,
            "--source", $Source, "--limit", $Limit, "--rounds", $Rounds,
            "--parallel", $Parallel, "--temperature", $Temperature)
if ($Resume) { $pyArgs += "--resume"; Write-Host "resume mode ON" -ForegroundColor Magenta }
& $Py @pyArgs
if (-not $?) { exit 1 }
$sw.Stop()
Write-Host ("experiment time: {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray

Write-Host ""
Write-Host "==== [2/3] Analyze + significance tests ====" -ForegroundColor Cyan
& $Py -m src.analyze

Write-Host ""
Write-Host "==== [3/3] Done ====" -ForegroundColor Green
Write-Host ("charts:  {0}\figures" -f $PSScriptRoot) -ForegroundColor Yellow
Write-Host ("logs:    {0}\results  (latest run_*.jsonl = this run)" -f $PSScriptRoot) -ForegroundColor Yellow
