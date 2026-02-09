param(
  [string]$Config = "configs/config.yaml",
  [switch]$PlanOnly,
  [int]$MaxTasks = 0,
  [string]$Tiers = "full_symbol,core_market,core_symbol"
)

$ErrorActionPreference = "Stop"

$cmd = @(
  "python", "-m", "src.models.train_stability_batch",
  "--config", $Config,
  "--tiers", $Tiers,
  "--max-tasks", "$MaxTasks"
)

if ($PlanOnly) {
  $cmd += "--plan-only"
}

Write-Host ("Running: " + ($cmd -join " ")) -ForegroundColor Cyan
& $cmd[0] $cmd[1..($cmd.Length - 1)]
if ($LASTEXITCODE -ne 0) {
  throw "Stability training command failed with exit code $LASTEXITCODE"
}

Write-Host "Stability training command completed." -ForegroundColor Green

