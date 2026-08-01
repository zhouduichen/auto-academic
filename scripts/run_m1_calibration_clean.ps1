param(
    [Parameter(Mandatory = $true)]
    [string]$ExpectedCommit,
    [string]$Repo = "C:\arw-m0",
    [string]$DataDir = "C:\arw-data",
    [string]$ResultsRoot = "C:\arw-results\m1-calibration-clean"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"

$actualCommit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $ExpectedCommit) {
    throw "source commit mismatch: expected=$ExpectedCommit actual=$actualCommit"
}
if (git status --porcelain --untracked-files=no) {
    throw "tracked worktree must be clean"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "experiment interpreter is unavailable: $Python"
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "data directory is unavailable: $DataDir"
}

powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 70 | Out-Null
powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
nvidia-smi -lgc 300,1500 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "GPU clock lock failed"
}

$process = [System.Diagnostics.Process]::GetCurrentProcess()
$process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
$process.ProcessorAffinity = [IntPtr]0x3FF
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

& $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA preflight failed"
}

foreach ($seed in @(101, 102)) {
    $outputDir = Join-Path $ResultsRoot ("clean-seed{0}" -f $seed)
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $summaryPath = Join-Path $outputDir "summary.json"
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        if ($summary.status -eq "succeeded" -and $summary.source_commit -eq $ExpectedCommit) {
            Write-Output "seed already complete; skipping: $seed"
            continue
        }
    }
    & $Python -m experiments.m1_calibration_clean `
        --seed ([string]$seed) `
        --data-dir $DataDir `
        --output-dir $outputDir `
        --device cuda 2>&1 | Tee-Object -FilePath (Join-Path $outputDir "stdout.log") -Append
    if ($LASTEXITCODE -ne 0) {
        throw "clean calibration failed: seed=$seed exit=$LASTEXITCODE"
    }
}

$completion = @{
    status = "succeeded"
    scope = "clean-only"
    source_commit = $ExpectedCommit
    seeds = @(101, 102)
    completed_at = (Get-Date).ToString("o")
}
$completion | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ResultsRoot "PARTIAL_COMPLETE.json")
Write-Output "M1 clean calibration seeds 101 and 102 complete"
