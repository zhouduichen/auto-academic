param(
    [Parameter(Mandatory = $true)][string]$ExpectedCommit,
    [Parameter(Mandatory = $true)][string]$Bundle101,
    [Parameter(Mandatory = $true)][string]$Bundle102,
    [string]$Repo = "C:\arw-m0",
    [string]$ImageStore = "C:\arw-data\m1-noise\image-store",
    [string]$TuningStore = "C:\arw-data\m1-noise\tuning-store",
    [string]$ResultsRoot = "C:\arw-results\m1-calibration-noisy"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if ((git rev-parse HEAD).Trim() -ne $ExpectedCommit) { throw "source commit mismatch" }
if (git status --porcelain --untracked-files=no) { throw "tracked worktree must be clean" }
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100 | Out-Null
powercfg /SETACTIVE SCHEME_CURRENT | Out-Null
nvidia-smi -rgc | Out-Null
$process = [System.Diagnostics.Process]::GetCurrentProcess()
$process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal
$process.ProcessorAffinity = [IntPtr]0xFFF
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
& $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed" }

$cells = @(
    @{Seed=101; Bundle=$Bundle101},
    @{Seed=102; Bundle=$Bundle102}
)
foreach ($cell in $cells) {
    $output = Join-Path $ResultsRoot ("noisy-seed{0}" -f $cell.Seed)
    New-Item -ItemType Directory -Force -Path $output | Out-Null
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -m experiments.m1_calibration_noisy `
        --seed ([string]$cell.Seed) `
        --training $cell.Bundle `
        --image-store $ImageStore `
        --tuning-store $TuningStore `
        --output-dir $output 2>&1 | Tee-Object -FilePath (Join-Path $output "stdout.log") -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($exitCode -ne 0) { throw "noisy calibration failed: seed=$($cell.Seed) exit=$exitCode" }
}
@{
    status="succeeded"; scope="noisy-only"; source_commit=$ExpectedCommit;
    seeds=@(101,102); completed_at=(Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ResultsRoot "NOISY_COMPLETE.json")
