param(
    [Parameter(Mandatory = $true)]
    [string]$ExpectedCommit,
    [string]$Repo = "C:\arw-m0",
    [string]$DataDir = "C:\arw-data",
    [string]$ResultsRoot = "C:\arw-results\m06",
    [string]$PilotSeed0 = "C:\arw-results\m05\seed0"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"

$actualCommit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $ExpectedCommit) {
    throw "source commit mismatch: expected=$ExpectedCommit actual=$actualCommit"
}
$trackedStatus = git status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $trackedStatus) {
    throw "tracked worktree must be clean"
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "data directory is unavailable: $DataDir"
}
if (-not (Test-Path -LiteralPath $PilotSeed0 -PathType Container)) {
    throw "pilot seed0 artifacts are unavailable: $PilotSeed0"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "experiment interpreter is unavailable: $Python"
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$process = [System.Diagnostics.Process]::GetCurrentProcess()
$process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
$process.ProcessorAffinity = [IntPtr]0x3FF

& $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA preflight failed"
}

function Invoke-Finalizer {
    param([Parameter(Mandatory = $true)][string]$BundleDir)
    & $Python -m experiments.m0_optimizer_state.finalize_bundle $BundleDir
    if ($LASTEXITCODE -ne 0) {
        throw "bundle finalization failed: $BundleDir"
    }
}

function Invoke-Bundle {
    param(
        [Parameter(Mandatory = $true)][string]$BundleDir,
        [Parameter(Mandatory = $true)][string[]]$RunArguments
    )
    if (Test-Path -LiteralPath $BundleDir -PathType Container) {
        try {
            Invoke-Finalizer -BundleDir $BundleDir
            Write-Output "bundle already valid; skipping: $BundleDir"
            return
        }
        catch {
            throw "existing bundle directory is incomplete or invalid: $BundleDir"
        }
    }
    New-Item -ItemType Directory -Path $BundleDir | Out-Null
    $stdoutPath = Join-Path $BundleDir "stdout.log"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -m experiments.m0_optimizer_state.m0_run @RunArguments 2>&1 |
        Tee-Object -FilePath $stdoutPath
    $runnerExit = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($runnerExit -ne 0) {
        throw "runner failed with exit code ${runnerExit}: $BundleDir"
    }
    Invoke-Finalizer -BundleDir $BundleDir
    Write-Output "bundle complete: $BundleDir"
}

$commonArguments = @(
    "--optimizer", "adamw",
    "--pulse", "label_flip",
    "--state-attribution",
    "--split-seed", "20260731",
    "--data-dir", $DataDir,
    "--device", "cuda"
)

$bridgeDir = Join-Path $ResultsRoot "bridge-seed0"
$bridgeArguments = $commonArguments + @(
    "--seed", "0",
    "--pulse-seed", "314159",
    "--output-dir", $bridgeDir
)
Invoke-Bundle -BundleDir $bridgeDir -RunArguments $bridgeArguments

foreach ($filename in @("trajectory_metrics.jsonl", "checkpoint_manifest.json")) {
    $oldPath = Join-Path $PilotSeed0 $filename
    $newPath = Join-Path $bridgeDir $filename
    if (-not (Test-Path -LiteralPath $oldPath -PathType Leaf)) {
        throw "pilot bridge reference is missing: $oldPath"
    }
    $oldHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $oldPath).Hash
    $newHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $newPath).Hash
    if ($oldHash -ne $newHash) {
        throw "bridge hash mismatch for $filename"
    }
}
Write-Output "bridge exact-match audit passed"

foreach ($trainSeed in 3..12) {
    foreach ($pulseSeed in @(271828, 161803)) {
        $bundleName = "train{0}-pulse{1}" -f $trainSeed, $pulseSeed
        $bundleDir = Join-Path $ResultsRoot $bundleName
        $runArguments = $commonArguments + @(
            "--seed", [string]$trainSeed,
            "--pulse-seed", [string]$pulseSeed,
            "--replay-seed", [string](420000 + $trainSeed),
            "--probe-seed", "20260806",
            "--output-dir", $bundleDir
        )
        Invoke-Bundle -BundleDir $bundleDir -RunArguments $runArguments
    }
}

$completion = @{
    status = "succeeded"
    source_commit = $ExpectedCommit
    completed_bundles = 20
    completed_at = (Get-Date).ToString("o")
}
$completion | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ResultsRoot "COMPLETE.json")
Write-Output "M0.6 matrix complete"
