$ErrorActionPreference = "Stop"

$repo = "C:\arw-m0"
$dataRoot = "C:\arw-data\m1-tuning-baselines"
$resultRoot = "C:\arw-results\m1-simple-baseline-tuning"
$noiseAuditRoot = "C:\arw-results\m1-simple-baseline-noise-audit"
$adamwRoot = "C:\arw-results\m1-adamw-tuning"
$archive = "C:\arw-data\cifar100\cifar-100-python.tar.gz"
$trainPayload = "C:\arw-data\cifar100\cifar-100-python\train"
$bundle201 = Join-Path $dataRoot "training\d3c8d08a62c44a8fb4c33f54aa854760"
$bundle202 = Join-Path $dataRoot "training\aa303d193bf74609a1de64379878e912"

New-Item -ItemType Directory -Force -Path $resultRoot, $noiseAuditRoot | Out-Null
$process = Get-Process -Id $PID
$process.ProcessorAffinity = [IntPtr]0xFFF
$process.PriorityClass = "Normal"
& nvidia-smi -rgc | Out-File (Join-Path $resultRoot "gpu-reset.log") -Append

Set-Location $repo
if (-not (Test-Path (Join-Path $dataRoot "image-store\manifest.json"))) {
    & uv run --locked python -m experiments.m1_noisy_data prepare `
        --archive $archive `
        --train-payload $trainPayload `
        --output $dataRoot
    if ($LASTEXITCODE -ne 0) { throw "baseline tuning store preparation failed" }
}

New-Item -ItemType Directory -Force -Path (Join-Path $dataRoot "training") | Out-Null
if (-not (Test-Path (Join-Path $bundle201 "manifest.json"))) {
    & uv run --locked python -m experiments.m1_noisy_data generate `
        --source (Join-Path $dataRoot "private-train-source") `
        --training $bundle201 `
        --audit (Join-Path $noiseAuditRoot "seed1201") `
        --noise-seed 1201 `
        --opaque-id "d3c8d08a62c44a8fb4c33f54aa854760"
    if ($LASTEXITCODE -ne 0) { throw "noise seed 1201 generation failed" }
}
if (-not (Test-Path (Join-Path $bundle202 "manifest.json"))) {
    & uv run --locked python -m experiments.m1_noisy_data generate `
        --source (Join-Path $dataRoot "private-train-source") `
        --training $bundle202 `
        --audit (Join-Path $noiseAuditRoot "seed1202") `
        --noise-seed 1202 `
        --opaque-id "aa303d193bf74609a1de64379878e912"
    if ($LASTEXITCODE -ne 0) { throw "noise seed 1202 generation failed" }
}

& uv run --locked python -m experiments.m1_simple_baseline_tuning `
    --data-dir "C:\arw-data\cifar100" `
    --image-store (Join-Path $dataRoot "image-store") `
    --tuning-store (Join-Path $dataRoot "tuning-store") `
    --bundle-201 $bundle201 `
    --bundle-202 $bundle202 `
    --adamw-root $adamwRoot `
    --output $resultRoot `
    --g-ref 2.017967104911804 `
    --learning-rate 0.0003 `
    --weight-decay 0.1 `
    *>> (Join-Path $resultRoot "dispatcher.log")
if ($LASTEXITCODE -ne 0) { throw "simple baseline tuning failed with exit code $LASTEXITCODE" }
