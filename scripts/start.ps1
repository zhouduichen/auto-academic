<#
.SYNOPSIS
    Starts the ARW Windows Server (arw-server) on port 8443 with HTTPS.

.DESCRIPTION
    Reads ARW_API_TOKEN from the environment (errors if missing), creates the
    log directory, and launches the server via `uv run arw-server` using
    Start-Process. Standard output and error streams are captured to timestamped
    log files. Outputs the PID and health-check URL.

    Fail-closed: the script halts on any error ($ErrorActionPreference = "Stop").

.EXAMPLE
    .\start.ps1

.NOTES
    The server binds to port 8443 with TLS.  Ensure the certificate and key
    are in place ($env:APPDATA\arw\cert.pem, $env:APPDATA\arw\key.pem) and
    the firewall rule was created by deploy.ps1.
#>

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helper: timestamped log line
# ---------------------------------------------------------------------------
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] [$Level] $Message"
}

# ---------------------------------------------------------------------------
# 1. Verify ARW_API_TOKEN is set
# ---------------------------------------------------------------------------
Write-Log "Checking ARW_API_TOKEN..."
$apiToken = $env:ARW_API_TOKEN
if ([string]::IsNullOrWhiteSpace($apiToken)) {
    Write-Log "ARW_API_TOKEN is not set in the current environment." "ERROR"
    Write-Log "Please run deploy.ps1 first, or set the variable and restart your shell." "ERROR"
    exit 1
}
Write-Log "ARW_API_TOKEN is set."

# ---------------------------------------------------------------------------
# 2. Build environment variable block
# ---------------------------------------------------------------------------
$dataDir       = "$env:LOCALAPPDATA\arw\data"
$worktreeDir   = "$env:LOCALAPPDATA\arw\worktrees"
$runsDir       = "$env:LOCALAPPDATA\arw\runs"

$dbPath        = if ([string]::IsNullOrWhiteSpace($env:ARW_DB_PATH))        { Join-Path $dataDir "arw_server.db" } else { $env:ARW_DB_PATH }
$worktreeRoot  = if ([string]::IsNullOrWhiteSpace($env:ARW_WORKTREE_ROOT))  { $worktreeDir }                    else { $env:ARW_WORKTREE_ROOT }
$cudaGate      = if ([string]::IsNullOrWhiteSpace($env:ARW_CUDA_GATE))      { "0" }                             else { $env:ARW_CUDA_GATE }

Write-Log "Configuration:"
Write-Log "  ARW_DB_PATH       = $dbPath"
Write-Log "  ARW_WORKTREE_ROOT = $worktreeRoot"
Write-Log "  ARW_CUDA_GATE     = $cudaGate"

# ---------------------------------------------------------------------------
# 3. Create log directory and log file paths
# ---------------------------------------------------------------------------
$logDir = "$env:LOCALAPPDATA\arw\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    Write-Log "Created log directory: $logDir"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = Join-Path $logDir "arw-server-$timestamp-stdout.log"
$stderrLog = Join-Path $logDir "arw-server-$timestamp-stderr.log"
$pidFile    = Join-Path $logDir "arw-server.pid"

Write-Log "Log files:"
Write-Log "  stdout : $stdoutLog"
Write-Log "  stderr : $stderrLog"

# ---------------------------------------------------------------------------
# 4. Start the server via Start-Process
# ---------------------------------------------------------------------------
Write-Log "Starting arw-server via uv run arw-server..."

# Launch with Start-Process, passing environment and capturing output
$proc = Start-Process `
    -FilePath "uv" `
    -ArgumentList "run", "arw-server" `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

# Wait a moment for it to start
Write-Log "Process launched with PID: $($proc.Id)"

# Save PID to file
$proc.Id | Out-File -FilePath $pidFile -Encoding utf8 -NoNewline

# Check that it's still running after a brief startup window
Start-Sleep -Seconds 3
if ($proc.HasExited) {
    Write-Log "Server process exited immediately (exit code: $($proc.ExitCode))." "ERROR"
    Write-Log "Check stderr log for details: $stderrLog" "ERROR"
    if (Test-Path $stderrLog) {
        Write-Log "--- stderr log tail ---" "ERROR"
        Get-Content -Path $stderrLog -Tail 30 | ForEach-Object { Write-Log $_ "ERROR" }
    }
    exit 1
}

# ---------------------------------------------------------------------------
# 5. Output PID and health check URL
# ---------------------------------------------------------------------------
$hostname = [System.Net.Dns]::GetHostName()
$healthUrl = "https://${hostname}:8443/healthz"

Write-Host ""
Write-Host "======================================================================"
Write-Host "  ARW Server started."
Write-Host "  PID            : $($proc.Id)"
Write-Host "  PID file       : $pidFile"
Write-Host "  Health check   : $healthUrl"
Write-Host "  Test command   : Invoke-WebRequest -Uri '$healthUrl' -SkipCertificateCheck"
Write-Host "                  (or run .\scripts\health-check.ps1)"
Write-Host "  Stdout log     : $stdoutLog"
Write-Host "  Stderr log     : $stderrLog"
Write-Host "======================================================================"
Write-Host ""

exit 0
