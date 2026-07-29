<#
.SYNOPSIS
    Stops a running ARW Windows Server process gracefully, falling back to a
    forceful kill after a timeout.  Cleans up orphan worktrees.

.DESCRIPTION
    Searches running processes for arw-server or uvicorn instances, sends a
    graceful shutdown via taskkill, waits up to 10 seconds, and forces a kill
    if the process has not exited.  Also cleans orphan git worktrees in the
    configured ARW_WORKTREE_ROOT.

    Fail-closed: the script halts on any error ($ErrorActionPreference = "Stop"),
    except during the worktree cleanup where errors are logged and skipped.

.EXAMPLE
    .\stop.ps1

.NOTES
    Requires PowerShell 5.1+ for Get-CimInstance (used to inspect command lines).
    On PowerShell 7+ the script also tries Get-Process -IncludeUserName for
    additional process metadata.
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
# Constants
# ---------------------------------------------------------------------------
$gracefulTimeoutSec  = 10
$killPollIntervalSec = 1
$pidFile             = "$env:LOCALAPPDATA\arw\logs\arw-server.pid"

# ---------------------------------------------------------------------------
# 1. Find the arw-server process
# ---------------------------------------------------------------------------
Write-Log "Searching for arw-server processes..."

$targetProcs = @()

# ----- Method A: look for the PID file -----
if (Test-Path $pidFile) {
    $savedPid = (Get-Content -Path $pidFile -Raw).Trim()
    if ($savedPid -match '^\d+$') {
        $proc = Get-Process -Id ([int]$savedPid) -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Log "Found process via PID file ($pidFile): PID $($proc.Id) — $($proc.ProcessName)"
            $targetProcs += $proc
        }
        else {
            Write-Log "PID file points to PID $savedPid but no such process is running (already stopped?)."
        }
    }
    else {
        Write-Log "PID file exists but content is not a valid PID: '$savedPid'"
    }
}

# ----- Method B: scan command lines for arw-server / uvicorn -----
$allProcs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'arw-server|uvicorn' }

foreach ($p in $allProcs) {
    # Skip if already captured
    if ($targetProcs.Id -contains $p.ProcessId) { continue }
    try {
        $proc = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Log "Found via command-line scan: PID $($proc.Id) — $($proc.ProcessName) — $($p.CommandLine)"
            $targetProcs += $proc
        }
    }
    catch {
        Write-Log "Could not retrieve process PID $($p.ProcessId): $_" "WARN"
    }
}

if ($targetProcs.Count -eq 0) {
    Write-Log "No arw-server or uvicorn processes found. Nothing to stop."
    Write-Log "Proceeding to worktree cleanup anyway..."
}
else {
    Write-Log "Found $($targetProcs.Count) matching process(es). Sending graceful shutdown..."
}

# ---------------------------------------------------------------------------
# 2. Graceful shutdown (taskkill without /F)
# ---------------------------------------------------------------------------
foreach ($proc in $targetProcs) {
    Write-Log "Sending graceful shutdown signal to PID $($proc.Id)..."

    # taskkill (no /F) sends WM_CLOSE — uvicorn / FastAPI may handle it
    $result = & taskkill /PID $proc.Id 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "  taskkill sent successfully to PID $($proc.Id)."
    }
    else {
        Write-Log "  taskkill returned exit code $LASTEXITCODE : $result" "WARN"
    }
}

# ---------------------------------------------------------------------------
# 3. Wait for processes to exit
# ---------------------------------------------------------------------------
$deadline = (Get-Date).AddSeconds($gracefulTimeoutSec)
$alive = @($targetProcs | Where-Object { -not $_.HasExited })

while ($alive.Count -gt 0 -and (Get-Date) -lt $deadline) {
    Write-Log "Waiting for exit... ($($alive.Count) process(es) still running)"
    Start-Sleep -Seconds $killPollIntervalSec
    # Refresh HasExited state
    foreach ($p in $alive) { $p.Refresh() }
    $alive = @($alive | Where-Object { -not $_.HasExited })
}

# ---------------------------------------------------------------------------
# 4. Forceful kill for stragglers
# ---------------------------------------------------------------------------
if ($alive.Count -gt 0) {
    Write-Log "Graceful timeout reached ($gracefulTimeoutSec s). Sending forceful kill..." "WARN"
    foreach ($proc in $alive) {
        Write-Log "Force-killing PID $($proc.Id)..."
        & taskkill /F /PID $proc.Id 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "  PID $($proc.Id) terminated with /F."
        }
        else {
            Write-Log "  Warning: taskkill /F PID $($proc.Id) returned exit code $LASTEXITCODE." "WARN"
        }
    }
}
else {
    Write-Log "All processes exited gracefully."
}

# ---------------------------------------------------------------------------
# 5. Remove stale PID file
# ---------------------------------------------------------------------------
if (Test-Path $pidFile) {
    Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
    Write-Log "Removed PID file: $pidFile"
}

# ---------------------------------------------------------------------------
# 6. Clean orphan worktrees
# ---------------------------------------------------------------------------
$worktreeRoot = if ([string]::IsNullOrWhiteSpace($env:ARW_WORKTREE_ROOT)) {
    "$env:LOCALAPPDATA\arw\worktrees"
}
else {
    $env:ARW_WORKTREE_ROOT
}

Write-Log "Cleaning orphan worktrees in: $worktreeRoot"

if (-not (Test-Path $worktreeRoot)) {
    Write-Log "Worktree root does not exist yet; nothing to clean."
    exit 0
}

# Prune git worktrees (if the root is a git repository or contains worktree dirs)
Push-Location $worktreeRoot
try {
    $gitCheck = & git rev-parse --is-inside-work-tree 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Worktree root is a git worktree; running 'git worktree prune'..."
        & git worktree prune 2>&1 | ForEach-Object { Write-Log "  git: $_" }
        if ($LASTEXITCODE -ne 0) {
            Write-Log "git worktree prune completed with warnings (exit $LASTEXITCODE)." "WARN"
        }
        else {
            Write-Log "git worktree prune: OK"
        }
    }
    else {
        Write-Log "Worktree root is not a git tree. Skipping git worktree prune."
    }
}
catch {
    Write-Log "Non-fatal: git worktree prune failed: $_" "WARN"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Log "Stop sequence complete."
exit 0
