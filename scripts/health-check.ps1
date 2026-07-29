<#
.SYNOPSIS
    Checks the health of the ARW Windows Server via its /healthz endpoint.

.DESCRIPTION
    Sends an HTTPS GET to https://localhost:8443/healthz using
    Invoke-WebRequest with -SkipCertificateCheck (the server may use a
    self-signed certificate).

    Exit codes:
        0 — ARW Server is HEALTHY (HTTP 200)
        1 — ARW Server is DOWN (unreachable, non-200, or TLS error)

    Also checks free disk space on the worktree drive and warns if below
    10 GB.

    Fail-closed: the script halts on any error ($ErrorActionPreference = "Stop").

.EXAMPLE
    .\health-check.ps1

.NOTES
    Requires PowerShell 6+ for -SkipCertificateCheck.
    On PowerShell 5.1, the script falls back to a .NET HttpClient that
    bypasses certificate validation.
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
# Configuration
# ---------------------------------------------------------------------------
$healthUrl = "https://localhost:8443/healthz"
$worktreeRoot = if ([string]::IsNullOrWhiteSpace($env:ARW_WORKTREE_ROOT)) {
    "$env:LOCALAPPDATA\arw\worktrees"
}
else {
    $env:ARW_WORKTREE_ROOT
}

# ---------------------------------------------------------------------------
# 1. Health-check HTTP request
# ---------------------------------------------------------------------------
Write-Log "Checking ARW server health at: $healthUrl"

$healthy = $false
$errorDetail = ""

try {
    # PowerShell 6+ supports -SkipCertificateCheck natively.
    # For PowerShell 5.1 we fall back to a .NET handler that bypasses validation.
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        $response = Invoke-WebRequest `
            -Uri $healthUrl `
            -Method GET `
            -SkipCertificateCheck `
            -TimeoutSec 10 `
            -UseBasicParsing `
            -ErrorAction Stop
    }
    else {
        # PowerShell 5.1 — use .NET to bypass SSL validation
        Add-Type -TypeDefinition @"
using System.Net;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
public static class ArwCertBypass {
    public static void Install() {
        ServicePointManager.ServerCertificateValidationCallback =
            (sender, cert, chain, sslPolicyErrors) => true;
    }
}
"@ -ErrorAction SilentlyContinue
        [ArwCertBypass]::Install()

        $response = Invoke-WebRequest `
            -Uri $healthUrl `
            -Method GET `
            -TimeoutSec 10 `
            -UseBasicParsing `
            -ErrorAction Stop
    }

    if ($response.StatusCode -eq 200) {
        $healthy = $true
        Write-Log "HTTP status: $($response.StatusCode)"
        Write-Log "Response body: $($response.Content.Trim())"
    }
    else {
        $healthy = $false
        $errorDetail = "HTTP status $($response.StatusCode) (expected 200)"
        Write-Log $errorDetail "WARN"
    }
}
catch {
    $healthy = $false
    $errorDetail = $_.Exception.Message
    Write-Log "Health check request failed: $errorDetail" "WARN"
}

# ---------------------------------------------------------------------------
# 2. Disk space check
# ---------------------------------------------------------------------------
Write-Log "Checking disk space on worktree volume..."

$worktreeRootResolved = $worktreeRoot
# If the directory doesn't exist yet, check the parent or fall back to LOCALAPPDATA
if (-not (Test-Path $worktreeRootResolved)) {
    $worktreeRootResolved = "$env:LOCALAPPDATA\arw"
    if (-not (Test-Path $worktreeRootResolved)) {
        $worktreeRootResolved = $env:LOCALAPPDATA
    }
}
if (-not (Test-Path $worktreeRootResolved)) {
    $worktreeRootResolved = $env:TEMP
}

try {
    $drive = (Get-Item $worktreeRootResolved).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    $totalGB = [math]::Round(($drive.Free + $drive.Used) / 1GB, 2)
    Write-Log "Disk on [$($drive.Name):] — Free: ${freeGB} GB / Total: ${totalGB} GB"

    if ($freeGB -lt 10) {
        Write-Log "WARNING: Less than 10 GB free on worktree drive ($freeGB GB).  Disk may fill up during worktree operations." "WARN"
    }
}
catch {
    Write-Log "Could not query disk space for '$worktreeRootResolved': $_" "WARN"
}

# ---------------------------------------------------------------------------
# 3. Final verdict
# ---------------------------------------------------------------------------
Write-Host ""
if ($healthy) {
    Write-Host "ARW Server: HEALTHY"
    Write-Host ""
    exit 0
}
else {
    Write-Host "ARW Server: DOWN"
    if ($errorDetail) {
        Write-Host "  Reason: $errorDetail"
    }
    Write-Host ""
    exit 1
}
