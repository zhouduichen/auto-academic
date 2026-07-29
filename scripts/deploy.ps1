<#
.SYNOPSIS
    Deploys the ARW Windows Server — verifies prerequisites, sets up directories,
    firewall rules, environment variables, CUDA gate, and validates the Karpathy clone.

.DESCRIPTION
    This script handles the one-time deployment of the ARW (Auto-Research Worktree)
    server on a Windows machine. It checks admin rights, verifies Python 3.12+ via
    uv, verifies TLS certificates, creates persistent data directories, configures
    the Windows Firewall for Tailscale access, prompts for ARW_API_TOKEN if unset,
    detects CUDA availability, and validates the Karpathy clone repository.

    Fail-closed: the script halts on any error ($ErrorActionPreference = "Stop").

.EXAMPLE
    .\deploy.ps1

.NOTES
    Must be run as Administrator for firewall rule creation.
    Requires uv (https://docs.astral.sh/uv/) installed and on PATH.
#>

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helper: emit a timestamped log line
# ---------------------------------------------------------------------------
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] [$Level] $Message"
}

# ---------------------------------------------------------------------------
# Helper: prompt for a secret string (SecureString → plaintext for env var)
# ---------------------------------------------------------------------------
function Read-Secret {
    param([string]$Prompt)
    Write-Host -NoNewline "$Prompt "
    $secure = Read-Host -AsSecureString
    $ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

# ---------------------------------------------------------------------------
# 1. Verify running as Administrator
# ---------------------------------------------------------------------------
Write-Log "Checking administrator privileges..."
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Log "This script must be run as Administrator (required for firewall rules)." "ERROR"
    exit 1
}
Write-Log "Running as Administrator: OK"

# ---------------------------------------------------------------------------
# 2. Verify Python 3.12+ via uv
# ---------------------------------------------------------------------------
Write-Log "Verifying uv is installed..."
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Log "uv is not installed or not on PATH. Install it from https://docs.astral.sh/uv/" "ERROR"
    exit 1
}
Write-Log "uv found at: $($uvPath.Source)"

Write-Log "Verifying Python 3.12+ via uv..."
$pythonVersion = & uv python find 3.12 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Log "Python 3.12+ not found via uv. Run 'uv python install 3.12' first." "ERROR"
    exit 1
}
Write-Log "Python resolved: $($pythonVersion.Trim())"

# ---------------------------------------------------------------------------
# 3. Verify TLS certificate and key
# ---------------------------------------------------------------------------
$certDir = "$env:APPDATA\arw"
$certFile = Join-Path $certDir "cert.pem"
$keyFile = Join-Path $certDir "key.pem"

Write-Log "Checking TLS certificate and key..."
if (-not (Test-Path $certFile)) {
    Write-Log "Certificate not found at $certFile" "ERROR"
    exit 1
}
if (-not (Test-Path $keyFile)) {
    Write-Log "Key not found at $keyFile" "ERROR"
    exit 1
}
Write-Log "Certificate: $certFile"
Write-Log "Key:         $keyFile"

# Detect self-signed certificate
try {
    $certContent = Get-Content -Raw -Path $certFile
    $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new()
    $cert.Import([System.Text.Encoding]::ASCII.GetBytes($certContent))
    if ($cert.Subject -eq $cert.Issuer) {
        Write-Log "WARNING: Certificate appears to be self-signed (Subject matches Issuer)." "WARN"
        Write-Log "Self-signed certs are acceptable for Tailscale-internal use but will require -SkipCertificateCheck on health checks." "WARN"
    }
    else {
        Write-Log "Certificate appears to be issued by a CA (Subject != Issuer)." "INFO"
    }
}
catch {
    Write-Log "WARNING: Could not parse certificate to check self-signed status: $_" "WARN"
}

# ---------------------------------------------------------------------------
# 4. Create persistent directories
# ---------------------------------------------------------------------------
$dataDir = "$env:LOCALAPPDATA\arw\data"
$worktreeDir = "$env:LOCALAPPDATA\arw\worktrees"
$runsDir = "$env:LOCALAPPDATA\arw\runs"
$logDir = "$env:LOCALAPPDATA\arw\logs"

Write-Log "Creating persistent directories..."
foreach ($dir in @($dataDir, $worktreeDir, $runsDir, $logDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Log "  Created: $dir"
    }
    else {
        Write-Log "  Exists:  $dir"
    }
}

# ---------------------------------------------------------------------------
# 5. Firewall rule — allow inbound TCP 8443 from Tailscale subnet
# ---------------------------------------------------------------------------
$firewallRuleName = "ARW Server HTTPS (Tailscale)"
$tailscaleSubnet = "100.64.0.0/10"

Write-Log "Configuring Windows Firewall..."
$existingRule = Get-NetFirewallRule -DisplayName $firewallRuleName -ErrorAction SilentlyContinue
if ($existingRule) {
    Write-Log "Firewall rule '$firewallRuleName' already exists; updating..."

    # Update remote address filter
    $existingFilter = Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $existingRule
    Set-NetFirewallAddressFilter -InputObject $existingFilter -RemoteAddress $tailscaleSubnet

    # Ensure protocol and port are correct
    $existingPortFilter = Get-NetFirewallPortFilter -AssociatedNetFirewallRule $existingRule
    Set-NetFirewallPortFilter -InputObject $existingPortFilter -Protocol TCP -LocalPort 8443

    # Ensure action is Allow
    Set-NetFirewallRule -InputObject $existingRule -Action Allow -Enabled True
    Write-Log "  Updated."
}
else {
    New-NetFirewallRule `
        -DisplayName $firewallRuleName `
        -Description "Allow inbound TCP 8443 for ARW server from Tailscale subnet" `
        -Direction Inbound `
        -Protocol TCP `
        -LocalPort 8443 `
        -RemoteAddress $tailscaleSubnet `
        -Action Allow `
        -Profile Any `
        -Enabled True | Out-Null
    Write-Log "  Created."
}
Write-Log "Firewall rule '$firewallRuleName': OK (RemoteAddress=$tailscaleSubnet, LocalPort=8443/TCP)"

# ---------------------------------------------------------------------------
# 6. ARW_API_TOKEN — set user env var if not already present
# ---------------------------------------------------------------------------
Write-Log "Checking ARW_API_TOKEN..."
$currentToken = [System.Environment]::GetEnvironmentVariable("ARW_API_TOKEN", "User")
if ([string]::IsNullOrWhiteSpace($currentToken)) {
    Write-Host ""
    Write-Host "======================================================================"
    Write-Host "  ARW_API_TOKEN is not set."
    Write-Host "  This is a pre-shared Bearer token required for API authentication."
    Write-Host "  The token will be stored as a user-level environment variable."
    Write-Host "======================================================================"
    $newToken = Read-Secret -Prompt "Enter ARW_API_TOKEN (or press Ctrl+C to abort):"
    if ([string]::IsNullOrWhiteSpace($newToken)) {
        Write-Log "No token provided. Aborting deployment." "ERROR"
        exit 1
    }
    [System.Environment]::SetEnvironmentVariable("ARW_API_TOKEN", $newToken, "User")
    Write-Log "ARW_API_TOKEN saved to user environment. Restart shells to pick up the change."
}
else {
    Write-Log "ARW_API_TOKEN is already set in user environment."
}

# ---------------------------------------------------------------------------
# 7. CUDA gate — detect nvidia-smi availability
# ---------------------------------------------------------------------------
Write-Log "Checking CUDA availability via nvidia-smi..."
$cudaAvailable = $false
try {
    $nvidiaOutput = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $gpuName = $nvidiaOutput.Trim()
        Write-Log "CUDA detected. GPU: $gpuName"
        $cudaAvailable = $true
    }
    else {
        throw "nvidia-smi exited with code $LASTEXITCODE"
    }
}
catch {
    Write-Log "nvidia-smi failed: $_" "WARN"
    Write-Log "Setting ARW_CUDA_GATE=0 — real executor disabled; falling back to fake executor only." "WARN"
    $cudaAvailable = $false
}

if ($cudaAvailable) {
    Write-Log "Setting ARW_CUDA_GATE=1 (real executor enabled)"
    [System.Environment]::SetEnvironmentVariable("ARW_CUDA_GATE", "1", "User")
    $env:ARW_CUDA_GATE = "1"
}
else {
    Write-Log "Setting ARW_CUDA_GATE=0 (fake executor only)"
    [System.Environment]::SetEnvironmentVariable("ARW_CUDA_GATE", "0", "User")
    $env:ARW_CUDA_GATE = "0"
}

# ---------------------------------------------------------------------------
# 8. Validate Karpathy clone exists at the configured path
# ---------------------------------------------------------------------------
$karPathDefault = "$env:LOCALAPPDATA\arw\karpathy-clone"
$karPath = [System.Environment]::GetEnvironmentVariable("ARW_KARPATHY_PATH", "User")
if ([string]::IsNullOrWhiteSpace($karPath)) {
    Write-Log "ARW_KARPATHY_PATH not set; using default: $karPathDefault"
    $karPath = $karPathDefault
}
Write-Log "Validating Karpathy clone at: $karPath"
if (-not (Test-Path $karPath)) {
    Write-Log "Karpathy clone directory not found at $karPath." "WARN"
    Write-Log "Please clone it and set ARW_KARPATHY_PATH environment variable if using a custom path." "WARN"
}
elseif (-not (Test-Path (Join-Path $karPath ".git"))) {
    Write-Log "Directory exists but does not appear to be a git repository (no .git folder)." "WARN"
}
else {
    Write-Log "Karpathy clone exists and is a git repository: OK"
}

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "======================================================================"
Write-Host "                       ARW DEPLOYMENT SUMMARY                        "
Write-Host "======================================================================"
Write-Host "  Admin check       : PASS"
Write-Host "  uv / Python 3.12+ : PASS"
Write-Host "  TLS certificate   : $certFile"
Write-Host "  TLS key           : $keyFile"
Write-Host "  Data directory    : $dataDir"
Write-Host "  Worktree root     : $worktreeDir"
Write-Host "  Runs directory    : $runsDir"
Write-Host "  Logs directory    : $logDir"
Write-Host "  Firewall rule     : '$firewallRuleName' (TCP 8443, $tailscaleSubnet)"
Write-Host "  CUDA gate         : $(if ($cudaAvailable) { '1 (real executor)' } else { '0 (fake executor only)' })"
Write-Host "  Karpathy clone    : $karPath"
Write-Host "======================================================================"
Write-Host ""
Write-Host "Deployment complete. Run scripts\start.ps1 to launch the server."
Write-Host ""

exit 0
