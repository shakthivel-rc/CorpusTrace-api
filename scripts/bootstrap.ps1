<#
.SYNOPSIS
    CorpusTrace setup for Windows.

.DESCRIPTION
    Installs what Windows is missing — Git and Docker Desktop — starts Docker if it is
    installed but not running, and then hands over to scripts/bootstrap.sh under Git Bash.

    THIS SCRIPT DELIBERATELY DOES NOT REIMPLEMENT SETUP.

    Everything that decides anything — which ports are free, what goes in .env, how to
    recover from a build failure, whether to fall back to a native install — lives in
    bootstrap.sh and its lib/ directory, and is the same code on all three platforms. A
    PowerShell port of that logic would be a second implementation to keep correct, and the
    two would drift the first time only one of them was tested.

    Bash is not an assumption on Windows: Docker Desktop requires WSL2 or Hyper-V, and Git
    for Windows ships Git Bash, so any machine that can run this stack at all has one. If
    Git is missing this script installs it, which is how it gets one.

.PARAMETER Native
    Skip Docker entirely and set up a Python virtualenv and npm install instead.

.PARAMETER Check
    Diagnose only. Changes nothing.

.PARAMETER Yes
    Never prompt; take the recommended answer every time.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

.EXAMPLE
    .\setup.cmd
#>
[CmdletBinding()]
param(
    [switch]$Native,
    [switch]$Check,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'

function Write-Step { param($Message) Write-Host "`n$Message" -ForegroundColor White }
function Write-Info { param($Message) Write-Host "  $Message" }
function Write-Ok   { param($Message) Write-Host "  OK  $Message" -ForegroundColor Green }
function Write-Warn { param($Message) Write-Host "  !   $Message" -ForegroundColor Yellow }
function Write-Fix  { param($Message) Write-Host "  ->  $Message" -ForegroundColor Cyan }
function Fail       { param($Message) Write-Host "`n  X  $Message" -ForegroundColor Red; exit 1 }

function Test-Command {
    param([string]$Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Confirm-Action {
    param([string]$Prompt)
    if ($Yes -or -not [Environment]::UserInteractive) { return $true }
    $answer = Read-Host "  $Prompt [Y/n]"
    return ($answer -eq '' -or $answer -match '^[Yy]')
}

# winget is the only package manager present on a stock modern Windows. It arrives with
# the App Installer package, which Windows 11 has out of the box and Windows 10 has after
# any Store update — but a freshly imaged or Store-disabled machine may not, and there is
# nothing this script can do about that except say so precisely.
function Install-WithWinget {
    param([string]$Id, [string]$Label)

    if (-not (Test-Command 'winget')) {
        Write-Warn "winget is not available, so $Label cannot be installed automatically."
        Write-Info "Install 'App Installer' from the Microsoft Store, or install $Label by hand, then re-run."
        return $false
    }
    if (-not (Confirm-Action "install $Label with winget?")) { return $false }

    Write-Fix "winget install --id $Id"
    # --silent so an installer UI does not sit waiting behind the console window, and
    # --accept-*-agreements because winget otherwise blocks on a prompt this script cannot
    # see or answer.
    & winget install --id $Id --silent --accept-package-agreements --accept-source-agreements | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "winget could not install $Label (exit $LASTEXITCODE)."
        return $false
    }

    # A fresh install is not on this process's PATH — the installer updated the machine
    # environment, which existing processes never see. Re-read it rather than telling the
    # operator to open a new terminal.
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [Environment]::GetEnvironmentVariable('Path', 'User')
    Write-Ok "$Label installed"
    return $true
}

function Find-GitBash {
    # Git Bash first and by explicit path. `where bash` on Windows 10+ finds
    # C:\Windows\System32\bash.exe — the WSL launcher — which would run the whole setup
    # inside a Linux distribution with its own filesystem, its own Docker socket and no
    # access to the Windows Docker Desktop CLI unless WSL integration happens to be on.
    # That failure is deeply confusing, so the WSL bash is never chosen by accident.
    $candidates = @(
        "$env:ProgramFiles\Git\bin\bash.exe",
        "${env:ProgramFiles(x86)}\Git\bin\bash.exe",
        "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    $onPath = Get-Command bash -ErrorAction SilentlyContinue
    if ($onPath -and $onPath.Source -notmatch 'System32') { return $onPath.Source }
    return $null
}

function Start-DockerDesktop {
    $exe = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $exe)) { return $false }
    Write-Fix "starting Docker Desktop"
    Start-Process -FilePath $exe | Out-Null

    # Docker Desktop reports nothing useful for the first half-minute of a cold start, and
    # the engine is up well after the window appears.
    for ($waited = 0; $waited -lt 180; $waited += 5) {
        Start-Sleep -Seconds 5
        & docker info 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { return $true }
        if ($waited -gt 0 -and $waited % 30 -eq 0) { Write-Info "still waiting for Docker ($waited s)" }
    }
    return $false
}

# Windows path -> the MSYS path Git Bash understands: C:\a\b becomes /c/a/b.
function ConvertTo-BashPath {
    param([string]$Path)
    $full = (Resolve-Path $Path).Path
    $drive = $full.Substring(0, 1).ToLower()
    return '/' + $drive + ($full.Substring(2) -replace '\\', '/')
}

# -------------------------------------------------------------------------------------
$apiDir = Split-Path -Parent $PSScriptRoot

Write-Host "CorpusTrace setup (Windows)" -ForegroundColor White
Write-Info "PowerShell $($PSVersionTable.PSVersion) on $([Environment]::OSVersion.VersionString)"

Write-Step "1. Git"
if (Test-Command 'git') {
    Write-Ok "git is installed"
} else {
    Write-Warn "git is not installed — it is needed to fetch the SPA, and it provides the shell the rest of setup runs in"
    if (-not (Install-WithWinget -Id 'Git.Git' -Label 'Git for Windows')) {
        Fail "Git for Windows is required. Install it from https://git-scm.com/download/win and re-run."
    }
}

Write-Step "2. Docker"
$dockerUsable = $false
if ($Native) {
    Write-Info "-Native was passed; not looking for Docker"
} elseif (Test-Command 'docker') {
    & docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Docker is running"
        $dockerUsable = $true
    } else {
        Write-Warn "Docker is installed but the engine is not responding"
        if (Start-DockerDesktop) {
            Write-Ok "Docker is up"
            $dockerUsable = $true
        } else {
            Write-Warn "Docker did not start. Setup will fall back to a native install."
        }
    }
} else {
    Write-Warn "Docker Desktop is not installed"
    if (Install-WithWinget -Id 'Docker.DockerDesktop' -Label 'Docker Desktop') {
        # Docker Desktop's first run needs a reboot on most machines to finish enabling
        # WSL2, and there is no way to work around that from here.
        if (Start-DockerDesktop) {
            $dockerUsable = $true
        } else {
            Write-Warn "Docker Desktop is installed but not running yet — it usually needs one reboot to finish setting up WSL2."
            Write-Info "Reboot, then re-run this script. Setup will continue natively for now."
        }
    }
}

Write-Step "3. Handing over to bootstrap.sh"
$bash = Find-GitBash
if (-not $bash) {
    Fail "no Git Bash found. Install Git for Windows (it includes it) and re-run:
       winget install --id Git.Git"
}
Write-Info "shell: $bash"

$arguments = @()
if ($Native -or -not $dockerUsable) { $arguments += '--native' }
if ($Check) { $arguments += '--check' }
if ($Yes)   { $arguments += '--yes' }

# `bash -lc` and not `bash -c`: a login shell is what puts Git's own bin directory, and
# anything a version manager installed, onto PATH. Without it, tools that are certainly
# installed appear missing.
$command = "cd '$(ConvertTo-BashPath $apiDir)' && ./scripts/bootstrap.sh $($arguments -join ' ')"
& $bash -lc $command
exit $LASTEXITCODE
