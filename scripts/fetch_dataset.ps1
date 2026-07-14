#requires -Version 5.1
<#
.SYNOPSIS
    Download the current dataset archive from Hugging Face into the local cache.

.DESCRIPTION
    Reads the repo + filename from scripts\_dataset_version.ps1.
    Requires HF_TOKEN env var (the dataset repo is private) and huggingface-cli
    on PATH.
    Idempotent: skips the download if the zip already exists.

.EXAMPLE
    $env:HF_TOKEN = 'hf_...'
    .\scripts\fetch_dataset.ps1
#>
param(
    [string]$OutDir
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot '_dataset_version.ps1')

if (-not $OutDir) {
    $OutDir = Join-Path $RepoRoot '_dataset_cache'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Out = Join-Path $OutDir $DatasetZip

if (Test-Path -LiteralPath $Out) {
    Write-Host "Dataset already present: $Out"
    Write-Host "(delete the file to force re-download)"
    return
}

if (-not (Get-Command huggingface-cli -ErrorAction SilentlyContinue)) {
    throw "huggingface-cli not on PATH. Install: pip install -U 'huggingface_hub[cli]'"
}
if (-not $env:HF_TOKEN) {
    throw "HF_TOKEN env var not set. Get one at https://huggingface.co/settings/tokens and: `$env:HF_TOKEN = 'hf_...'`"
}

if (-not $env:HF_XET_HIGH_PERFORMANCE) { $env:HF_XET_HIGH_PERFORMANCE = '1' }
Write-Host "Downloading $DatasetZip"
Write-Host "  from: $DatasetHfRepo"
Write-Host "  into: $OutDir"
& huggingface-cli download $DatasetHfRepo $DatasetZip `
    --repo-type dataset `
    --local-dir $OutDir `
    --token $env:HF_TOKEN
if ($LASTEXITCODE -ne 0) { throw "huggingface-cli download failed (exit $LASTEXITCODE)" }
Write-Host "Done: $Out"
