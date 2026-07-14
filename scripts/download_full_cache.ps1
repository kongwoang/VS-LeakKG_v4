#requires -Version 5.1
<#
VS-LeakKG controlled-download / cache pass.

Downloads only the exact URLs / bucket paths listed in the project spec. Does
NOT search for alternatives. Does NOT extract archives (except the LIT-PCBA
split tarball, which is small and safe). Designed to be safe to re-run:
already-complete files are detected by Content-Length and skipped, partial
files are resumed with curl -C -.

Run:
    pwsh -File scripts\download_full_cache.ps1
    powershell -ExecutionPolicy Bypass -File scripts\download_full_cache.ps1
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

$ROOT      = 'D:\hoangpc\VS-LeakKG'
$LOG       = Join-Path $ROOT 'outputs\logs\full_dataset_download.log'
$MANUAL    = Join-Path $ROOT 'data\raw\manual_downloads_needed'
$MIN_FREE_GB_DOWNLOAD = 150
$MIN_FREE_GB_PLINDER  = 200

New-Item -ItemType Directory -Path (Split-Path $LOG) -Force | Out-Null
New-Item -ItemType Directory -Path $MANUAL -Force | Out-Null


# ---------------- helpers ----------------

function Get-FreeSpaceGB {
    param([string]$Drive = 'D')
    $d = Get-PSDrive -Name $Drive -ErrorAction SilentlyContinue
    if (-not $d) { return -1 }
    return [math]::Round($d.Free / 1GB, 2)
}

function Get-ProjectSizeGB {
    try {
        $sum = (Get-ChildItem $ROOT -Recurse -Force -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum).Sum
        if (-not $sum) { $sum = 0 }
        return [math]::Round($sum / 1GB, 2)
    } catch { return -1 }
}

function Log-Step {
    param(
        [Parameter(Mandatory=$true)][string]$Step,
        [Parameter(Mandatory=$true)][string]$Target
    )
    $ts   = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    $cwd  = (Get-Location).Path
    $free = Get-FreeSpaceGB 'D'
    $proj = Get-ProjectSizeGB
    $lines = @(
        "==== $ts ====",
        "step: $Step",
        "target: $Target",
        "cwd: $cwd",
        ("-- drives --"),
        ("  D: free={0}GB" -f $free)
    )
    foreach ($pd in Get-PSDrive -PSProvider FileSystem) {
        if ($pd.Name -ne 'D') {
            $lines += ("  {0}: free={1}GB" -f $pd.Name, [math]::Round($pd.Free/1GB,2))
        }
    }
    $lines += ("-- project size: {0} GB ({1})" -f $proj, $ROOT)
    Add-Content -Path $LOG -Value ($lines -join "`n") -Encoding utf8
    Add-Content -Path $LOG -Value '' -Encoding utf8
}

function Get-RemoteSize {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest -Uri $Url -Method Head -MaximumRedirection 5 `
             -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop
        $len = $r.Headers.'Content-Length'
        if ($len) { return [int64]$len }
    } catch { }
    return $null
}

function Download-Resume {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$Target,
        [int]$Attempts = 3,
        [int]$MinFreeGB = $MIN_FREE_GB_DOWNLOAD
    )
    New-Item -ItemType Directory -Path (Split-Path $Target) -Force | Out-Null

    $free = Get-FreeSpaceGB 'D'
    if ($free -ge 0 -and $free -lt $MinFreeGB) {
        Log-Step 'skip_lowdisk' $Target
        return [pscustomobject]@{ ok=$false; reason="low_disk free=$($free)GB < $($MinFreeGB)GB"; size=0 }
    }

    $remoteLen = Get-RemoteSize $Url
    if ((Test-Path $Target) -and $remoteLen -and ((Get-Item $Target).Length -eq $remoteLen)) {
        Log-Step 'already_complete' $Target
        return [pscustomobject]@{ ok=$true; reason='already_complete'; size=$remoteLen }
    }

    Log-Step 'pre_download' $Target
    $err = ''
    for ($i = 1; $i -le $Attempts; $i++) {
        # curl.exe — resumable, follows redirects, fails on 4xx/5xx.
        & curl.exe -L -C - --retry 2 --retry-delay 5 --connect-timeout 60 `
            --max-time 7200 -A 'Mozilla/5.0 (VS-LeakKG cache)' `
            -f -o $Target $Url
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $Target)) {
            $sz = (Get-Item $Target).Length
            if (-not $remoteLen -or $sz -eq $remoteLen) {
                Log-Step ("post_download_attempt_$i") $Target
                return [pscustomobject]@{ ok=$true; reason="ok_attempt_$i"; size=$sz }
            } else {
                $err = "size mismatch local=$sz remote=$remoteLen"
            }
        } else {
            $err = "curl exit=$code"
        }
        Log-Step ("attempt_${i}_fail") "$Target ($err)"
    }
    return [pscustomobject]@{ ok=$false; reason=$err; size=0 }
}

function Write-Todo {
    param(
        [string]$Name,
        [string]$Url,
        [string]$Target,
        [string]$ManualCmd,
        [string]$Error
    )
    $ts = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    $p = Join-Path $MANUAL "$Name.md"
    @"
# $Name — manual download required

- **timestamp:** $ts
- **failed URL / command:** ``$Url``
- **intended target path:** ``$Target``
- **error message:** $Error

## Exact command to run manually

```powershell
$ManualCmd
```

## Notes
- After the manual download completes, re-run ``scripts\download_full_cache.ps1`` —
  it will detect the existing file via Content-Length and record it as complete.
- If the URL pattern in this file is broken at the source, update the script
  and this TODO together.
"@ | Out-File -FilePath $p -Encoding utf8
}


# ---------------- main ----------------

Log-Step 'cache_start' 'full_dataset_download_pass'

$DOWNLOADS = @(
    @{ name = 'LIT_PCBA_SPLITS'
       url    = 'https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz'
       target = 'D:\hoangpc\VS-LeakKG\data\raw\LIT-PCBA\splits\AVE_unbiased.tgz'
       todo_cmd = 'curl.exe -L -C - -o D:\hoangpc\VS-LeakKG\data\raw\LIT-PCBA\splits\AVE_unbiased.tgz https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz'
       extract_safe = $true }
    @{ name = 'DEKOIS'
       url    = 'https://zenodo.org/records/8131256/files/DEKOIS2.zip?download=1'
       target = 'D:\hoangpc\VS-LeakKG\data\raw\DEKOIS\DEKOIS2.zip'
       todo_cmd = 'curl.exe -L -C - -o D:\hoangpc\VS-LeakKG\data\raw\DEKOIS\DEKOIS2.zip "https://zenodo.org/records/8131256/files/DEKOIS2.zip?download=1"'
       extract_safe = $false }
    @{ name = 'BindingDB'
       url    = 'https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202605_tsv.zip'
       target = 'D:\hoangpc\VS-LeakKG\data\raw\BindingDB\BindingDB_All_202605_tsv.zip'
       todo_cmd = 'curl.exe -L -C - -o D:\hoangpc\VS-LeakKG\data\raw\BindingDB\BindingDB_All_202605_tsv.zip "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202605_tsv.zip"'
       extract_safe = $false }
    @{ name = 'BayesBind'
       url    = 'https://storage.googleapis.com/bigbind_data/BayesBindV1.5.tar.gz'
       target = 'D:\hoangpc\VS-LeakKG\data\raw\BayesBind\BayesBindV1.5.tar.gz'
       todo_cmd = 'curl.exe -L -C - -o D:\hoangpc\VS-LeakKG\data\raw\BayesBind\BayesBindV1.5.tar.gz https://storage.googleapis.com/bigbind_data/BayesBindV1.5.tar.gz'
       extract_safe = $false }
    @{ name = 'BigBind'
       url    = 'https://storage.googleapis.com/bigbind_data/BigBindV1.5.tar.gz'
       target = 'D:\hoangpc\VS-LeakKG\data\raw\BigBind\BigBindV1.5.tar.gz'
       todo_cmd = 'curl.exe -L -C - -o D:\hoangpc\VS-LeakKG\data\raw\BigBind\BigBindV1.5.tar.gz https://storage.googleapis.com/bigbind_data/BigBindV1.5.tar.gz'
       extract_safe = $false }
)

$results = @{}
foreach ($d in $DOWNLOADS) {
    $r = Download-Resume -Url $d.url -Target $d.target
    $results[$d.name] = $r
    if (-not $r.ok) {
        Write-Todo -Name ("$($d.name)_TODO") -Url $d.url -Target $d.target `
                   -ManualCmd $d.todo_cmd -Error $r.reason
    }
}

# LIT-PCBA split archive: safe to extract in place if download succeeded.
$litSplit = $DOWNLOADS | Where-Object { $_.name -eq 'LIT_PCBA_SPLITS' } | Select-Object -First 1
if ($results['LIT_PCBA_SPLITS'].ok) {
    $tgz = $litSplit.target
    $outDir = Join-Path (Split-Path $tgz) 'AVE_unbiased'
    if (-not (Test-Path (Join-Path $outDir '.extracted_ok'))) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        Log-Step 'pre_extract' $tgz
        & tar.exe -xzf $tgz -C $outDir
        if ($LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Path (Join-Path $outDir '.extracted_ok') -Force | Out-Null
            Log-Step 'post_extract' $tgz
        } else {
            Log-Step 'extract_fail' $tgz
        }
    } else {
        Log-Step 'already_extracted' $tgz
    }
}

# PLINDER — only if gsutil is available AND free space >= 200 GB.
$gsutilCmd = Get-Command gsutil -ErrorAction SilentlyContinue
$freeGB = Get-FreeSpaceGB 'D'
$plinderTarget = 'D:\hoangpc\VS-LeakKG\data\raw\PLINDER'
$plinderManualCmd = @'
# 1) Install Google Cloud SDK + gsutil (https://cloud.google.com/sdk/docs/install).
# 2) From any shell, run:
cd D:\hoangpc\VS-LeakKG\data\raw\PLINDER
gsutil -m cp -r "gs://plinder/2024-04" "gs://plinder/2024-06" "gs://plinder/manifest.md" .
# 3) If the bucket is fully public over HTTPS, try also:
curl.exe -L -C - -o manifest.md "https://storage.googleapis.com/plinder/manifest.md"
'@
$plinderError = ''
if (-not $gsutilCmd) {
    $plinderError = "gsutil not installed on this host (gcloud SDK missing)"
} elseif ($freeGB -lt $MIN_FREE_GB_PLINDER) {
    $plinderError = "free=${freeGB}GB < ${MIN_FREE_GB_PLINDER}GB threshold for PLINDER"
}

if ($plinderError) {
    # Try anonymous HTTPS for the manifest only (cheap if the bucket is public).
    try {
        $man = Join-Path $plinderTarget 'manifest.md'
        if (-not (Test-Path $man)) {
            $r = Invoke-WebRequest -Uri 'https://storage.googleapis.com/plinder/manifest.md' `
                 -OutFile $man -UseBasicParsing -TimeoutSec 60 -ErrorAction Stop
            Log-Step 'plinder_manifest_only' $man
        }
    } catch {
        Log-Step 'plinder_manifest_fail' "$plinderTarget ($($_.Exception.Message))"
    }
    Write-Todo -Name 'PLINDER_GSUTIL_TODO' `
               -Url 'gsutil -m cp -r "gs://plinder/2024-04" "gs://plinder/2024-06" "gs://plinder/manifest.md" .' `
               -Target $plinderTarget `
               -ManualCmd $plinderManualCmd `
               -Error $plinderError
    $results['PLINDER'] = [pscustomobject]@{ ok=$false; reason=$plinderError; size=0 }
} else {
    Log-Step 'pre_plinder_gsutil' $plinderTarget
    Push-Location $plinderTarget
    & gsutil -m cp -r 'gs://plinder/2024-04' 'gs://plinder/2024-06' 'gs://plinder/manifest.md' .
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -eq 0) {
        Log-Step 'post_plinder_gsutil' $plinderTarget
        $results['PLINDER'] = [pscustomobject]@{ ok=$true; reason='gsutil_ok'; size=0 }
    } else {
        Log-Step 'plinder_gsutil_fail' "$plinderTarget (exit=$code)"
        Write-Todo -Name 'PLINDER_GSUTIL_TODO' `
                   -Url 'gsutil -m cp -r ...' `
                   -Target $plinderTarget `
                   -ManualCmd $plinderManualCmd `
                   -Error "gsutil exit=$code"
        $results['PLINDER'] = [pscustomobject]@{ ok=$false; reason="gsutil exit=$code"; size=0 }
    }
}

Log-Step 'cache_end' 'full_dataset_download_pass'

# Compact summary on stdout — final report is written elsewhere.
"== summary =="
foreach ($k in $results.Keys) {
    $r = $results[$k]
    "  {0,-20} ok={1} size={2,12:N0}B  reason={3}" -f $k, $r.ok, $r.size, $r.reason
}
"free_after_GB=$(Get-FreeSpaceGB 'D'); project_GB=$(Get-ProjectSizeGB)"
