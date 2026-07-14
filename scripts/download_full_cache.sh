#!/usr/bin/env bash
# Bash equivalent of scripts/download_full_cache.ps1.
# Same URLs, same target paths, same resume + skip-if-complete behavior.
# Run from any shell with curl in PATH. On Windows use Git Bash / WSL.

set -euo pipefail

ROOT="${VSLEAKKG_ROOT:-/d/hoangpc/VS-LeakKG}"
LOG="$ROOT/outputs/logs/full_dataset_download.log"
MANUAL="$ROOT/data/raw/manual_downloads_needed"
MIN_FREE_GB_DOWNLOAD=150
MIN_FREE_GB_PLINDER=200

mkdir -p "$(dirname "$LOG")" "$MANUAL"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

get_free_gb() {
    df -Pk "$ROOT" | awk 'NR==2 {printf "%.2f", $4/1024/1024}'
}

get_project_gb() {
    du -sb "$ROOT" 2>/dev/null | awk '{printf "%.2f", $1/1024/1024/1024}'
}

log_step() {
    local step="$1" target="$2"
    {
        echo "==== $(ts) ===="
        echo "step: $step"
        echo "target: $target"
        echo "cwd: $(pwd)"
        echo "-- df -h --"
        df -h "$ROOT" 2>/dev/null || true
        echo "-- project size --"
        echo "  $(get_project_gb) GB"
        echo ""
    } >> "$LOG"
}

remote_size() {
    local url="$1"
    curl -sIL --max-time 60 -A 'Mozilla/5.0 (VS-LeakKG cache)' "$url" \
        | awk 'BEGIN{IGNORECASE=1} /^content-length:/{n=$2} END{gsub(/[^0-9]/,"",n); print n}'
}

download_resume() {
    local url="$1" target="$2" min_free="${3:-$MIN_FREE_GB_DOWNLOAD}"
    mkdir -p "$(dirname "$target")"
    local free; free=$(get_free_gb)
    awk -v f="$free" -v m="$min_free" 'BEGIN{ if (f+0 < m+0) exit 1 }' || {
        log_step "skip_lowdisk" "$target"
        echo "SKIP_LOW_DISK $target"
        return 2
    }
    local remote_len; remote_len=$(remote_size "$url" || echo "")
    if [ -f "$target" ] && [ -n "$remote_len" ]; then
        local local_sz; local_sz=$(stat -c %s "$target" 2>/dev/null || echo 0)
        if [ "$local_sz" = "$remote_len" ]; then
            log_step "already_complete" "$target"
            echo "ALREADY_COMPLETE $target"
            return 0
        fi
    fi
    log_step "pre_download" "$target"
    local code=1
    for i in 1 2 3; do
        if curl -L -C - --retry 2 --retry-delay 5 --connect-timeout 60 \
                --max-time 7200 -A 'Mozilla/5.0 (VS-LeakKG cache)' \
                -f -o "$target" "$url"; then
            log_step "post_download_attempt_$i" "$target"
            echo "OK $target"
            return 0
        fi
        log_step "attempt_${i}_fail" "$target"
    done
    return 3
}

write_todo() {
    local name="$1" url="$2" target="$3" manual="$4" err="$5"
    cat > "$MANUAL/${name}.md" <<EOF
# $name — manual download required

- **timestamp:** $(ts)
- **failed URL / command:** \`$url\`
- **intended target path:** \`$target\`
- **error message:** $err

## Exact command to run manually

\`\`\`bash
$manual
\`\`\`
EOF
}

log_step "cache_start" "full_dataset_download_pass"

# --- HTTP downloads ---
declare -A RESULTS
download_pair() {
    local name="$1" url="$2" target="$3"
    if download_resume "$url" "$target"; then
        RESULTS[$name]=ok
    else
        RESULTS[$name]=fail
        write_todo "${name}_TODO" "$url" "$target" \
            "curl -L -C - -o \"$target\" \"$url\"" \
            "curl exit / size mismatch / low disk"
    fi
}

download_pair LIT_PCBA_SPLITS \
    "https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz" \
    "$ROOT/data/raw/LIT-PCBA/splits/AVE_unbiased.tgz"
download_pair DEKOIS \
    "https://zenodo.org/records/8131256/files/DEKOIS2.zip?download=1" \
    "$ROOT/data/raw/DEKOIS/DEKOIS2.zip"
download_pair BindingDB \
    "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202605_tsv.zip" \
    "$ROOT/data/raw/BindingDB/BindingDB_All_202605_tsv.zip"
download_pair BayesBind \
    "https://storage.googleapis.com/bigbind_data/BayesBindV1.5.tar.gz" \
    "$ROOT/data/raw/BayesBind/BayesBindV1.5.tar.gz"
download_pair BigBind \
    "https://storage.googleapis.com/bigbind_data/BigBindV1.5.tar.gz" \
    "$ROOT/data/raw/BigBind/BigBindV1.5.tar.gz"

# LIT-PCBA split: safe extract.
LIT_SPLIT="$ROOT/data/raw/LIT-PCBA/splits/AVE_unbiased.tgz"
if [ "${RESULTS[LIT_PCBA_SPLITS]:-}" = "ok" ] && [ -f "$LIT_SPLIT" ]; then
    out="$ROOT/data/raw/LIT-PCBA/splits/AVE_unbiased"
    if [ ! -f "$out/.extracted_ok" ]; then
        mkdir -p "$out"
        log_step "pre_extract" "$LIT_SPLIT"
        if tar xzf "$LIT_SPLIT" -C "$out"; then
            : > "$out/.extracted_ok"
            log_step "post_extract" "$LIT_SPLIT"
        else
            log_step "extract_fail" "$LIT_SPLIT"
        fi
    else
        log_step "already_extracted" "$LIT_SPLIT"
    fi
fi

# PLINDER — only if gsutil is on PATH AND free GB >= threshold.
free=$(get_free_gb)
PLINDER_TARGET="$ROOT/data/raw/PLINDER"
PLINDER_MANUAL='# 1) Install Google Cloud SDK + gsutil (https://cloud.google.com/sdk/docs/install).
# 2) Run:
cd '"$PLINDER_TARGET"'
gsutil -m cp -r "gs://plinder/2024-04" "gs://plinder/2024-06" "gs://plinder/manifest.md" .
# 3) If the bucket is fully public over HTTPS, try:
curl -L -C - -o manifest.md "https://storage.googleapis.com/plinder/manifest.md"'

plinder_err=""
if ! command -v gsutil >/dev/null 2>&1; then
    plinder_err="gsutil not installed"
elif awk -v f="$free" -v m="$MIN_FREE_GB_PLINDER" 'BEGIN{ if (f+0 < m+0) exit 1 }'; then
    plinder_err="free=${free}GB < ${MIN_FREE_GB_PLINDER}GB"
fi

if [ -n "$plinder_err" ]; then
    # Best-effort anonymous manifest pull.
    mkdir -p "$PLINDER_TARGET"
    if [ ! -f "$PLINDER_TARGET/manifest.md" ]; then
        curl -L -sS -o "$PLINDER_TARGET/manifest.md" \
             "https://storage.googleapis.com/plinder/manifest.md" || true
    fi
    write_todo PLINDER_GSUTIL_TODO \
        'gsutil -m cp -r "gs://plinder/2024-04" "gs://plinder/2024-06" "gs://plinder/manifest.md" .' \
        "$PLINDER_TARGET" "$PLINDER_MANUAL" "$plinder_err"
    RESULTS[PLINDER]=fail
else
    log_step "pre_plinder_gsutil" "$PLINDER_TARGET"
    ( cd "$PLINDER_TARGET" && gsutil -m cp -r "gs://plinder/2024-04" "gs://plinder/2024-06" "gs://plinder/manifest.md" . ) \
        && { RESULTS[PLINDER]=ok; log_step "post_plinder_gsutil" "$PLINDER_TARGET"; } \
        || { RESULTS[PLINDER]=fail; log_step "plinder_gsutil_fail" "$PLINDER_TARGET"; }
fi

log_step "cache_end" "full_dataset_download_pass"

echo "== summary =="
for k in "${!RESULTS[@]}"; do
    echo "  $k = ${RESULTS[$k]}"
done
echo "free_after_GB=$(get_free_gb)  project_GB=$(get_project_gb)"
