#!/usr/bin/env bash
# VS-LeakKG — reproducible setup script.
#
# Mirrors what was run during the initial setup on 2026-05-18 from a Windows
# PowerShell session. On Windows, prefer running the equivalent PowerShell
# helpers in this directory (log_disk.ps1, fetch_dude.ps1) — this bash script
# is the cross-platform / Linux/macOS reference.
#
# Defaults assume the project lives at /d/hoangpc/VS-LeakKG (Git Bash) or
# /mnt/d/hoangpc/VS-LeakKG (WSL). Override with $VSLEAKKG_ROOT.

set -euo pipefail

VSLEAKKG_ROOT="${VSLEAKKG_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"

LOG_DIR="$VSLEAKKG_ROOT/outputs/logs"
DISK_LOG="$LOG_DIR/disk_usage.log"
RUN_LOG="$LOG_DIR/setup_data.log"
mkdir -p "$LOG_DIR"

# Redirect all stdout/stderr to the run log AND console.
exec > >(tee -a "$RUN_LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

check_disk() {
    # Args: required_gb context_label. Exits with code 2 if not enough free space.
    local need="${1:-$MIN_FREE_GB}"
    local label="${2:-(unspecified)}"
    local target_dir="$VSLEAKKG_ROOT"
    local free_kb
    free_kb=$(df -Pk "$target_dir" | awk 'NR==2 {print $4}')
    local free_gb=$(( free_kb / 1024 / 1024 ))
    echo "[$(ts)] check_disk: free=${free_gb}GB need>=${need}GB target='$target_dir' label='$label'"
    if [ "$free_gb" -lt "$need" ]; then
        echo "[$(ts)] check_disk: NOT ENOUGH FREE SPACE — stopping before '$label'" >&2
        exit 2
    fi
}

log_disk() {
    # Args: event target. Appends a structured block to disk_usage.log.
    local event="${1:-event}"
    local target="${2:-(unspecified)}"
    {
        echo "==== $(ts) ===="
        echo "event: $event"
        echo "target: $target"
        echo "cwd: $(pwd)"
        echo "-- df -h --"
        df -h "$VSLEAKKG_ROOT" 2>/dev/null || true
        echo "-- du -sh project --"
        du -sh "$VSLEAKKG_ROOT" 2>/dev/null || true
        if command -v lsblk >/dev/null 2>&1; then
            echo "-- lsblk --"
            lsblk 2>/dev/null || true
        fi
        if command -v free >/dev/null 2>&1; then
            echo "-- free -h --"
            free -h 2>/dev/null || true
        fi
        echo ""
    } >> "$DISK_LOG"
}

clone_or_status() {
    # Args: url dest_dir.
    local url="$1"
    local dest="$2"
    if [ -d "$dest/.git" ]; then
        echo "[$(ts)] repo exists: $dest"
        (cd "$dest" && git status --short && echo "HEAD=$(git rev-parse HEAD)")
    else
        log_disk pre_clone "$dest"
        git clone --depth 50 "$url" "$dest"
        log_disk post_clone "$dest"
    fi
}

download_resume() {
    # Args: url dest_path min_free_gb label.
    local url="$1"
    local dest="$2"
    local need="${3:-$MIN_FREE_GB}"
    local label="${4:-$(basename "$dest")}"
    mkdir -p "$(dirname "$dest")"
    check_disk "$need" "$label"
    log_disk pre_download "$label"
    # -C - resumes; --retry handles transient errors; -f fails on HTTP errors.
    curl -L -C - -o "$dest" --retry 3 --retry-delay 5 --connect-timeout 30 -f "$url"
    log_disk post_download "$label"
}

# ---------------- main ----------------
echo "[$(ts)] VS-LeakKG setup starting at $VSLEAKKG_ROOT"
log_disk setup_start "VS-LeakKG"

# Step 1 — directory tree.
mkdir -p "$VSLEAKKG_ROOT"/{external,data/processed,notebooks,outputs/logs,outputs/reports,outputs/tables,scripts,src/vsleakkg}
mkdir -p "$VSLEAKKG_ROOT"/data/raw/{LIT-PCBA,DUD-E,ChEMBL,BindingDB,BayesBind,BigBind,PLINDER,manual_downloads_needed}

# Step 3 — clone reference repos.
clone_or_status https://github.com/sievestack/LIT-PCBA-audit.git    "$VSLEAKKG_ROOT/external/LIT-PCBA-audit"
clone_or_status https://github.com/cthoyt/chembl-downloader.git      "$VSLEAKKG_ROOT/external/chembl-downloader"
clone_or_status https://github.com/plinder-org/plinder.git           "$VSLEAKKG_ROOT/external/plinder"

# Step 5A — ChEMBL 35 SQLite.
# Direct URL is the same one chembl-downloader uses (EBI mirror).
download_resume \
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_sqlite.tar.gz" \
    "$VSLEAKKG_ROOT/data/raw/ChEMBL/chembl_35_sqlite.tar.gz" \
    "$MIN_FREE_GB" "chembl_35_sqlite"

# Step 5B — BindingDB.
# WARNING: as of 2026-05-18 the static URL returns 404; BindingDB serves dumps
# via a JSP that mints date-suffixed filenames. See
# data/raw/manual_downloads_needed/BindingDB_TODO.md for the manual procedure.
echo "[$(ts)] BindingDB: manual download required — see data/raw/manual_downloads_needed/BindingDB_TODO.md"

# Step 5C — LIT-PCBA full_data archive (receptors + actives.smi + inactives.smi).
# The Unistra HEAD timed out on the first attempt but succeeded on retry (2026-05-18).
# Split files (train/val/test) are NOT in this archive — see the TODO.
download_resume \
    "https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz" \
    "$VSLEAKKG_ROOT/data/raw/LIT-PCBA/full_data.tgz" \
    "$MIN_FREE_GB" "litpcba_full_data"
echo "[$(ts)] LIT-PCBA splits still manual — see data/raw/manual_downloads_needed/LIT-PCBA_TODO.md"

# Step 5D — DUD-E ligand lists for all 102 targets.
DUDE_DIR="$VSLEAKKG_ROOT/data/raw/DUD-E"
mkdir -p "$DUDE_DIR"
log_disk pre_download "DUD-E ligand lists (102 targets)"
DUDE_TARGETS="aa2ar abl1 ace aces ada ada17 adrb1 adrb2 akt1 akt2 aldr ampc andr aofb bace1 braf cah2 casp3 cdk2 comt cp2c9 cp3a4 csf1r cxcr4 def dhi1 dpp4 drd3 dyr egfr esr1 esr2 fa10 fa7 fabp4 fak1 fgfr1 fkb1a fnta fpps gcr glcm gria2 grik1 hdac2 hdac8 hivint hivpr hivrt hmdh hs90a hxk4 igf1r inha ital jak2 kif11 kit kith kpcb lck lkha4 mapk2 mcr met mk01 mk10 mk14 mmp13 mp2k1 nos1 nram pa2ga parp1 pde5a pgh1 pgh2 plk1 pnph ppara ppard pparg prgr ptn1 pur2 pygm pyrd reni rock1 rxra sahh src tgfr1 thb thrb try1 tryb1 tysy urok vgfr2 wee1 xiap"
for t in $DUDE_TARGETS; do
    mkdir -p "$DUDE_DIR/$t"
    for f in actives_final.ism decoys_final.ism; do
        out="$DUDE_DIR/$t/$f"
        if [ -s "$out" ]; then
            continue
        fi
        curl -L -sS -f --retry 2 --retry-delay 2 --connect-timeout 30 \
            -o "$out" "https://dude.docking.org/targets/$t/$f" || \
            echo "[$(ts)] WARN: failed to download $t/$f"
    done
done
log_disk post_download "DUD-E ligand lists (102 targets)"

# Step 5E + 5F — manual TODOs only.
echo "[$(ts)] BayesBind/BigBind: see data/raw/manual_downloads_needed/BayesBind_BigBind_TODO.md"
echo "[$(ts)] PLINDER:           see data/raw/manual_downloads_needed/PLINDER_TODO.md"

log_disk setup_end "VS-LeakKG"
echo "[$(ts)] VS-LeakKG setup complete."
