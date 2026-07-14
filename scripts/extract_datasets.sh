#!/usr/bin/env bash
# Restore data/raw/ from the dataset archive on Hugging Face.
#
# Usage:
#   bash scripts/extract_datasets.sh                          # use the cached zip
#   bash scripts/extract_datasets.sh /path/to/<dataset>.zip   # explicit path
#
# If no argument is given, looks for the current version's zip in
# $DATASET_CACHE_DIR (default: _dataset_cache/<zip>). The version + filename
# come from scripts/dataset_version.sh — bump that file to release a new one.
#
# What this does:
#   1. Unzips the outer archive into a temp staging dir.
#   2. Merges raw/ into data/raw/ (existing files are preserved).
#   3. Re-extracts every inner archive into <dataset>/extracted/.
#
# Idempotent: skips any inner archive whose extracted/ target is non-empty.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/dataset_version.sh
. "${REPO_ROOT}/scripts/dataset_version.sh"

ZIP="${1:-${DATASET_CACHE_DIR}/${DATASET_ZIP}}"
if [[ ! -f "${ZIP}" ]]; then
  echo "Dataset archive not found: ${ZIP}" >&2
  echo "Fetch it first:" >&2
  echo "  bash scripts/fetch_dataset.sh" >&2
  exit 2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

echo "[1/3] Unzipping outer archive -> ${STAGE}"
unzip -q -o "${ZIP}" -d "${STAGE}"

# The archive may root at either "raw/..." or "<dated_dir>/raw/..." depending
# on which packaging run produced it. Auto-detect.
RAW_SRC=""
if [[ -d "${STAGE}/raw" ]]; then
  RAW_SRC="${STAGE}/raw"
else
  RAW_SRC="$(find "${STAGE}" -maxdepth 3 -type d -name raw | head -n 1 || true)"
fi
if [[ -z "${RAW_SRC}" ]] || [[ ! -d "${RAW_SRC}" ]]; then
  echo "Could not find raw/ inside the zip" >&2
  exit 3
fi
STAGE_ROOT="$(dirname "${RAW_SRC}")"

echo "[2/3] Merging raw/ into ${REPO_ROOT}/data/raw/"
mkdir -p "${REPO_ROOT}/data/raw"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${RAW_SRC}/" "${REPO_ROOT}/data/raw/"
else
  cp -a "${RAW_SRC}/." "${REPO_ROOT}/data/raw/"
fi

# Surface the proposal PDF and run-specific manifest at the repo root
# (these never go into Git — .gitignore excludes them).
[[ -f "${STAGE_ROOT}/VS_LeakKG_proposal.pdf" ]] && cp -n "${STAGE_ROOT}/VS_LeakKG_proposal.pdf" "${REPO_ROOT}/" || true
[[ -f "${STAGE_ROOT}/data_MANIFEST_run_specific.md" ]] && cp -n "${STAGE_ROOT}/data_MANIFEST_run_specific.md" "${REPO_ROOT}/data/MANIFEST.run_specific.md" || true

ROOT="${REPO_ROOT}/data/raw"

extract_tar() {
  local archive="$1" target="$2"
  if [[ -f "${archive}" ]]; then
    if [[ ! -d "${target}" ]] || [[ -z "$(ls -A "${target}" 2>/dev/null)" ]]; then
      echo "  tar -> ${target}"
      mkdir -p "${target}"
      tar -xf "${archive}" -C "${target}"
    else
      echo "  skip (already extracted): ${target}"
    fi
  fi
}

extract_zip() {
  local archive="$1" target="$2"
  if [[ -f "${archive}" ]]; then
    if [[ ! -d "${target}" ]] || [[ -z "$(ls -A "${target}" 2>/dev/null)" ]]; then
      echo "  zip -> ${target}"
      mkdir -p "${target}"
      unzip -q "${archive}" -d "${target}"
    else
      echo "  skip (already extracted): ${target}"
    fi
  fi
}

echo "[3/3] Extracting inner archives"
extract_tar "${ROOT}/ChEMBL/chembl_35_sqlite.tar.gz"          "${ROOT}/ChEMBL/extracted"
extract_zip "${ROOT}/BindingDB/BindingDB_All_202605_tsv.zip"  "${ROOT}/BindingDB/extracted"
extract_tar "${ROOT}/PBDBind/P-L.tar.gz"                      "${ROOT}/PBDBind/extracted"
extract_tar "${ROOT}/PBDBind/index.tar.gz"                    "${ROOT}/PBDBind/extracted"
extract_tar "${ROOT}/LIT-PCBA/full_data.tgz"                  "${ROOT}/LIT-PCBA/extracted"
extract_tar "${ROOT}/BayesBind/BayesBindV1.5.tar.gz"          "${ROOT}/BayesBind/extracted"
extract_zip "${ROOT}/DEKOIS/DEKOIS2.zip"                      "${ROOT}/DEKOIS/extracted"
extract_tar "${ROOT}/BigBind/BigBindV1.5.tar.gz"              "${ROOT}/BigBind/extracted"

echo "Done. data/raw/ is ready."
