#!/usr/bin/env bash
# Download the current dataset archive from Hugging Face into the local cache.
#
# Usage:
#   bash scripts/fetch_dataset.sh                       # → _dataset_cache/<zip>
#   bash scripts/fetch_dataset.sh /target/dir           # → /target/dir/<zip>
#
# Reads version + repo from scripts/dataset_version.sh.
# Requires:
#   - huggingface-cli on PATH  (pip install -U "huggingface_hub[cli]")
#   - HF_TOKEN env var         (repo is private)
#
# Idempotent: skips the download if the zip is already present.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/dataset_version.sh
. "${REPO_ROOT}/scripts/dataset_version.sh"

OUT_DIR="${1:-${DATASET_CACHE_DIR}}"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/${DATASET_ZIP}"

if [[ -f "${OUT}" ]]; then
  echo "Dataset already present: ${OUT}"
  echo "(delete the file to force re-download)"
  exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not on PATH." >&2
  echo "  pip install -U 'huggingface_hub[cli]'" >&2
  exit 4
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN env var not set — required because the dataset repo is private." >&2
  echo "  https://huggingface.co/settings/tokens" >&2
  echo "  export HF_TOKEN=hf_..." >&2
  exit 5
fi

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
echo "Downloading ${DATASET_ZIP}"
echo "  from: ${DATASET_HF_REPO}"
echo "  into: ${OUT_DIR}"
huggingface-cli download "${DATASET_HF_REPO}" "${DATASET_ZIP}" \
  --repo-type dataset \
  --local-dir "${OUT_DIR}" \
  --token "${HF_TOKEN}"
echo "Done: ${OUT}"
