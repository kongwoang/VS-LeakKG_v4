# Single source of truth for the dataset archive currently on Hugging Face.
# Bump these two variables to release a new version — every other script
# (fetch_dataset.sh, extract_datasets.sh, reproduce.sh) reads them from here.
#
# This file is sourced, not executed.
export DATASET_HF_REPO="kongwoang/VS_LeakKG"
export DATASET_ZIP="VS-LeakKG_raw_datasets_20260519.zip"

# Default download location. Override with $DATASET_CACHE_DIR if you want to
# keep the archive on a different mount.
: "${DATASET_CACHE_DIR:=${REPO_ROOT:-$PWD}/_dataset_cache}"
export DATASET_CACHE_DIR
