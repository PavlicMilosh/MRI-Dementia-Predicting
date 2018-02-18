#!/usr/bin/env bash
# Script to preprocess the dementia data set.
#
# usage:
#  ./preprocess_dementia.sh [data-dir]
set -e

if [ -z "$1" ]; then
  echo "Usage: download_and_preprocess_flowers.sh [data dir]"
  exit
fi

# Create the output and temporary directories.
DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}/raw-data"
mkdir -p "${DATA_DIR}"
mkdir -p "${SCRATCH_DIR}"

CURRENT_DIR=$(pwd)
cd "${DATA_DIR}"


# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}/train"
VALIDATION_DIRECTORY="${SCRATCH_DIR}/validation"

# Expands the data into the flower_photos/ directory and rename it as the
# train directory.
rm -rf "${TRAIN_DIRECTORY}" "${VALIDATION_DIRECTORY}"
mv images "${TRAIN_DIRECTORY}"


# Generate a list of 2 labels: demented, nondemented
LABELS_FILE="${SCRATCH_DIR}/labels.txt"
ls -1 "${TRAIN_DIRECTORY}" | grep -v 'LICENSE' | sed 's/\///' | sort > "${LABELS_FILE}"

# Generate the validation data set.
while read LABEL; do
  VALIDATION_DIR_FOR_LABEL="${VALIDATION_DIRECTORY}/${LABEL}"
  TRAIN_DIR_FOR_LABEL="${TRAIN_DIRECTORY}/${LABEL}"

  # Move the first randomly selected 75 images(20%) to the validation set.
  mkdir -p "${VALIDATION_DIR_FOR_LABEL}"
  VALIDATION_IMAGES=$(ls -1 "${TRAIN_DIR_FOR_LABEL}" | shuf | head -75)
  for IMAGE in ${VALIDATION_IMAGES}; do
    mv -f "${TRAIN_DIRECTORY}/${LABEL}/${IMAGE}" "${VALIDATION_DIR_FOR_LABEL}"
  done
done < "${LABELS_FILE}"


