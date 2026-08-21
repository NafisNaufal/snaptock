#!/usr/bin/env bash
# Fetch the nota-pembelian COCO export into data/raw/.
#
#   cp .env.example .env && $EDITOR .env      # set ROBOFLOW_API_KEY
#   bash ml/download_dataset.sh
#
# ~331 MB. Idempotent: skips the download if data/raw/train already exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

[[ -f .env ]] && set -a && source .env && set +a

: "${ROBOFLOW_API_KEY:?set ROBOFLOW_API_KEY in .env (see .env.example)}"
WORKSPACE="${ROBOFLOW_WORKSPACE:-ocr-fqwqd}"
PROJECT="${ROBOFLOW_PROJECT:-nota-pembelian}"
VERSION="${ROBOFLOW_VERSION:-9}"

DEST="data/raw"
if [[ -d "${DEST}/train" ]]; then
  echo "${DEST}/train already present ($(ls "${DEST}/train" | wc -l | tr -d ' ') files) — skipping."
  exit 0
fi

mkdir -p "${DEST}"
echo "resolving export link for ${WORKSPACE}/${PROJECT} v${VERSION}..."
LINK=$(curl -s --fail \
  "https://api.roboflow.com/${WORKSPACE}/${PROJECT}/${VERSION}/coco?api_key=${ROBOFLOW_API_KEY}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["export"]["link"])')

echo "downloading..."
curl -L --fail --progress-bar -o "${DEST}/export.zip" "${LINK}"
unzip -q -o "${DEST}/export.zip" -d "${DEST}"
rm -f "${DEST}/export.zip"

echo "done: ${DEST}/train ($(ls "${DEST}/train" | wc -l | tr -d ' ') files)"
