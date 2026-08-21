#!/usr/bin/env bash
# Fine-tune PP-OCRv5 mobile recognition on handwritten Indonesian nota.
#
#   bash ml/finetune_rec.sh /path/to/workdir
#
# Idempotent: re-running skips the clone and the weight download.
# Everything below is pinned. PaddleOCR's config schema changed materially
# between 2.x and 3.x, so an unpinned clone will eventually break this config.

set -euo pipefail

PADDLEOCR_TAG="v3.7.0"
PRETRAIN_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams"
WORKDIR="${1:-$(pwd)/work}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_SRC="${REPO_ROOT}/ml/configs/nota_rec_v5_mobile.yml"
REC_DATA="${REC_DATA:-${REPO_ROOT}/data/rec}"

echo "workdir   : ${WORKDIR}"
echo "rec data  : ${REC_DATA}"

for f in train_rec.txt val_rec.txt test_rec.txt dict.txt split_manifest.json; do
  [[ -f "${REC_DATA}/${f}" ]] || {
    echo "ERROR: ${REC_DATA}/${f} missing. Run ml/prepare_dataset.py first." >&2
    exit 1
  }
done

# Refuse to train on a leaking split. The notebook's split put 96% of val
# crops in train; this is the guard that stops that ever shipping again.
python3 - "${REC_DATA}/split_manifest.json" <<'PYEOF'
import json, sys, itertools
m = json.load(open(sys.argv[1]))
r = m["receipts"]
for a, b in itertools.combinations(("train", "val", "test"), 2):
    overlap = set(r[a]) & set(r[b])
    if overlap:
        sys.exit(f"FATAL: {len(overlap)} receipts shared between {a} and {b}")
print(f"split OK  : grouped by {m['grouped_by']}; "
      f"{len(r['train'])}/{len(r['val'])}/{len(r['test'])} receipts, disjoint")
if m["grouped_by"] != "writer":
    print("WARNING   : split is grouped by receipt, not writer. The same few "
          "hands appear in every split, so val accuracy will overstate\n"
          "            real-world performance. Supply --writer-map for the honest number.")
PYEOF

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

if [[ ! -d PaddleOCR ]]; then
  git clone --depth 1 --branch "${PADDLEOCR_TAG}" \
    https://github.com/PaddlePaddle/PaddleOCR.git
fi
cd PaddleOCR

mkdir -p pretrain
PRETRAIN_FILE="pretrain/$(basename "${PRETRAIN_URL}")"
[[ -f "${PRETRAIN_FILE}" ]] || curl -L --fail -o "${PRETRAIN_FILE}" "${PRETRAIN_URL}"

# The config uses ./data/rec/ relative to the PaddleOCR root.
mkdir -p data
[[ -e data/rec ]] || ln -s "${REC_DATA}" data/rec

cp "${CONFIG_SRC}" configs/rec/nota_rec_v5_mobile.yml

echo
echo "=== train ==="
python3 tools/train.py -c configs/rec/nota_rec_v5_mobile.yml

echo
echo "=== export inference model ==="
python3 tools/export_model.py -c configs/rec/nota_rec_v5_mobile.yml \
  -o Global.pretrained_model=./output/nota_rec_v5_mobile/best_accuracy \
     Global.save_inference_dir=./output/nota_rec_v5_mobile_infer/

echo
echo "inference model -> ${WORKDIR}/PaddleOCR/output/nota_rec_v5_mobile_infer/"
echo "next: evaluate on the held-out TEST split (never used during training):"
echo "  python3 tools/eval.py -c configs/rec/nota_rec_v5_mobile.yml \\"
echo "    -o Global.checkpoints=./output/nota_rec_v5_mobile/best_accuracy \\"
echo "       Eval.dataset.label_file_list=[./data/rec/test_rec.txt]"
