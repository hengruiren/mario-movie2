#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
: "${DATA_ROOT:?Set DATA_ROOT to the MAGB directory}"
: "${LLM_MODEL_PATH:?Set LLM_MODEL_PATH to local Llama-3.1-8B-Instruct}"
DATA_ROOT="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$DATA_ROOT")"
LLM_MODEL_PATH="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$LLM_MODEL_PATH")"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/movies20_original_8epochs_seed42_run1}"
RUN_DIR="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RUN_DIR")"
MOVIES_SUBSET="$PROJECT_ROOT/splits/Movies/stage2_train20_seed42.json"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR already exists; choose a fresh directory: $RUN_DIR" >&2
  exit 1
fi
"$PYTHON" "$PROJECT_ROOT/scripts/audit_original.py"
"$PYTHON" "$PROJECT_ROOT/scripts/prepare_movies_subset.py" \
  --data-root "$DATA_ROOT" --output "$MOVIES_SUBSET" --subset-seed 42
"$PYTHON" - <<'PY'
import torch
import dgl
assert torch.cuda.is_available(), "A CUDA GPU is required for this launcher"
try:
    probe = dgl.graph(([0], [1]), num_nodes=2).to('cuda:0')
except dgl.DGLError as exc:
    raise RuntimeError('Install CUDA-enabled DGL; see MOVIES20_5EPOCH.md') from exc
print('GPU:', torch.cuda.get_device_name(0), '| DGL:', dgl.__version__)
PY

mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
cp "$MOVIES_SUBSET" "$RUN_DIR/stage2_targets.json"
cp "$PROJECT_ROOT/UPSTREAM.json" "$RUN_DIR/upstream.json"

if [[ -z "${STAGE1_FEATURE:-}" ]]; then
  MOVIES_TEXT_FEATURE="${MOVIES_TEXT_FEATURE:-$DATA_ROOT/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy}"
  "$PYTHON" "$PROJECT_ROOT/scripts/prepare_movies_inputs.py" \
    --data-root "$DATA_ROOT" --work-dir "$RUN_DIR" --text-feature "$MOVIES_TEXT_FEATURE"
  (
    cd "$RUN_DIR/stage1"
    DATA_ROOT="$RUN_DIR/data" "$PYTHON" "$PROJECT_ROOT/stage1/main.py" \
      --dataset Movies \
      --use_large_features \
      --checkpoint_folder "$RUN_DIR/stage1/check" \
      --n_epochs 80 \
      --batch_size 500 \
      --lr 0.001 \
      --lr_scheduler_gamma 0.95 \
      2>&1 | tee "$RUN_DIR/stage1/train.log"
  )
  STAGE1_FEATURE="$RUN_DIR/stage1/Movies_stage1_mlp.pth"
else
  STAGE1_FEATURE="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$STAGE1_FEATURE")"
  "$PYTHON" - "$STAGE1_FEATURE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
sidecar = Path(str(path) + '.json')
if sidecar.is_file():
    meta = json.loads(sidecar.read_text())
    if meta.get('method') != 'official_mario_infonce':
        raise ValueError('Use original InfoNCE embeddings; this file is not marked as original Mario')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if meta.get('feature_sha256') != digest.hexdigest():
        raise ValueError('Embedding differs from its provenance sidecar')
else:
    print('External embedding has no provenance sidecar. Its original-InfoNCE training origin must be known.')
PY
  printf '%s\n' "$STAGE1_FEATURE" > "$RUN_DIR/stage1_feature.txt"
fi

mkdir "$RUN_DIR/stage2"
"$PYTHON" "$PROJECT_ROOT/stage2/train_llm.py" \
  --dataset Movies \
  --data-root "$DATA_ROOT" \
  --seed 42 \
  --movies-subset "$RUN_DIR/stage2_targets.json" \
  --stage1-feature "$STAGE1_FEATURE" \
  --llm-model-path "$LLM_MODEL_PATH" \
  --output-dir "$RUN_DIR/stage2" \
  --epochs 8 --patience 2 \
  --batch-size 4 \
  --inference-policy router \
  2>&1 | tee "$RUN_DIR/stage2/train.log"
