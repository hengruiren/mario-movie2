#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
: "${DATA_ROOT:?Set DATA_ROOT to the MAGB directory}"
: "${LLM_MODEL_PATH:?Set LLM_MODEL_PATH to local Llama-3.1-8B-Instruct}"
DATA_ROOT="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$DATA_ROOT")"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/runs/movies20_clip_class_8epochs_run1}"
RUN_DIR="$("$PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RUN_DIR")"
ASSET_CACHE="${ASSET_CACHE:-$PROJECT_ROOT/cache/prototype_patch}"
CLIP_ALL_CACHE="${CLIP_ALL_CACHE:-$PROJECT_ROOT/cache/clip_all_nodes.pt}"
LLAMA_VISION_MODEL_PATH="${LLAMA_VISION_MODEL_PATH:-/workspace/models/Llama-3.2-11B-Vision-Instruct}"
LLAMA_VISION_REVISION="${LLAMA_VISION_REVISION:-9eb2daaa8597bf192a8b0e73f848f3a102794df5}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/workspace/models/clip-vit-base-patch16}"
CLIP_REVISION="${CLIP_REVISION:-57c216476eefef5ab752ec549e440a49ae4ae5f3}"
MOVIES_IMAGE_DIR="${MOVIES_IMAGE_DIR:-$DATA_ROOT/Movies/MoviesImages}"
MOVIES_TEXT_FEATURE="${MOVIES_TEXT_FEATURE:-$DATA_ROOT/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy}"
MOVIES_IMAGE_FEATURE="${MOVIES_IMAGE_FEATURE:-$DATA_ROOT/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
cd "$PROJECT_ROOT"
if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR already exists; choose a fresh directory: $RUN_DIR" >&2
  exit 1
fi
"$PYTHON" scripts/audit_original.py
"$PYTHON" scripts/prepare_movies_subset.py \
  --data-root "$DATA_ROOT" \
  --output "$PROJECT_ROOT/splits/Movies/stage2_train20_seed42.json" \
  --subset-seed 42
"$PYTHON" - <<'PY'
import torch
import dgl
assert torch.cuda.is_available(), 'A CUDA GPU is required for this launcher'
dgl.graph(([0], [1]), num_nodes=2).to('cuda:0')
print('GPU:', torch.cuda.get_device_name(0), '| DGL:', dgl.__version__)
PY
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
cp "$PROJECT_ROOT/splits/Movies/stage2_train20_seed42.json" "$RUN_DIR/stage2_targets.json"
cp "$PROJECT_ROOT/UPSTREAM.json" "$RUN_DIR/upstream.json"
cp "$PROJECT_ROOT/configs/prototype_patch.json" "$RUN_DIR/stage1_settings.json"
cp "$PROJECT_ROOT/configs/class_guided_patch.json" "$RUN_DIR/clip_class_settings.json"

# Frozen class descriptions and CLIP training-patch teacher. No node Llama
# feature re-extraction is performed; existing MAGB globals are the inputs.
"$PYTHON" scripts/prepare_prototype_patch.py \
  --mode all --data-root "$DATA_ROOT" \
  --movies-subset "$RUN_DIR/stage2_targets.json" \
  --llama-model "$LLAMA_VISION_MODEL_PATH" --llama-revision "$LLAMA_VISION_REVISION" \
  --clip-model "$CLIP_MODEL_PATH" --clip-revision "$CLIP_REVISION" \
  --image-dir "$MOVIES_IMAGE_DIR" --image-extension .jpg \
  --llama-batch-size 1 --clip-batch-size 32 --dtype bfloat16 \
  --local-files-only --output-dir "$ASSET_CACHE" \
  2>&1 | tee "$RUN_DIR/prepare.log"

"$PYTHON" stage1/train_prototype_patch.py \
  --data-root "$DATA_ROOT" --movies-subset "$RUN_DIR/stage2_targets.json" \
  --text-feature "$MOVIES_TEXT_FEATURE" --image-feature "$MOVIES_IMAGE_FEATURE" \
  --class-cache "$ASSET_CACHE/class_features.pt" --patch-cache "$ASSET_CACHE/clip_patches.pt" \
  --config "$RUN_DIR/stage1_settings.json" --output-dir "$RUN_DIR/stage1" \
  --method full --seed 123 \
  --base-epochs 80 --base-batch-size 500 \
  --finetune-epochs 20 --finetune-batch-size 128 --finetune-lr 0.0001 \
  --lambda-class 1.0 --lambda-fine 0.1 --class-temperature 0.07 \
  --class-only-epochs 5 --fine-ramp-epochs 5 --top-k 5 \
  2>&1 | tee "$RUN_DIR/stage1.log"

# Intermediate CLIP -> Mario ridge bridge -> CLASS-weighted refinement.
# This is the earlier CLIP Class method, with no Llama crops or CLIP pre-screen.
mkdir "$RUN_DIR/clip_class"
"$PYTHON" scripts/class_guided_patch.py \
  --mode all --data-root "$DATA_ROOT" --movies-subset "$RUN_DIR/stage2_targets.json" \
  --stage1-feature "$RUN_DIR/stage1/Movies_stage1_mlp.pth" \
  --class-prototypes "$RUN_DIR/stage1/class_prototypes.pt" \
  --clip-model "$CLIP_MODEL_PATH" --clip-revision "$CLIP_REVISION" \
  --clip-cache "$CLIP_ALL_CACHE" --image-dir "$MOVIES_IMAGE_DIR" \
  --image-extension .jpg --clip-batch-size 32 --local-files-only \
  --patch-query class --config "$RUN_DIR/clip_class_settings.json" \
  --output-dir "$RUN_DIR/clip_class" \
  2>&1 | tee "$RUN_DIR/clip_class/refine.log"

# Keep a dedicated pack of the source embeddings, trained Encoder and prototypes
# so later patch variants can be recombined without repeating Stage 1.
"$PYTHON" - "$RUN_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import tarfile
run = Path(sys.argv[1])
names = ['stage2_targets.json', 'stage1_settings.json', 'clip_class_settings.json',
         'stage1/last_encoder.pt', 'stage1/class_prototypes.pt',
         'stage1/Movies_stage1_mlp.pth', 'stage1/Movies_stage1_mlp.pth.json',
         'stage1/run_config.json', 'stage1/stage1_metrics.json',
         'clip_class/Movies_stage1_mlp.pth', 'clip_class/Movies_stage1_mlp.pth.json',
         'clip_class/clip_to_mario_bridge.pt', 'clip_class/refinement_metrics.json',
         'clip_class/run_config.json']
digests = {}
for name in names:
    h = hashlib.sha256()
    with (run / name).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    digests[name] = h.hexdigest()
(run / 'embedding_assets_sha256.json').write_text(json.dumps(digests, indent=2) + '\n')
with tarfile.open(run / 'embedding_assets.tar.gz', 'w:gz', compresslevel=1) as tar:
    for name in names + ['embedding_assets_sha256.json']:
        tar.add(run / name, arcname=name, recursive=False)
print('Saved embedding_assets.tar.gz; copy this archive off the GPU server after training.')
PY

# Keep the original Stage-2 model, objective, neighbor selection and inference.
mkdir "$RUN_DIR/stage2"
"$PYTHON" stage2/train_llm.py \
  --dataset Movies --data-root "$DATA_ROOT" --seed 42 \
  --movies-subset "$RUN_DIR/stage2_targets.json" \
  --stage1-feature "$RUN_DIR/clip_class/Movies_stage1_mlp.pth" \
  --llm-model-path "$LLM_MODEL_PATH" --output-dir "$RUN_DIR/stage2" \
  --epochs 8 --patience 2 --batch-size 4 --lr 0.00001 --top-k 10 \
  --inference-policy router \
  2>&1 | tee "$RUN_DIR/stage2/train.log"
