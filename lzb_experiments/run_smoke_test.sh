#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/data0/hl/lzb-experiments/lzb_smoke_outputs}"
TRAIN_ROOT="${TRAIN_ROOT:-/data0/lzb-change-vmunet/FinalTrainData/}"
TEST_JSON="${TEST_JSON:-/data0/lzb-change-vmunet/FMAE5.0/test_datasets_loc_small_mid_big.json}"
ROBUST_DATASET="${ROBUST_DATASET:-Casiav1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2026}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MANTRA_BATCH_SIZE="${MANTRA_BATCH_SIZE:-1}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-4}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-2}"
SMOKE_TEST_SAMPLES="${SMOKE_TEST_SAMPLES:-2}"
REBUILD_LISTS="${REBUILD_LISTS:-0}"

run_py() {
  local env_spec="$1"
  shift
  if [[ -n "$env_spec" ]]; then
    if [[ "$env_spec" == */* ]]; then
      conda run -p "$env_spec" "$@"
    else
      conda run -n "$env_spec" "$@"
    fi
  else
    "$@"
  fi
}

base_lists_ready() {
  [[ -s "$WORK_DIR/lists/filter_report.json" ]] || return 1
  [[ -s "$WORK_DIR/lists/train.txt" ]] || return 1
  [[ -s "$WORK_DIR/lists/val.txt" ]] || return 1
  [[ -s "$WORK_DIR/lists/tests/${ROBUST_DATASET}.txt" ]] || return 1
  [[ -s "$WORK_DIR/lists/robust/${ROBUST_DATASET}_jpeg_q70.txt" ]] || return 1
}

smoke_lists_ready() {
  local smoke_dir="$WORK_DIR/lists/smoke"
  [[ -s "$smoke_dir/train.txt" ]] || return 1
  [[ -s "$smoke_dir/val.txt" ]] || return 1
  [[ -s "$smoke_dir/${ROBUST_DATASET}_clean.txt" ]] || return 1
  [[ -s "$smoke_dir/${ROBUST_DATASET}_jpeg_q70.txt" ]] || return 1
}

make_smoke_lists() {
  if [[ "$REBUILD_LISTS" != "1" ]] && smoke_lists_ready; then
    echo "Reusing existing smoke lists under: $WORK_DIR/lists/smoke"
    echo "Set REBUILD_LISTS=1 to rebuild base and smoke lists."
    return 0
  fi

  if [[ "$REBUILD_LISTS" != "1" ]] && base_lists_ready; then
    echo "Reusing existing base lists under: $WORK_DIR/lists"
  else
    echo "Preparing base train/val/test/robust lists under: $WORK_DIR/lists"
    "$PYTHON_BIN" -m lzb_experiments.prepare_lists \
      --train-root "$TRAIN_ROOT" \
      --test-json "$TEST_JSON" \
      --work-dir "$WORK_DIR" \
      --robust-dataset "$ROBUST_DATASET" \
      --seed "$SEED"
  fi

  local smoke_dir="$WORK_DIR/lists/smoke"
  mkdir -p "$smoke_dir"
  head -n "$SMOKE_TRAIN_SAMPLES" "$WORK_DIR/lists/train.txt" > "$smoke_dir/train.txt"
  head -n "$SMOKE_VAL_SAMPLES" "$WORK_DIR/lists/val.txt" > "$smoke_dir/val.txt"
  head -n "$SMOKE_TEST_SAMPLES" "$WORK_DIR/lists/tests/${ROBUST_DATASET}.txt" > "$smoke_dir/${ROBUST_DATASET}_clean.txt"
  head -n "$SMOKE_TEST_SAMPLES" "$WORK_DIR/lists/robust/${ROBUST_DATASET}_jpeg_q70.txt" > "$smoke_dir/${ROBUST_DATASET}_jpeg_q70.txt"
}

predict_eval_smoke() {
  local method="$1"
  local env_name="$2"
  local project_dir="$3"
  local pred_script="$4"
  local weight_arg="$5"
  local weight_path="$6"
  local predict_batch="$7"

  local smoke_dir="$WORK_DIR/lists/smoke"
  local pred_root="$WORK_DIR/predictions_smoke/$method"
  local result_root="$WORK_DIR/results_smoke/$method"
  mkdir -p "$pred_root" "$result_root"

  run_py "$env_name" "$PYTHON_BIN" "$project_dir/$pred_script" \
    --list-file "$smoke_dir/${ROBUST_DATASET}_clean.txt" \
    "$weight_arg" "$weight_path" \
    --out-dir "$pred_root/clean" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$predict_batch" \
    --workers 0
  "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
    --list-file "$smoke_dir/${ROBUST_DATASET}_clean.txt" \
    --pred-dir "$pred_root/clean" \
    --out "$result_root/clean.json"

  run_py "$env_name" "$PYTHON_BIN" "$project_dir/$pred_script" \
    --list-file "$smoke_dir/${ROBUST_DATASET}_jpeg_q70.txt" \
    "$weight_arg" "$weight_path" \
    --out-dir "$pred_root/jpeg_q70" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$predict_batch" \
    --workers 0
  "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
    --list-file "$smoke_dir/${ROBUST_DATASET}_jpeg_q70.txt" \
    --pred-dir "$pred_root/jpeg_q70" \
    --out "$result_root/jpeg_q70.json"
}

make_smoke_lists

run_py "${CAT_ENV:-}" "$PYTHON_BIN" "$ROOT/CAT-Net/CAT-Net-main/tools/train_lzb.py" \
  --train-list "$WORK_DIR/lists/smoke/train.txt" \
  --val-list "$WORK_DIR/lists/smoke/val.txt" \
  --out-dir "$WORK_DIR/checkpoints_smoke/catnet" \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --workers 0 \
  --seed "$SEED"
predict_eval_smoke "CAT-Net" "${CAT_ENV:-}" "$ROOT/CAT-Net/CAT-Net-main" "tools/predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints_smoke/catnet/best.pth.tar" "$BATCH_SIZE"

run_py "${MVSS_ENV:-}" "$PYTHON_BIN" "$ROOT/MVSS-Net/MVSS-Net-master/train_lzb.py" \
  --train-list "$WORK_DIR/lists/smoke/train.txt" \
  --val-list "$WORK_DIR/lists/smoke/val.txt" \
  --out-dir "$WORK_DIR/checkpoints_smoke/mvssnet" \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --workers 0 \
  --seed "$SEED"
predict_eval_smoke "MVSS-Net" "${MVSS_ENV:-}" "$ROOT/MVSS-Net/MVSS-Net-master" "predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints_smoke/mvssnet/best.pth" "$BATCH_SIZE"

run_py "${PSCC_ENV:-}" "$PYTHON_BIN" "$ROOT/PSCC-Net/PSCC-Net-main/train_lzb.py" \
  --train-list "$WORK_DIR/lists/smoke/train.txt" \
  --val-list "$WORK_DIR/lists/smoke/val.txt" \
  --out-dir "$WORK_DIR/checkpoints_smoke/psccnet" \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --workers 0 \
  --seed "$SEED"
predict_eval_smoke "PSCC-Net" "${PSCC_ENV:-}" "$ROOT/PSCC-Net/PSCC-Net-main" "predict_lzb.py" "--checkpoint-dir" "$WORK_DIR/checkpoints_smoke/psccnet" "$BATCH_SIZE"

run_py "${SPAN_ENV:-}" "$PYTHON_BIN" "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main/train_lzb.py" \
  --train-list "$WORK_DIR/lists/smoke/train.txt" \
  --val-list "$WORK_DIR/lists/smoke/val.txt" \
  --out-dir "$WORK_DIR/checkpoints_smoke/span" \
  --epochs 1 \
  --batch-size 1 \
  --image-size "$IMAGE_SIZE" \
  --workers 0 \
  --seed "$SEED"
predict_eval_smoke "IRIS0-SPAN" "${SPAN_ENV:-}" "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main" "predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints_smoke/span/best.h5" "1"

run_py "${MANTRA_ENV:-}" "$PYTHON_BIN" "$ROOT/ManTraNet/ManTraNet-pytorch-main/train_lzb.py" \
  --train-list "$WORK_DIR/lists/smoke/train.txt" \
  --val-list "$WORK_DIR/lists/smoke/val.txt" \
  --out-dir "$WORK_DIR/checkpoints_smoke/mantranet" \
  --epochs 1 \
  --batch-size "$MANTRA_BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --workers 0 \
  --seed "$SEED"
predict_eval_smoke "ManTraNet" "${MANTRA_ENV:-}" "$ROOT/ManTraNet/ManTraNet-pytorch-main" "predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints_smoke/mantranet/best.pth" "$MANTRA_BATCH_SIZE"

"$PYTHON_BIN" -m lzb_experiments.summarize_results \
  --results-dir "$WORK_DIR/results_smoke" \
  --out-csv "$WORK_DIR/summary/smoke_summary.csv"

echo "Smoke test finished."
echo "Smoke results: $WORK_DIR/results_smoke"
echo "Smoke summary: $WORK_DIR/summary/smoke_summary.csv"
