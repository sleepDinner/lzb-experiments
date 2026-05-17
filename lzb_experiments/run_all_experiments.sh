#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/data0/hl/lzb-experiments/lzb_outputs}"
TRAIN_ROOT="${TRAIN_ROOT:-/data0/lzb-change-vmunet/FinalTrainData/}"
TEST_JSON="${TEST_JSON:-/data0/lzb-change-vmunet/FMAE5.0/test_datasets_loc_small_mid_big.json}"
ROBUST_DATASET="${ROBUST_DATASET:-Casiav1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MANTRA_BATCH_SIZE="${MANTRA_BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"

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

prepare() {
  "$PYTHON_BIN" -m lzb_experiments.prepare_lists \
    --train-root "$TRAIN_ROOT" \
    --test-json "$TEST_JSON" \
    --work-dir "$WORK_DIR" \
    --robust-dataset "$ROBUST_DATASET"
}

predict_and_eval() {
  local method="$1"
  local env_name="$2"
  local project_dir="$3"
  local pred_script="$4"
  local weight_arg="$5"
  local weight_path="$6"
  local predict_batch="$7"

  local lists_dir="$WORK_DIR/lists"
  local pred_root="$WORK_DIR/predictions/$method"
  local result_root="$WORK_DIR/results/$method"
  local test_result_root="$WORK_DIR/test_results/$method"
  mkdir -p "$pred_root" "$result_root" "$test_result_root"

  for list_file in "$lists_dir"/tests/*.txt; do
    local dataset_name
    dataset_name="$(basename "$list_file" .txt)"
    run_py "$env_name" "$PYTHON_BIN" "$project_dir/$pred_script" \
      --list-file "$list_file" \
      "$weight_arg" "$weight_path" \
      --out-dir "$pred_root/tests/$dataset_name" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$predict_batch"
    "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
      --list-file "$list_file" \
      --pred-dir "$pred_root/tests/$dataset_name" \
      --out "$test_result_root/$dataset_name.json"
  done

  run_py "$env_name" "$PYTHON_BIN" "$project_dir/$pred_script" \
    --list-file "$lists_dir/tests/${ROBUST_DATASET}.txt" \
    "$weight_arg" "$weight_path" \
    --out-dir "$pred_root/clean" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$predict_batch"
  "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
    --list-file "$lists_dir/tests/${ROBUST_DATASET}.txt" \
    --pred-dir "$pred_root/clean" \
    --out "$result_root/clean.json"

  for variant in jpeg_q100 jpeg_q70 jpeg_q50 gaussian_s5 gaussian_s10 gaussian_s15; do
    run_py "$env_name" "$PYTHON_BIN" "$project_dir/$pred_script" \
      --list-file "$lists_dir/robust/${ROBUST_DATASET}_${variant}.txt" \
      "$weight_arg" "$weight_path" \
      --out-dir "$pred_root/$variant" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$predict_batch"
    "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
      --list-file "$lists_dir/robust/${ROBUST_DATASET}_${variant}.txt" \
      --pred-dir "$pred_root/$variant" \
      --out "$result_root/$variant.json"
  done
}

prepare

run_py "${CAT_ENV:-}" "$PYTHON_BIN" "$ROOT/CAT-Net/CAT-Net-main/tools/train_lzb.py" \
  --train-list "$WORK_DIR/lists/train.txt" \
  --val-list "$WORK_DIR/lists/val.txt" \
  --out-dir "$WORK_DIR/checkpoints/catnet" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE"
predict_and_eval "CAT-Net" "${CAT_ENV:-}" "$ROOT/CAT-Net/CAT-Net-main" "tools/predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints/catnet/best.pth.tar" "$BATCH_SIZE"

run_py "${MVSS_ENV:-}" "$PYTHON_BIN" "$ROOT/MVSS-Net/MVSS-Net-master/train_lzb.py" \
  --train-list "$WORK_DIR/lists/train.txt" \
  --val-list "$WORK_DIR/lists/val.txt" \
  --out-dir "$WORK_DIR/checkpoints/mvssnet" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE"
predict_and_eval "MVSS-Net" "${MVSS_ENV:-}" "$ROOT/MVSS-Net/MVSS-Net-master" "predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints/mvssnet/best.pth" "$BATCH_SIZE"

run_py "${PSCC_ENV:-}" "$PYTHON_BIN" "$ROOT/PSCC-Net/PSCC-Net-main/train_lzb.py" \
  --train-list "$WORK_DIR/lists/train.txt" \
  --val-list "$WORK_DIR/lists/val.txt" \
  --out-dir "$WORK_DIR/checkpoints/psccnet" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE"
predict_and_eval "PSCC-Net" "${PSCC_ENV:-}" "$ROOT/PSCC-Net/PSCC-Net-main" "predict_lzb.py" "--checkpoint-dir" "$WORK_DIR/checkpoints/psccnet" "$BATCH_SIZE"

run_py "${SPAN_ENV:-}" "$PYTHON_BIN" "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main/train_lzb.py" \
  --train-list "$WORK_DIR/lists/train.txt" \
  --val-list "$WORK_DIR/lists/val.txt" \
  --out-dir "$WORK_DIR/checkpoints/span" \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --image-size "$IMAGE_SIZE"
predict_and_eval "IRIS0-SPAN" "${SPAN_ENV:-}" "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main" "predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints/span/best.h5" "1"

run_py "${MANTRA_ENV:-}" "$PYTHON_BIN" "$ROOT/ManTraNet/ManTraNet-pytorch-main/train_lzb.py" \
  --train-list "$WORK_DIR/lists/train.txt" \
  --val-list "$WORK_DIR/lists/val.txt" \
  --out-dir "$WORK_DIR/checkpoints/mantranet" \
  --epochs "$EPOCHS" \
  --batch-size "$MANTRA_BATCH_SIZE" \
  --image-size "$IMAGE_SIZE"
predict_and_eval "ManTraNet" "${MANTRA_ENV:-}" "$ROOT/ManTraNet/ManTraNet-pytorch-main" "predict_lzb.py" "--model-file" "$WORK_DIR/checkpoints/mantranet/best.pth" "$MANTRA_BATCH_SIZE"

"$PYTHON_BIN" -m lzb_experiments.summarize_results \
  --results-dir "$WORK_DIR/results" \
  --out-csv "$WORK_DIR/summary/robust_summary.csv"

echo "All experiments finished."
echo "Test results: $WORK_DIR/test_results"
echo "Robustness summary: $WORK_DIR/summary/robust_summary.csv"
