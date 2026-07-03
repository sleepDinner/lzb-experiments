#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/data0/hl/lzb-experiments/lzb_outputs_adaptive}"
TEST_JSON="${TEST_JSON:-/data0/lzb-change-vmunet/FMAE5.0/test_datasets_loc_small_mid_big.json}"
ROBUST_DATASETS="${ROBUST_DATASETS:-Columbia,NIST16,IMD2020,DSO-1,Korus}"
METHODS="${METHODS:-CAT-Net,MVSS-Net,PSCC-Net,IRIS0-SPAN,ManTraNet}"
PROFILE_FILE="${PROFILE_FILE:-$WORK_DIR/summary/adaptive_selected_profiles.tsv}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2026}"
PREDICT_WORKERS="${PREDICT_WORKERS:-8}"
ROBUST_PREP_WORKERS="${ROBUST_PREP_WORKERS:-8}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
REBUILD_ROBUST_LISTS="${REBUILD_ROBUST_LISTS:-0}"

export CUDA_VISIBLE_DEVICES
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

run_py() {
  local env_spec="$1"
  shift
  if [[ -n "$env_spec" ]]; then
    if [[ "$env_spec" == */* ]]; then
      conda run --no-capture-output -p "$env_spec" "$@"
    else
      conda run --no-capture-output -n "$env_spec" "$@"
    fi
  else
    "$@"
  fi
}

split_list() {
  printf "%s\n" "$1" | tr ',' ' '
}

profile_value() {
  local method="$1"
  local field="$2"
  awk -F '\t' -v method="$method" -v field="$field" '$1 == method {print $field; found=1} END {exit found ? 0 : 1}' "$PROFILE_FILE"
}

variant_list_file() {
  local dataset="$1"
  local variant="$2"
  if [[ "$variant" == "clean" ]]; then
    printf "%s/lists/tests/%s.txt" "$WORK_DIR" "$dataset"
  else
    printf "%s/lists/robust/%s_%s.txt" "$WORK_DIR" "$dataset" "$variant"
  fi
}

method_results_complete() {
  local dataset="$1"
  local method="$2"
  local result_root="$WORK_DIR/results_robust_datasets/$dataset/$method"
  for variant in clean jpeg_q100 jpeg_q70 jpeg_q50 gaussian_s5 gaussian_s10 gaussian_s15; do
    [[ -s "$result_root/$variant.json" ]] || return 1
  done
}

predict_batch_for_method() {
  local method="$1"
  local selected_batch="$2"
  case "$method" in
    "CAT-Net") printf "%s" "${CAT_PREDICT_BATCH_SIZE:-$selected_batch}" ;;
    "MVSS-Net") printf "%s" "${MVSS_PREDICT_BATCH_SIZE:-$selected_batch}" ;;
    "PSCC-Net") printf "%s" "${PSCC_PREDICT_BATCH_SIZE:-$selected_batch}" ;;
    "IRIS0-SPAN") printf "%s" "${SPAN_PREDICT_BATCH_SIZE:-$selected_batch}" ;;
    "ManTraNet") printf "%s" "${MANTRA_PREDICT_BATCH_SIZE:-$selected_batch}" ;;
    *) printf "%s" "$selected_batch" ;;
  esac
}

check_weights() {
  local method="$1"
  local checkpoint_dir="$2"
  case "$method" in
    "CAT-Net") [[ -s "$checkpoint_dir/best.pth.tar" ]] ;;
    "MVSS-Net") [[ -s "$checkpoint_dir/best.pth" ]] ;;
    "PSCC-Net") [[ -s "$checkpoint_dir/best_FENet.pth" && -s "$checkpoint_dir/best_SegNet.pth" && -s "$checkpoint_dir/best_ClsNet.pth" ]] ;;
    "IRIS0-SPAN") [[ -s "$checkpoint_dir/best.h5" ]] ;;
    "ManTraNet") [[ -s "$checkpoint_dir/best.pth" ]] ;;
    *) return 1 ;;
  esac
}

predict_one_variant() {
  local dataset="$1"
  local method="$2"
  local variant="$3"
  local list_file="$4"
  local image_size="$5"
  local batch_size="$6"
  local checkpoint_dir="$7"

  local pred_root="$WORK_DIR/predictions_robust_datasets/$dataset/$method/$variant"
  local result_root="$WORK_DIR/results_robust_datasets/$dataset/$method"
  mkdir -p "$pred_root" "$result_root"

  case "$method" in
    "CAT-Net")
      run_py "${CAT_ENV:-}" "$PYTHON_BIN" "$ROOT/CAT-Net/CAT-Net-main/tools/predict_lzb.py" \
        --list-file "$list_file" \
        --model-file "$checkpoint_dir/best.pth.tar" \
        --out-dir "$pred_root" \
        --image-size "$image_size" \
        --batch-size "$batch_size" \
        --workers "$PREDICT_WORKERS"
      ;;
    "MVSS-Net")
      run_py "${MVSS_ENV:-}" "$PYTHON_BIN" "$ROOT/MVSS-Net/MVSS-Net-master/predict_lzb.py" \
        --list-file "$list_file" \
        --model-file "$checkpoint_dir/best.pth" \
        --out-dir "$pred_root" \
        --image-size "$image_size" \
        --batch-size "$batch_size" \
        --workers "$PREDICT_WORKERS"
      ;;
    "PSCC-Net")
      run_py "${PSCC_ENV:-}" "$PYTHON_BIN" "$ROOT/PSCC-Net/PSCC-Net-main/predict_lzb.py" \
        --list-file "$list_file" \
        --checkpoint-dir "$checkpoint_dir" \
        --out-dir "$pred_root" \
        --image-size "$image_size" \
        --batch-size "$batch_size" \
        --workers "$PREDICT_WORKERS"
      ;;
    "IRIS0-SPAN")
      run_py "${SPAN_ENV:-}" "$PYTHON_BIN" "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main/predict_lzb.py" \
        --list-file "$list_file" \
        --model-file "$checkpoint_dir/best.h5" \
        --out-dir "$pred_root" \
        --image-size "$image_size" \
        --batch-size "$batch_size" \
        --workers "$PREDICT_WORKERS"
      ;;
    "ManTraNet")
      run_py "${MANTRA_ENV:-}" "$PYTHON_BIN" "$ROOT/ManTraNet/ManTraNet-pytorch-main/predict_lzb.py" \
        --list-file "$list_file" \
        --model-file "$checkpoint_dir/best.pth" \
        --out-dir "$pred_root" \
        --image-size "$image_size" \
        --batch-size "$batch_size" \
        --workers "$PREDICT_WORKERS"
      ;;
    *)
      echo "Unknown method: $method"
      return 1
      ;;
  esac

  "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
    --list-file "$list_file" \
    --pred-dir "$pred_root" \
    --out "$result_root/$variant.json"
}

predict_dataset_method() {
  local dataset="$1"
  local method="$2"

  if [[ "$SKIP_COMPLETED" == "1" ]] && method_results_complete "$dataset" "$method"; then
    echo "[$dataset][$method] completed robust outputs found; skipping."
    return 0
  fi

  local image_size batch_size checkpoint_dir predict_batch
  image_size="$(profile_value "$method" 2)" || {
    echo "Missing selected profile for $method in $PROFILE_FILE"
    return 1
  }
  batch_size="$(profile_value "$method" 3)"
  checkpoint_dir="$(profile_value "$method" 5)"
  predict_batch="$(predict_batch_for_method "$method" "$batch_size")"

  if ! check_weights "$method" "$checkpoint_dir"; then
    echo "Missing best checkpoint for $method under: $checkpoint_dir"
    return 1
  fi

  echo "[$dataset][$method] image_size=$image_size batch_size=$predict_batch checkpoint=$checkpoint_dir"
  for variant in clean jpeg_q100 jpeg_q70 jpeg_q50 gaussian_s5 gaussian_s10 gaussian_s15; do
    local list_file
    list_file="$(variant_list_file "$dataset" "$variant")"
    if [[ ! -s "$list_file" ]]; then
      echo "Missing list file: $list_file"
      return 1
    fi
    predict_one_variant "$dataset" "$method" "$variant" "$list_file" "$image_size" "$predict_batch" "$checkpoint_dir"
  done
}

if [[ ! -s "$PROFILE_FILE" ]]; then
  echo "Selected profile file not found: $PROFILE_FILE"
  echo "This script expects formal adaptive training outputs, not smoke-test checkpoints."
  exit 1
fi

prepare_args=(
  -m lzb_experiments.prepare_robust_lists
  --test-json "$TEST_JSON"
  --work-dir "$WORK_DIR"
  --datasets "$ROBUST_DATASETS"
  --seed "$SEED"
  --workers "$ROBUST_PREP_WORKERS"
)
if [[ "$REBUILD_ROBUST_LISTS" == "1" ]]; then
  prepare_args+=(--rebuild)
fi
"$PYTHON_BIN" "${prepare_args[@]}"

for dataset in $(split_list "$ROBUST_DATASETS"); do
  for method in $(split_list "$METHODS"); do
    predict_dataset_method "$dataset" "$method"
  done
  "$PYTHON_BIN" -m lzb_experiments.summarize_results \
    --results-dir "$WORK_DIR/results_robust_datasets/$dataset" \
    --out-csv "$WORK_DIR/summary/robust_${dataset}_summary.csv"
done

"$PYTHON_BIN" -m lzb_experiments.summarize_robust_datasets \
  --results-dir "$WORK_DIR/results_robust_datasets" \
  --out-csv "$WORK_DIR/summary/robust_all_datasets_summary.csv"

echo "Robust datasets finished."
echo "Results root: $WORK_DIR/results_robust_datasets"
echo "Predictions root: $WORK_DIR/predictions_robust_datasets"
echo "Combined summary: $WORK_DIR/summary/robust_all_datasets_summary.csv"
