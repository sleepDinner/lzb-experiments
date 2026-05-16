#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-/data0/hl/lzb-experiments/lzb_outputs_adaptive}"
TRAIN_ROOT="${TRAIN_ROOT:-/data0/lzb-change-vmunet/FinalTrainData/}"
TEST_JSON="${TEST_JSON:-/data0/lzb-change-vmunet/FMAE5.0/test_datasets_loc_small_mid_big.json}"
ROBUST_DATASET="${ROBUST_DATASET:-Casiav1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GLOBAL_EPOCHS="${EPOCHS:-}"
CAT_EPOCHS="${CAT_EPOCHS:-${GLOBAL_EPOCHS:-200}}"
MVSS_EPOCHS="${MVSS_EPOCHS:-${GLOBAL_EPOCHS:-100}}"
PSCC_EPOCHS="${PSCC_EPOCHS:-${GLOBAL_EPOCHS:-25}}"
SPAN_EPOCHS="${SPAN_EPOCHS:-${GLOBAL_EPOCHS:-500}}"
MANTRA_EPOCHS="${MANTRA_EPOCHS:-${GLOBAL_EPOCHS:-100}}"
WORKERS="${WORKERS:-8}"
CAT_WORKERS="${CAT_WORKERS:-4}"
SPAN_WORKERS="${SPAN_WORKERS:-4}"
PROFILE_FILE="$WORK_DIR/summary/adaptive_selected_profiles.tsv"
STREAM_ATTEMPT_LOGS="${STREAM_ATTEMPT_LOGS:-1}"
REBUILD_LISTS="${REBUILD_LISTS:-0}"

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

lists_ready() {
  local lists_dir="$WORK_DIR/lists"
  [[ -s "$lists_dir/filter_report.json" ]] || return 1
  [[ -s "$lists_dir/train.txt" ]] || return 1
  [[ -s "$lists_dir/val.txt" ]] || return 1
  [[ -s "$lists_dir/tests/${ROBUST_DATASET}.txt" ]] || return 1
  for variant in jpeg_q100 jpeg_q70 jpeg_q50 gaussian_s5 gaussian_s10 gaussian_s15; do
    [[ -s "$lists_dir/robust/${ROBUST_DATASET}_${variant}.txt" ]] || return 1
  done
  compgen -G "$lists_dir/tests/*.txt" > /dev/null || return 1
}

prepare() {
  if [[ "$REBUILD_LISTS" != "1" ]] && lists_ready; then
    echo "Reusing existing lists under: $WORK_DIR/lists"
    echo "Set REBUILD_LISTS=1 to rebuild train/val/test/robust lists."
    return 0
  fi
  echo "Preparing train/val/test/robust lists under: $WORK_DIR/lists"
  "$PYTHON_BIN" -m lzb_experiments.prepare_lists \
    --train-root "$TRAIN_ROOT" \
    --test-json "$TEST_JSON" \
    --work-dir "$WORK_DIR" \
    --robust-dataset "$ROBUST_DATASET"
}

is_oom_log() {
  local log_file="$1"
  grep -Eiq \
    "CUDA out of memory|out of memory|ResourceExhaustedError|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|failed to allocate|Cannot allocate memory|DefaultCPUAllocator|OOM|DataLoader worker .* is killed by signal: Killed|worker.*killed by signal|Bus error|No space left on device|shared memory" \
    "$log_file"
}

record_profile() {
  local method="$1"
  local image_size="$2"
  local batch_size="$3"
  local lr="$4"
  local out_dir="$5"
  local workers="$6"
  mkdir -p "$(dirname "$PROFILE_FILE")"
  if [[ ! -f "$PROFILE_FILE" ]]; then
    printf "method\timage_size\tbatch_size\tlr\tcheckpoint_dir\tworkers\n" > "$PROFILE_FILE"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$method" "$image_size" "$batch_size" "$lr" "$out_dir" "$workers" >> "$PROFILE_FILE"
}

train_adaptive() {
  local method="$1"
  local env_spec="$2"
  local train_script="$3"
  local checkpoint_prefix="$4"
  local profile_string="$5"
  local epochs="$6"
  local workers="$7"
  local extra_args="${8:-}"

  local train_list="$WORK_DIR/lists/train.txt"
  local val_list="$WORK_DIR/lists/val.txt"
  local attempt_root="$WORK_DIR/adaptive_attempt_logs/$method"
  mkdir -p "$attempt_root"

  local selected_file="$WORK_DIR/adaptive_attempt_logs/${method}.selected"
  rm -f "$selected_file"

  IFS=';' read -ra profiles <<< "$profile_string"
  for profile in "${profiles[@]}"; do
    [[ -z "$profile" ]] && continue
    IFS=',' read -r image_size batch_size lr profile_workers <<< "$profile"
    local attempt_workers="${profile_workers:-$workers}"
    local out_dir="$WORK_DIR/checkpoints/${checkpoint_prefix}_${image_size}_bs${batch_size}_w${attempt_workers}"
    local log_file="$attempt_root/image${image_size}_bs${batch_size}_w${attempt_workers}_lr${lr}.log"
    mkdir -p "$out_dir"

    echo "[$method] trying image_size=$image_size batch_size=$batch_size workers=$attempt_workers lr=$lr epochs=$epochs"
    set +e
    if [[ "$STREAM_ATTEMPT_LOGS" == "1" ]]; then
      run_py "$env_spec" "$PYTHON_BIN" "$train_script" \
        --train-list "$train_list" \
        --val-list "$val_list" \
        --out-dir "$out_dir" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --image-size "$image_size" \
        --lr "$lr" \
        --workers "$attempt_workers" \
        $extra_args 2>&1 | tee "$log_file"
      local status=${PIPESTATUS[0]}
    else
      run_py "$env_spec" "$PYTHON_BIN" "$train_script" \
        --train-list "$train_list" \
        --val-list "$val_list" \
        --out-dir "$out_dir" \
        --epochs "$epochs" \
        --batch-size "$batch_size" \
        --image-size "$image_size" \
        --lr "$lr" \
        --workers "$attempt_workers" \
        $extra_args > "$log_file" 2>&1
      local status=$?
    fi
    set -e

    if [[ "$status" -eq 0 ]]; then
      echo "[$method] selected image_size=$image_size batch_size=$batch_size workers=$attempt_workers lr=$lr"
      record_profile "$method" "$image_size" "$batch_size" "$lr" "$out_dir" "$attempt_workers"
      printf "%s\t%s\t%s\t%s\n" "$image_size" "$batch_size" "$lr" "$out_dir" > "$selected_file"
      return 0
    fi

    if is_oom_log "$log_file"; then
      echo "[$method] OOM or memory allocation failure; retrying lower profile. See $log_file"
      continue
    fi

    echo "[$method] failed for a non-memory reason. See $log_file"
    tail -n 80 "$log_file" || true
    return "$status"
  done

  echo "[$method] all adaptive profiles failed."
  return 1
}

predict_and_eval() {
  local method="$1"
  local env_spec="$2"
  local project_dir="$3"
  local pred_script="$4"
  local weight_arg="$5"
  local weight_path="$6"
  local image_size="$7"

  local lists_dir="$WORK_DIR/lists"
  local pred_root="$WORK_DIR/predictions/$method"
  local result_root="$WORK_DIR/results/$method"
  local test_result_root="$WORK_DIR/test_results/$method"
  mkdir -p "$pred_root" "$result_root" "$test_result_root"

  for list_file in "$lists_dir"/tests/*.txt; do
    local dataset_name
    dataset_name="$(basename "$list_file" .txt)"
    run_py "$env_spec" "$PYTHON_BIN" "$project_dir/$pred_script" \
      --list-file "$list_file" \
      "$weight_arg" "$weight_path" \
      --out-dir "$pred_root/tests/$dataset_name" \
      --image-size "$image_size"
    "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
      --list-file "$list_file" \
      --pred-dir "$pred_root/tests/$dataset_name" \
      --out "$test_result_root/$dataset_name.json"
  done

  run_py "$env_spec" "$PYTHON_BIN" "$project_dir/$pred_script" \
    --list-file "$lists_dir/tests/${ROBUST_DATASET}.txt" \
    "$weight_arg" "$weight_path" \
    --out-dir "$pred_root/clean" \
    --image-size "$image_size"
  "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
    --list-file "$lists_dir/tests/${ROBUST_DATASET}.txt" \
    --pred-dir "$pred_root/clean" \
    --out "$result_root/clean.json"

  for variant in jpeg_q100 jpeg_q70 jpeg_q50 gaussian_s5 gaussian_s10 gaussian_s15; do
    run_py "$env_spec" "$PYTHON_BIN" "$project_dir/$pred_script" \
      --list-file "$lists_dir/robust/${ROBUST_DATASET}_${variant}.txt" \
      "$weight_arg" "$weight_path" \
      --out-dir "$pred_root/$variant" \
      --image-size "$image_size"
    "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
      --list-file "$lists_dir/robust/${ROBUST_DATASET}_${variant}.txt" \
      --pred-dir "$pred_root/$variant" \
      --out "$result_root/$variant.json"
  done
}

selected_field() {
  local method="$1"
  local field_index="$2"
  cut -f "$field_index" "$WORK_DIR/adaptive_attempt_logs/${method}.selected"
}

prepare
mkdir -p "$WORK_DIR/summary"
rm -f "$PROFILE_FILE"

# Profiles are ordered from closest-to-original/highest-detail to safer fallbacks.
# Each entry is image_size,batch_size,lr or image_size,batch_size,lr,workers.

train_adaptive \
  "CAT-Net" \
  "${CAT_ENV:-}" \
  "$ROOT/CAT-Net/CAT-Net-main/tools/train_lzb.py" \
  "catnet" \
  "512,22,0.005,4;512,22,0.005,2;512,16,0.005,4;512,16,0.005,2;512,11,0.005,4;512,11,0.005,2;512,8,0.005,4;512,8,0.005,2;512,4,0.005,2;512,4,0.005,0;384,8,0.005,4;384,4,0.005,2;256,8,0.005,4;256,4,0.005,2" \
  "$CAT_EPOCHS" \
  "$CAT_WORKERS"
cat_img="$(selected_field CAT-Net 1)"
cat_dir="$(selected_field CAT-Net 4)"
predict_and_eval "CAT-Net" "${CAT_ENV:-}" "$ROOT/CAT-Net/CAT-Net-main" "tools/predict_lzb.py" "--model-file" "$cat_dir/best.pth.tar" "$cat_img"

train_adaptive \
  "MVSS-Net" \
  "${MVSS_ENV:-}" \
  "$ROOT/MVSS-Net/MVSS-Net-master/train_lzb.py" \
  "mvssnet" \
  "512,8,0.0001;512,4,0.0001;512,2,0.0001;512,1,0.0001;384,8,0.0001;384,4,0.0001;256,8,0.0001;256,4,0.0001" \
  "$MVSS_EPOCHS" \
  "$WORKERS"
mvss_img="$(selected_field MVSS-Net 1)"
mvss_dir="$(selected_field MVSS-Net 4)"
predict_and_eval "MVSS-Net" "${MVSS_ENV:-}" "$ROOT/MVSS-Net/MVSS-Net-master" "predict_lzb.py" "--model-file" "$mvss_dir/best.pth" "$mvss_img"

train_adaptive \
  "PSCC-Net" \
  "${PSCC_ENV:-}" \
  "$ROOT/PSCC-Net/PSCC-Net-main/train_lzb.py" \
  "psccnet" \
  "256,10,0.0002;256,8,0.0002;256,4,0.0002;256,2,0.0002;256,1,0.0002" \
  "$PSCC_EPOCHS" \
  "$WORKERS"
pscc_img="$(selected_field PSCC-Net 1)"
pscc_dir="$(selected_field PSCC-Net 4)"
predict_and_eval "PSCC-Net" "${PSCC_ENV:-}" "$ROOT/PSCC-Net/PSCC-Net-main" "predict_lzb.py" "--checkpoint-dir" "$pscc_dir" "$pscc_img"

train_adaptive \
  "IRIS0-SPAN" \
  "${SPAN_ENV:-}" \
  "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main/train_lzb.py" \
  "span" \
  "512,1,0.0001;384,1,0.0001;256,1,0.0001" \
  "$SPAN_EPOCHS" \
  "$SPAN_WORKERS"
span_img="$(selected_field IRIS0-SPAN 1)"
span_dir="$(selected_field IRIS0-SPAN 4)"
predict_and_eval "IRIS0-SPAN" "${SPAN_ENV:-}" "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main" "predict_lzb.py" "--model-file" "$span_dir/best.h5" "$span_img"

train_adaptive \
  "ManTraNet" \
  "${MANTRA_ENV:-}" \
  "$ROOT/ManTraNet/ManTraNet-pytorch-main/train_lzb.py" \
  "mantranet" \
  "512,1,0.00001;384,1,0.00001;256,2,0.00001;256,1,0.00001" \
  "$MANTRA_EPOCHS" \
  "$WORKERS"
mantra_img="$(selected_field ManTraNet 1)"
mantra_dir="$(selected_field ManTraNet 4)"
predict_and_eval "ManTraNet" "${MANTRA_ENV:-}" "$ROOT/ManTraNet/ManTraNet-pytorch-main" "predict_lzb.py" "--model-file" "$mantra_dir/best.pth" "$mantra_img"

"$PYTHON_BIN" -m lzb_experiments.summarize_results \
  --results-dir "$WORK_DIR/results" \
  --out-csv "$WORK_DIR/summary/robust_summary.csv"

echo "Adaptive experiments finished."
echo "Selected profiles: $PROFILE_FILE"
echo "Test results: $WORK_DIR/test_results"
echo "Robustness summary: $WORK_DIR/summary/robust_summary.csv"
