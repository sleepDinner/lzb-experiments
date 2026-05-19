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
SEED="${SEED:-2026}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
GLOBAL_EPOCHS="${EPOCHS:-}"
CAT_EPOCHS="${CAT_EPOCHS:-${GLOBAL_EPOCHS:-200}}"
MVSS_EPOCHS="${MVSS_EPOCHS:-${GLOBAL_EPOCHS:-100}}"
PSCC_EPOCHS="${PSCC_EPOCHS:-${GLOBAL_EPOCHS:-25}}"
SPAN_EPOCHS="${SPAN_EPOCHS:-${GLOBAL_EPOCHS:-500}}"
MANTRA_EPOCHS="${MANTRA_EPOCHS:-${GLOBAL_EPOCHS:-100}}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
CAT_EARLY_STOP_MIN_EPOCHS="${CAT_EARLY_STOP_MIN_EPOCHS:-20}"
CAT_EARLY_STOP_PATIENCE="${CAT_EARLY_STOP_PATIENCE:-12}"
MVSS_EARLY_STOP_MIN_EPOCHS="${MVSS_EARLY_STOP_MIN_EPOCHS:-15}"
MVSS_EARLY_STOP_PATIENCE="${MVSS_EARLY_STOP_PATIENCE:-10}"
PSCC_EARLY_STOP_MIN_EPOCHS="${PSCC_EARLY_STOP_MIN_EPOCHS:-8}"
PSCC_EARLY_STOP_PATIENCE="${PSCC_EARLY_STOP_PATIENCE:-6}"
SPAN_EARLY_STOP_MIN_EPOCHS="${SPAN_EARLY_STOP_MIN_EPOCHS:-25}"
SPAN_EARLY_STOP_PATIENCE="${SPAN_EARLY_STOP_PATIENCE:-15}"
MANTRA_EARLY_STOP_MIN_EPOCHS="${MANTRA_EARLY_STOP_MIN_EPOCHS:-15}"
MANTRA_EARLY_STOP_PATIENCE="${MANTRA_EARLY_STOP_PATIENCE:-10}"
WORKERS="${WORKERS:-8}"
CAT_WORKERS="${CAT_WORKERS:-4}"
SPAN_WORKERS="${SPAN_WORKERS:-4}"
PROFILE_FILE="$WORK_DIR/summary/adaptive_selected_profiles.tsv"
STREAM_ATTEMPT_LOGS="${STREAM_ATTEMPT_LOGS:-1}"
REBUILD_LISTS="${REBUILD_LISTS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_PROFILE_FILE="${RESET_PROFILE_FILE:-0}"
SAVE_LAST_EVERY="${SAVE_LAST_EVERY:-1}"
CAT_SAVE_LAST_EVERY="${CAT_SAVE_LAST_EVERY:-$SAVE_LAST_EVERY}"
MVSS_SAVE_LAST_EVERY="${MVSS_SAVE_LAST_EVERY:-$SAVE_LAST_EVERY}"
PSCC_SAVE_LAST_EVERY="${PSCC_SAVE_LAST_EVERY:-$SAVE_LAST_EVERY}"
SPAN_SAVE_LAST_EVERY="${SPAN_SAVE_LAST_EVERY:-$SAVE_LAST_EVERY}"
MANTRA_SAVE_LAST_EVERY="${MANTRA_SAVE_LAST_EVERY:-$SAVE_LAST_EVERY}"
CAT_BEST_SAVE_START_EPOCH="${CAT_BEST_SAVE_START_EPOCH:-10}"
MVSS_BEST_SAVE_START_EPOCH="${MVSS_BEST_SAVE_START_EPOCH:-10}"
PSCC_BEST_SAVE_START_EPOCH="${PSCC_BEST_SAVE_START_EPOCH:-6}"
SPAN_BEST_SAVE_START_EPOCH="${SPAN_BEST_SAVE_START_EPOCH:-10}"
MANTRA_BEST_SAVE_START_EPOCH="${MANTRA_BEST_SAVE_START_EPOCH:-10}"

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
    --robust-dataset "$ROBUST_DATASET" \
    --seed "$SEED"
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
  else
    local tmp_file
    tmp_file="${PROFILE_FILE}.tmp"
    awk -v method="$method" 'NR == 1 || $1 != method' "$PROFILE_FILE" > "$tmp_file"
    mv "$tmp_file" "$PROFILE_FILE"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$method" "$image_size" "$batch_size" "$lr" "$out_dir" "$workers" >> "$PROFILE_FILE"
}

method_outputs_complete() {
  local method="$1"
  local result_root="$WORK_DIR/results/$method"
  local test_result_root="$WORK_DIR/test_results/$method"
  [[ -s "$result_root/clean.json" ]] || return 1
  for variant in jpeg_q100 jpeg_q70 jpeg_q50 gaussian_s5 gaussian_s10 gaussian_s15; do
    [[ -s "$result_root/$variant.json" ]] || return 1
  done
  compgen -G "$test_result_root/*.json" > /dev/null || return 1
}

method_complete() {
  local method="$1"
  [[ "$SKIP_COMPLETED" == "1" ]] || return 1
  method_outputs_complete "$method"
}

mark_complete() {
  local method="$1"
  local checkpoint_dir="$2"
  local image_size="$3"
  mkdir -p "$WORK_DIR/completed"
  {
    printf "method=%s\n" "$method"
    printf "checkpoint_dir=%s\n" "$checkpoint_dir"
    printf "image_size=%s\n" "$image_size"
    date '+completed_at=%Y-%m-%d %H:%M:%S'
  } > "$WORK_DIR/completed/${method}.done"
}

resume_args_for_method() {
  local method="$1"
  local out_dir="$2"
  case "$method" in
    "CAT-Net")
      if [[ -s "$out_dir/last.pth.tar" ]]; then
        printf -- "--resume-from %s" "$out_dir/last.pth.tar"
      elif [[ -s "$out_dir/best.pth.tar" ]]; then
        printf -- "--resume-from %s" "$out_dir/best.pth.tar"
      fi
      ;;
    "MVSS-Net"|"ManTraNet")
      if [[ -s "$out_dir/last.pth" ]]; then
        printf -- "--resume-from %s" "$out_dir/last.pth"
      elif [[ -s "$out_dir/best.pth" ]]; then
        printf -- "--resume-from %s" "$out_dir/best.pth"
      fi
      ;;
    "PSCC-Net")
      if [[ -s "$out_dir/last_FENet.pth" && -s "$out_dir/last_SegNet.pth" && -s "$out_dir/last_ClsNet.pth" ]]; then
        printf -- "--resume-from %s" "$out_dir"
      elif [[ -s "$out_dir/best_FENet.pth" && -s "$out_dir/best_SegNet.pth" && -s "$out_dir/best_ClsNet.pth" ]]; then
        printf -- "--resume-from %s" "$out_dir"
      fi
      ;;
    "IRIS0-SPAN")
      if [[ -s "$out_dir/last.h5" ]]; then
        printf -- "--resume-from %s" "$out_dir/last.h5"
      elif [[ -s "$out_dir/best.h5" ]]; then
        printf -- "--resume-from %s" "$out_dir/best.h5"
      fi
      ;;
  esac
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
    local resume_args
    resume_args="$(resume_args_for_method "$method" "$out_dir")"

    echo "[$method] trying image_size=$image_size batch_size=$batch_size workers=$attempt_workers lr=$lr epochs=$epochs"
    if [[ -n "$resume_args" ]]; then
      echo "[$method] resume args: $resume_args"
    fi
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
        --seed "$SEED" \
        $resume_args \
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
        --seed "$SEED" \
        $resume_args \
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
  local predict_batch="$8"

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
      --image-size "$image_size" \
      --batch-size "$predict_batch"
    "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
      --list-file "$list_file" \
      --pred-dir "$pred_root/tests/$dataset_name" \
      --out "$test_result_root/$dataset_name.json"
  done

  run_py "$env_spec" "$PYTHON_BIN" "$project_dir/$pred_script" \
    --list-file "$lists_dir/tests/${ROBUST_DATASET}.txt" \
    "$weight_arg" "$weight_path" \
    --out-dir "$pred_root/clean" \
    --image-size "$image_size" \
    --batch-size "$predict_batch"
  "$PYTHON_BIN" -m lzb_experiments.evaluate_predictions \
    --list-file "$lists_dir/tests/${ROBUST_DATASET}.txt" \
    --pred-dir "$pred_root/clean" \
    --out "$result_root/clean.json"

  for variant in jpeg_q100 jpeg_q70 jpeg_q50 gaussian_s5 gaussian_s10 gaussian_s15; do
    run_py "$env_spec" "$PYTHON_BIN" "$project_dir/$pred_script" \
      --list-file "$lists_dir/robust/${ROBUST_DATASET}_${variant}.txt" \
      "$weight_arg" "$weight_path" \
      --out-dir "$pred_root/$variant" \
      --image-size "$image_size" \
      --batch-size "$predict_batch"
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
if [[ "$RESET_PROFILE_FILE" == "1" ]]; then
  rm -f "$PROFILE_FILE"
fi

# Profiles are ordered from closest-to-original/highest-detail to safer fallbacks.
# Each entry is image_size,batch_size,lr or image_size,batch_size,lr,workers.

if method_complete "CAT-Net"; then
  echo "[CAT-Net] completed outputs found; skipping training/test/robust. Set SKIP_COMPLETED=0 to rerun."
else
  train_adaptive \
    "CAT-Net" \
    "${CAT_ENV:-}" \
    "$ROOT/CAT-Net/CAT-Net-main/tools/train_lzb.py" \
    "catnet" \
    "${CAT_PROFILES:-512,22,0.0001,2;512,16,0.000073,4;512,16,0.000073,2;512,11,0.00005,4;512,11,0.00005,2;512,8,0.000036,4;512,8,0.000036,2;512,4,0.000018,2;512,4,0.000018,0;384,8,0.000036,4;384,4,0.000018,2;256,8,0.000036,4;256,4,0.000018,2}" \
    "$CAT_EPOCHS" \
    "$CAT_WORKERS" \
    "--save-last-every $CAT_SAVE_LAST_EVERY --best-save-start-epoch $CAT_BEST_SAVE_START_EPOCH --early-stop-min-epochs $CAT_EARLY_STOP_MIN_EPOCHS --early-stop-patience $CAT_EARLY_STOP_PATIENCE --early-stop-min-delta $EARLY_STOP_MIN_DELTA"
  cat_img="$(selected_field CAT-Net 1)"
  cat_batch="${CAT_PREDICT_BATCH_SIZE:-$(selected_field CAT-Net 2)}"
  cat_dir="$(selected_field CAT-Net 4)"
  predict_and_eval "CAT-Net" "${CAT_ENV:-}" "$ROOT/CAT-Net/CAT-Net-main" "tools/predict_lzb.py" "--model-file" "$cat_dir/best.pth.tar" "$cat_img" "$cat_batch"
  mark_complete "CAT-Net" "$cat_dir" "$cat_img"
fi

if method_complete "MVSS-Net"; then
  echo "[MVSS-Net] completed outputs found; skipping training/test/robust. Set SKIP_COMPLETED=0 to rerun."
else
  train_adaptive \
    "MVSS-Net" \
    "${MVSS_ENV:-}" \
    "$ROOT/MVSS-Net/MVSS-Net-master/train_lzb.py" \
    "mvssnet" \
    "${MVSS_PROFILES:-512,16,0.0001;512,12,0.000075;512,8,0.00005;512,6,0.0000375;512,4,0.000025;512,2,0.0000125;384,16,0.0001;384,12,0.000075;384,8,0.00005;384,6,0.0000375;384,4,0.000025;384,2,0.0000125;256,16,0.0001;256,12,0.000075;256,8,0.00005;256,6,0.0000375;256,4,0.000025;256,2,0.0000125}" \
    "$MVSS_EPOCHS" \
    "$WORKERS" \
    "--save-last-every $MVSS_SAVE_LAST_EVERY --best-save-start-epoch $MVSS_BEST_SAVE_START_EPOCH --early-stop-min-epochs $MVSS_EARLY_STOP_MIN_EPOCHS --early-stop-patience $MVSS_EARLY_STOP_PATIENCE --early-stop-min-delta $EARLY_STOP_MIN_DELTA"
  mvss_img="$(selected_field MVSS-Net 1)"
  mvss_batch="${MVSS_PREDICT_BATCH_SIZE:-$(selected_field MVSS-Net 2)}"
  mvss_dir="$(selected_field MVSS-Net 4)"
  predict_and_eval "MVSS-Net" "${MVSS_ENV:-}" "$ROOT/MVSS-Net/MVSS-Net-master" "predict_lzb.py" "--model-file" "$mvss_dir/best.pth" "$mvss_img" "$mvss_batch"
  mark_complete "MVSS-Net" "$mvss_dir" "$mvss_img"
fi

if method_complete "PSCC-Net"; then
  echo "[PSCC-Net] completed outputs found; skipping training/test/robust. Set SKIP_COMPLETED=0 to rerun."
else
  train_adaptive \
    "PSCC-Net" \
    "${PSCC_ENV:-}" \
    "$ROOT/PSCC-Net/PSCC-Net-main/train_lzb.py" \
    "psccnet" \
    "${PSCC_PROFILES:-256,10,0.0002;256,8,0.00016;256,4,0.00008;256,2,0.00004;256,1,0.00002}" \
    "$PSCC_EPOCHS" \
    "$WORKERS" \
    "--save-last-every $PSCC_SAVE_LAST_EVERY --best-save-start-epoch $PSCC_BEST_SAVE_START_EPOCH --early-stop-min-epochs $PSCC_EARLY_STOP_MIN_EPOCHS --early-stop-patience $PSCC_EARLY_STOP_PATIENCE --early-stop-min-delta $EARLY_STOP_MIN_DELTA"
  pscc_img="$(selected_field PSCC-Net 1)"
  pscc_batch="${PSCC_PREDICT_BATCH_SIZE:-$(selected_field PSCC-Net 2)}"
  pscc_dir="$(selected_field PSCC-Net 4)"
  predict_and_eval "PSCC-Net" "${PSCC_ENV:-}" "$ROOT/PSCC-Net/PSCC-Net-main" "predict_lzb.py" "--checkpoint-dir" "$pscc_dir" "$pscc_img" "$pscc_batch"
  mark_complete "PSCC-Net" "$pscc_dir" "$pscc_img"
fi

if method_complete "IRIS0-SPAN"; then
  echo "[IRIS0-SPAN] completed outputs found; skipping training/test/robust. Set SKIP_COMPLETED=0 to rerun."
else
  train_adaptive \
    "IRIS0-SPAN" \
    "${SPAN_ENV:-}" \
    "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main/train_lzb.py" \
    "span" \
    "${SPAN_PROFILES:-512,16,0.0001;512,12,0.000075;512,8,0.00005;512,6,0.0000375;512,4,0.000025;512,2,0.0000125;384,16,0.0001;384,12,0.000075;384,8,0.00005;384,6,0.0000375;384,4,0.000025;384,2,0.0000125;256,16,0.0001;256,12,0.000075;256,8,0.00005;256,6,0.0000375;256,4,0.000025;256,2,0.0000125}" \
    "$SPAN_EPOCHS" \
    "$SPAN_WORKERS" \
    "--save-last-every $SPAN_SAVE_LAST_EVERY --best-save-start-epoch $SPAN_BEST_SAVE_START_EPOCH --early-stop-min-epochs $SPAN_EARLY_STOP_MIN_EPOCHS --early-stop-patience $SPAN_EARLY_STOP_PATIENCE --early-stop-min-delta $EARLY_STOP_MIN_DELTA"
  span_img="$(selected_field IRIS0-SPAN 1)"
  span_batch="${SPAN_PREDICT_BATCH_SIZE:-$(selected_field IRIS0-SPAN 2)}"
  span_dir="$(selected_field IRIS0-SPAN 4)"
  predict_and_eval "IRIS0-SPAN" "${SPAN_ENV:-}" "$ROOT/IRIS0-SPAN/IRIS0-SPAN-main" "predict_lzb.py" "--model-file" "$span_dir/best.h5" "$span_img" "$span_batch"
  mark_complete "IRIS0-SPAN" "$span_dir" "$span_img"
fi

if method_complete "ManTraNet"; then
  echo "[ManTraNet] completed outputs found; skipping training/test/robust. Set SKIP_COMPLETED=0 to rerun."
else
  train_adaptive \
    "ManTraNet" \
    "${MANTRA_ENV:-}" \
    "$ROOT/ManTraNet/ManTraNet-pytorch-main/train_lzb.py" \
    "mantranet" \
    "${MANTRA_PROFILES:-512,16,0.00001;512,12,0.0000075;512,8,0.000005;512,6,0.00000375;512,4,0.0000025;512,2,0.00000125;384,16,0.00001;384,12,0.0000075;384,8,0.000005;384,6,0.00000375;384,4,0.0000025;384,2,0.00000125;256,16,0.00001;256,12,0.0000075;256,8,0.000005;256,6,0.00000375;256,4,0.0000025;256,2,0.00000125}" \
    "$MANTRA_EPOCHS" \
    "$WORKERS" \
    "--save-last-every $MANTRA_SAVE_LAST_EVERY --best-save-start-epoch $MANTRA_BEST_SAVE_START_EPOCH --early-stop-min-epochs $MANTRA_EARLY_STOP_MIN_EPOCHS --early-stop-patience $MANTRA_EARLY_STOP_PATIENCE --early-stop-min-delta $EARLY_STOP_MIN_DELTA"
  mantra_img="$(selected_field ManTraNet 1)"
  mantra_batch="${MANTRA_PREDICT_BATCH_SIZE:-$(selected_field ManTraNet 2)}"
  mantra_dir="$(selected_field ManTraNet 4)"
  predict_and_eval "ManTraNet" "${MANTRA_ENV:-}" "$ROOT/ManTraNet/ManTraNet-pytorch-main" "predict_lzb.py" "--model-file" "$mantra_dir/best.pth" "$mantra_img" "$mantra_batch"
  mark_complete "ManTraNet" "$mantra_dir" "$mantra_img"
fi

"$PYTHON_BIN" -m lzb_experiments.summarize_results \
  --results-dir "$WORK_DIR/results" \
  --out-csv "$WORK_DIR/summary/robust_summary.csv"

echo "Adaptive experiments finished."
echo "Selected profiles: $PROFILE_FILE"
echo "Test results: $WORK_DIR/test_results"
echo "Robustness summary: $WORK_DIR/summary/robust_summary.csv"
