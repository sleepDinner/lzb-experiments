# LZB Unified Comparison Experiments

This folder adds a unified wrapper around the five downloaded projects:

- CAT-Net
- IRIS0-SPAN
- ManTraNet
- MVSS-Net
- PSCC-Net

The wrapper assumes the server workspace is `/data0/hl/lzb-experiments/`.

## Experimental Protocol

1. Build one deterministic split from `/data0/lzb-change-vmunet/FinalTrainData/`.
   - `images/` and `masks/` are paired by file stem.
   - Samples are skipped before training/testing if the image or mask cannot be read, has an invalid shape, or if image/mask height and width do not match.
   - Filtering details are written to `lists/filter_report.json` and `lists/filter_skipped.tsv`.
   - 90% is used for training and 10% for validation.
   - Split seed defaults to `2026`.
2. Train every method on the same generated train/val lists.
   - Best checkpoints are selected by validation pixel F1.
   - PyTorch wrappers show a per-epoch `tqdm` batch progress bar with running loss.
   - PyTorch wrappers save `best_epochXXX` only when validation F1 improves, keep only the current best epoch checkpoint, and save `last` once at the final epoch by default.
   - Stable aliases such as `best.pth`, `best.pth.tar`, and `best_FENet.pth` point to the epoch-named best checkpoints so existing test scripts keep working.
   - Set `--save-last-every N` on an individual training wrapper if periodic recovery checkpoints are needed.
   - CAT-Net, ManTraNet, MVSS-Net, and PSCC-Net save PyTorch checkpoints.
   - IRIS0-SPAN is a Keras project and saves `best.h5`.
   - The unified script uses each project's pretrained initialization by default.
3. Predict masks on the configured test datasets.
   - Test roots are read from `/data0/lzb-change-vmunet/FMAE5.0/test_datasets_loc_small_mid_big.json`.
4. Run robustness evaluation on the `Casiav1` list.
   - Clean
   - JPEG compression: QF 100, 70, 50
   - Gaussian noise: sigma 5, 10, 15
5. Compute pixel-level F1, IoU/Jaccard, and AUC with the same evaluator for every method.

## Output Layout

Default output root:

```bash
/data0/hl/lzb-experiments/lzb_outputs
```

Important outputs:

- `lists/`: generated train, val, test, and robustness lists
- `checkpoints/`: best and last checkpoints per method
- `predictions/`: predicted masks
- `test_results/`: clean evaluation on every dataset in the JSON file
- `results/`: per-method JSON metrics
- `summary/robust_summary.csv`: final comparison table source

## One-Command Run

Create one environment first:

```bash
conda create -n lzb4 python=3.10 -y
conda activate lzb4
pip install -r requirements_unified.txt
```

From `/data0/hl/lzb-experiments/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 EPOCHS=100 BATCH_SIZE=8 MANTRA_BATCH_SIZE=2 IMAGE_SIZE=256 bash lzb_experiments/run_all_experiments.sh
```

If the environment was created with a prefix, use `conda run -p` instead of
`conda run -n`. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 EPOCHS=100 BATCH_SIZE=8 MANTRA_BATCH_SIZE=2 IMAGE_SIZE=256 conda run -p /data0/hl/conda_envs/lzb bash lzb_experiments/run_all_experiments.sh
```

The smoke test uses the same pattern:

```bash
CUDA_VISIBLE_DEVICES=0,1 WORK_DIR=/data0/hl/lzb-experiments/lzb_smoke_outputs BATCH_SIZE=1 MANTRA_BATCH_SIZE=1 IMAGE_SIZE=256 SMOKE_TRAIN_SAMPLES=4 SMOKE_VAL_SAMPLES=2 SMOKE_TEST_SAMPLES=2 conda run -p /data0/hl/conda_envs/lzb bash lzb_experiments/run_smoke_test.sh
```

If each project uses a separate conda environment, set environment names in the same command:

```bash
CUDA_VISIBLE_DEVICES=0,1 CAT_ENV=catnet MVSS_ENV=mvss PSCC_ENV=pscc SPAN_ENV=span MANTRA_ENV=mantra EPOCHS=100 BATCH_SIZE=8 MANTRA_BATCH_SIZE=2 IMAGE_SIZE=256 bash lzb_experiments/run_all_experiments.sh
```

The `*_ENV` variables can also be full conda environment prefixes, such as
`CAT_ENV=/data0/hl/conda_envs/catnet`.

`IMAGE_SIZE=256` is the safe default because PSCC-Net's non-local block has quadratic memory growth. If memory allows and all methods are stable, rerun with `IMAGE_SIZE=512`.

## Adaptive Original-First Run

Use this when you want each method to start from a closer-to-original/high-detail
training profile and only reduce settings after a memory failure.

```bash
cd /data0/hl/lzb-experiments
LOG=/data0/hl/lzb-experiments/adaptive_$(date +%Y%m%d_%H%M%S).log
nohup conda run --no-capture-output -p /data0/hl/conda_envs/lzb bash -lc '
cd /data0/hl/lzb-experiments
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0,1 \
WORK_DIR=/data0/hl/lzb-experiments/lzb_outputs_adaptive \
bash lzb_experiments/run_adaptive_experiments.sh
' > "$LOG" 2>&1 &
echo "log: $LOG"
```

The adaptive script reuses existing files under `lzb_outputs_adaptive/lists` on
restart only when the strict filtering report is present. Set `REBUILD_LISTS=1`
to force list regeneration. Training attempt logs
are streamed to the main `nohup` log by default and are also saved under
`adaptive_attempt_logs/`; set `STREAM_ATTEMPT_LOGS=0` for quieter main logs.

The adaptive runner records the profile that actually succeeded in:

```bash
/data0/hl/lzb-experiments/lzb_outputs_adaptive/summary/adaptive_selected_profiles.tsv
```

Default model-specific epochs are:

- CAT-Net: 200
- MVSS-Net: 100
- PSCC-Net: 25
- IRIS0-SPAN: 500
- ManTraNet: 100

PSCC-Net uses its original five-stage learning-rate strategy by default:
`2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5`. If `PSCC_EPOCHS` or global `EPOCHS`
is changed, the five stages are spread evenly over the requested epoch count.

Set `EPOCHS=100` to force the same epoch count for every method, or set
`CAT_EPOCHS`, `MVSS_EPOCHS`, `PSCC_EPOCHS`, `SPAN_EPOCHS`, and
`MANTRA_EPOCHS` separately.

Adaptive retry only handles memory failures such as CUDA OOM or TensorFlow
`ResourceExhaustedError`. Other errors stop the script so real code/data issues
are not hidden.

## Pretrained Weights

Default training expects project pretraining to be available.

- CAT-Net: place `hrnetv2_w48_imagenet_pretrained.pth` and `DCT_djpeg.pth.tar` under `CAT-Net/CAT-Net-main/pretrained_models/`. These are not downloaded automatically by the wrapper.
- MVSS-Net: place the official MVSS-Net checkpoint at `MVSS-Net/MVSS-Net-master/ckpt/mvssnet.pth`. The ResNet50 ImageNet weights used by its backbone are downloaded automatically by PyTorch model zoo on first run, then cached under `~/.cache/torch/hub/checkpoints/`; if the server has no internet, download `resnet50-19c8e357.pth` manually to that cache path first.
- PSCC-Net: place pretrained checkpoints at `PSCC-Net/PSCC-Net-main/checkpoint/HRNet_checkpoint/HRNet.pth`, `PSCC-Net/PSCC-Net-main/checkpoint/NLCDetection_checkpoint/NLCDetection.pth`, and `PSCC-Net/PSCC-Net-main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth`. These are not downloaded automatically.
- IRIS0-SPAN: place ManTraNet pretrain at `IRIS0-SPAN/IRIS0-SPAN-main/pretrained_weights/ManTraNet_Ptrain4.h5`; place `PixelAttention32.h5` in `IRIS0-SPAN/IRIS0-SPAN-main/`. These are not downloaded automatically. The original config points to `../ManTraNet`, but the LZB wrapper overrides that path so the pretrain stays inside `IRIS0-SPAN-main`.
- ManTraNet: this repository already contains `ManTraNet/ManTraNet-pytorch-main/MantraNet/IMTFEv4.pt` and `ManTraNet/ManTraNet-pytorch-main/MantraNet/MantraNetv4.pt`; no extra download is needed unless those files are missing.

If you intentionally want a from-scratch ablation, call the individual training scripts with their `--no-pretrain` options where available and set `--init-weight ""` for IRIS0-SPAN.
