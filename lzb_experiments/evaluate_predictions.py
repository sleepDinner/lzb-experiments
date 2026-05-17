import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import jaccard_score

from lzb_experiments.common import ensure_dir, prediction_filename, read_mask, read_pairs, read_pred


def prediction_path(pred_dir, image_path):
    return Path(pred_dir) / prediction_filename(image_path)


def imdlbenco_f1_from_confusion(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def imdlbenco_pixel_f1(mask, pred, threshold=0.5, mode="origin"):
    mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None, ...]
    pred_tensor = torch.from_numpy(pred.astype(np.float32))[None, None, ...]
    pred_binary = (pred_tensor > threshold).float()
    tp = torch.sum(pred_binary * mask_tensor, dim=(1, 2, 3))
    tn = torch.sum((1.0 - pred_binary) * (1.0 - mask_tensor), dim=(1, 2, 3))
    fp = torch.sum(pred_binary * (1.0 - mask_tensor), dim=(1, 2, 3))
    fn = torch.sum((1.0 - pred_binary) * mask_tensor, dim=(1, 2, 3))
    if mode == "origin":
        value = imdlbenco_f1_from_confusion(tp, fp, fn)
    elif mode == "double":
        origin = imdlbenco_f1_from_confusion(tp, fp, fn)
        reverse = imdlbenco_f1_from_confusion(fn, tn, tp)
        value = torch.maximum(origin, reverse)
    else:
        raise RuntimeError(f"Cal_F1 no mode name {mode}")
    return float(value[0].item())


def imdlbenco_cal_auc(mask, pred):
    y_true = torch.from_numpy(mask.astype(np.float32).reshape(-1))
    y_score = torch.from_numpy(pred.astype(np.float32).reshape(-1))
    desc_score_indices = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[desc_score_indices]
    n_pos = torch.sum(y_true_sorted).item()
    n_neg = len(y_true_sorted) - n_pos
    if n_pos <= 0.0 or n_neg <= 0.0:
        return float("nan")
    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_true_sorted, dim=0)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return float(torch.trapz(tpr, fpr).item())


def imdlbenco_pixel_auc(mask, pred, mode="origin"):
    if mode == "origin":
        return imdlbenco_cal_auc(mask, pred)
    if mode == "double":
        origin = imdlbenco_cal_auc(mask, pred)
        reverse = imdlbenco_cal_auc(mask, 1.0 - pred)
        if not np.isfinite(origin) and not np.isfinite(reverse):
            return float("nan")
        if not np.isfinite(origin):
            return float(reverse)
        if not np.isfinite(reverse):
            return float(origin)
        return float(max(origin, reverse))
    raise RuntimeError(f"Cal_AUC no mode name {mode}")


def mean_or_nan(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def evaluate_pair_list(list_file, pred_dir, threshold=0.5):
    pairs = read_pairs(list_file)
    f1_origin_values = []
    f1_double_values = []
    iou_values = []
    auc_origin_values = []
    auc_double_values = []
    missing = []
    for image_path, mask_path, _ in pairs:
        pred_path = prediction_path(pred_dir, image_path)
        if not pred_path.is_file():
            missing.append(str(pred_path))
            continue
        mask = read_mask(mask_path)
        pred = read_pred(pred_path, size=mask.shape)
        y_true = mask.reshape(-1)
        y_score = pred.reshape(-1)
        y_pred = (y_score > threshold).astype(np.uint8)
        f1_origin_values.append(imdlbenco_pixel_f1(mask, pred, threshold=threshold, mode="origin"))
        f1_double_values.append(imdlbenco_pixel_f1(mask, pred, threshold=threshold, mode="double"))
        iou_values.append(jaccard_score(y_true, y_pred, zero_division=0))
        auc_origin_values.append(imdlbenco_pixel_auc(mask, pred, mode="origin"))
        auc_double_values.append(imdlbenco_pixel_auc(mask, pred, mode="double"))

    result = {
        "list_file": str(list_file),
        "pred_dir": str(pred_dir),
        "samples": len(pairs),
        "evaluated": len(f1_origin_values),
        "missing": len(missing),
        "f1": mean_or_nan(f1_origin_values),
        "f1_origin": mean_or_nan(f1_origin_values),
        "f1_double": mean_or_nan(f1_double_values),
        "iou": float(np.mean(iou_values)) if iou_values else float("nan"),
        "auc": mean_or_nan(auc_origin_values),
        "auc_origin": mean_or_nan(auc_origin_values),
        "auc_double": mean_or_nan(auc_double_values),
        "threshold": threshold,
        "metric_impl": "IMDLBenCo PixelF1/PixelAUC",
        "metric_modes": ["origin", "double"],
    }
    if missing:
        result["first_missing"] = missing[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = evaluate_pair_list(args.list_file, args.pred_dir, threshold=args.threshold)
    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
