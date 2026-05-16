import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score

from lzb_experiments.common import ensure_dir, prediction_filename, read_mask, read_pairs, read_pred


def prediction_path(pred_dir, image_path):
    return Path(pred_dir) / prediction_filename(image_path)


def evaluate_pair_list(list_file, pred_dir, threshold=0.5):
    pairs = read_pairs(list_file)
    f1_values = []
    iou_values = []
    auc_values = []
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
        y_pred = (y_score >= threshold).astype(np.uint8)
        f1_values.append(f1_score(y_true, y_pred, zero_division=0))
        iou_values.append(jaccard_score(y_true, y_pred, zero_division=0))
        if len(np.unique(y_true)) > 1:
            auc_values.append(roc_auc_score(y_true, y_score))

    result = {
        "list_file": str(list_file),
        "pred_dir": str(pred_dir),
        "samples": len(pairs),
        "evaluated": len(f1_values),
        "missing": len(missing),
        "f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "iou": float(np.mean(iou_values)) if iou_values else float("nan"),
        "auc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "threshold": threshold,
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
