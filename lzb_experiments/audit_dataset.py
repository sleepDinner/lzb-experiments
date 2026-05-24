import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


RATIO_BINS = (
    ("empty", 0.0, 0.0),
    ("0_0.0001", 0.0, 0.0001),
    ("0.0001_0.001", 0.0001, 0.001),
    ("0.001_0.01", 0.001, 0.01),
    ("0.01_0.05", 0.01, 0.05),
    ("0.05_0.1", 0.05, 0.1),
    ("0.1_0.5", 0.1, 0.5),
    ("0.5_0.95", 0.5, 0.95),
    ("0.95_1", 0.95, 1.0),
)


def ratio_bin(value):
    if value <= 0.0:
        return "empty"
    for name, lo, hi in RATIO_BINS[1:]:
        if lo < value <= hi:
            return name
    return "gt_1"


def require_cv2():
    if cv2 is None:
        raise SystemExit("OpenCV is required for dataset audit. Install opencv-python or run inside the lzb conda environment.")
    return cv2


def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) == 1:
                pairs.append((parts[0], "None", -1))
            elif len(parts) == 2:
                pairs.append((parts[0], parts[1], 1))
            else:
                pairs.append((parts[0], parts[1], int(float(parts[2]))))
    return pairs


def read_image_shape(path):
    cv = require_cv2()
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        return None
    return tuple(int(v) for v in image.shape[:2])


def read_mask(path):
    cv = require_cv2()
    mask = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask


def empty_stats():
    return {
        "total": 0,
        "ok": 0,
        "missing_image": 0,
        "missing_mask": 0,
        "unreadable_image": 0,
        "unreadable_mask": 0,
        "shape_mismatch": 0,
        "empty_mask": 0,
        "tiny_mask": 0,
        "large_mask": 0,
        "label_mask_mismatch": 0,
        "duplicate_image_path": 0,
        "duplicate_mask_path": 0,
        "positive_ratio_bins": Counter(),
        "image_hw_top": Counter(),
        "mask_hw_top": Counter(),
        "mask_unique_values_top": Counter(),
        "positive_ratio_min": None,
        "positive_ratio_max": None,
        "positive_ratio_mean": 0.0,
    }


def update_ratio_stats(stats, ratio):
    stats["positive_ratio_bins"][ratio_bin(ratio)] += 1
    if stats["positive_ratio_min"] is None or ratio < stats["positive_ratio_min"]:
        stats["positive_ratio_min"] = ratio
    if stats["positive_ratio_max"] is None or ratio > stats["positive_ratio_max"]:
        stats["positive_ratio_max"] = ratio
    stats["positive_ratio_mean"] += ratio


def add_problem(problems, limit, row):
    if len(problems) < limit:
        problems.append(row)


def audit_list(name, list_path, tiny_threshold, large_threshold, max_problem_examples, max_samples):
    pairs = read_pairs(list_path)
    if max_samples is not None:
        pairs = pairs[:max_samples]

    stats = empty_stats()
    problems = []
    seen_images = set()
    seen_masks = set()

    for index, (image_path, mask_path, label) in enumerate(pairs):
        stats["total"] += 1
        image_path = str(image_path)
        mask_path = str(mask_path)
        label = int(label)

        if image_path in seen_images:
            stats["duplicate_image_path"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "duplicate_image_path",
                "image": image_path,
                "mask": mask_path,
                "detail": "",
            })
        seen_images.add(image_path)

        if mask_path in seen_masks:
            stats["duplicate_mask_path"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "duplicate_mask_path",
                "image": image_path,
                "mask": mask_path,
                "detail": "",
            })
        seen_masks.add(mask_path)

        if not Path(image_path).is_file():
            stats["missing_image"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "missing_image",
                "image": image_path,
                "mask": mask_path,
                "detail": "",
            })
            continue
        if not Path(mask_path).is_file():
            stats["missing_mask"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "missing_mask",
                "image": image_path,
                "mask": mask_path,
                "detail": "",
            })
            continue

        image_hw = read_image_shape(image_path)
        if image_hw is None:
            stats["unreadable_image"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "unreadable_image",
                "image": image_path,
                "mask": mask_path,
                "detail": "",
            })
            continue

        mask = read_mask(mask_path)
        if mask is None:
            stats["unreadable_mask"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "unreadable_mask",
                "image": image_path,
                "mask": mask_path,
                "detail": "",
            })
            continue

        mask_hw = tuple(int(v) for v in mask.shape[:2])
        stats["image_hw_top"][str(image_hw)] += 1
        stats["mask_hw_top"][str(mask_hw)] += 1
        unique_values = np.unique(mask)
        if unique_values.size <= 8:
            unique_key = ",".join(str(int(v)) for v in unique_values.tolist())
        else:
            unique_key = "many_values"
        stats["mask_unique_values_top"][unique_key] += 1

        if image_hw != mask_hw:
            stats["shape_mismatch"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "shape_mismatch",
                "image": image_path,
                "mask": mask_path,
                "detail": f"image_hw={image_hw} mask_hw={mask_hw}",
            })
            continue

        positive = mask > 0
        positive_pixels = int(positive.sum())
        total_pixels = int(positive.size)
        positive_ratio = float(positive_pixels / total_pixels) if total_pixels > 0 else 0.0
        update_ratio_stats(stats, positive_ratio)

        if positive_ratio <= 0.0:
            stats["empty_mask"] += 1
        elif positive_ratio < tiny_threshold:
            stats["tiny_mask"] += 1
        elif positive_ratio > large_threshold:
            stats["large_mask"] += 1

        mask_label = 1 if positive_ratio > 0.0 else 0
        if label >= 0 and label != mask_label:
            stats["label_mask_mismatch"] += 1
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": "label_mask_mismatch",
                "image": image_path,
                "mask": mask_path,
                "detail": f"label={label} mask_label={mask_label} positive_ratio={positive_ratio:.8f}",
            })

        if positive_ratio <= 0.0 or positive_ratio < tiny_threshold or positive_ratio > large_threshold:
            reason = "empty_mask" if positive_ratio <= 0.0 else "tiny_mask" if positive_ratio < tiny_threshold else "large_mask"
            add_problem(problems, max_problem_examples, {
                "list": name,
                "index": index,
                "reason": reason,
                "image": image_path,
                "mask": mask_path,
                "detail": f"positive_ratio={positive_ratio:.8f}",
            })

        stats["ok"] += 1

    if stats["ok"] > 0:
        stats["positive_ratio_mean"] /= stats["ok"]
    stats["positive_ratio_bins"] = dict(stats["positive_ratio_bins"])
    stats["image_hw_top"] = dict(stats["image_hw_top"].most_common(20))
    stats["mask_hw_top"] = dict(stats["mask_hw_top"].most_common(20))
    stats["mask_unique_values_top"] = dict(stats["mask_unique_values_top"].most_common(20))
    return stats, problems


def discover_lists(list_dir):
    list_dir = Path(list_dir)
    candidates = []
    for rel in ["train.txt", "val.txt"]:
        path = list_dir / rel
        if path.is_file():
            candidates.append((rel.replace(".txt", ""), path))
    for subdir in ["tests", "robust"]:
        root = list_dir / subdir
        if root.is_dir():
            for path in sorted(root.glob("*.txt")):
                candidates.append((f"{subdir}/{path.stem}", path))
    return candidates


def write_problem_tsv(path, problems_by_list):
    rows = []
    for _name, problems in problems_by_list.items():
        rows.extend(problems)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["list", "index", "reason", "image", "mask", "detail"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-dir", default="", help="Audit generated lists under this directory.")
    parser.add_argument("--list-file", action="append", default=[], help="Audit one list file. Can be repeated.")
    parser.add_argument("--out-dir", required=True, help="Directory for audit_summary.json and audit_problems.tsv.")
    parser.add_argument("--tiny-threshold", type=float, default=0.0001, help="Mask positive ratio below this is suspicious.")
    parser.add_argument("--large-threshold", type=float, default=0.95, help="Mask positive ratio above this is suspicious.")
    parser.add_argument("--max-problem-examples", type=int, default=200, help="Max problem examples stored per list.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional first-N sample limit for quick checks.")
    args = parser.parse_args()

    list_items = []
    if args.list_dir:
        list_items.extend(discover_lists(args.list_dir))
    for item in args.list_file:
        path = Path(item)
        list_items.append((path.stem, path))
    if not list_items:
        raise SystemExit("Provide --list-dir or at least one --list-file.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tiny_threshold": args.tiny_threshold,
        "large_threshold": args.large_threshold,
        "max_samples": args.max_samples,
        "lists": {},
    }
    problems_by_list = defaultdict(list)
    max_samples = args.max_samples if args.max_samples > 0 else None

    for name, path in list_items:
        stats, problems = audit_list(
            name=name,
            list_path=path,
            tiny_threshold=args.tiny_threshold,
            large_threshold=args.large_threshold,
            max_problem_examples=args.max_problem_examples,
            max_samples=max_samples,
        )
        summary["lists"][name] = {
            "path": str(path),
            **stats,
        }
        problems_by_list[name].extend(problems)
        print(
            f"{name}: total={stats['total']} ok={stats['ok']} empty={stats['empty_mask']} "
            f"tiny={stats['tiny_mask']} large={stats['large_mask']} mismatch={stats['shape_mismatch']} "
            f"missing={stats['missing_image'] + stats['missing_mask']} unreadable={stats['unreadable_image'] + stats['unreadable_mask']}"
        )

    summary_path = out_dir / "audit_summary.json"
    problems_path = out_dir / "audit_problems.tsv"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_problem_tsv(problems_path, problems_by_list)
    print(f"Audit summary: {summary_path}")
    print(f"Problem examples: {problems_path}")


if __name__ == "__main__":
    main()
