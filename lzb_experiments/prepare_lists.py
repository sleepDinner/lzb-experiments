import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from lzb_experiments.common import (
    add_common_args,
    build_test_lists,
    create_robust_lists,
    find_image_mask_pairs,
    split_pairs,
    write_mvss_list,
    write_pairs,
)


def write_filter_report(list_dir, skipped_records, train_count, val_count, test_lists):
    list_dir = Path(list_dir)
    summary = {
        "strict_pair_filter": True,
        "rule": "Skip samples whose image or mask cannot be read, has invalid shape, or whose image/mask height/width differ.",
        "train_samples": int(train_count),
        "val_samples": int(val_count),
        "test_datasets": sorted(test_lists.keys()),
        "skipped_total": len(skipped_records),
        "skipped_by_reason": dict(Counter(item["reason"] for item in skipped_records)),
        "skipped_by_dataset": dict(Counter(item["dataset"] for item in skipped_records)),
        "skipped_tsv": str(list_dir / "filter_skipped.tsv"),
    }
    with open(list_dir / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(list_dir / "filter_skipped.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "reason", "image", "mask", "detail"], delimiter="\t")
        writer.writeheader()
        writer.writerows(skipped_records)


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--robust-dataset", default="Casiav1")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    list_dir = work_dir / "lists"

    skipped_records = []
    pairs = find_image_mask_pairs(args.train_root, dataset_name="train", skipped_records=skipped_records)
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.seed)
    write_pairs(list_dir / "train.txt", train_pairs)
    write_pairs(list_dir / "val.txt", val_pairs)
    write_mvss_list(list_dir / "train_mvss.txt", train_pairs)
    write_mvss_list(list_dir / "val_mvss.txt", val_pairs)

    test_lists = build_test_lists(args.test_json, list_dir, skipped_records=skipped_records)
    if args.robust_dataset not in test_lists:
        raise KeyError(f"Robust dataset {args.robust_dataset!r} not found in {args.test_json}. Available: {sorted(test_lists)}")
    create_robust_lists(test_lists[args.robust_dataset], list_dir, dataset_name=args.robust_dataset, seed=args.seed)
    write_filter_report(list_dir, skipped_records, len(train_pairs), len(val_pairs), test_lists)

    print(f"Prepared {len(train_pairs)} train and {len(val_pairs)} val samples")
    print(f"Lists written under: {list_dir}")
    print(f"Strict pair filter skipped {len(skipped_records)} samples. Report: {list_dir / 'filter_report.json'}")


if __name__ == "__main__":
    main()
