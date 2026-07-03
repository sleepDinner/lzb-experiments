import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from lzb_experiments.common import (
    add_common_args,
    build_test_lists,
    create_robust_lists,
)


ROBUST_VARIANTS = (
    "jpeg_q100",
    "jpeg_q70",
    "jpeg_q50",
    "gaussian_s5",
    "gaussian_s10",
    "gaussian_s15",
)


def parse_names(value):
    names = []
    for item in str(value).replace(",", " ").split():
        item = item.strip()
        if item:
            names.append(item)
    return names


def robust_lists_ready(list_dir, dataset_name):
    return all((list_dir / "robust" / f"{dataset_name}_{variant}.txt").is_file() for variant in ROBUST_VARIANTS)


def write_filter_report(list_dir, skipped_records):
    if not skipped_records:
        return
    list_dir = Path(list_dir)
    report_path = list_dir / "filter_report_robust_extra.json"
    skipped_path = list_dir / "filter_skipped_robust_extra.tsv"
    summary = {
        "strict_pair_filter": True,
        "scope": "extra robust test datasets",
        "skipped_total": len(skipped_records),
        "skipped_by_reason": dict(Counter(item["reason"] for item in skipped_records)),
        "skipped_by_dataset": dict(Counter(item["dataset"] for item in skipped_records)),
        "skipped_tsv": str(skipped_path),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(skipped_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "reason", "image", "mask", "detail"], delimiter="\t")
        writer.writeheader()
        writer.writerows(skipped_records)
    print(f"Extra robust filter report: {report_path}")


def main():
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument(
        "--datasets",
        default="Columbia,NIST16,IMD2020,DSO-1,Korus",
        help="Comma or space separated dataset names from the test json.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Regenerate robust images/lists even if they already exist.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for image-only robust variant generation.")
    args = parser.parse_args()

    dataset_names = parse_names(args.datasets)
    if not dataset_names:
        raise RuntimeError("No robust datasets were requested.")

    list_dir = Path(args.work_dir) / "lists"
    test_dir = list_dir / "tests"
    skipped_records = []

    missing_test_lists = [name for name in dataset_names if not (test_dir / f"{name}.txt").is_file()]
    if missing_test_lists:
        print(f"Missing test lists for {missing_test_lists}; rebuilding test lists from {args.test_json}")
        build_test_lists(args.test_json, list_dir, skipped_records=skipped_records)

    for dataset_name in dataset_names:
        source_list = test_dir / f"{dataset_name}.txt"
        if not source_list.is_file():
            raise FileNotFoundError(f"Missing test list for {dataset_name}: {source_list}")
        if not args.rebuild and robust_lists_ready(list_dir, dataset_name):
            print(f"Reusing robust lists for {dataset_name}")
            continue
        print(f"Creating image-only robust lists for {dataset_name} from {source_list} with workers={args.workers}")
        print("Masks are not generated or copied; robust lists keep the original mask paths.")
        create_robust_lists(source_list, list_dir, dataset_name=dataset_name, seed=args.seed, workers=args.workers)

    write_filter_report(list_dir, skipped_records)
    print(f"Prepared robust lists for: {', '.join(dataset_names)}")


if __name__ == "__main__":
    main()
