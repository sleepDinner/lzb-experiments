import argparse
import csv
import json
from pathlib import Path


ORDER = ["clean", "jpeg_q100", "jpeg_q70", "jpeg_q50", "gaussian_s5", "gaussian_s10", "gaussian_s15"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    rows = []
    for result_file in sorted(Path(args.results_dir).rglob("*.json")):
        with open(result_file, "r", encoding="utf-8") as f:
            item = json.load(f)
        rel = result_file.relative_to(args.results_dir)
        if len(rel.parts) < 2:
            continue
        method = rel.parts[0]
        variant = result_file.stem
        rows.append({
            "method": method,
            "variant": variant,
            "f1": item.get("f1"),
            "iou": item.get("iou"),
            "auc": item.get("auc"),
            "evaluated": item.get("evaluated"),
            "missing": item.get("missing"),
        })

    order_index = {name: idx for idx, name in enumerate(ORDER)}
    rows.sort(key=lambda r: (r["method"], order_index.get(r["variant"], 999), r["variant"]))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "variant", "f1", "iou", "auc", "evaluated", "missing"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()
