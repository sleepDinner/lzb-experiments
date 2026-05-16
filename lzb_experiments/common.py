import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def natural_key(path):
    return str(path).replace("\\", "/").lower()


def unique_stem(path):
    path_str = str(path).replace("\\", "/")
    digest = hashlib.md5(path_str.encode("utf-8")).hexdigest()[:10]
    return f"{Path(path).stem}_{digest}"


def prediction_filename(image_path):
    return f"{unique_stem(image_path)}.png"


def iter_images(directory):
    directory = Path(directory)
    files = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return sorted(files, key=natural_key)


def _mask_candidates(mask_dir, image_path):
    stem = image_path.stem
    rel_stem = image_path.with_suffix("").name
    names = [stem, rel_stem, stem + "_mask", stem + "_gt", stem + "_groundtruth"]
    for name in names:
        for ext in MASK_EXTS:
            yield Path(mask_dir) / f"{name}{ext}"


def _add_skip(skipped_records, dataset_name, reason, image_path, mask_path=None, detail=None):
    if skipped_records is None:
        return
    skipped_records.append({
        "dataset": str(dataset_name) if dataset_name is not None else "",
        "reason": reason,
        "image": str(image_path) if image_path is not None else "",
        "mask": str(mask_path) if mask_path is not None else "",
        "detail": str(detail) if detail is not None else "",
    })


def validate_image_mask_pair(image_path, mask_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False, 0, "unreadable_image", "cv2.imread returned None"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, 0, "unreadable_mask", "cv2.imread returned None"
    if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        return False, 0, "invalid_image_shape", image.shape
    if mask.ndim != 2 or mask.shape[0] <= 0 or mask.shape[1] <= 0:
        return False, 0, "invalid_mask_shape", mask.shape
    if image.shape[:2] != mask.shape[:2]:
        return False, 0, "shape_mismatch", f"image_hw={image.shape[:2]} mask_hw={mask.shape[:2]}"
    return True, int(mask.max() > 0), "", ""


def find_image_mask_pairs(dataset_root, dataset_name=None, skipped_records=None):
    dataset_root = Path(dataset_root)
    dataset_name = dataset_name or dataset_root.name
    image_dir = dataset_root / "images"
    mask_dir = dataset_root / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing masks directory: {mask_dir}")

    masks_by_stem = {}
    for mask_path in mask_dir.rglob("*"):
        if mask_path.is_file() and mask_path.suffix.lower() in MASK_EXTS:
            masks_by_stem.setdefault(mask_path.stem, mask_path)

    pairs = []
    missing = []
    for image_path in iter_images(image_dir):
        mask_path = None
        for candidate in _mask_candidates(mask_dir, image_path):
            if candidate.is_file():
                mask_path = candidate
                break
        if mask_path is None:
            mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            missing.append(str(image_path))
            _add_skip(skipped_records, dataset_name, "missing_mask", image_path)
            continue
        ok, label, reason, detail = validate_image_mask_pair(image_path, mask_path)
        if not ok:
            _add_skip(skipped_records, dataset_name, reason, image_path, mask_path, detail)
            continue
        pairs.append((str(image_path), str(mask_path), label))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {dataset_root}")
    if missing:
        print(f"WARNING: skipped {len(missing)} images without masks. First skipped: {missing[0]}")
    if skipped_records:
        skipped_here = [item for item in skipped_records if item.get("dataset") == str(dataset_name)]
        if skipped_here:
            print(f"WARNING: skipped {len(skipped_here)} invalid pairs in {dataset_name}. First skipped: {skipped_here[0]}")
    return pairs


def split_pairs(pairs, val_ratio=0.1, seed=2026):
    rng = random.Random(seed)
    groups = {}
    for pair in pairs:
        groups.setdefault(int(pair[2]), []).append(pair)
    train_pairs = []
    val_pairs = []
    for group_pairs in groups.values():
        group_pairs = list(group_pairs)
        rng.shuffle(group_pairs)
        val_count = int(round(len(group_pairs) * val_ratio))
        if len(group_pairs) > 1:
            val_count = max(1, val_count)
        val_pairs.extend(group_pairs[:val_count])
        train_pairs.extend(group_pairs[val_count:])
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    return train_pairs, val_pairs


def write_pairs(path, pairs):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for image, mask, label in pairs:
            writer.writerow([image, mask, int(label)])
    return path


def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()
            if len(parts) == 1:
                pairs.append((parts[0], "None", -1))
            elif len(parts) == 2:
                pairs.append((parts[0], parts[1], 1))
            else:
                pairs.append((parts[0], parts[1], int(float(parts[2]))))
    return pairs


def write_mvss_list(path, pairs):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for image, mask, label in pairs:
            f.write(f"{image} {mask} {int(label)}\n")
    return path


def load_test_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_test_lists(test_json, out_dir, skipped_records=None):
    out_dir = Path(out_dir)
    test_roots = load_test_json(test_json)
    written = {}
    for name, root in test_roots.items():
        pairs = find_image_mask_pairs(root, dataset_name=name, skipped_records=skipped_records)
        list_path = out_dir / "tests" / f"{name}.txt"
        write_pairs(list_path, pairs)
        write_mvss_list(out_dir / "tests_mvss" / f"{name}.txt", pairs)
        written[name] = str(list_path)
    return written


def imread_color(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def make_jpeg_variant(image_bgr, quality):
    ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def make_noise_variant(image_bgr, sigma, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, float(sigma), image_bgr.shape)
    out = np.clip(image_bgr.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def create_robust_lists(source_list, out_dir, dataset_name="Casiav1", seed=2026):
    pairs = read_pairs(source_list)
    out_dir = Path(out_dir)
    variants = []
    variants.extend(("jpeg", q, f"jpeg_q{q}") for q in (100, 70, 50))
    variants.extend(("noise", s, f"gaussian_s{s}") for s in (5, 10, 15))

    written = {}
    for kind, value, variant_name in variants:
        variant_pairs = []
        image_out_dir = out_dir / "robust_images" / dataset_name / variant_name / "images"
        ensure_dir(image_out_dir)
        for idx, (image_path, mask_path, label) in enumerate(pairs):
            image = imread_color(image_path)
            if kind == "jpeg":
                perturbed = make_jpeg_variant(image, value)
                ext = ".jpg"
            else:
                perturbed = make_noise_variant(image, value, seed + idx)
                ext = ".png"
            out_image = image_out_dir / f"{unique_stem(image_path)}{ext}"
            cv2.imwrite(str(out_image), perturbed)
            variant_pairs.append((str(out_image), mask_path, label))
        list_path = out_dir / "robust" / f"{dataset_name}_{variant_name}.txt"
        write_pairs(list_path, variant_pairs)
        write_mvss_list(out_dir / "robust_mvss" / f"{dataset_name}_{variant_name}.txt", variant_pairs)
        written[variant_name] = str(list_path)
    return written


def read_mask(path, size=None):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if size is not None and mask.shape[:2] != size:
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def read_pred(path, size=None):
    pred = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise FileNotFoundError(f"Cannot read prediction: {path}")
    if size is not None and pred.shape[:2] != size:
        pred = cv2.resize(pred, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    pred = pred.astype(np.float32)
    if pred.max() > 1.0:
        pred /= 255.0
    return np.clip(pred, 0.0, 1.0)


def add_common_args(parser):
    parser.add_argument("--train-root", default="/data0/lzb-change-vmunet/FinalTrainData/")
    parser.add_argument("--test-json", default="/data0/lzb-change-vmunet/FMAE5.0/test_datasets_loc_small_mid_big.json")
    parser.add_argument("--work-dir", default="/data0/hl/lzb-experiments/lzb_outputs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {value}")
