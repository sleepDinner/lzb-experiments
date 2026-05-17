import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from keras_compat import apply_compat
apply_compat()

from lzb_experiments.common import prediction_filename
from lzb_data import read_pairs
from models import ManTraNetv3 as mm

os.chdir(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/config_CASIA_RESIZE_02.json")
    parser.add_argument("--mantranet-pretrain", default="pretrained_weights/ManTraNet_Ptrain4.h5")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mm.json = json
    project = mm.ManTraNet(args.config)
    if not os.path.isfile(args.mantranet_pretrain):
        raise FileNotFoundError("IRIS0-SPAN ManTraNet pretrain not found: {}".format(args.mantranet_pretrain))
    project.weight_file = args.mantranet_pretrain
    model = project.get_model_1010_resize()
    model.load_weights(args.model_file)
    pairs = read_pairs(args.list_file)
    batch_size = max(1, args.batch_size)
    for start in tqdm(range(0, len(pairs), batch_size)):
        batch = pairs[start:start + batch_size]
        images = []
        shapes = []
        paths = []
        for image_path, _mask_path, _label in batch:
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(image_rgb, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
            images.append(resized.astype("float32") / 255.0 * 2.0 - 1.0)
            shapes.append((h, w))
            paths.append(image_path)
        preds = model.predict(np.stack(images, axis=0), batch_size=len(images), verbose=0)
        for idx, image_path in enumerate(paths):
            h, w = shapes[idx]
            pred = preds[idx, ..., 0]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_dir / prediction_filename(image_path)), (np.clip(pred, 0, 1) * 255).astype("uint8"))


if __name__ == "__main__":
    main()
