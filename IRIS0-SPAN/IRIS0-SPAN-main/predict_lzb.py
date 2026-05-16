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
    for image_path, _mask_path, _label in tqdm(read_pairs(args.list_file)):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        x = np.expand_dims(resized.astype("float32") / 255.0 * 2.0 - 1.0, axis=0)
        pred = model.predict(x, verbose=0)[0, ..., 0]
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / prediction_filename(image_path)), (np.clip(pred, 0, 1) * 255).astype("uint8"))


if __name__ == "__main__":
    main()
