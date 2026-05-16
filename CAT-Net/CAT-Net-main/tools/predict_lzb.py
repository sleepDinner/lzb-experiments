import argparse
import os
import sys
from pathlib import Path

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
if path not in sys.path:
    sys.path.insert(0, path)
os.chdir(path)

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lzb_experiments.common import prediction_filename
from lib import models
from lib.config import config, update_config
from Splicing.data.dataset_lzb import LZBPairDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cfg", default="experiments/CAT_full.yaml")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def cat_collate(batch):
    images, labels, qtables = zip(*batch)
    images = torch.stack([item.contiguous() for item in images], dim=0)
    labels = torch.stack([item.contiguous() for item in labels], dim=0)
    qtables = torch.stack([item.contiguous() for item in qtables], dim=0)
    return images, labels, qtables


def main():
    args = parse_args()
    cfg_args = argparse.Namespace(cfg=args.cfg, opts=None)
    update_config(config, cfg_args)
    config.defrost()
    config.TEST.IMAGE_SIZE = [args.image_size, args.image_size]
    config.TRAIN.IMAGE_SIZE = [args.image_size, args.image_size]
    config.freeze()
    cudnn.benchmark = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LZBPairDataset(
        None,
        True,
        ("RGB", "DCTvol", "qtable"),
        1,
        args.list_file,
        read_from_jpeg=True,
        resize_to=(args.image_size, args.image_size),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=cat_collate)
    model = eval("models." + config.MODEL.NAME + ".get_seg_model")(config)
    checkpoint = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
    model.eval()

    with torch.no_grad():
        for index, (image, _label, qtable) in enumerate(tqdm(loader)):
            image_path, _ = dataset.tamp_list[index]
            image = image.cuda(non_blocking=True)
            qtable = qtable.cuda(non_blocking=True)
            logits = model(image, qtable)
            prob = torch.softmax(logits, dim=1)[:, 1]
            raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            target_size = raw.shape[:2] if raw is not None else _label.shape[-2:]
            prob = F.interpolate(prob.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False)
            pred = prob.squeeze().detach().cpu().numpy()
            cv2.imwrite(str(out_dir / prediction_filename(image_path)), (pred * 255).clip(0, 255).astype("uint8"))


if __name__ == "__main__":
    main()
