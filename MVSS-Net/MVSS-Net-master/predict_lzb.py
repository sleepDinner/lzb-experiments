import argparse
import os
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lzb_experiments.common import prediction_filename
from lzb_data import LZBSegDataset, save_prob_png
from models.mvssnet import get_mvss

os.chdir(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    dataset = LZBSegDataset(args.list_file, args.image_size, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = get_mvss(backbone="resnet50", pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3)
    checkpoint = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
    model.eval()
    with torch.no_grad():
        for image, _mask, _label, paths in tqdm(loader):
            image = image.cuda(non_blocking=True)
            _edge, seg = model(image)
            seg = F.interpolate(seg, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            prob = torch.sigmoid(seg)[0, 0].detach().cpu().numpy()
            image_path = paths[0]
            raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if raw is not None and raw.shape[:2] != prob.shape:
                prob = cv2.resize(prob, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
            save_prob_png(Path(args.out_dir) / prediction_filename(image_path), prob)


if __name__ == "__main__":
    main()
