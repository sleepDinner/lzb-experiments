import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lzb_experiments.common import prediction_filename
from lzb_data import LZBManTraDataset, save_prob_png
from train_lzb import load_model, move_plain_tensors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LZBManTraDataset(args.list_file, args.image_size, train=False)
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.workers, pin_memory=True)
    model = load_model(use_pretrain=False)
    checkpoint = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    move_plain_tensors(model, device)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    model.eval()

    with torch.no_grad():
        for image, _mask, _label, paths in tqdm(loader):
            image = image.to(device, non_blocking=True)
            pred = model(image)
            for idx, image_path in enumerate(paths):
                raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                target_size = raw.shape[:2] if raw is not None else (args.image_size, args.image_size)
                prob = F.interpolate(pred[idx:idx + 1], size=target_size, mode="bilinear", align_corners=False)
                prob = prob[0, 0].detach().cpu().numpy()
                save_prob_png(Path(args.out_dir) / prediction_filename(image_path), prob)


if __name__ == "__main__":
    main()
