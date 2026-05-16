import argparse
import os
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lzb_experiments.common import prediction_filename
from lzb_data import LZBPSCCDataset, save_prob_png
from models.NLCDetection import NLCDetection
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args

os.chdir(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-file", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args_cli = parse_args()
    Path(args_cli.out_dir).mkdir(parents=True, exist_ok=True)
    pscc_args = get_pscc_args()
    pscc_args.defrost()
    pscc_args.crop_size = [args_cli.image_size, args_cli.image_size]
    pscc_args.freeze()

    dataset = LZBPSCCDataset(args_cli.list_file, args_cli.image_size, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args_cli.workers, pin_memory=True)
    FENet = get_seg_model(get_hrnet_cfg()).cuda()
    SegNet = NLCDetection(pscc_args).cuda()
    FENet.load_state_dict(torch.load(Path(args_cli.checkpoint_dir) / "best_FENet.pth", map_location="cpu")["state_dict"])
    SegNet.load_state_dict(torch.load(Path(args_cli.checkpoint_dir) / "best_SegNet.pth", map_location="cpu")["state_dict"])
    FENet = torch.nn.DataParallel(FENet, device_ids=list(range(torch.cuda.device_count())))
    SegNet = torch.nn.DataParallel(SegNet, device_ids=list(range(torch.cuda.device_count())))
    FENet.eval()
    SegNet.eval()
    with torch.no_grad():
        for image, _mask, _cls, paths in tqdm(loader):
            image = image.cuda(non_blocking=True)
            feat = FENet(image)
            pred = SegNet(feat)[0]
            raw = cv2.imread(paths[0], cv2.IMREAD_COLOR)
            target_size = raw.shape[:2] if raw is not None else (args_cli.image_size, args_cli.image_size)
            pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=True)
            prob = pred[0, 0].detach().cpu().numpy()
            save_prob_png(Path(args_cli.out_dir) / prediction_filename(paths[0]), prob)


if __name__ == "__main__":
    main()
