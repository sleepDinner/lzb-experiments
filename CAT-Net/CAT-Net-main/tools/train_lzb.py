import argparse
import glob
import os
import shutil
import sys
import time

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
if path not in sys.path:
    sys.path.insert(0, path)
os.chdir(path)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib import models
from lib.config import config, update_config
from lib.core.criterion import CrossEntropy
from Splicing.data.dataset_lzb import LZBPairDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cfg", default="experiments/CAT_full.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-last-every", type=int, default=0, help="Save last checkpoint every N epochs; 0 means final epoch only.")
    parser.add_argument("--early-stop-min-epochs", type=int, default=20, help="Minimum epochs before early stopping can trigger.")
    parser.add_argument("--early-stop-patience", type=int, default=12, help="Stop after this many epochs without meaningful val_f1 improvement; 0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum val_f1 improvement considered meaningful for early stopping.")
    parser.add_argument("--no-pretrain", dest="use_pretrain", action="store_false")
    parser.set_defaults(use_pretrain=True)
    return parser.parse_args()


def configure(args):
    cfg_args = argparse.Namespace(cfg=args.cfg, opts=None)
    update_config(config, cfg_args)
    config.defrost()
    config.TRAIN.IMAGE_SIZE = [args.image_size, args.image_size]
    config.TEST.IMAGE_SIZE = [args.image_size, args.image_size]
    config.TRAIN.BATCH_SIZE_PER_GPU = args.batch_size
    config.TRAIN.END_EPOCH = args.epochs
    config.TRAIN.LR = args.lr
    config.WORKERS = args.workers
    if not args.use_pretrain:
        config.MODEL.PRETRAINED_RGB = ""
        config.MODEL.PRETRAINED_DCT = ""
    config.freeze()


def f1_from_logits(logits, labels):
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    valid = labels >= 0
    pred = (torch.softmax(logits, dim=1)[:, 1] >= 0.5) & valid
    gt = (labels > 0) & valid
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    return (2 * tp / (2 * tp + fp + fn + 1e-6)).item()


def validate(model, loader):
    model.eval()
    values = []
    with torch.no_grad():
        for image, label, qtable in loader:
            image = image.cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            qtable = qtable.cuda(non_blocking=True)
            logits = model(image, qtable)
            values.append(f1_from_logits(logits, label))
    return float(np.mean(values)) if values else 0.0


def cat_collate(batch):
    images, labels, qtables = zip(*batch)
    images = torch.stack([item.contiguous() for item in images], dim=0)
    labels = torch.stack([item.contiguous() for item in labels], dim=0)
    qtables = torch.stack([item.contiguous() for item in qtables], dim=0)
    return images, labels, qtables


def replace_alias(target_path, alias_path):
    if os.path.lexists(alias_path):
        os.remove(alias_path)
    rel_target = os.path.basename(target_path)
    try:
        os.symlink(rel_target, alias_path)
        return
    except OSError:
        pass
    try:
        os.link(target_path, alias_path)
        return
    except OSError:
        pass
    shutil.copy2(target_path, alias_path)


def save_best_checkpoint(state, out_dir, epoch):
    filename = "best_epoch{:03d}.pth.tar".format(epoch)
    target_path = os.path.join(out_dir, filename)
    for old_path in glob.glob(os.path.join(out_dir, "best_epoch*.pth.tar")):
        if os.path.basename(old_path) != filename:
            os.remove(old_path)
    torch.save(state, target_path)
    replace_alias(target_path, os.path.join(out_dir, "best.pth.tar"))
    return filename


def main():
    args = parse_args()
    configure(args)
    if args.use_pretrain:
        for weight_path in (config.MODEL.PRETRAINED_RGB, config.MODEL.PRETRAINED_DCT):
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    "CAT-Net pretrained weight not found: {}. "
                    "Download CAT-Net pretrained_models or pass --no-pretrain.".format(weight_path)
                )
    os.makedirs(args.out_dir, exist_ok=True)

    cudnn.benchmark = True
    crop_size = (args.image_size, args.image_size)
    blocks = ("RGB", "DCTvol", "qtable")
    train_set = LZBPairDataset(crop_size, True, blocks, 1, args.train_list, read_from_jpeg=True)
    val_set = LZBPairDataset(
        None,
        True,
        blocks,
        1,
        args.val_list,
        read_from_jpeg=True,
        resize_to=(args.image_size, args.image_size),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=cat_collate)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=max(0, args.workers // 2), pin_memory=True, collate_fn=cat_collate)

    model = eval("models." + config.MODEL.NAME + ".get_seg_model")(config)
    device_ids = list(range(torch.cuda.device_count()))
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL, weight=torch.FloatTensor([0.5, 2.5])).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    best_f1 = -1.0
    early_best_f1 = -1.0
    epochs_without_improvement = 0
    last_state = None
    best_path = os.path.join(args.out_dir, "best.pth.tar")
    last_path = os.path.join(args.out_dir, "last.pth.tar")
    for epoch in range(args.epochs):
        model.train()
        loss_meter = []
        started = time.time()
        progress = tqdm(train_loader, desc="CAT-Net epoch {}/{}".format(epoch + 1, args.epochs), leave=False, dynamic_ncols=True, mininterval=5)
        for image, label, qtable in progress:
            image = image.cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            qtable = qtable.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits = model(image, qtable)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            loss_meter.append(loss_value)
            progress.set_postfix(loss="{:.4f}".format(float(np.mean(loss_meter))))
        val_f1 = validate(model, val_loader)
        print("epoch={} loss={:.6f} val_f1={:.6f} time={:.1f}s".format(
            epoch + 1, float(np.mean(loss_meter)), val_f1, time.time() - started
        ))
        state = {
            "epoch": epoch + 1,
            "state_dict": model.module.state_dict(),
            "val_f1": val_f1,
            "config": args.cfg,
        }
        last_state = state
        if args.save_last_every > 0 and (epoch + 1) % args.save_last_every == 0:
            torch.save(state, last_path)
            print("saved periodic last.pth.tar epoch={}".format(epoch + 1))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = save_best_checkpoint(state, args.out_dir, epoch + 1)
            print("updated {} val_f1={:.6f}".format(best_name, best_f1))
        if args.early_stop_patience > 0:
            if val_f1 > early_best_f1 + args.early_stop_min_delta:
                early_best_f1 = val_f1
                epochs_without_improvement = 0
            elif epoch + 1 >= args.early_stop_min_epochs:
                epochs_without_improvement += 1
                print(
                    "early_stop patience={}/{} best_val_f1={:.6f} min_delta={:.6g}".format(
                        epochs_without_improvement,
                        args.early_stop_patience,
                        early_best_f1,
                        args.early_stop_min_delta,
                    )
                )
                if epochs_without_improvement >= args.early_stop_patience:
                    print("early stopping at epoch {} best_val_f1={:.6f}".format(epoch + 1, best_f1))
                    break
    if last_state is not None:
        torch.save(last_state, last_path)
        print("saved final last.pth.tar epoch={}".format(last_state["epoch"]))
        if not os.path.isfile(best_path):
            best_name = save_best_checkpoint(last_state, args.out_dir, last_state["epoch"])
            print("best.pth.tar was missing; saved {} as best".format(best_name))


if __name__ == "__main__":
    main()
