import argparse
import glob
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lzb_data import LZBSegDataset
from models.mvssnet import get_mvss

os.chdir(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-weight", default="ckpt/mvssnet.pth")
    parser.add_argument("--resume-from", default="", help="Resume training from an MVSS-Net LZB checkpoint.")
    parser.add_argument("--save-last-every", type=int, default=0, help="Save last checkpoint every N epochs; 0 means final epoch only.")
    parser.add_argument("--best-save-start-epoch", type=int, default=10, help="Do not write best checkpoints before this epoch.")
    parser.add_argument("--early-stop-min-epochs", type=int, default=15, help="Minimum epochs before early stopping can trigger.")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Stop after this many epochs without meaningful val_f1 improvement; 0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum val_f1 improvement considered meaningful for early stopping.")
    return parser.parse_args()


def seed_everything(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def pixel_f1(prob, mask):
    pred = prob >= 0.5
    gt = mask > 0.5
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    return (2 * tp / (2 * tp + fp + fn + 1e-6)).item()


def validate(model, loader):
    model.eval()
    values = []
    with torch.no_grad():
        for image, mask, _label, _path in loader:
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            _edge, seg = model(image)
            seg = F.interpolate(seg, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            values.append(pixel_f1(torch.sigmoid(seg), mask))
    return float(np.mean(values)) if values else 0.0


def seg_collate(batch):
    images, masks, labels, paths = zip(*batch)
    images = torch.stack([item.contiguous() for item in images], dim=0)
    masks = torch.stack([item.contiguous() for item in masks], dim=0)
    labels = torch.stack([item for item in labels], dim=0)
    return images, masks, labels, list(paths)


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
    filename = "best_epoch{:03d}.pth".format(epoch)
    target_path = os.path.join(out_dir, filename)
    for old_path in glob.glob(os.path.join(out_dir, "best_epoch*.pth")):
        if os.path.basename(old_path) != filename:
            os.remove(old_path)
    torch.save(state, target_path)
    replace_alias(target_path, os.path.join(out_dir, "best.pth"))
    return filename


def load_resume_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state_dict)
    if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("resumed optimizer state from {}".format(path))
    else:
        print("resume checkpoint has no optimizer state; optimizer is reinitialized")
    start_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    best_f1 = float(checkpoint.get("val_f1", -1.0)) if isinstance(checkpoint, dict) else -1.0
    print("resumed MVSS-Net from {} at epoch={} val_f1={:.6f}".format(path, start_epoch, best_f1))
    return start_epoch, best_f1


def main():
    args = parse_args()
    seed_everything(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train_set = LZBSegDataset(args.train_list, args.image_size, train=True)
    val_set = LZBSegDataset(args.val_list, args.image_size, train=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=seg_collate, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=max(0, args.workers // 2), pin_memory=True, collate_fn=seg_collate, worker_init_fn=seed_worker)

    model = get_mvss(backbone="resnet50", pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3)
    if args.init_weight:
        if not os.path.isfile(args.init_weight):
            raise FileNotFoundError(
                "MVSS-Net pretrained checkpoint not found: {}. "
                "Download the official MVSS-Net checkpoint to ckpt/ or pass --init-weight '' to use only ResNet50 ImageNet pretrain.".format(args.init_weight)
            )
        checkpoint = torch.load(args.init_weight, map_location="cpu")
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        print("MVSS-Net pretrained checkpoint loaded:", args.init_weight)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    early_best_f1 = -1.0
    epochs_without_improvement = 0
    start_epoch = 0
    if args.resume_from:
        start_epoch, best_f1 = load_resume_checkpoint(args.resume_from, model, optimizer)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        print("MVSS-Net optimizer lr set to {}".format(args.lr))
        early_best_f1 = best_f1
    last_state = None
    best_path = os.path.join(args.out_dir, "best.pth")
    last_path = os.path.join(args.out_dir, "last.pth")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc="MVSS-Net epoch {}/{}".format(epoch + 1, args.epochs), leave=False, dynamic_ncols=True, mininterval=5)
        for image, mask, _label, _path in progress:
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            optimizer.zero_grad()
            edge, seg = model(image)
            edge = F.interpolate(edge, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            seg = F.interpolate(seg, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(seg, mask) + 0.2 * criterion(edge, mask)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            progress.set_postfix(loss="{:.4f}".format(float(np.mean(losses))))
        val_f1 = validate(model, val_loader)
        print("epoch={} loss={:.6f} val_f1={:.6f}".format(epoch + 1, float(np.mean(losses)), val_f1))
        state = {
            "epoch": epoch + 1,
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_f1": val_f1,
        }
        last_state = state
        if args.save_last_every > 0 and (epoch + 1) % args.save_last_every == 0:
            torch.save(state, last_path)
            print("saved periodic last.pth epoch={}".format(epoch + 1))
        if epoch + 1 < args.best_save_start_epoch:
            if val_f1 > best_f1:
                print(
                    "best candidate epoch={} val_f1={:.6f} not saved before best_save_start_epoch={}".format(
                        epoch + 1, val_f1, args.best_save_start_epoch
                    )
                )
        elif val_f1 > best_f1:
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
        print("saved final last.pth epoch={}".format(last_state["epoch"]))
        if not os.path.isfile(best_path):
            best_name = save_best_checkpoint(last_state, args.out_dir, last_state["epoch"])
            print("best.pth was missing; saved {} as best".format(best_name))


if __name__ == "__main__":
    main()
