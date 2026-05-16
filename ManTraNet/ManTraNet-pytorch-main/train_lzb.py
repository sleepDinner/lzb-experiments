import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lzb_data import LZBManTraDataset


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "MantraNet"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def patch_cv2_import():
    import cv2
    if not hasattr(cv2, "detail_ImageFeatures"):
        cv2.detail_ImageFeatures = object


def load_model(weight_path=None, use_pretrain=True):
    patch_cv2_import()
    old_cwd = os.getcwd()
    os.chdir(MODEL_DIR)
    try:
        from MantraNet.mantranet import MantraNet, pre_trained_model

        if use_pretrain:
            weight_path = weight_path or str(MODEL_DIR / "MantraNetv4.pt")
            if not Path(weight_path).is_file():
                raise FileNotFoundError(
                    "ManTraNet pretrained weight not found: {}. "
                    "Keep MantraNetv4.pt under MantraNet/ or pass --no-pretrain.".format(weight_path)
                )
            model = pre_trained_model(weight_path=weight_path, device=torch.device("cpu"))
        else:
            imtfe_weight = MODEL_DIR / "IMTFEv4.pt"
            if not imtfe_weight.is_file():
                raise FileNotFoundError("ManTraNet IMTFEv4.pt is required by the original model constructor.")
            model = MantraNet(device=torch.device("cpu"))
    finally:
        os.chdir(old_cwd)
    return model


def move_plain_tensors(module, device):
    for child in module.modules():
        for name, value in list(vars(child).items()):
            if isinstance(value, torch.Tensor):
                setattr(child, name, value.to(device))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--init-weight", default=str(MODEL_DIR / "MantraNetv4.pt"))
    parser.add_argument("--save-last-every", type=int, default=0, help="Save last checkpoint every N epochs; 0 means final epoch only.")
    parser.add_argument("--no-pretrain", dest="use_pretrain", action="store_false")
    parser.set_defaults(use_pretrain=True)
    return parser.parse_args()


def pixel_f1(prob, mask):
    pred = prob >= 0.5
    gt = mask > 0.5
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    return (2 * tp / (2 * tp + fp + fn + 1e-6)).item()


def validate(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for image, mask, _label, _path in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred = model(image)
            if pred.shape[-2:] != mask.shape[-2:]:
                pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            values.append(pixel_f1(pred, mask))
    return float(np.mean(values)) if values else 0.0


def mantra_collate(batch):
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
    out_dir = Path(out_dir)
    filename = "best_epoch{:03d}.pth".format(epoch)
    target_path = out_dir / filename
    for old_path in glob.glob(str(out_dir / "best_epoch*.pth")):
        if os.path.basename(old_path) != filename:
            os.remove(old_path)
    torch.save(state, target_path)
    replace_alias(str(target_path), str(out_dir / "best.pth"))
    return filename


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = LZBManTraDataset(args.train_list, args.image_size, train=True)
    val_set = LZBManTraDataset(args.val_list, args.image_size, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=mantra_collate)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=max(0, args.workers // 2), pin_memory=True, collate_fn=mantra_collate)

    model = load_model(args.init_weight, use_pretrain=args.use_pretrain)
    move_plain_tensors(model, device)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    best_f1 = -1.0
    last_state = None
    best_path = Path(args.out_dir) / "best.pth"
    last_path = Path(args.out_dir) / "last.pth"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc="ManTraNet epoch {}/{}".format(epoch + 1, args.epochs), leave=False, dynamic_ncols=True, mininterval=5)
        for image, mask, _label, _path in progress:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(image)
            if pred.shape[-2:] != mask.shape[-2:]:
                pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            progress.set_postfix(loss="{:.4f}".format(float(np.mean(losses))))

        val_f1 = validate(model, val_loader, device)
        print("epoch={} loss={:.6f} val_f1={:.6f}".format(epoch + 1, float(np.mean(losses)), val_f1))
        state = {"epoch": epoch + 1, "state_dict": model.module.state_dict(), "val_f1": val_f1}
        last_state = state
        if args.save_last_every > 0 and (epoch + 1) % args.save_last_every == 0:
            torch.save(state, last_path)
            print("saved periodic last.pth epoch={}".format(epoch + 1))
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = save_best_checkpoint(state, args.out_dir, epoch + 1)
            print("updated {} val_f1={:.6f}".format(best_name, best_f1))
    if last_state is not None:
        torch.save(last_state, last_path)
        print("saved final last.pth epoch={}".format(last_state["epoch"]))
        if not best_path.is_file():
            best_name = save_best_checkpoint(last_state, args.out_dir, last_state["epoch"])
            print("best.pth was missing; saved {} as best".format(best_name))


if __name__ == "__main__":
    main()
