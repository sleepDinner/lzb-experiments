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

from lzb_data import LZBPSCCDataset, multiscale_masks
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args

os.chdir(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--lr-strategy",
        default="original",
        help="Comma-separated epoch-stage learning rates. Use 'original' for PSCC-Net config or 'none' for fixed --lr.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume-from", default="", help="Resume training from a PSCC-Net checkpoint directory, prefix, or component file.")
    parser.add_argument("--save-last-every", type=int, default=0, help="Save last checkpoints every N epochs; 0 means final epoch only.")
    parser.add_argument("--best-save-start-epoch", type=int, default=6, help="Do not write best checkpoints before this epoch.")
    parser.add_argument("--early-stop-min-epochs", type=int, default=8, help="Minimum epochs before early stopping can trigger.")
    parser.add_argument("--early-stop-patience", type=int, default=6, help="Stop after this many epochs without meaningful val_f1 improvement; 0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum val_f1 improvement considered meaningful for early stopping.")
    parser.add_argument("--no-pretrain", dest="use_pretrain", action="store_false")
    parser.set_defaults(use_pretrain=True)
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


def match_mask_size(pred, mask):
    if pred.shape[-2:] != mask.shape[-2:]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=True)
    return pred


def validate(FENet, SegNet, loader):
    FENet.eval()
    SegNet.eval()
    values = []
    with torch.no_grad():
        for image, mask, _cls, _path in loader:
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            feat = FENet(image)
            pred = SegNet(feat)[0].squeeze(1)
            pred = F.interpolate(pred.unsqueeze(1), size=mask.shape[-2:], mode="bilinear", align_corners=True).squeeze(1)
            values.append(pixel_f1(pred, mask))
    return float(np.mean(values)) if values else 0.0


def pscc_collate(batch):
    images, masks, labels, paths = zip(*batch)
    images = torch.stack([item.contiguous() for item in images], dim=0)
    masks = torch.stack([item.contiguous() for item in masks], dim=0)
    labels = torch.stack([item for item in labels], dim=0)
    return images, masks, labels, list(paths)


def save_state(out_dir, FENet, SegNet, ClsNet, epoch, val_f1, lr, name, optimizer=None):
    state = {"epoch": epoch, "val_f1": val_f1, "lr": lr}
    paths = {
        "FENet": os.path.join(out_dir, f"{name}_FENet.pth"),
        "SegNet": os.path.join(out_dir, f"{name}_SegNet.pth"),
        "ClsNet": os.path.join(out_dir, f"{name}_ClsNet.pth"),
    }
    torch.save({**state, "state_dict": FENet.module.state_dict()}, paths["FENet"])
    torch.save({**state, "state_dict": SegNet.module.state_dict()}, paths["SegNet"])
    cls_state = {**state, "state_dict": ClsNet.module.state_dict()}
    if optimizer is not None:
        cls_state["optimizer"] = optimizer.state_dict()
    torch.save(cls_state, paths["ClsNet"])
    return paths


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


def save_best_state(out_dir, FENet, SegNet, ClsNet, epoch, val_f1, lr, optimizer=None):
    prefix = "best_epoch{:03d}".format(epoch)
    for old_path in glob.glob(os.path.join(out_dir, "best_epoch*_*.pth")):
        os.remove(old_path)
    paths = save_state(out_dir, FENet, SegNet, ClsNet, epoch, val_f1, lr, prefix, optimizer=optimizer)
    for component, path in paths.items():
        replace_alias(path, os.path.join(out_dir, f"best_{component}.pth"))
    return prefix


def load_state_dict_flexible(net, state):
    targets = [net]
    if hasattr(net, "module"):
        targets.append(net.module)
    variants = [state]
    if all(str(key).startswith("module.") for key in state.keys()):
        variants.append({key[len("module."):]: value for key, value in state.items()})
    last_error = None
    for target in targets:
        for variant in variants:
            try:
                target.load_state_dict(variant)
                return
            except RuntimeError as exc:
                last_error = exc
    raise last_error


def load_pretrained_or_fail(FENet, SegNet, ClsNet):
    weight_specs = [
        (FENet, Path("checkpoint/HRNet_checkpoint/HRNet.pth"), "HRNet"),
        (SegNet, Path("checkpoint/NLCDetection_checkpoint/NLCDetection.pth"), "NLCDetection"),
        (ClsNet, Path("checkpoint/DetectionHead_checkpoint/DetectionHead.pth"), "DetectionHead"),
    ]
    missing = [str(path) for _net, path, _name in weight_specs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "PSCC-Net pretrained weights not found: {}. "
            "Download PSCC-Net checkpoint weights or pass --no-pretrain.".format(", ".join(missing))
        )
    for net, path, name in weight_specs:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        load_state_dict_flexible(net, state)
        print("{} pretrained weight loaded: {}".format(name, path))


def resolve_resume_paths(resume_from):
    path = Path(resume_from)
    components = ("FENet", "SegNet", "ClsNet")
    if path.is_dir():
        for name in ("last", "best"):
            paths = {component: path / f"{name}_{component}.pth" for component in components}
            if all(item.is_file() for item in paths.values()):
                return paths
        epoch_candidates = sorted(path.glob("best_epoch*_FENet.pth"))
        for fenet_path in reversed(epoch_candidates):
            prefix = str(fenet_path)[: -len("_FENet.pth")]
            paths = {component: Path(f"{prefix}_{component}.pth") for component in components}
            if all(item.is_file() for item in paths.values()):
                return paths
    elif path.is_file() and path.name.endswith("_FENet.pth"):
        prefix = str(path)[: -len("_FENet.pth")]
        paths = {component: Path(f"{prefix}_{component}.pth") for component in components}
        if all(item.is_file() for item in paths.values()):
            return paths
    else:
        paths = {component: Path(f"{resume_from}_{component}.pth") for component in components}
        if all(item.is_file() for item in paths.values()):
            return paths
    raise FileNotFoundError("Could not resolve PSCC-Net resume checkpoint from: {}".format(resume_from))


def load_resume_checkpoint(resume_from, FENet, SegNet, ClsNet, optimizer):
    paths = resolve_resume_paths(resume_from)
    nets = {"FENet": FENet, "SegNet": SegNet, "ClsNet": ClsNet}
    checkpoints = {}
    for component, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu")
        checkpoints[component] = checkpoint
        state_dict = checkpoint.get("state_dict", checkpoint)
        load_state_dict_flexible(nets[component], state_dict)
        print("resumed {} from {}".format(component, path))
    optimizer_loaded = False
    for checkpoint in checkpoints.values():
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_loaded = True
            break
    if optimizer_loaded:
        print("resumed optimizer state from PSCC-Net checkpoint")
    else:
        print("resume checkpoint has no optimizer state; optimizer is reinitialized")
    start_epoch = max(int(item.get("epoch", 0)) for item in checkpoints.values() if isinstance(item, dict))
    best_f1 = max(float(item.get("val_f1", -1.0)) for item in checkpoints.values() if isinstance(item, dict))
    print("resumed PSCC-Net at epoch={} val_f1={:.6f}".format(start_epoch, best_f1))
    return start_epoch, best_f1


def parse_lr_strategy(value, pscc_args):
    value = str(value).strip()
    if not value or value.lower() in {"none", "fixed", "off"}:
        return []
    if value.lower() == "original":
        return [float(lr) for lr in pscc_args.lr_strategy]
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def lr_for_epoch(epoch, total_epochs, base_lr, lr_strategy):
    if not lr_strategy:
        return base_lr
    stage_epochs = max(1, int(np.ceil(float(total_epochs) / float(len(lr_strategy)))))
    stage_idx = min(epoch // stage_epochs, len(lr_strategy) - 1)
    return lr_strategy[stage_idx]


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def mask_balance(mask):
    balance = torch.ones_like(mask)
    if (mask == 1).sum():
        balance[mask == 1] = 0.5 / ((mask == 1).sum().to(torch.float) / mask.numel())
        balance[mask == 0] = 0.5 / ((mask == 0).sum().to(torch.float) / mask.numel())
    else:
        print("Mask balance is not working!")
    return balance


def main():
    args_cli = parse_args()
    seed_everything(args_cli.seed)
    Path(args_cli.out_dir).mkdir(parents=True, exist_ok=True)
    pscc_args = get_pscc_args()
    pscc_args.defrost()
    pscc_args.crop_size = [args_cli.image_size, args_cli.image_size]
    pscc_args.freeze()
    lr_strategy = parse_lr_strategy(args_cli.lr_strategy, pscc_args)
    if lr_strategy:
        stage_epochs = max(1, int(np.ceil(float(args_cli.epochs) / float(len(lr_strategy)))))
        print("PSCC-Net lr_strategy={} stage_epochs={}".format(lr_strategy, stage_epochs))
    else:
        print("PSCC-Net fixed lr={}".format(args_cli.lr))

    train_set = LZBPSCCDataset(args_cli.train_list, args_cli.image_size, train=True)
    val_set = LZBPSCCDataset(args_cli.val_list, args_cli.image_size, train=False)
    generator = torch.Generator()
    generator.manual_seed(args_cli.seed)
    train_loader = DataLoader(train_set, batch_size=args_cli.batch_size, shuffle=True, num_workers=args_cli.workers, pin_memory=True, collate_fn=pscc_collate, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=max(0, args_cli.workers // 2), pin_memory=True, collate_fn=pscc_collate, worker_init_fn=seed_worker)

    FENet = get_seg_model(get_hrnet_cfg()).cuda()
    SegNet = NLCDetection(pscc_args).cuda()
    ClsNet = DetectionHead(pscc_args).cuda()
    params = list(FENet.parameters()) + list(SegNet.parameters()) + list(ClsNet.parameters())
    optimizer = torch.optim.Adam(params, lr=lr_for_epoch(0, args_cli.epochs, args_cli.lr, lr_strategy))
    FENet = torch.nn.DataParallel(FENet, device_ids=list(range(torch.cuda.device_count())))
    SegNet = torch.nn.DataParallel(SegNet, device_ids=list(range(torch.cuda.device_count())))
    ClsNet = torch.nn.DataParallel(ClsNet, device_ids=list(range(torch.cuda.device_count())))
    if args_cli.use_pretrain:
        load_pretrained_or_fail(FENet, SegNet, ClsNet)
    authentic_ratio = float(pscc_args.train_ratio[0])
    fake_ratio = 1.0 - authentic_ratio
    ce_weights = torch.tensor([1.0 / authentic_ratio, 1.0 / fake_ratio], dtype=torch.float32).cuda()
    bce_full = torch.nn.BCELoss(reduction="none")
    ce = torch.nn.CrossEntropyLoss(weight=ce_weights)

    best_f1 = -1.0
    early_best_f1 = -1.0
    epochs_without_improvement = 0
    start_epoch = 0
    if args_cli.resume_from:
        start_epoch, best_f1 = load_resume_checkpoint(args_cli.resume_from, FENet, SegNet, ClsNet, optimizer)
        early_best_f1 = best_f1
    last_state = None
    best_fenet_path = os.path.join(args_cli.out_dir, "best_FENet.pth")
    for epoch in range(start_epoch, args_cli.epochs):
        current_lr = lr_for_epoch(epoch, args_cli.epochs, args_cli.lr, lr_strategy)
        set_optimizer_lr(optimizer, current_lr)
        FENet.train()
        SegNet.train()
        ClsNet.train()
        losses = []
        progress = tqdm(train_loader, desc="PSCC-Net epoch {}/{}".format(epoch + 1, args_cli.epochs), leave=False, dynamic_ncols=True, mininterval=5)
        for image, mask, cls, _path in progress:
            image = image.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            cls[cls != 0] = 1
            cls = cls.cuda(non_blocking=True)
            mask1, mask2, mask3, mask4 = multiscale_masks(mask)
            mask1_balance = mask_balance(mask1)
            mask2_balance = mask_balance(mask2)
            mask3_balance = mask_balance(mask3)
            mask4_balance = mask_balance(mask4)
            optimizer.zero_grad()
            feat = FENet(image)
            pred1, pred2, pred3, pred4 = SegNet(feat)
            pred1 = match_mask_size(pred1, mask1.unsqueeze(1))
            pred2 = match_mask_size(pred2, mask2.unsqueeze(1))
            pred3 = match_mask_size(pred3, mask3.unsqueeze(1))
            pred4 = match_mask_size(pred4, mask4.unsqueeze(1))
            pred_logit = ClsNet(feat)
            loss = (
                torch.mean(bce_full(pred1.squeeze(1), mask1) * mask1_balance)
                + torch.mean(bce_full(pred2.squeeze(1), mask2) * mask2_balance)
                + torch.mean(bce_full(pred3.squeeze(1), mask3) * mask3_balance)
                + torch.mean(bce_full(pred4.squeeze(1), mask4) * mask4_balance)
                + ce(pred_logit, cls)
            )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            progress.set_postfix(lr="{:.2g}".format(current_lr), loss="{:.4f}".format(float(np.mean(losses))))
        val_f1 = validate(FENet, SegNet, val_loader)
        print("epoch={} lr={:.8g} loss={:.6f} val_f1={:.6f}".format(epoch + 1, current_lr, float(np.mean(losses)), val_f1))
        last_state = (epoch + 1, val_f1, current_lr)
        if args_cli.save_last_every > 0 and (epoch + 1) % args_cli.save_last_every == 0:
            save_state(args_cli.out_dir, FENet, SegNet, ClsNet, epoch + 1, val_f1, current_lr, "last", optimizer=optimizer)
            print("saved periodic last checkpoints epoch={}".format(epoch + 1))
        if epoch + 1 < args_cli.best_save_start_epoch:
            if val_f1 > best_f1:
                print(
                    "best candidate epoch={} val_f1={:.6f} not saved before best_save_start_epoch={}".format(
                        epoch + 1, val_f1, args_cli.best_save_start_epoch
                    )
                )
        elif val_f1 > best_f1:
            best_f1 = val_f1
            best_prefix = save_best_state(args_cli.out_dir, FENet, SegNet, ClsNet, epoch + 1, val_f1, current_lr, optimizer=optimizer)
            print("updated {} checkpoints val_f1={:.6f}".format(best_prefix, best_f1))
        if args_cli.early_stop_patience > 0:
            if val_f1 > early_best_f1 + args_cli.early_stop_min_delta:
                early_best_f1 = val_f1
                epochs_without_improvement = 0
            elif epoch + 1 >= args_cli.early_stop_min_epochs:
                epochs_without_improvement += 1
                print(
                    "early_stop patience={}/{} best_val_f1={:.6f} min_delta={:.6g}".format(
                        epochs_without_improvement,
                        args_cli.early_stop_patience,
                        early_best_f1,
                        args_cli.early_stop_min_delta,
                    )
                )
                if epochs_without_improvement >= args_cli.early_stop_patience:
                    print("early stopping at epoch {} best_val_f1={:.6f}".format(epoch + 1, best_f1))
                    break
    if last_state is not None:
        epoch, val_f1, current_lr = last_state
        save_state(args_cli.out_dir, FENet, SegNet, ClsNet, epoch, val_f1, current_lr, "last", optimizer=optimizer)
        print("saved final last checkpoints epoch={}".format(epoch))
        if not os.path.isfile(best_fenet_path):
            best_prefix = save_best_state(args_cli.out_dir, FENet, SegNet, ClsNet, epoch, val_f1, current_lr, optimizer=optimizer)
            print("best checkpoints were missing; saved {} checkpoints as best".format(best_prefix))


if __name__ == "__main__":
    main()
