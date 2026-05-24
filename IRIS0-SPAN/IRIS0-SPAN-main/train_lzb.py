import argparse
import glob
import json
import os
import random
import re
import shutil
from pathlib import Path

from keras_compat import apply_compat
apply_compat()

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from lzb_data import LZBSequence
from models import ManTraNetv3 as mm

os.chdir(Path(__file__).resolve().parent)


class SafeF1(keras.metrics.Metric):
    def __init__(self, name="F1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.inv_tp = self.add_weight(name="inv_tp", initializer="zeros")
        self.inv_fp = self.add_weight(name="inv_fp", initializer="zeros")
        self.inv_fn = self.add_weight(name="inv_fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt = tf.cast(y_true > 0.5, tf.bool)
        pred = tf.cast(y_pred > 0.5, tf.bool)
        inv_pred = tf.cast((1.0 - y_pred) > 0.5, tf.bool)

        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(pred, gt), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(pred, tf.logical_not(gt)), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(pred), gt), tf.float32)))
        self.inv_tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(inv_pred, gt), tf.float32)))
        self.inv_fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(inv_pred, tf.logical_not(gt)), tf.float32)))
        self.inv_fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(inv_pred), gt), tf.float32)))

    def result(self):
        denom = 2.0 * self.tp + self.fp + self.fn
        inv_denom = 2.0 * self.inv_tp + self.inv_fp + self.inv_fn
        f1 = tf.where(denom > 0.0, 2.0 * self.tp / denom, 0.0)
        inv_f1 = tf.where(inv_denom > 0.0, 2.0 * self.inv_tp / inv_denom, 0.0)
        return tf.maximum(f1, inv_f1)

    def reset_state(self):
        for value in (self.tp, self.fp, self.fn, self.inv_tp, self.inv_fp, self.inv_fn):
            value.assign(0.0)

    def reset_states(self):
        self.reset_state()


def _np_safe_auc(y_true, y_pred):
    scores = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for yy_true, yy_pred in zip(y_true, y_pred):
        gt = (yy_true > 0.5).astype("int").ravel()
        if gt.size == 0 or gt.min() == gt.max():
            continue
        pred = yy_pred.ravel()
        this = roc_auc_score(gt, pred)
        that = roc_auc_score(gt, 1.0 - pred)
        scores.append(max(this, that))
    return np.float32(np.mean(scores) if scores else 0.5)


def auroc(y_true, y_pred):
    return tf.py_func(_np_safe_auc, [y_true, y_pred], "float32")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/config_CASIA_RESIZE_02.json")
    parser.add_argument("--mantranet-pretrain", default="pretrained_weights/ManTraNet_Ptrain4.h5")
    parser.add_argument("--init-weight", default="PixelAttention32.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume-from", default="", help="Resume training from an IRIS0-SPAN .h5 checkpoint.")
    parser.add_argument("--resume-epoch", type=int, default=-1, help="Epoch to resume from when checkpoint metadata is unavailable.")
    parser.add_argument("--save-last-every", type=int, default=1, help="Save last.h5 every N epochs; 0 means final epoch only.")
    parser.add_argument("--best-save-start-epoch", type=int, default=10, help="Do not write best checkpoints before this epoch.")
    parser.add_argument("--early-stop-min-epochs", type=int, default=25, help="Minimum epochs before early stopping can trigger.")
    parser.add_argument("--early-stop-patience", type=int, default=15, help="Stop after this many epochs without meaningful val_F1 improvement; 0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum val_F1 improvement considered meaningful for early stopping.")
    return parser.parse_args()


def seed_everything(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except AttributeError:
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass


def load_initial_weights(model, weight_path):
    try:
        model.load_weights(weight_path)
        print("loaded initial weight:", weight_path)
        return
    except ValueError as exc:
        if "Layer count mismatch" not in str(exc):
            raise

    # Some original IRIS0-SPAN checkpoints were saved from an outer wrapper
    # model that contains the actual sigNet model as a single nested layer.
    wrapper_input = keras.layers.Input(shape=model.input_shape[1:], name="lzb_init_input")
    wrapper = keras.models.Model(wrapper_input, model(wrapper_input), name="lzb_init_wrapper")
    wrapper.load_weights(weight_path)
    print("loaded initial weight through nested wrapper:", weight_path)


def metadata_path_for(weight_path):
    base = os.path.splitext(weight_path)[0]
    return base + ".json"


def infer_resume_epoch(weight_path, explicit_epoch=-1):
    if explicit_epoch >= 0:
        return int(explicit_epoch)
    meta_path = metadata_path_for(weight_path)
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("epoch", 0))
    match = re.search(r"epoch(\d+)", os.path.basename(weight_path))
    if match:
        return int(match.group(1))
    return 0


def write_checkpoint_metadata(weight_path, epoch, logs=None):
    logs = logs or {}
    serializable = {}
    for key, value in logs.items():
        try:
            serializable[key] = float(value)
        except (TypeError, ValueError):
            serializable[key] = str(value)
    serializable["epoch"] = int(epoch)
    with open(metadata_path_for(weight_path), "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)


class LastCheckpoint(keras.callbacks.Callback):
    def __init__(self, out_dir, save_every=1):
        super().__init__()
        self.path = os.path.join(out_dir, "last.h5")
        self.save_every = int(save_every)
        self.last_epoch = 0
        self.last_logs = {}

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        self.last_epoch = current_epoch
        self.last_logs = logs or {}
        if self.save_every > 0 and current_epoch % self.save_every == 0:
            self.model.save_weights(self.path)
            write_checkpoint_metadata(self.path, current_epoch, logs)
            print("saved periodic last.h5 epoch={}".format(current_epoch))

    def on_train_end(self, logs=None):
        if self.last_epoch > 0:
            self.model.save_weights(self.path)
            write_checkpoint_metadata(self.path, self.last_epoch, self.last_logs)
            print("saved final last.h5 epoch={}".format(self.last_epoch))


class DelayedBestCheckpoint(keras.callbacks.Callback):
    def __init__(self, out_dir, monitor="val_F1", start_epoch=1, min_delta=0.0):
        super().__init__()
        self.out_dir = out_dir
        self.monitor = monitor
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.best = float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        current_epoch = epoch + 1
        if current is None:
            return
        current = float(current)
        if current_epoch < self.start_epoch:
            if current > self.best + self.min_delta:
                print(
                    "best candidate epoch={} {}={:.6f} not saved before best_save_start_epoch={}".format(
                        current_epoch, self.monitor, current, self.start_epoch
                    )
                )
            return
        if current <= self.best + self.min_delta:
            return
        self.best = current
        filename = "best_epoch{:03d}.h5".format(current_epoch)
        target_path = os.path.join(self.out_dir, filename)
        for old_path in glob.glob(os.path.join(self.out_dir, "best_epoch*.h5")):
            if os.path.basename(old_path) != filename:
                os.remove(old_path)
        self.model.save_weights(target_path)
        write_checkpoint_metadata(target_path, current_epoch, logs)
        alias_path = os.path.join(self.out_dir, "best.h5")
        shutil.copy2(target_path, alias_path)
        write_checkpoint_metadata(alias_path, current_epoch, logs)
        print("updated {} {}={:.6f}".format(filename, self.monitor, current))


class MinEpochEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor="val_F1", min_epochs=25, patience=15, min_delta=1e-4):
        super().__init__()
        self.monitor = monitor
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("-inf")
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None or self.patience <= 0:
            return
        current = float(current)
        current_epoch = epoch + 1
        if current > self.best + self.min_delta:
            self.best = current
            self.wait = 0
            return
        if current_epoch < self.min_epochs:
            return
        self.wait += 1
        print(
            "early_stop patience={}/{} best_{}={:.6f} min_delta={:.6g}".format(
                self.wait,
                self.patience,
                self.monitor,
                self.best,
                self.min_delta,
            )
        )
        if self.wait >= self.patience:
            print("early stopping at epoch {} best_{}={:.6f}".format(current_epoch, self.monitor, self.best))
            self.model.stop_training = True


def main():
    args = parse_args()
    seed_everything(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    mm.json = json
    project = mm.ManTraNet(args.config)
    if not os.path.isfile(args.mantranet_pretrain):
        raise FileNotFoundError("IRIS0-SPAN ManTraNet pretrain not found: {}".format(args.mantranet_pretrain))
    project.weight_file = args.mantranet_pretrain
    model = project.get_model_1010_resize()
    output_shape = model.output_shape
    mask_size = int(output_shape[1]) if output_shape[1] is not None else args.image_size
    train_gen = LZBSequence(args.train_list, args.image_size, args.batch_size, shuffle=True, mask_size=mask_size)
    val_gen = LZBSequence(args.val_list, args.image_size, args.batch_size, shuffle=False, mask_size=mask_size)
    print("IRIS0-SPAN input_size={} mask_size={} output_shape={}".format(args.image_size, mask_size, output_shape))
    if args.init_weight and os.path.isfile(args.init_weight):
        load_initial_weights(model, args.init_weight)
    elif args.init_weight:
        raise FileNotFoundError("IRIS0-SPAN init weight not found: {}".format(args.init_weight))
    initial_epoch = 0
    if args.resume_from:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError("IRIS0-SPAN resume checkpoint not found: {}".format(args.resume_from))
        load_initial_weights(model, args.resume_from)
        initial_epoch = infer_resume_epoch(args.resume_from, args.resume_epoch)
        print("resumed IRIS0-SPAN from {} at epoch={}".format(args.resume_from, initial_epoch))

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[SafeF1(name="F1"), auroc],
    )
    last_checkpoint = LastCheckpoint(args.out_dir, save_every=args.save_last_every)
    callbacks = [
        last_checkpoint,
        DelayedBestCheckpoint(
            args.out_dir,
            monitor="val_F1",
            start_epoch=args.best_save_start_epoch,
            min_delta=0.0,
        ),
        keras.callbacks.CSVLogger(os.path.join(args.out_dir, "train_log.csv")),
    ]
    if args.early_stop_patience > 0:
        callbacks.append(
            MinEpochEarlyStopping(
                monitor="val_F1",
                min_epochs=args.early_stop_min_epochs,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
        )
    model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=initial_epoch,
        epochs=args.epochs,
        workers=args.workers,
        use_multiprocessing=False,
        callbacks=callbacks,
    )
    if not os.path.isfile(os.path.join(args.out_dir, "best.h5")):
        model.save_weights(os.path.join(args.out_dir, "best.h5"))
        fallback_epoch = last_checkpoint.last_epoch or initial_epoch or args.epochs
        write_checkpoint_metadata(os.path.join(args.out_dir, "best.h5"), fallback_epoch, {})


if __name__ == "__main__":
    main()
