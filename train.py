# -*- coding: utf-8 -*-
"""
No-Visdom Main (TensorBoard + Console logging)
- Every epoch prints: train_loss, val_loss, OA/AA/Kappa + per-class accuracy
- Writes all scalars to TensorBoard and CSV

Usage (example):
python main.py \
  --dataset Berlin \
  --model JViT \
  --patch_size 7 \
  --epoch 150 \
  --lr 3e-4 \
  --batch_size 128 \
  --cuda 0 \
  --flip_augmentation \
  --train_set ./Datasets/Berlin_C20_TRLabel.mat \
  --test_set  ./Datasets/Berlin_C20_TSLabel.mat
"""
from __future__ import print_function, division
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import numpy as np
import torch
import torch.utils.data as data
from torchsummary import summary
import seaborn as sns

from utils import (
    sample_gt,
    compute_imf_weights,
    get_device,
    restore_from_padding,
    seed_torch,
)
from datasets import get_dataset, MultiModalX, open_file, DATASETS_CONFIG
from model_utils import get_model, test  # 训练循环在本文件里写
import argparse

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# -------------------------
# Logger (TensorBoard + CSV + Console)
# -------------------------
class RunLogger:
    def __init__(self, dataset, model, seed, label_values, base_dir="runs"):
        run_tag = f"{dataset}_{model}_seed{seed}"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(base_dir, run_tag, stamp)
        os.makedirs(self.log_dir, exist_ok=True)

        self.label_values = list(label_values)

        self.tb = SummaryWriter(self.log_dir) if SummaryWriter is not None else None
        self.csv_path = os.path.join(self.log_dir, "metrics_epoch.csv")
        self.per_class_csv = os.path.join(self.log_dir, "per_class_epoch.csv")

        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,lr,train_loss,val_loss,val_OA,val_AA,val_Kappa\n")
        with open(self.per_class_csv, "w", encoding="utf-8") as f:
            f.write("split,epoch,class_id,class_name,accuracy,support\n")

        print("[Logger] logdir =", self.log_dir)
        print("[Logger] epoch CSV =", self.csv_path)
        print("[Logger] per-class CSV =", self.per_class_csv)
        if self.tb is None:
            print("[Logger] TensorBoard not available, using CSV only.")
    

    def log_epoch(self, epoch, lr, train_loss, val_loss, oa, aa, kappa):
        # console
        print(
            f"[Epoch {epoch:03d}] lr={lr:.3e} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"OA={oa*100:.2f}% AA={aa*100:.2f}% Kappa={kappa:.4f}"
        )

        # csv
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{lr},{train_loss},{val_loss},{oa},{aa},{kappa}\n")

        # tensorboard
        if self.tb is not None:
            self.tb.add_scalar("lr", lr, epoch)
            self.tb.add_scalar("loss/train", train_loss, epoch)
            self.tb.add_scalar("loss/val", val_loss, epoch)
            self.tb.add_scalar("metrics/OA", oa, epoch)
            self.tb.add_scalar("metrics/AA", aa, epoch)
            self.tb.add_scalar("metrics/Kappa", kappa, epoch)

    def log_per_class(self, split, epoch, per_class_acc, per_class_support):
        """
        per_class_acc: array [n_classes], NaN for ignored/invalid
        per_class_support: array [n_classes] number of GT pixels in that class for this split
        """
        # console block
        print("  Per-class accuracy (GT support):")
        for cid, (acc, sup) in enumerate(zip(per_class_acc, per_class_support)):
            name = self.label_values[cid] if cid < len(self.label_values) else f"class_{cid}"
            if np.isnan(acc):
                continue
            print(f"    [{cid:02d}] {name:20s}  acc={acc*100:6.2f}%   n={int(sup)}")

        # csv + tensorboard
        with open(self.per_class_csv, "a", encoding="utf-8") as f:
            for cid, (acc, sup) in enumerate(zip(per_class_acc, per_class_support)):
                if np.isnan(acc):
                    continue
                name = self.label_values[cid] if cid < len(self.label_values) else f"class_{cid}"
                f.write(f"{split},{epoch},{cid},{name},{acc},{int(sup)}\n")

        if self.tb is not None:
            for cid, (acc, sup) in enumerate(zip(per_class_acc, per_class_support)):
                if np.isnan(acc):
                    continue
                name = self.label_values[cid] if cid < len(self.label_values) else f"class_{cid}"
                # 每类一条曲线：metrics_per_class/<class_name>
                safe_name = name.replace(" ", "_").replace("/", "_")
                self.tb.add_scalar(f"metrics_per_class/{split}_{safe_name}", acc, epoch)

    def log_final_test(self, oa, aa, kappa, per_class_acc, per_class_support):
        print(f"[Final Test] OA={oa*100:.2f}% AA={aa*100:.2f}% Kappa={kappa:.4f}")
        if self.tb is not None:
            self.tb.add_scalar("final_test/OA", oa, 0)
            self.tb.add_scalar("final_test/AA", aa, 0)
            self.tb.add_scalar("final_test/Kappa", kappa, 0)
        self.log_per_class("test", 0, per_class_acc, per_class_support)

    def close(self):
        if self.tb is not None:
            self.tb.close()


# -------------------------
# Confusion + OA/AA/Kappa + per-class acc
# -------------------------
def confusion_from_preds(y_true, y_pred, n_classes, ignored_labels):
    """
    y_true/y_pred: 1D numpy int arrays
    returns conf[n_classes, n_classes] row=gt col=pred
    """
    y_true = y_true.astype(np.int64, copy=False).reshape(-1)
    y_pred = y_pred.astype(np.int64, copy=False).reshape(-1)

    mask = np.ones_like(y_true, dtype=bool)
    for ig in ignored_labels:
        mask &= (y_true != int(ig))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)

    k = n_classes
    idx = y_true * k + y_pred
    conf = np.bincount(idx, minlength=k * k).reshape(k, k)
    return conf.astype(np.int64)


def oa_aa_kappa_and_perclass(conf, ignored_labels):
    conf = conf.astype(np.float64)
    total = conf.sum()
    n_classes = conf.shape[0]
    ignored = set(int(x) for x in ignored_labels)

    if total <= 0:
        per_class_acc = np.full((n_classes,), np.nan, dtype=np.float64)
        per_class_support = np.zeros((n_classes,), dtype=np.int64)
        return 0.0, 0.0, 0.0, per_class_acc, per_class_support

    # OA
    oa = np.trace(conf) / total

    # per-class accuracy = diag / row_sum (i.e., recall)
    row_sum = conf.sum(axis=1)
    per_class_support = row_sum.astype(np.int64)
    per_class_acc = np.full((n_classes,), np.nan, dtype=np.float64)
    for c in range(n_classes):
        if c in ignored:
            continue
        if row_sum[c] > 0:
            per_class_acc[c] = conf[c, c] / row_sum[c]

    # AA = mean(per-class acc) excluding ignored and empty
    valid = ~np.isnan(per_class_acc)
    aa = float(np.mean(per_class_acc[valid])) if np.any(valid) else 0.0

    # Kappa
    col_sum = conf.sum(axis=0)
    pe = (row_sum * col_sum).sum() / (total * total + 1e-12)
    kappa = (oa - pe) / (1 - pe + 1e-12)

    return float(oa), float(aa), float(kappa), per_class_acc, per_class_support


@torch.no_grad()
def eval_on_loader(model, loader, loss_fn, device, n_classes, ignored_labels):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x1, x2)
        if isinstance(out, (tuple, list)):
            out = out[0]

        # 若模型输出是 [B,K,P,P]，取中心像素
        if out.ndim == 4:
            p = out.shape[-1] // 2
            out = out[:, :, p, p]

        # 若 label 是 [B,P,P]，取中心像素；若已是 [B] 则不动
        if y.ndim >= 2:
            p = y.shape[-1] // 2
            y_use = y[..., p, p]
        else:
            y_use = y

        loss = loss_fn(out, y_use.long())
        total_loss += float(loss.item())
        n_batches += 1

        pred = out.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        gt_np = y_use.detach().cpu().numpy().astype(np.int64)

        conf += confusion_from_preds(gt_np.reshape(-1), pred.reshape(-1), n_classes, ignored_labels)

    avg_loss = total_loss / max(1, n_batches)
    oa, aa, kappa, per_class_acc, per_class_support = oa_aa_kappa_and_perclass(conf, ignored_labels)
    return avg_loss, oa, aa, kappa, per_class_acc, per_class_support


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x1, x2)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim == 4:
            p = out.shape[-1] // 2
            out = out[:, :, p, p]

        if y.ndim >= 2:
            p = y.shape[-1] // 2
            y_use = y[..., p, p]
        else:
            y_use = y

        loss = loss_fn(out, y_use.long())
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


# -------------------------
# CLI
# -------------------------
dataset_names = [v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()]

parser = argparse.ArgumentParser(description="Run deep learning experiments (No Visdom, TB+Console).")
parser.add_argument("--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use.")
parser.add_argument("--model", type=str, default=None, help="Model to train.")
parser.add_argument("--folder", type=str, default="./Datasets/", help="Dataset folder.")
parser.add_argument("--cuda", type=int, default=1, help="CUDA device index (-1 for CPU).")
parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument("--restore", type=str, default=None, help="Checkpoint to load.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")

group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument("--train_val_split", type=float, default=0.8, help="Train/val split inside training mask.")
group_dataset.add_argument("--training_sample", type=float, default=0.99, help="Training ratio when sampling from gt.")
group_dataset.add_argument("--sampling_mode", type=str, default="random", help="random or disjoint.")
group_dataset.add_argument("--train_set", type=str, default=None, help="Path to train mask (.mat with TRLabel).")
group_dataset.add_argument("--test_set", type=str, default=None, help="Path to test mask (.mat with TSLabel).")

group_train = parser.add_argument_group("Training")
group_train.add_argument("--epoch", type=int, help="Training epochs.")
group_train.add_argument("--patch_size", type=int, help="Patch size.")
group_train.add_argument("--lr", type=float, help="Learning rate.")
group_train.add_argument("--class_balancing", action="store_true", help="Inverse median frequency weighting.")
group_train.add_argument("--batch_size", type=int, help="Batch size.")
group_train.add_argument("--test_stride", type=int, default=1, help="Test stride for sliding window inference.")

group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument("--flip_augmentation", action="store_true")
group_da.add_argument("--radiation_augmentation", action="store_true")
group_da.add_argument("--mixture_augmentation", action="store_true")

parser.add_argument("--download", type=str, default=None, nargs="+", choices=dataset_names, help="Download datasets and quit.")

args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)
seed_torch(seed=args.seed)

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=args.folder)
    quit()

# Load dataset
img1, img2, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.dataset, args.folder)

# palette kept minimal
if palette is None:
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

N_CLASSES = len(LABEL_VALUES)
N_BANDS = (img1.shape[-1], img2.shape[-1])

hyperparams = vars(args)
hyperparams.update({
    "n_classes": N_CLASSES,
    "n_bands": N_BANDS,
    "ignored_labels": IGNORED_LABELS,
    "device": CUDA_DEVICE,
})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

logger = RunLogger(args.dataset, args.model, args.seed, LABEL_VALUES)

for run in range(args.runs):
    # Train/Test masks
    if args.train_set is not None and args.test_set is not None:
        train_gt = open_file(args.train_set)["TRLabel"]
        test_gt = open_file(args.test_set)["TSLabel"]
    elif args.train_set is not None:
        train_gt = open_file(args.train_set)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    else:
        train_gt, test_gt = sample_gt(gt, args.training_sample, mode=args.sampling_mode)

    print(f"[Run {run+1}/{args.runs}] train points={np.count_nonzero(train_gt)} / total labeled={np.count_nonzero(gt)}")

    # class balancing
    if args.class_balancing:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        hyperparams["weights"] = torch.from_numpy(weights)

    # model
    model, optimizer, loss_fn, hyperparams = get_model(args.model, **hyperparams)
    scheduler = hyperparams.get("scheduler", None)

    # split train/val inside train_gt
    if args.train_val_split != 1:
        train_gt2, val_gt = sample_gt(train_gt, args.train_val_split, mode="random")
    else:
        _, val_gt = sample_gt(train_gt, 0.95, mode="random")
        train_gt2 = train_gt

    train_dataset = MultiModalX(img1, img2, train_gt2, **hyperparams)
    train_loader = data.DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)

    val_dataset = MultiModalX(img1, img2, val_gt, **hyperparams)
    val_loader = data.DataLoader(val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    # summary
    print("Network summary:")
    with torch.no_grad():
        for x1, x2, _ in train_loader:
            break
        summary(model.to(hyperparams["device"]), [x1.size()[1:], x2.size()[1:]])

    if args.restore is not None:
        model.load_state_dict(torch.load(args.restore, map_location="cpu"))

    epochs = hyperparams["epoch"]
    for ep in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, hyperparams["device"])
        lr = float(optimizer.param_groups[0]["lr"])

        if len(val_dataset) > 0:
            val_loss, oa, aa, kappa, per_class_acc, per_class_support = eval_on_loader(
                model, val_loader, loss_fn, hyperparams["device"], N_CLASSES, IGNORED_LABELS
            )
        else:
            val_loss, oa, aa, kappa = 0.0, 0.0, 0.0, 0.0
            per_class_acc = np.full((N_CLASSES,), np.nan, dtype=np.float64)
            per_class_support = np.zeros((N_CLASSES,), dtype=np.int64)

        logger.log_epoch(ep, lr, train_loss, val_loss, oa, aa, kappa)
        logger.log_per_class("val", ep, per_class_acc, per_class_support)

        if scheduler is not None:
            scheduler.step()

    # Final full-image test
    probabilities = test(model, img1, img2, hyperparams)

    try:
        prediction = np.argmax(probabilities, axis=-1)
    except Exception:
        probabilities = restore_from_padding(
            probabilities,
            patch_size=[hyperparams["patch_size"], hyperparams["patch_size"]],
        )
        prediction = np.argmax(probabilities, axis=-1)

    # 只在 test_gt!=ignored 的位置评估
    test_gt_int = np.asarray(test_gt).astype(np.int64, copy=False)
    pred_int = np.asarray(prediction).astype(np.int64, copy=False)

    conf = confusion_from_preds(test_gt_int.reshape(-1), pred_int.reshape(-1), N_CLASSES, IGNORED_LABELS)
    oa, aa, kappa, per_class_acc, per_class_support = oa_aa_kappa_and_perclass(conf, IGNORED_LABELS)

    logger.log_final_test(oa, aa, kappa, per_class_acc, per_class_support)

logger.close()

print("\nDone. View curves:")
print("  tensorboard --logdir runs --port 6006")
print("  open http://localhost:6006")
