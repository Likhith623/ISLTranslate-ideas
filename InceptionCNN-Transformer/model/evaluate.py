"""
Milestone 2: Test-Set Evaluation

Loads best_model.pt and evaluates on the locked test split.
Reports: Top-1 accuracy, Top-5 accuracy, per-class accuracy table,
and saves a confusion matrix heatmap as confusion_matrix.png.
"""

import json
import os
import sys

import numpy as np
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from dataset import get_dataloaders, load_class_map
from model import ASLTransformer

CHECKPOINT = os.path.join(MODEL_DIR, "checkpoints", "best_model.pt")


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_evaluation():
    device = get_device()
    print(f"\n{'='*64}")
    print(f"  ASL Transformer — Test Set Evaluation")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"{'='*64}\n")

    # Load model
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model = ASLTransformer().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded from epoch {checkpoint['epoch']} "
          f"(val acc at save: {checkpoint['val_acc']:.2f}%)\n")

    # Load data
    _, _, test_loader = get_dataloaders(batch_size=64)
    class_map   = load_class_map()
    id_to_gloss = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    # Collect predictions
    all_preds  = []
    all_labels = []
    all_logits = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y)

    logits_all = torch.cat(all_logits)    # (N, 100)
    preds_all  = torch.cat(all_preds)     # (N,)
    labels_all = torch.cat(all_labels)    # (N,)
    N = len(labels_all)

    # Top-1 Accuracy
    top1 = (preds_all == labels_all).float().mean().item() * 100

    # Top-5 Accuracy
    top5_preds = logits_all.topk(5, dim=1).indices
    top5 = (top5_preds == labels_all.unsqueeze(1)).any(dim=1).float().mean().item() * 100

    print(f"  Top-1 Accuracy : {top1:.2f}%")
    print(f"  Top-5 Accuracy : {top5:.2f}%")
    print(f"  Test samples   : {N}\n")

    # Per-class Accuracy
    class_correct = np.zeros(num_classes)
    class_total   = np.zeros(num_classes)
    for pred, label in zip(preds_all.numpy(), labels_all.numpy()):
        class_total[label]   += 1
        class_correct[label] += int(pred == label)

    per_class = [
        (id_to_gloss.get(i, str(i)),
         class_correct[i], class_total[i],
         class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    ]
    per_class.sort(key=lambda x: x[3])   # worst first

    print(f"  {'Gloss':<28} {'Correct':>8} {'Total':>7} {'Acc':>8}")
    print("  " + "-" * 56)
    for gloss, correct, total, acc in per_class:
        bar = "█" * int(acc / 10)
        print(f"  {gloss:<28} {int(correct):>8} {int(total):>7} {acc:>7.1f}%  {bar}")

    # Confusion Matrix
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for pred, label in zip(preds_all.numpy(), labels_all.numpy()):
            cm[label][pred] += 1

        fig, ax = plt.subplots(figsize=(22, 22))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Confusion Matrix — Top-1: {top1:.1f}%  Top-5: {top5:.1f}%", fontsize=14)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        labels = [id_to_gloss.get(i, str(i)) for i in range(num_classes)]
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(labels, fontsize=6)
        plt.tight_layout()
        out_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        plt.savefig(out_path, dpi=150)
        print(f"\n[Plot] Confusion matrix saved to {out_path}")
    except ImportError:
        print("\n[Plot] matplotlib not installed — skipping confusion matrix.")

    print(f"\n{'='*64}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Top-1 : {top1:.2f}%   Top-5 : {top5:.2f}%")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    run_evaluation()
