"""
Milestone 2: Training Script

Optimizer  : AdamW (lr=3e-4, weight_decay=1e-4)
Scheduler  : CosineAnnealingWarmRestarts (T_0=30)
Loss       : CrossEntropyLoss (label_smoothing=0.1)
Early stop : patience=20 epochs on val accuracy
Saves best_model.pt whenever val accuracy improves.
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR # UPDATED: Removed WarmRestarts

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from dataset import get_dataloaders
from model import ASLTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "batch_size":      32,
    "max_epochs":      150,
    "lr":              3e-4,
    "weight_decay":    1e-3,
    "label_smoothing": 0.1,
    "patience":        25,
    # REMOVED T_0 entirely
    "checkpoint_path": os.path.join(MODEL_DIR, "checkpoints", "best_model.pt"),
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon GPU
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return top-1 accuracy (0–100) on a DataLoader."""
    model.eval()
    correct = total = 0
    
    # Use AMP for faster evaluation too
    use_amp = device.type == 'cuda'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            preds = model(x).argmax(dim=1)
            
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total * 100


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train():
    device = get_device()
    print(f"\n{'='*64}")
    print(f"  ASL Transformer — Milestone 2 Training")
    print(f"  Device : {device}")
    print(f"{'='*64}\n")

    # OPTIMIZATION: Enable cuDNN benchmarking for fixed input sizes (speeds up CNNs)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _ = get_dataloaders(batch_size=CONFIG["batch_size"])

    model = ASLTransformer().to(device)
    print(f"[Model] Parameters: {model.count_parameters():,}\n")

    # Shape sanity check before training starts
    dummy = next(iter(train_loader))[0][:2].to(device)
    out   = model(dummy)
    assert out.shape == (2, 100), f"Shape error: {out.shape}"
    print(f"[Sanity] Forward pass OK — output shape: {out.shape}\n")

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    
    # UPDATED: Standard Cosine Annealing over max epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["max_epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # OPTIMIZATION: Automatic Mixed Precision (AMP) Scaler
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)

    best_val_acc  = 0.0
    patience_left = CONFIG["patience"]

    print(f"{'Epoch':>6}  {'Loss':>8}  {'Val Acc':>8}  {'LR':>10}  {'Time':>7}  Note")
    print("-" * 62)

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # OPTIMIZATION: set_to_none=True is faster than default zero_grad
            optimizer.zero_grad(set_to_none=True)
            
            # OPTIMIZATION: Forward pass with Mixed Precision
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x)
                loss = criterion(outputs, y)
                
            # Backward pass with Scaler
            scaler.scale(loss).backward()
            
            # Unscale before clipping gradients
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step and update scaler
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader, device)
        lr_now   = optimizer.param_groups[0]["lr"]
        elapsed  = time.time() - t0

        is_best = val_acc > best_val_acc
        note    = "⬆ best" if is_best else ""

        print(f"{epoch:>5}/{CONFIG['max_epochs']}  "
              f"{avg_loss:>8.4f}  "
              f"{val_acc:>7.2f}%  "
              f"{lr_now:>10.2e}  "
              f"{fmt_time(elapsed):>7}  {note}")

        if is_best:
            best_val_acc  = val_acc
            patience_left = CONFIG["patience"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "config":           CONFIG,
            }, CONFIG["checkpoint_path"])
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"\n[EarlyStop] No improvement for {CONFIG['patience']} epochs. Stopping.")
                break

    print(f"\n{'='*64}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"  Checkpoint saved  : {CONFIG['checkpoint_path']}")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    train()