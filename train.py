"""
Fine-tunes prithivMLmods/Deep-Fake-Detector-v2-Model on the Kaggle deepfake dataset.
Saves the fine-tuned model to ./model/ for use in app.py.
"""

import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DATASET    = Path("/Users/kenmarfrancisco/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset")
SAVE_DIR   = Path("./model")
EPOCHS     = 3
BATCH_SIZE = 32
LR         = 2e-5

# ── Processor ─────────────────────────────────────────────────────────────────
processor = ViTImageProcessor.from_pretrained(MODEL_ID)

# ImageFolder sorts classes alphabetically: Fake=0, Real=1
# The HF model uses: Deepfake=0, Realism=1 — same order, labels match.
def transform(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ImageFolder(DATASET / "Train",      transform=transform)
    val_ds   = ImageFolder(DATASET / "Validation", transform=transform)

    # num_workers=0 avoids multiprocessing issues on macOS
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"Classes: {train_ds.classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for step, (pixels, labels) in enumerate(train_loader):
            pixels, labels = pixels.to(device), labels.to(device)
            outputs = model(pixel_values=pixels, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f} | Acc: {correct/total:.4f}")

        train_acc = correct / total
        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for pixels, labels in val_loader:
                pixels, labels = pixels.to(device), labels.to(device)
                outputs = model(pixel_values=pixels)
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)
            print(f"  Saved best model (val_acc={val_acc:.4f}) to {SAVE_DIR}\n")

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {SAVE_DIR}")
