"""
Evaluates the fine-tuned model on the test set.
Prints accuracy, precision, recall, F1, and a confusion matrix.
"""

import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import torch
import numpy as np

MODEL_DIR = "./model" if os.path.exists("./model") else "prithivMLmods/Deep-Fake-Detector-v2-Model"
DATASET   = Path("/Users/kenmarfrancisco/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset")
BATCH_SIZE = 64

processor = ViTImageProcessor.from_pretrained(MODEL_DIR)

def transform(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_DIR}")

    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    model = model.to(device)
    model.eval()

    test_ds = ImageFolder(DATASET / "Test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Test images: {len(test_ds)} | Classes: {test_ds.classes}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for step, (pixels, labels) in enumerate(test_loader):
            pixels = pixels.to(device)
            outputs = model(pixel_values=pixels)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            if (step + 1) % 50 == 0:
                print(f"  Processed {(step+1) * BATCH_SIZE}/{len(test_ds)} images...")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Results
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

    cm = confusion_matrix(all_labels, all_preds)
    print("CONFUSION MATRIX")
    print("-" * 30)
    print(f"{'':>10} {'Pred Fake':>10} {'Pred Real':>10}")
    print(f"{'True Fake':>10} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"{'True Real':>10} {cm[1][0]:>10} {cm[1][1]:>10}")

    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
