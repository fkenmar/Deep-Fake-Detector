"""
Downloads OpenRL/DeepFakeFace from HuggingFace and arranges it into an
ImageFolder-compatible layout that train.py can consume alongside DS1/DS2.

The HF dataset ships 4 zip files, each ~30k images:
  - wiki.zip       → real (from IMDB-WIKI)
  - inpainting.zip → fake (Stable Diffusion inpainting)
  - insight.zip    → fake (InsightFace swap)
  - text2img.zip   → fake (Stable Diffusion v1.5)

Output layout (symlinked to save disk):
  ~/.cache/huggingface/datasets/openrl_deepfakeface/
    Train/{Real,Fake}/
    Validation/{Real,Fake}/   # deterministic 5% holdout

Idempotent: re-running after a successful prep is a no-op.
"""

import os
import random
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID   = "OpenRL/DeepFakeFace"
OUT_ROOT  = Path("~/.cache/huggingface/datasets/openrl_deepfakeface").expanduser()
VAL_FRAC  = 0.05
SEED      = 42

REAL_ZIPS = ["wiki.zip"]
FAKE_ZIPS = ["inpainting.zip", "insight.zip", "text2img.zip"]
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link_images(source_dir: Path, target_dir: Path, prefix: str) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMG_EXTS:
            continue
        rel = path.relative_to(source_dir)
        name = f"{prefix}__" + "_".join(rel.parts)
        dest = target_dir / name
        if dest.exists() or dest.is_symlink():
            continue
        os.symlink(path.resolve(), dest)
        count += 1
    return count


def holdout_validation(train_dir: Path, val_dir: Path, rng: random.Random) -> int:
    val_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in train_dir.iterdir() if p.is_file() or p.is_symlink())
    n_val = int(len(files) * VAL_FRAC)
    if n_val == 0:
        return 0
    for p in rng.sample(files, n_val):
        p.rename(val_dir / p.name)
    return n_val


def main() -> None:
    train_real = OUT_ROOT / "Train" / "Real"
    train_fake = OUT_ROOT / "Train" / "Fake"
    val_real   = OUT_ROOT / "Validation" / "Real"
    val_fake   = OUT_ROOT / "Validation" / "Fake"

    populated = (
        train_real.exists() and any(train_real.iterdir())
        and train_fake.exists() and any(train_fake.iterdir())
        and val_real.exists() and any(val_real.iterdir())
        and val_fake.exists() and any(val_fake.iterdir())
    )
    if populated:
        print(f"[skip] already populated at {OUT_ROOT}")
        for d in (train_real, train_fake, val_real, val_fake):
            print(f"  {d.relative_to(OUT_ROOT)}: {sum(1 for _ in d.iterdir())}")
        return

    print(f"Downloading {REPO_ID} ...")
    repo_path = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset"))
    print(f"  → {repo_path}")

    extract_root = OUT_ROOT / ".extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    for zip_name in REAL_ZIPS + FAKE_ZIPS:
        zip_path = repo_path / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {zip_path} — HF layout may have changed")
        extract_dir = extract_root / zip_path.stem
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  [extracted] {zip_name}")
            continue
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [extracting] {zip_name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    for zn in REAL_ZIPS:
        stem = Path(zn).stem
        n = link_images(extract_root / stem, train_real, prefix=stem)
        print(f"[real] {zn}: linked {n}")

    for zn in FAKE_ZIPS:
        stem = Path(zn).stem
        n = link_images(extract_root / stem, train_fake, prefix=stem)
        print(f"[fake] {zn}: linked {n}")

    rng = random.Random(SEED)
    n_val_real = holdout_validation(train_real, val_real, rng)
    n_val_fake = holdout_validation(train_fake, val_fake, rng)
    print(f"[val] moved {n_val_real} real + {n_val_fake} fake to Validation/")

    print("\nFinal counts:")
    for d in (train_real, train_fake, val_real, val_fake):
        print(f"  {d.relative_to(OUT_ROOT)}: {sum(1 for _ in d.iterdir())}")


if __name__ == "__main__":
    main()
