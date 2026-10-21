#!/usr/bin/env python3
"""
Keep only bark photos from a folder of per-species images using CLIP zero-shot scoring.

Usage:
    pip install torch transformers pillow
    python filter_bark_clip.py --src bark_data --dst bark_filtered --threshold 0.5

Each image is scored against "bark" prompts vs. "not bark" prompts (leaves, flowers,
fruit, whole tree, people, etc.). Images whose bark probability >= threshold are copied
to --dst/<species>/. All scores go to scores.csv so you can adjust the threshold and
review borderline images (0.3-0.6) by hand.
"""
import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BARK = [
    "a close-up photo of tree bark",
    "a photo of the rough bark texture of a tree trunk",
    "a macro photo of bark",
    
]
OTHER = [
    "a photo of green leaves",
    "a photo of flowers",
    "a photo of fruits or seeds",
    "a photo of a whole tree",
    "a photo of a landscape with trees",
    "a photo of a person",
    "a photo of an insect or animal",
    "a photo of a road or a building",
    "a photo of a seedling in a pot",
    "a photo of a dried leaf, twig or herbarium sheet",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bark_data")
    ap.add_argument("--dst", default="bark_filtered")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.model)

    prompts = BARK + OTHER
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        txt = model.get_text_features(**t)
        if hasattr(txt, "pooler_output"):
            txt = txt.pooler_output
        txt = txt / txt.norm(dim=-1, keepdim=True)

    src, dst = Path(args.src), Path(args.dst)
    files = sorted(p for p in src.glob("*/*.jpg"))
    print(f"Scoring {len(files)} images ...")
    kept = {}
    with open(dst.with_name(dst.name + "_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "file", "bark_prob", "kept"])
        for i in range(0, len(files), args.batch):
            chunk, imgs = [], []
            for p in files[i:i + args.batch]:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    chunk.append(p)
                except Exception:
                    continue
            if not imgs:
                continue
            with torch.no_grad():
                x = proc(images=imgs, return_tensors="pt").to(device)
                emb = model.get_image_features(**x)
                if hasattr(emb, "pooler_output"):
                    emb = emb.pooler_output
                emb = emb / emb.norm(dim=-1, keepdim=True)
                probs = (100.0 * emb @ txt.T).softmax(dim=-1)
                bark_prob = probs[:, :len(BARK)].sum(dim=-1).cpu().tolist()
            for p, s in zip(chunk, bark_prob):
                keep = s >= args.threshold
                w.writerow([p.parent.name, p.name, f"{s:.3f}", int(keep)])
                if keep:
                    out = dst / p.parent.name
                    out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, out / p.name)
                    kept[p.parent.name] = kept.get(p.parent.name, 0) + 1
            print(f"  {min(i + args.batch, len(files))}/{len(files)}")

    print("\nBark photos kept per species:")
    for sp in sorted({p.parent.name for p in files}):
        print(f"  {sp:30s} {kept.get(sp, 0)}")
    print("\nReview the kept images by hand, then split train/val/test by observation_id "
          "(the number before the underscore in each filename).")


if __name__ == "__main__":
    main()
