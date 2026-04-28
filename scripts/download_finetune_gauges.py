"""
Download gauge images from HuggingFace dataset Francesco/gauge-u2lwv
and extract raw images into Dataset_FineTune/ folder for manual annotation.
"""
import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset_FineTune")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading Francesco/gauge-u2lwv from HuggingFace...")
ds = load_dataset("Francesco/gauge-u2lwv")

# Dataset has train/test/validation splits - extract images from all
total_saved = 0
for split_name in ds:
    split = ds[split_name]
    print(f"\nProcessing split: {split_name} ({len(split)} images)")
    
    for i, sample in enumerate(split):
        image = sample["image"]
        filename = f"hf_gauge_{split_name}_{i+1:04d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Convert to RGB and save
        image.convert("RGB").save(filepath, "JPEG", quality=95)
        total_saved += 1

print(f"\nDone. Saved {total_saved} images to {OUTPUT_DIR}/")
print("Next step: manually review images, read gauge values, and annotate.")
