"""
Augment fine-tuning gauge images and expand metadata.

Augmentations applied (preserving gauge reading):
1. Rotation: ±10-15° (simulates camera tilt)
2. Brightness jitter: ±20-30% (simulates lighting variation)
3. CLAHE: Contrast-limited adaptive histogram equalization (simulates exposure)
4. Center crop + resize: 85% crop then back to original size (simulates zoom)

Each original image produces 3 augmented variants → 100 originals → 400 total images.
Each augmented image inherits all metadata from its parent.
"""
import os
import json
import csv
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_DIR = os.path.join(BASE_DIR, "Dataset_FineTune")
SELECTED_DIR = os.path.join(FINETUNE_DIR, "Selected images")
AUGMENTED_DIR = os.path.join(FINETUNE_DIR, "Augmented")
INPUT_JSON = os.path.join(FINETUNE_DIR, "finetune_metadata.json")
OUTPUT_JSON = os.path.join(FINETUNE_DIR, "finetune_metadata_augmented.json")
OUTPUT_CSV = os.path.join(FINETUNE_DIR, "finetune_metadata_augmented.csv")

os.makedirs(AUGMENTED_DIR, exist_ok=True)

# --- Augmentation functions ---

def augment_rotate(img, angle):
    """Rotate image by angle degrees, fill background with edge color."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

def augment_brightness(img, factor):
    """Adjust brightness. factor > 1 = brighter, < 1 = darker."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def augment_contrast(img, factor):
    """Adjust contrast via PIL. factor > 1 = more contrast."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def augment_center_crop(img, crop_ratio=0.85):
    """Center crop to crop_ratio of original, then resize back."""
    w, h = img.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


def generate_augmentations(img):
    """Generate 3 augmented variants of an image."""
    augmented = []
    
    # Variant 1: Rotation + brightness
    angle = random.choice([-15, -12, -10, 10, 12, 15])
    brightness = random.uniform(0.7, 0.85)
    aug1 = augment_brightness(augment_rotate(img, angle), brightness)
    augmented.append(("rot_dark", aug1))
    
    # Variant 2: Contrast boost + slight rotation
    angle = random.choice([-8, -5, 5, 8])
    contrast = random.uniform(1.2, 1.5)
    aug2 = augment_contrast(augment_rotate(img, angle), contrast)
    augmented.append(("contrast", aug2))
    
    # Variant 3: Center crop + brightness boost
    brightness = random.uniform(1.15, 1.35)
    aug3 = augment_brightness(augment_center_crop(img, crop_ratio=0.85), brightness)
    augmented.append(("crop_bright", aug3))
    
    return augmented


# --- Main ---
print("Loading metadata...")
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# Group metadata by image_id (each image has 2 rows: Rule A + Rule B)
image_groups = {}
for row in metadata:
    img_id = row["image_id"]
    if img_id not in image_groups:
        image_groups[img_id] = []
    image_groups[img_id].append(row)

print(f"  {len(image_groups)} unique images, {len(metadata)} metadata rows")

# Process each image
all_rows = list(metadata)  # Start with originals
aug_count = 0
new_idx = len(metadata) + 1

for img_id, rows in sorted(image_groups.items(), key=lambda x: int(x[0].split("(")[1].split(")")[0])):
    img_path = os.path.join(SELECTED_DIR, img_id)
    
    if not os.path.exists(img_path):
        print(f"  WARNING: {img_id} not found, skipping")
        continue
    
    img = Image.open(img_path).convert("RGB")
    augmentations = generate_augmentations(img)
    
    for aug_suffix, aug_img in augmentations:
        # Generate augmented image filename
        base_name = img_id.replace(".jpg", "")
        aug_name = f"{base_name}_{aug_suffix}.jpg"
        aug_path = os.path.join(AUGMENTED_DIR, aug_name)
        
        # Save augmented image
        aug_img.save(aug_path, "JPEG", quality=92)
        aug_count += 1
        
        # Duplicate metadata rows for augmented image
        for orig_row in rows:
            new_row = dict(orig_row)
            new_row["index"] = new_idx
            new_row["image_id"] = aug_name
            new_row["full_path"] = f"Dataset_FineTune/Augmented/{aug_name}"
            new_row["processed_path"] = f"Dataset_FineTune/Augmented/{aug_name}"
            all_rows.append(new_row)
            new_idx += 1

print(f"\nGenerated {aug_count} augmented images")
print(f"Total metadata rows: {len(all_rows)} (originals: {len(metadata)}, augmented: {len(all_rows) - len(metadata)})")

# --- Write augmented JSON ---
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_rows, f, indent=4, ensure_ascii=False)
print(f"\nWrote: {OUTPUT_JSON}")

# --- Write augmented CSV ---
csv_columns = [
    "index", "image_id", "category", "artifact_tag", "ground_truth_value",
    "unit", "logic_constraint", "expected_value", "rule_id", "reasoning",
    "hard_case_flag", "expected_verdict", "full_path", "processed_path", "constraint"
]
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Wrote: {OUTPUT_CSV}")

# --- Summary ---
orig_images = len(image_groups)
aug_images = aug_count
total_images = orig_images + aug_images
total_rows = len(all_rows)

print(f"\n--- Final Dataset Summary ---")
print(f"  Original images:   {orig_images}")
print(f"  Augmented images:  {aug_images} (3 per original)")
print(f"  Total images:      {total_images}")
print(f"  Total eval rows:   {total_rows} ({total_images} images × 2 rules)")
