"""
Prepare fine-tuning dataset:
1. Rename selected images sequentially: ft_gauge (1).jpg, ft_gauge (2).jpg, ...
2. Create a CSV skeleton with all columns matching clean_metadata.json schema
3. Each image appears TWICE (Rule A + Rule B), just like the Golden 100
4. Remove non-selected images from Dataset_FineTune/
"""
import os
import csv
import shutil
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_DIR = os.path.join(BASE_DIR, "Dataset_FineTune")
SELECTED_DIR = os.path.join(FINETUNE_DIR, "Selected images")
CSV_PATH = os.path.join(FINETUNE_DIR, "finetune_metadata.csv")

# --- Step 1: Rename selected images sequentially ---
selected_files = sorted([
    f for f in os.listdir(SELECTED_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"Found {len(selected_files)} selected images.")

rename_map = {}  # old_name -> new_name
for i, old_name in enumerate(selected_files, start=1):
    new_name = f"ft_gauge ({i}).jpg"
    old_path = os.path.join(SELECTED_DIR, old_name)
    new_path = os.path.join(SELECTED_DIR, new_name)
    
    # Avoid collision during rename by using temp names first
    rename_map[old_name] = (old_path, new_path, new_name)

# Rename in two passes to avoid collisions
# Pass 1: rename to temp names
temp_paths = []
for old_name, (old_path, new_path, new_name) in rename_map.items():
    temp_name = f"__temp__{new_name}"
    temp_path = os.path.join(SELECTED_DIR, temp_name)
    os.rename(old_path, temp_path)
    temp_paths.append((temp_path, new_path, new_name))

# Pass 2: rename temp to final names
final_names = []
for temp_path, new_path, new_name in temp_paths:
    os.rename(temp_path, new_path)
    final_names.append(new_name)
    print(f"  Renamed -> {new_name}")

print(f"\nRenamed {len(final_names)} images.")

# --- Step 2: Create CSV skeleton with duplicate rows ---
columns = [
    "index",
    "image_id",
    "category",
    "artifact_tag",
    "ground_truth_value",
    "unit",
    "logic_constraint",
    "expected_value",
    "rule_id",
    "reasoning",
    "hard_case_flag",
    "expected_verdict",
    "full_path",
    "processed_path",
    "constraint"
]

rows = []
idx = 1
for img_name in sorted(final_names, key=lambda x: int(x.split("(")[1].split(")")[0])):
    img_num = int(img_name.split("(")[1].split(")")[0])
    full_path = f"Dataset_FineTune/Selected images/{img_name}"
    processed_path = f"Dataset_FineTune/Preprocessed/{img_name}"
    
    # Rule A row
    rows.append({
        "index": idx,
        "image_id": img_name,
        "category": "guage",
        "artifact_tag": "",
        "ground_truth_value": "",
        "unit": "",
        "logic_constraint": "",
        "expected_value": "",
        "rule_id": "Rule A",
        "reasoning": "",
        "hard_case_flag": "",
        "expected_verdict": "",
        "full_path": full_path,
        "processed_path": processed_path,
        "constraint": ""
    })
    idx += 1
    
    # Rule B row
    rows.append({
        "index": idx,
        "image_id": img_name,
        "category": "guage",
        "artifact_tag": "",
        "ground_truth_value": "",
        "unit": "",
        "logic_constraint": "",
        "expected_value": "",
        "rule_id": "Rule B",
        "reasoning": "",
        "hard_case_flag": "",
        "expected_verdict": "",
        "full_path": full_path,
        "processed_path": processed_path,
        "constraint": ""
    })
    idx += 1

with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCreated CSV: {CSV_PATH}")
print(f"  {len(final_names)} images x 2 rules = {len(rows)} rows")

# --- Step 3: Remove non-selected images from Dataset_FineTune root ---
removed = 0
for f in os.listdir(FINETUNE_DIR):
    fpath = os.path.join(FINETUNE_DIR, f)
    if os.path.isfile(fpath) and f.lower().endswith(('.jpg', '.jpeg', '.png')):
        os.remove(fpath)
        removed += 1

print(f"\nCleaned up {removed} non-selected images from Dataset_FineTune/ root.")
print("\nDone! Open finetune_metadata.csv and fill in:")
print("  - ground_truth_value (the gauge reading)")
print("  - unit (bar, psi, kPa, etc.)")
print("  - artifact_tag (Glare, Oblique Angle, Low Resolution, etc. or leave blank if clean)")
print("  - hard_case_flag (1 for hard, 0 for clean)")
print("After that, the safety constraints will be auto-generated.")
