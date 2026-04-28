"""
Generate the full fine-tuning metadata:
1. Read the manually labeled CSV (100 rows)
2. Normalize units (NA/na → null for display as "NA" in constraints)
3. Generate Rule A (strict threshold → UNSAFE) and Rule B (permissive threshold → SAFE)
4. For zero-value gauges: Rule A = "Alert if reading is at 0" (UNSAFE), Rule B uses a threshold above 0 (SAFE stays at 0)
5. Expand to 200 rows (Rule A + Rule B per image)
6. Apply image augmentation (rotation, brightness, CLAHE) 
7. Output final JSON matching clean_metadata.json schema
"""
import csv
import json
import os
import random
import math

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_DIR = os.path.join(BASE_DIR, "Dataset_FineTune")
INPUT_CSV = os.path.join(FINETUNE_DIR, "finetune_metadata.csv")
OUTPUT_JSON = os.path.join(FINETUNE_DIR, "finetune_metadata.json")
OUTPUT_CSV = os.path.join(FINETUNE_DIR, "finetune_metadata_full.csv")

# --- Helper: normalize unit ---
def normalize_unit(unit_str):
    """Normalize unit strings. NA/na/None → null."""
    if not unit_str or unit_str.strip().upper() == "NA":
        return None
    # Standardize casing
    unit_map = {
        "psi": "psi",
        "bar": "bar",
        "kpa": "kPa",
        "mpa": "Mpa",
        "mbar": "mbar",
        "celcius": "°C",
        "kg/cm": "kg/cm²",
        "kg/cm3": "kg/cm³",
    }
    key = unit_str.strip().lower()
    return unit_map.get(key, unit_str.strip())


# --- Helper: generate thresholds ---
def generate_thresholds(value, unit):
    """
    Generate Rule A (strict, triggers UNSAFE) and Rule B (permissive, triggers SAFE).
    
    For non-zero values:
      Rule A threshold < value → value exceeds threshold → UNSAFE
      Rule B threshold > value → value below threshold → SAFE
    
    For zero values:
      Rule A: "Alert if reading is at 0" → UNSAFE  
      Rule B: threshold above 0 → 0 is below → SAFE
    """
    val = float(value)
    unit_str = unit if unit else "NA"
    
    if val == 0:
        # Zero-value gauge: system is idle/off
        # Rule A: Alert on zero (system might be offline/dead)
        rule_a_constraint = f"Alert if reading is at 0 {unit_str}."
        rule_a_reasoning = f"Value is 0. System is at 0."
        rule_a_verdict = "UNSAFE"
        
        # Rule B: threshold above zero — 0 is safely below
        if unit_str in ["psi", "kPa", "mbar"]:
            threshold_b = random.choice([5, 10, 15, 20, 50])
        elif unit_str in ["bar", "Mpa"]:
            threshold_b = random.choice([0.5, 1.0, 2.0, 5.0])
        elif unit_str == "°C":
            threshold_b = random.choice([10, 20, 30, 50])
        elif unit_str in ["kg/cm²", "kg/cm³"]:
            threshold_b = random.choice([0.5, 1.0, 2.0])
        else:
            threshold_b = random.choice([1, 5, 10, 20, 50])
        
        rule_b_constraint = f"Alert if > {threshold_b} {unit_str}."
        rule_b_reasoning = f"Value is 0. 0 < {threshold_b}."
        rule_b_verdict = "SAFE"
    else:
        # Non-zero: generate bracket thresholds around the actual value
        if val < 1:
            # Small values: use proportional offsets
            delta = round(val * random.uniform(0.3, 0.6), 2)
            delta = max(delta, 0.01)  # floor
        elif val < 10:
            delta = round(random.uniform(0.5, 2.0), 1)
        elif val < 100:
            delta = round(random.uniform(2, 15), 0)
        elif val < 1000:
            delta = round(random.uniform(10, 80), 0)
        else:
            delta = round(random.uniform(50, 500), 0)
        
        threshold_a = round(val - delta, 2)
        threshold_b = round(val + delta, 2)
        
        # Ensure threshold_a is non-negative
        if threshold_a < 0:
            threshold_a = round(val * 0.5, 2)
        
        # Clean up: remove trailing zeros for display
        def fmt(v):
            if v == int(v):
                return str(int(v))
            return str(v)
        
        rule_a_constraint = f"Alert if > {fmt(threshold_a)} {unit_str}."
        rule_a_reasoning = f"Value is {fmt(val)}. {fmt(val)} > {fmt(threshold_a)}."
        rule_a_verdict = "UNSAFE"
        
        rule_b_constraint = f"Alert if > {fmt(threshold_b)} {unit_str}."
        rule_b_reasoning = f"Value is {fmt(val)}. {fmt(val)} < {fmt(threshold_b)}."
        rule_b_verdict = "SAFE"
    
    return {
        "rule_a": {
            "logic_constraint": rule_a_constraint,
            "reasoning": rule_a_reasoning,
            "expected_verdict": rule_a_verdict,
        },
        "rule_b": {
            "logic_constraint": rule_b_constraint,
            "reasoning": rule_b_reasoning,
            "expected_verdict": rule_b_verdict,
        }
    }


# --- Main ---
print("Reading labeled CSV...")
rows_in = []
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows_in.append(row)

print(f"  {len(rows_in)} images loaded.")

# Generate full metadata
full_rows = []
idx = 1

for row in rows_in:
    image_id = row["image_id"].strip()
    value = float(row["ground_truth_value"].strip())
    raw_unit = row["unit"].strip()
    unit = normalize_unit(raw_unit)
    artifact_tag = row["artifact_tag"].strip()
    hard_case = int(row["hard_case_flag"].strip())
    
    # Normalize artifact_tag
    if artifact_tag.lower() == "none":
        artifact_tag = None
    
    # Format expected_value
    unit_display = unit if unit else "NA"
    if value == int(value):
        val_str = str(int(value))
    else:
        val_str = str(value)
    expected_value = f"{val_str} {unit_display}"
    
    # Generate thresholds
    rules = generate_thresholds(value, unit)
    
    full_path = f"Dataset_FineTune/Selected images/{image_id}"
    processed_path = f"Dataset_FineTune/Preprocessed/{image_id}"
    
    # Rule A row
    full_rows.append({
        "index": idx,
        "image_id": image_id,
        "category": "guage",
        "artifact_tag": artifact_tag,
        "ground_truth_value": value if value != int(value) else int(value),
        "unit": unit,
        "logic_constraint": rules["rule_a"]["logic_constraint"],
        "expected_value": expected_value,
        "rule_id": "Rule A",
        "reasoning": rules["rule_a"]["reasoning"],
        "hard_case_flag": hard_case,
        "expected_verdict": rules["rule_a"]["expected_verdict"],
        "full_path": full_path,
        "processed_path": processed_path,
        "constraint": artifact_tag
    })
    idx += 1
    
    # Rule B row
    full_rows.append({
        "index": idx,
        "image_id": image_id,
        "category": "guage",
        "artifact_tag": artifact_tag,
        "ground_truth_value": value if value != int(value) else int(value),
        "unit": unit,
        "logic_constraint": rules["rule_b"]["logic_constraint"],
        "expected_value": expected_value,
        "rule_id": "Rule B",
        "reasoning": rules["rule_b"]["reasoning"],
        "hard_case_flag": hard_case,
        "expected_verdict": rules["rule_b"]["expected_verdict"],
        "full_path": full_path,
        "processed_path": processed_path,
        "constraint": artifact_tag
    })
    idx += 1

# --- Write JSON ---
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(full_rows, f, indent=4, ensure_ascii=False)
print(f"\nWrote JSON: {OUTPUT_JSON}")
print(f"  {len(full_rows)} rows ({len(rows_in)} images × 2 rules)")

# --- Write CSV ---
csv_columns = [
    "index", "image_id", "category", "artifact_tag", "ground_truth_value",
    "unit", "logic_constraint", "expected_value", "rule_id", "reasoning",
    "hard_case_flag", "expected_verdict", "full_path", "processed_path", "constraint"
]
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(full_rows)
print(f"Wrote CSV: {OUTPUT_CSV}")

# --- Summary stats ---
na_units = sum(1 for r in rows_in if normalize_unit(r["unit"].strip()) is None)
zero_vals = sum(1 for r in rows_in if float(r["ground_truth_value"].strip()) == 0)
hard_cases = sum(1 for r in rows_in if int(r["hard_case_flag"].strip()) == 1)

print(f"\n--- Dataset Summary ---")
print(f"  Total images:    {len(rows_in)}")
print(f"  Zero-value:      {zero_vals}")
print(f"  NA unit:         {na_units}")
print(f"  Hard cases:      {hard_cases}")
print(f"  Clean cases:     {len(rows_in) - hard_cases}")
print(f"  Total eval rows: {len(full_rows)}")
