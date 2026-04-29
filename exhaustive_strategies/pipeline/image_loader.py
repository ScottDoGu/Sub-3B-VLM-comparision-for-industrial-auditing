import os, re, glob
import pandas as pd
from PIL import Image

LABEL_MAP = {}
FOUND_BASE_DIR = "."

excel_candidates = glob.glob("*.xlsx") + glob.glob("Dataset/*.xlsx") + glob.glob("/content/drive/**/*.xlsx", recursive=True)
excel_path = next((f for f in excel_candidates if "data_label" in f.lower() and not f.startswith("~")), None)

if excel_path and os.path.exists(excel_path):
    print(f"[INFO] Successfully loaded Excel mapping from: {excel_path}")
    FOUND_BASE_DIR = os.path.dirname(excel_path) 
    try:
        df = pd.read_excel(excel_path)
        for _, row in df.iterrows():
            if 'image_id' in row and 'expected_verdict' in row:
                raw_name = str(row['image_id']).strip()
                fixed_name = raw_name.replace("guage", "gauge")
                raw_verdict = str(row['expected_verdict']).strip().lower()
                if raw_verdict == "safe": clean_verdict = "pass"
                elif raw_verdict == "unsafe": clean_verdict = "fail"
                else: clean_verdict = raw_verdict
                LABEL_MAP[fixed_name] = clean_verdict
    except Exception as e:
        print(f"\u26A0\uFE0F Warning: Failed to parse Excel file: {e}")
else:
    print("\u26A0\uFE0F Warning: Could not locate data_label_constraint_image.xlsx.")

def discover(data_dir=None):
    if data_dir and os.path.isdir(data_dir): root = data_dir
    else:
        root = os.path.join(FOUND_BASE_DIR, "Data_Preprocessed")
        if not os.path.isdir(root): root = "Data_Preprocessed"

    print(f"[INFO] Scanning for images in: {root}")
    found = {"gauge": [], "pipe_corroded": [], "pipe_non_corroded": []}
    if not os.path.isdir(root):
        print(f"[ERROR] Directory '{root}' not found. Cannot load images.")
        return found

    for f in sorted(os.listdir(root)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
        f_lower = f.lower()
        if "guage" in f_lower or "gauge" in f_lower: cat = "gauge"
        elif "non_corroded" in f_lower: cat = "pipe_non_corroded"
        elif "corroded" in f_lower: cat = "pipe_corroded"
        else: continue

        m = re.search(r"(\d+)", f)
        idx = int(m.group(1)) if m else 0
        corrected_name = f.replace("guage", "gauge")
        found[cat].append({"path": os.path.join(root, f), "file": corrected_name, "idx": idx, "cat": cat})
    return found

def load_subset(n_gauge=10, n_pipe=10, data_dir=None):
    imgs = discover(data_dir)
    if n_gauge == 0 and n_pipe == 0: return imgs["gauge"] + imgs["pipe_corroded"] + imgs["pipe_non_corroded"]
    out = imgs["gauge"][:n_gauge]
    n_corr = n_pipe // 2
    out += imgs["pipe_corroded"][:n_corr] + imgs["pipe_non_corroded"][:n_pipe - n_corr]
    return out

def ground_truth(entry): return LABEL_MAP.get(entry["file"])
def load_image(path): return Image.open(path).convert("RGB")
