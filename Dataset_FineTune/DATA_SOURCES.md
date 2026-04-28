# Dataset Sources Reference

All image data used across this project, including both the evaluation benchmark and the fine-tuning training set.

---

## 1. Golden 100 — Evaluation Benchmark (DO NOT USE FOR TRAINING)

**Role:** Held-out test set for all zero-shot, prompt intervention, and fine-tuning evaluations.

| Category | Count | Description |
|----------|-------|-------------|
| Analog Gauges | 50 images | Pressure, temperature, flow meters with various visual stressors |
| Corroded Pipelines | 25 images | Structural corrosion, leaks, oxidation |
| Non-Corroded Pipelines | 25 images | Clean pipe surfaces as control group |

- **Total Evaluation Rows:** 200 (each image tested under two opposing SOP rules)
- **Annotation:** Hand-labeled with ground-truth readings, units, artifact tags, and contradictory Rule A/B constraint pairs
- **Schema:** `src/ingestion/clean_metadata.json`
- **Documentation:** `Dataset/Metadata.Rmd`, `Dataset/Metadata.pdf`

### Original Image Sources (3 Roboflow Datasets)

#### 1a. gauge-kzdux (Roboflow Universe) — Gauge Images

- **URL:** https://universe.roboflow.com/workspace-vem8e/gauge-kzdux
- **Images Used:** 50 (selected from ~302 available)
- **Content:** Analog dial images with oblique viewing angles, glare, and partial occlusions
- **Original Task:** Object detection (classes: needle, box_gauge, center, gauge)
- **Our Use:** Images only — gauge readings manually annotated by us with ground-truth values, units, and artifact tags
- **License:** CC BY 4.0

#### 1b. Classi1 (Roboflow Universe) — Corroded Pipe Images

- **URL:** https://universe.roboflow.com/skander/classi1-tvpwr
- **Images Used:** 25 (selected from ~125 available)
- **Content:** Defective pipeline surface images showing corrosion, rust, and structural damage
- **Original Task:** Binary classification (defected vs. non-defected)
- **Our Use:** "Defected" class images paired with SOP-style corrosion detection rules
- **License:** See Roboflow homepage

#### 1c. Non-Corroded-Pipe (Roboflow Universe) — Clean Pipe Images

- **URL:** https://universe.roboflow.com/search?q=Non-Corroded-Pipe
- **Images Used:** 25
- **Content:** Clean, non-corroded pipeline surface images as control group
- **Original Task:** Classification baseline
- **Our Use:** Paired with permissive SOP rules to test model resistance to false positives
- **License:** See Roboflow homepage

---

## 2. Fine-Tuning Dataset — Training Data (Gauges Only)

### 2a. Francesco/gauge-u2lwv (HuggingFace / Roboflow 100)

- **URL:** https://huggingface.co/datasets/Francesco/gauge-u2lwv
- **Original Source:** https://universe.roboflow.com/object-detection/gauge-u2lwv
- **Total Images:** 235 (158 train + 25 validation + 52 test)
- **Format:** JPEG images extracted from Parquet files
- **Local Path:** `Dataset_FineTune/hf_gauge_*.jpg`
- **Original Task:** Object detection (bounding boxes only — no gauge readings)
- **Our Use:** Images only — gauge readings manually annotated by us
- **License:** See original Roboflow homepage
- **Citation:**
  ```bibtex
  @misc{gauge-u2lwv,
    title   = {gauge u2lwv Dataset},
    type    = {Open Source Dataset},
    author  = {Roboflow 100},
    howpublished = {\url{https://universe.roboflow.com/object-detection/gauge-u2lwv}},
    url     = {https://universe.roboflow.com/object-detection/gauge-u2lwv},
    journal = {Roboflow Universe},
    publisher = {Roboflow},
    year    = {2022},
    month   = {nov},
    note    = {visited on 2023-03-29}
  }
  ```

### 2b. YOLO-lxxyl/analog-gauge-meter (Roboflow Universe)

- **URL:** https://universe.roboflow.com/yolo-lxxyl/analog-gauge-meter/dataset/1
- **Status:** Reference only — images to be downloaded and selected manually
- **Original Task:** Analog gauge meter detection
- **Our Use:** Supplementary gauge images for fine-tuning training set
- **Local Path:** TBD (images to be added to `Dataset_FineTune/` after selection)

---

## Data Integrity Rules

1. **Golden 100 is NEVER used for training.** It remains the held-out evaluation benchmark across all experiments (zero-shot baseline, prompt interventions, and fine-tuning).
2. **Fine-tuning data** comes exclusively from the sources listed in Section 2 above.
3. All gauge readings in the fine-tuning set are **manually annotated** — no automated labels from the original datasets are used for our compliance task.
4. Constraint pairs (Rule A / Rule B) are generated programmatically from the manual annotations to match the schema in `clean_metadata.json`.
