🧩 Exhaustive Strategies Extension

Inference‑Time Strategy Suite for Sub‑3B VLMs in Industrial Auditing

This module, exhaustive_strategies, extends the original Sub‑3B VLM benchmarking project by introducing a comprehensive suite of inference‑time strategies designed to probe, stress, and mitigate the failure modes observed in the first five ensemble runs.
It applies uniformly across all six models in the project and uses the same execution setup as the original ensembles to ensure strict comparability.

This extension supports the project’s final research paper, to be published on or before May 1, 2026.

📦 Folder Structure
```
.
├── exhaustive_strategies/
│   ├── preprocessing/
│   ├── pipeline/
│   ├── innovations/
│   ├── evaluation/
│   ├── config.py
│   └── run_extension.py
├── scripts/
├── results/
├── figures/
├── analysis/
├── models/
└── Dataset/
```
🧠 Models Covered

This extension applies to the six models used in the main project:

SmolVLM‑2.2B

Qwen2‑VL‑2B

MiniCPM‑V 2.6B

Janus‑Pro‑1.6B

Phi‑3.5‑Vision‑2.2B

InternVL‑2B

All models are run using the same setup as the first ensembles, including:

T4‑safe precision

model‑appropriate quantization

deterministic seeds

batch size = 1

identical evaluation loop

identical output structure

This ensures the extension’s results are directly comparable to the original method's runs.

🧩 Strategy Index (File Mapping)
| Strategy Name          | File                   | Purpose                                                |
| ---------------------- | ---------------------- | ------------------------------------------------------ |
| CLAHE Enhancement      | s01_clahe.py           | Contrast normalization for low‑visibility gauges/pipes |
| Chain‑of‑Thought       | s02_cot.py             | Structured reasoning to reduce shallow hallucinations  |
| Contrast‑CoT           | s03_contrast_cot.py    | CLAHE + CoT hybrid for difficult lighting              |
| Rule Decomposition     | s04_decomposition.py   | Breaks tasks into smaller, safer steps                 |
| Patch Sampling         | s05_patches.py         | Multi‑crop sampling for small gauges/labels            |
| Dual‑Pass Verification | s06_dual_pass.py       | Two identical prompts to detect instability            | 
| Confidence Calibration | s07_calibration.py     | Adds confidence estimation to outputs                  |
| Negative Prompting     | s08_negative.py        | Suppresses hallucination‑prone behaviors               |
| ROI Cropping           | s09_roi.py             | Focuses on gauge/pipe regions only                     |
| Structured Output      | s10_structured.py      | Enforces strict output formatting                      |
| Few‑Shot Grounding     | s11_few_shot.py        | Adds domain‑specific examples                          |
| Ensemble Aggregation   | s12_ensemble.py        | Aggregates multiple strategy outputs                   |
| Grid Overlay           | s13_grid.py            | Multi‑view grid for complex scenes                     |
| Divergence Prompting   | s14_divergence.py      | Two *different* prompts to detect semantic drift       |
| Noise Injection        | s15_noise_injection.py | Stability test via controlled prompt noise             |
| Vision‑Only Baseline   | s16_vision_only.py     | Tests reliance on visual vs textual cues               |

🚀 Running the Extension
```
python -m exhaustive_strategies.run_extension 
 --model <model_name> 
 --dataset Dataset/ 
 --results <output_dir>
 ```

Example:
```
python -m exhaustive_strategies.run_extension 
    --model Qwen/Qwen2-VL-2B-Instruct 
    --dataset Dataset/ 
    --results results/qwen2vl/
```
🔁 Reproducibility Protocol (3× Runs)
```
for i in 1 2 3 \
do
    python -m exhaustive_strategies.run_extension 
        --model Qwen/Qwen2-VL-2B-Instruct 
        --dataset Dataset/ 
        --results results/qwen2vl/run_$i/ 
done
```

This matches the reproducibility protocol used in the first ensembles.

📊 Evaluation

Each run produces:

predictions.json

metrics.json

catastrophic_summary.json

Evaluation uses the same catastrophic‑failure‑focused metrics as the original project, ensuring continuity and comparability.

🧪 Dataset
```
Dataset/
 image_001.jpg
 image_002.jpg
 …
```
The extension automatically loads all images.

🛠️ Checkpointing

Resume an interrupted run:
```
python -m exhaustive_strategies.run_extension --resume
```

📜 Note

This extension is designed to exhaustively probe inference‑time behavior in air‑gapped industrial auditing scenarios, addressing failure modes observed in the first five method runs.
It expands the project’s experimental depth.

📄 Publication

This extension supports the project’s final research paper, to be published on or before May 1, 2026.

📬 Contact

For replication support, open an issue in this repository.
