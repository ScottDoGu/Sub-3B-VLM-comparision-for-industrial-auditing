# qwen_extended

A modular benchmarking package for evaluating the **Qwen Vision-Language Model (VLM)** family across all available model sizes. Designed to support reproducible, multi-run evaluation pipelines for academic research, with a built-in **triple-run protocol** that produces publication-ready aggregate statistics.

> **Project Context:** Sub-3B Vision-Language Model Benchmarking Study — University of Houston–Downtown, 2026.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Qwen VLM Family](#qwen-vlm-family)
  - [Package Architecture](#package-architecture)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Single Model Run](#single-model-run)
  - [Multi-Model Sweep](#multi-model-sweep)
  - [Triple-Run Protocol](#triple-run-protocol)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

`qwen_extended` provides a unified interface for:

1. **Loading and running** any Qwen-family VLM (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) across all released parameter sizes.
2. **Standardized benchmarking** on vision-language tasks with consistent preprocessing, prompting, and metric collection.
3. **Statistical rigor** via the triple-run protocol — each model configuration is evaluated three times with controlled random seeds, producing mean, standard deviation, and confidence intervals for every metric.

The package abstracts away model-specific loading differences and provides a clean CLI and Python API for both interactive exploration and batch evaluation.

---

## Architecture

### Qwen VLM Family

All Qwen VLMs share a three-component architecture:

```
┌─────────────────────────────────────────────────┐
│                  Qwen VLM Pipeline              │
│                                                 │
│  ┌──────────────┐   ┌────────────────────────┐  │
│  │ Vision       │   │  MLP-Based             │  │
│  │ Encoder      │──▶│  Vision-Language Merger │  │
│  │ (ViT)        │   └───────────┬────────────┘  │
│  └──────────────┘               │               │
│                                 ▼               │
│                     ┌───────────────────────┐   │
│      Text Input ──▶ │  LLM Backbone         │   │
│                     │  (Qwen2.5 / Qwen3)    │   │
│                     └───────────────────────┘   │
└─────────────────────────────────────────────────┘
```

| Component | Role |
|---|---|
| **Vision Encoder (ViT)** | Processes images at native dynamic resolution. Uses 2D-RoPE and window attention for efficient multi-scale feature extraction. Patch stride of 14 on images resized to multiples of 28. |
| **Vision-Language Merger** | An MLP-based projection layer that maps variable-length visual token sequences into the LLM's embedding space. |
| **LLM Backbone** | A pretrained Qwen language model with Multimodal Rotary Position Embedding (M-RoPE) decomposing positional encoding into 1D (text), 2D (image), and 3D (video) components. |

**Key architectural features across generations:**
- **Qwen2-VL:** Naive Dynamic Resolution, M-RoPE for multimodal positional encoding.
- **Qwen2.5-VL:** Window Attention in ViT, absolute time encoding, native dynamic-resolution ViT trained from scratch.
- **Qwen3-VL:** Interleaved-MRoPE, DeepStack integration (multi-level ViT features), text-based time alignment for video.

### Package Architecture

```
qwen_extended/
├── __init__.py              # Package entry point, version
├── config.py                # RunConfig, ModelConfig, BenchmarkConfig
├── loader.py                # Unified model loading across all Qwen VLM generations
├── runner.py                # Single-run inference engine
├── multi_runner.py          # Multi-model sweep orchestrator
├── triple_run.py            # Triple-run protocol controller
├── metrics.py               # Metric computation (accuracy, F1, BLEU, CIDEr, etc.)
├── aggregator.py            # Cross-run statistical aggregation
├── preprocessing.py         # Image/video normalization and prompt formatting
├── benchmarks/              # Benchmark dataset loaders and prompt templates
│   ├── __init__.py
│   ├── mmmu.py
│   ├── mathvista.py
│   ├── docvqa.py
│   ├── realworldqa.py
│   ├── ocrbench.py
│   └── custom.py            # User-defined benchmark support
├── utils/
│   ├── seeds.py             # Seed management for reproducibility
│   ├── logging.py           # Structured logging and progress tracking
│   ├── gpu.py               # GPU memory management and multi-GPU distribution
│   └── export.py            # Results export (CSV, JSON, LaTeX)
└── cli.py                   # Command-line interface
```

---

## Supported Models

`qwen_extended` supports every publicly released Qwen VLM variant:

| Generation | Model ID | Parameters | Release | License |
|---|---|---|---|---|
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | 2B | 2024 | Apache 2.0 |
| Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` | 7B | 2024 | Apache 2.0 |
| Qwen2-VL | `Qwen/Qwen2-VL-72B-Instruct` | 72B | 2024 | Qwen License |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | 3B | Feb 2025 | Apache 2.0 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | Feb 2025 | Apache 2.0 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-72B-Instruct` | 72B | Feb 2025 | Qwen License |
| Qwen3-VL | `Qwen/Qwen3-VL-2B` | 2B (Dense) | Nov 2025 | Apache 2.0 |
| Qwen3-VL | `Qwen/Qwen3-VL-4B` | 4B (Dense) | Nov 2025 | Apache 2.0 |
| Qwen3-VL | `Qwen/Qwen3-VL-8B` | 8B (Dense) | Nov 2025 | Apache 2.0 |
| Qwen3-VL | `Qwen/Qwen3-VL-32B` | 32B (Dense) | Nov 2025 | Apache 2.0 |
| Qwen3-VL | `Qwen/Qwen3-VL-30B-A3B` | 30B MoE (3B active) | Nov 2025 | Apache 2.0 |
| Qwen3-VL | `Qwen/Qwen3-VL-235B-A22B` | 235B MoE (22B active) | Nov 2025 | Qwen License |

> **Sub-3B Focus:** For the primary benchmarking study, the target models are those with ≤3B total (or active) parameters: `Qwen2-VL-2B`, `Qwen2.5-VL-3B`, `Qwen3-VL-2B`, and `Qwen3-VL-30B-A3B`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/qwen_extended.git
cd qwen_extended

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install the package with dependencies
pip install -e .
```

### Dependencies

```
torch>=2.1.0
transformers>=4.40.0
accelerate>=0.30.0
qwen-vl-utils>=0.0.9
Pillow>=10.0
pandas>=2.0
scipy>=1.11
tqdm
```

> **GPU Requirements:** Sub-3B models run on a single GPU with ≥8 GB VRAM. The 7B models require ≥16 GB. For 32B+ models, multi-GPU or quantized inference is required.

---

## Quick Start

```python
from qwen_extended import RunConfig, run_single

config = RunConfig(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    benchmark="mathvista",
    seed=42,
    output_dir="./results"
)

results = run_single(config)
print(f"Accuracy: {results.accuracy:.4f}")
```

---

## Usage

### Single Model Run

Run a single model on a single benchmark with one seed:

```bash
# CLI
qwen-extended run \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --benchmark mathvista \
    --seed 42 \
    --output-dir ./results
```

```python
# Python API
from qwen_extended import RunConfig, run_single

config = RunConfig(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    benchmark="mathvista",
    seed=42,
    max_new_tokens=512,
    temperature=0.0,       # Greedy decoding for reproducibility
    output_dir="./results"
)

results = run_single(config)
```

### Multi-Model Sweep

Run every supported model (or a filtered subset) across one or more benchmarks:

```bash
# Run all sub-3B models across all benchmarks
qwen-extended sweep \
    --filter "sub3b" \
    --benchmarks mathvista,mmmu,docvqa,ocrbench,realworldqa \
    --output-dir ./results/sweep
```

```python
# Python API
from qwen_extended import SweepConfig, run_sweep

sweep = SweepConfig(
    model_filter="sub3b",                  # "sub3b" | "all" | list of model IDs
    benchmarks=["mathvista", "mmmu", "docvqa", "ocrbench", "realworldqa"],
    seed=42,
    output_dir="./results/sweep"
)

sweep_results = run_sweep(sweep)
sweep_results.to_csv("sweep_summary.csv")
```

**Filter options:**

| Filter | Models Included |
|---|---|
| `sub3b` | Qwen2-VL-2B, Qwen2.5-VL-3B, Qwen3-VL-2B, Qwen3-VL-30B-A3B |
| `dense` | All dense (non-MoE) variants |
| `moe` | All MoE variants |
| `gen2` | All Qwen2-VL models |
| `gen2.5` | All Qwen2.5-VL models |
| `gen3` | All Qwen3-VL models |
| `all` | Every supported model |

### Triple-Run Protocol

The triple-run protocol is the **required evaluation methodology for the final paper**. It executes three independent runs per model-benchmark pair, each with a distinct random seed, to produce statistically robust results.

#### Protocol Specification

```
For each (model M, benchmark B):
    Run 1: seed = 42
    Run 2: seed = 123
    Run 3: seed = 7
    ────────────────────────────
    Report: mean ± std across 3 runs
            95% confidence interval
            per-run breakdowns
```

#### Why Three Runs?

1. **Variance Quantification:** Even with greedy decoding (`temperature=0`), non-determinism in GPU floating-point operations, batch ordering, and dynamic resolution padding can produce run-to-run variance. Three runs capture this.
2. **Outlier Detection:** A single anomalous run is identifiable and reportable when flanked by two consistent runs.
3. **Publication Standard:** Reporting mean ± std from multiple runs is the accepted norm for VLM benchmarking in NeurIPS/CVPR-tier venues.

#### Running the Triple-Run Protocol

```bash
# CLI — full triple-run for all sub-3B models
qwen-extended triple-run \
    --filter "sub3b" \
    --benchmarks mathvista,mmmu,docvqa,ocrbench,realworldqa \
    --output-dir ./results/triple_run
```

```python
# Python API
from qwen_extended import TripleRunConfig, run_triple

config = TripleRunConfig(
    model_filter="sub3b",
    benchmarks=["mathvista", "mmmu", "docvqa", "ocrbench", "realworldqa"],
    seeds=[42, 123, 7],             # Fixed seed triplet
    output_dir="./results/triple_run",
    export_formats=["csv", "latex"]  # Auto-generate LaTeX tables
)

triple_results = run_triple(config)

# Access aggregate statistics
for model, stats in triple_results.items():
    print(f"{model}:")
    for benchmark, metrics in stats.items():
        print(f"  {benchmark}: {metrics.mean:.4f} ± {metrics.std:.4f} "
              f"(95% CI: [{metrics.ci_lower:.4f}, {metrics.ci_upper:.4f}])")
```

#### Triple-Run Output Structure

```
results/triple_run/
├── raw/
│   ├── Qwen2-VL-2B_mathvista_seed42.json
│   ├── Qwen2-VL-2B_mathvista_seed123.json
│   ├── Qwen2-VL-2B_mathvista_seed7.json
│   ├── ...
├── aggregate/
│   ├── aggregate_summary.csv        # Mean, std, CI for all model-benchmark pairs
│   ├── per_model_breakdown.csv      # Per-run detail for each model
│   └── tables/
│       ├── main_results.tex         # LaTeX table for paper
│       └── sub3b_comparison.tex     # Sub-3B focused comparison table
└── logs/
    ├── run_manifest.json            # Full run metadata and reproducibility record
    └── gpu_utilization.log
```

---

## Configuration

All configuration is handled through dataclasses (Python API) or CLI flags:

```python
from qwen_extended import RunConfig

config = RunConfig(
    # Model
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",          # "float16" | "bfloat16" | "float32"
    device_map="auto",               # Automatic multi-GPU distribution
    quantization=None,               # None | "4bit" | "8bit" (via bitsandbytes)

    # Generation
    max_new_tokens=512,
    temperature=0.0,                 # 0.0 = greedy (recommended for benchmarks)
    top_p=1.0,
    repetition_penalty=1.0,

    # Benchmark
    benchmark="mathvista",
    split="test",                    # "val" | "test" | "mini" (subset for debugging)
    max_samples=None,                # Limit sample count (None = full dataset)

    # Reproducibility
    seed=42,

    # Output
    output_dir="./results",
    save_predictions=True,           # Save per-sample predictions
    export_formats=["csv", "json"],  # "csv" | "json" | "latex"
)
```

---

## Output Format

### Per-Run JSON

Each individual run produces a JSON file:

```json
{
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "benchmark": "mathvista",
  "seed": 42,
  "timestamp": "2026-04-26T20:30:00",
  "metrics": {
    "accuracy": 0.6234,
    "f1": 0.6102,
    "exact_match": 0.5891
  },
  "runtime_seconds": 1847.3,
  "gpu_peak_memory_gb": 7.2,
  "num_samples": 1000,
  "config": { "...": "..." }
}
```

### Aggregate CSV (Triple-Run)

| model | benchmark | metric | run_1 | run_2 | run_3 | mean | std | ci_lower | ci_upper |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | mathvista | accuracy | 0.6234 | 0.6198 | 0.6251 | 0.6228 | 0.0027 | 0.6160 | 0.6295 |

### LaTeX Export

The `tables/` directory contains pre-formatted LaTeX tables ready for insertion into the paper:

```latex
\begin{table}[h]
\centering
\caption{Sub-3B VLM Benchmark Results (mean $\pm$ std, 3 runs)}
\begin{tabular}{lccccc}
\toprule
Model & MathVista & MMMU & DocVQA & OCRBench & RealWorldQA \\
\midrule
Qwen2-VL-2B   & 62.3 $\pm$ 0.3 & ... & ... & ... & ... \\
Qwen2.5-VL-3B & 64.1 $\pm$ 0.2 & ... & ... & ... & ... \\
Qwen3-VL-2B   & 66.8 $\pm$ 0.4 & ... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Reproducing Paper Results

To reproduce all results reported in the final paper:

```bash
# Step 1: Run the full triple-run protocol on sub-3B models
qwen-extended triple-run \
    --filter "sub3b" \
    --benchmarks mathvista,mmmu,docvqa,ocrbench,realworldqa \
    --output-dir ./results/paper_final

# Step 2: Generate LaTeX tables and summary statistics
qwen-extended export \
    --input-dir ./results/paper_final \
    --formats csv,latex \
    --output-dir ./results/paper_final/export

# Step 3 (optional): Run extended sweep including larger models for context
qwen-extended triple-run \
    --filter "all" \
    --benchmarks mathvista,mmmu \
    --output-dir ./results/extended
```

**Reproducibility guarantees:**
- Fixed seed triplet `[42, 123, 7]` is hardcoded as the default.
- All run metadata (library versions, GPU info, timestamps) is logged in `run_manifest.json`.
- The `--deterministic` flag enables `torch.use_deterministic_algorithms(True)` for maximum reproducibility (may reduce performance).

---

## Project Structure

```
qwen_extended/
├── qwen_extended/           # Source package (see Package Architecture above)
├── benchmarks_data/         # Cached benchmark datasets (gitignored)
├── results/                 # Run outputs (gitignored)
├── scripts/
│   ├── run_paper_final.sh   # One-command paper reproduction
│   └── analyze_results.py   # Post-hoc analysis and visualization
├── tests/
│   ├── test_loader.py
│   ├── test_runner.py
│   └── test_triple_run.py
├── pyproject.toml
├── setup.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `OutOfMemoryError` on 7B+ models | Use `--quantization 4bit` or set `device_map="auto"` for multi-GPU sharding. |
| Inconsistent results across runs | Ensure `temperature=0.0`. Enable `--deterministic` mode. Check that `qwen-vl-utils>=0.0.9` is installed. |
| Slow image preprocessing | Verify Pillow is compiled with SIMD support. Consider reducing `max_samples` for debugging. |
| Model download failures | Set `HF_HOME` to a directory with sufficient storage. Use `huggingface-cli login` for gated models (72B). |
| LaTeX export formatting issues | Ensure `pandas>=2.0` is installed. Run `qwen-extended export --format latex` separately after runs complete. |

---

## Citation

If you use `qwen_extended` in your research, please cite:

```bibtex
@software{qwen_extended_2026,
  title   = {qwen\_extended: A Multi-Run Benchmarking Package for Qwen Vision-Language Models},
  author  = {Scott},
  year    = {2026},
  url     = {https://github.com/<your-org>/qwen_extended}
}
```

For the Qwen model family, please also cite the relevant technical reports:

```bibtex
@article{qwen2.5vl_2025,
  title   = {Qwen2.5-VL Technical Report},
  author  = {Bai, Shuai and Chen, Keqin and Liu, Xuejing and others},
  journal = {arXiv preprint arXiv:2502.13923},
  year    = {2025}
}

@article{qwen3vl_2025,
  title   = {Qwen3-VL Technical Report},
  author  = {Bai, Shuai and Cai, Yuxuan and Chen, Ruizhe and others},
  journal = {arXiv preprint arXiv:2511.21631},
  year    = {2025}
}
```

---

## License

This package is released under the **MIT License**. See [LICENSE](./LICENSE) for details.

> **Note:** Individual Qwen models are subject to their own licenses (Apache 2.0 for smaller variants; Qwen License for 72B+ models). Refer to each model's Hugging Face page for terms.
