# =============================================================================
#  LoRA Fine-Tune Evaluation Runner
#  Runs all 4 LoRA configurations (3 runs each = 12 total inferences)
#  against the Golden 100 benchmark.
#
#  Prerequisites:
#    1. four_models venv with `peft` installed
#    2. LoRA adapter saved at: models/qwen2vl_gauge_lora/
#    3. CUDA-capable GPU with at least 6GB VRAM
#
#  Usage:  .\run_lora.ps1
#  Estimated runtime: ~3-4 hours (4 configs × 3 runs × 100 images)
# =============================================================================

$fourModels = "four_models\Scripts\python.exe"

Write-Host "`n============================================"
Write-Host "  LoRA EVALUATION: 4 Configurations × 3 Runs"
Write-Host "============================================"

# --- Config 1: LoRA Baseline (standard prompt, no CLAHE) ---
Write-Host "`n----> [1/4] LoRA Baseline (3 runs)..."
& $fourModels src\generation_lora\run_qwen2_vl_lora_baseline.py

# --- Config 2: LoRA + CLAHE ---
Write-Host "`n----> [2/4] LoRA + CLAHE (3 runs)..."
& $fourModels src\generation_lora\run_qwen2_vl_lora_clahe.py

# --- Config 3: LoRA + Rule Decomposition ---
Write-Host "`n----> [3/4] LoRA + Decomposition (3 runs)..."
& $fourModels src\generation_lora\run_qwen2_vl_lora_decomp.py

# --- Config 4: LoRA + CLAHE + Decomposition (Full Stack) ---
Write-Host "`n----> [4/4] LoRA + CLAHE + Decomposition (3 runs)..."
& $fourModels src\generation_lora\run_qwen2_vl_lora_contrast.py

Write-Host "`n============================================"
Write-Host "  LoRA EVALUATION COMPLETE"
Write-Host "  Results saved to: results/innovation/lora/"
Write-Host "============================================"
