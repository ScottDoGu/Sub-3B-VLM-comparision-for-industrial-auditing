# =============================================================================
#  Gemma 4 Innovation Pipeline Reproducibility Script
#  Runs the remaining experimental phases sequentially for Gemma-4-E2B-it.
#  (Baseline was already run via its own script)
#
#  Prerequisites:
#    1. "gemma_env" virtual environment created and installed
#    2. Gemma-4-E2B-it model downloaded
# =============================================================================

$gemma = "gemma_env\Scripts\python.exe"

# Write-Host "`n============================================"
# Write-Host "  PHASE 2: Chain-of-Thought Inference"
# Write-Host "============================================"
# Write-Host "`n---> Gemma-4-E2B-it CoT..."
# & $gemma src\generation_cot\run_gemma4_e2b_cot.py

# Write-Host "`n============================================"
# Write-Host "  PHASE 3: Rule Decomposition Inference"
# Write-Host "============================================"
# Write-Host "`n---> Gemma-4-E2B-it Decomposition..."
# & $gemma src\generation_decomposition\run_gemma4_e2b_decomp.py

# Write-Host "`n============================================"
# Write-Host "  PHASE 4: CLAHE + Decomposition Inference"
# Write-Host "============================================"
# Write-Host "`n---> Gemma-4-E2B-it Contrast..."
# & $gemma src\generation_contrast\run_gemma4_e2b_contrast.py

# Write-Host "`n============================================"
# Write-Host "  PHASE 5: CLAHE + CoT Inference"
# Write-Host "============================================"
# Write-Host "`n---> Gemma-4-E2B-it Contrast+CoT..."
# & $gemma src\generation_contrast_cot\run_gemma4_e2b_contrast_cot.py

Write-Host "`n============================================"
Write-Host "  PHASE 6: Hardware Profiling"
Write-Host "============================================"
Write-Host "`n---> Profiling Gemma-4-E2B-it..."
& $gemma src\generation_profiling\run_gemma4_e2b_profile.py

Write-Host "`n============================================"
Write-Host "  GEMMA INNOVATION PHASES COMPLETE"
Write-Host "  Results saved to: results/"
Write-Host "============================================"
