import os
DATA_DIR   = os.getenv("QE_DATA",   "Data_Preprocessed")
OUTPUT_DIR = os.getenv("QE_OUTPUT", "results/innovation/qwen_extended")
MODEL_ID    = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
MAX_TOKENS  = 512
TEMPERATURE = 0.1
MIN_PIXELS  = 256 * 28 * 28
MAX_PIXELS  = 768 * 28 * 28
SWEEP_STRATEGIES = [
    "clahe_contrast", "chain_of_thought", "contrast_cot",
    "task_decomposition", "multi_scale_patch", "dual_pass_verification",
    "confidence_calibration", "negative_prompting", "roi_cropping",
    "structured_output", "few_shot_grounding", "grid_overlay",
]
STRATEGY_WEIGHTS = {
    "clahe_contrast": 1.0,  "chain_of_thought": 1.2, "contrast_cot": 1.3,
    "task_decomposition": 1.5, "multi_scale_patch": 1.4, "dual_pass_verification": 1.6,
    "confidence_calibration": 1.1, "negative_prompting": 1.0, "roi_cropping": 1.2,
    "structured_output": 1.1, "few_shot_grounding": 1.3, "grid_overlay": 0.8,
}
FN_WEIGHT = 10.0
FP_WEIGHT = 1.0
