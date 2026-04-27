from qwen_extended.innovations.s01_clahe import CLAHEContrast
from qwen_extended.innovations.s02_cot import ChainOfThought
from qwen_extended.innovations.s03_contrast_cot import ContrastCoT
from qwen_extended.innovations.s04_decomposition import TaskDecomposition
from qwen_extended.innovations.s05_patches import MultiScalePatch
from qwen_extended.innovations.s06_dual_pass import DualPassVerification
from qwen_extended.innovations.s07_calibration import ConfidenceCalibration
from qwen_extended.innovations.s08_negative import NegativePrompting
from qwen_extended.innovations.s09_roi import ROICropping
from qwen_extended.innovations.s10_structured import StructuredOutput
from qwen_extended.innovations.s11_few_shot import FewShotGrounding
from qwen_extended.innovations.s12_ensemble import AdaptiveEnsembleRouter
from qwen_extended.innovations.s13_grid import GridOverlay

STRATEGY_REGISTRY = {
    "clahe_contrast": CLAHEContrast, "chain_of_thought": ChainOfThought, "contrast_cot": ContrastCoT,
    "task_decomposition": TaskDecomposition, "multi_scale_patch": MultiScalePatch,
    "dual_pass_verification": DualPassVerification, "confidence_calibration": ConfidenceCalibration,
    "negative_prompting": NegativePrompting, "roi_cropping": ROICropping,
    "structured_output": StructuredOutput, "few_shot_grounding": FewShotGrounding,
    "adaptive_ensemble_router": AdaptiveEnsembleRouter, "grid_overlay": GridOverlay,
}
