from exhaustive_strategies.innovations.s01_clahe import CLAHEContrast
from exhaustive_strategies.innovations.s02_cot import ChainOfThought
from exhaustive_strategies.innovations.s03_contrast_cot import ContrastCoT
from exhaustive_strategies.innovations.s04_decomposition import TaskDecomposition
from exhaustive_strategies.innovations.s05_patches import MultiScalePatch
from exhaustive_strategies.innovations.s06_dual_pass import DualPassVerification
from exhaustive_strategies.innovations.s07_calibration import ConfidenceCalibration
from exhaustive_strategies.innovations.s08_negative import NegativePrompting
from exhaustive_strategies.innovations.s09_roi import ROICropping
from exhaustive_strategies.innovations.s10_structured import StructuredOutput
from exhaustive_strategies.innovations.s11_few_shot import FewShotGrounding
from exhaustive_strategies.innovations.s12_ensemble import AdaptiveEnsembleRouter
from exhaustive_strategies.innovations.s13_grid import GridOverlay

STRATEGY_REGISTRY = {
    "clahe_contrast": CLAHEContrast, "chain_of_thought": ChainOfThought, "contrast_cot": ContrastCoT,
    "task_decomposition": TaskDecomposition, "multi_scale_patch": MultiScalePatch,
    "dual_pass_verification": DualPassVerification, "confidence_calibration": ConfidenceCalibration,
    "negative_prompting": NegativePrompting, "roi_cropping": ROICropping,
    "structured_output": StructuredOutput, "few_shot_grounding": FewShotGrounding,
    "adaptive_ensemble_router": AdaptiveEnsembleRouter, "grid_overlay": GridOverlay,
}
