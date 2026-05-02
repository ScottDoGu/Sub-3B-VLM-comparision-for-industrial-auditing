from .s01_clahe import CLAHEContrast
from .s02_cot import ChainOfThought
from .s03_contrast_cot import ContrastCoT
from .s04_decomposition import TaskDecomposition
from .s05_patches import MultiScalePatch
from .s06_dual_pass import DualPassVerification
from .s07_calibration import ConfidenceCalibration
from .s08_negative import NegativePrompting
from .s09_roi import ROICropping
from .s10_structured import StructuredOutput
from .s11_few_shot import FewShotGrounding
from .s12_ensemble import AdaptiveEnsembleRouter
from .s13_grid import GridOverlay
from .s14_multipass_verification import MultiPassVerification
from .s15_negative_evidence import NegativeEvidence
from .s16_self_consistency import SelfConsistency

def get_all_strategies():
    return {
        "clahe_contrast": CLAHEContrast(),
        "chain_of_thought": ChainOfThought(),
        "contrast_cot": ContrastCoT(),
        "task_decomposition": TaskDecomposition(),
        "multi_scale_patch": MultiScalePatch(),
        "dual_pass_verification": DualPassVerification(),
        "confidence_calibration": ConfidenceCalibration(),
        "negative_prompting": NegativePrompting(),
        "roi_cropping": ROICropping(),
        "structured_output": StructuredOutput(),
        "few_shot_grounding": FewShotGrounding(),
        "adaptive_ensemble": AdaptiveEnsembleRouter(),
        "grid_overlay": GridOverlay(),
        "multipass_verification": MultiPassVerification(),
        "negative_evidence": NegativeEvidence(),
        "self_consistency": SelfConsistency()
    }
