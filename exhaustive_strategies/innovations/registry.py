from s1_baseline import Baseline
from s2_cot import ChainOfThought
from s3_decomposition import Decomposition
from s4_contrast import Contrast
from s5_contrast_cot import ContrastCoT
from s6_vision_only import VisionOnly
from s7_noise_injection import NoiseInjectedPrompt
from s8_divergence import DivergencePrompting
from s9_verification_only import VerificationOnly
from s10_structured_output import StructuredOutput
from s11_few_shot import FewShotGrounding
from s12_adaptive_router import AdaptiveEnsembleRouter
from s13_cross_model import MultiModelCrossCheck
from s14_multi_pass import MultiPassVerification
from s15_negative_evidence import NegativeEvidence
from s16_self_consistency import SelfConsistency

STRATEGY_REGISTRY = {
    "baseline": Baseline,
    "cot": ChainOfThought,
    "decomposition": Decomposition,
    "contrast": Contrast,
    "contrast_cot": ContrastCoT,
    "vision_only": VisionOnly,
    "noise_injection": NoiseInjectedPrompt,
    "divergence": DivergencePrompting,
    "verification_only": VerificationOnly,
    "structured_output": StructuredOutput,
    "few_shot_grounding": FewShotGrounding,
    "adaptive_ensemble_router": AdaptiveEnsembleRouter,
    "cross_model_check": MultiModelCrossCheck,
    "multi_pass_verification": MultiPassVerification,
    "negative_evidence": NegativeEvidence,
    "self_consistency": SelfConsistency,
}

