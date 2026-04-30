from .innovations.s1_baseline import Baseline
from .innovations.s2_cot import ChainOfThought
from .innovations.s3_decomposition import Decomposition
from .innovations.s4_contrast import Contrast
from .innovations.s5_contrast_cot import ContrastCoT
from .innovations.s6_vision_only import VisionOnly
from .innovations.s7_noise_injection import NoiseInjectedPrompt
from .innovations.s8_divergence import DivergencePrompting
from .innovations.s9_verification_only import VerificationOnly
from .innovations.s10_structured_output import StructuredOutput
from .innovations.s11_few_shot import FewShotGrounding
from .innovations.s12_adaptive_router import AdaptiveEnsembleRouter
from .innovations.s13_cross_model import MultiModelCrossCheck
from .innovations.s14_multi_pass import MultiPassVerification
from .innovations.s15_negative_evidence import NegativeEvidence
from .innovations.s16_self_consistency import SelfConsistency

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

