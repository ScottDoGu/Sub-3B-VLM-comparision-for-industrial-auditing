from abc import ABC, abstractmethod
from typing import Dict, Any


class Strategy(ABC):
    name: str  # used in results/<model>/<strategy>/...

    @abstractmethod
    def build_prompt(
        self,
        base_prompt: str,
        metadata: Dict[str, Any],
        model_name: str,
    ) -> str:
        """
        Take the baseline prompt and optionally modify it
        (CoT, contrast, decomposition, etc.).
        """
        ...

    @abstractmethod
    def postprocess(
        self,
        model_output: str,
        parsed_output: str,
        metadata: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Optionally verify/correct the parsed output.

        Must return at least:
          - "model_output": str        # possibly same as input
          - "parsed_output": str       # possibly same as input
          - "is_correct": bool
          - "is_wrong": bool
          - "is_no_verdict": bool
        Can add extra fields (e.g., "verification_trace").
        """
        ...
