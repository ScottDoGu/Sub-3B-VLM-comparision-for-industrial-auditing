from abc import ABC, abstractmethod
from typing import Dict, Any

class Strategy(ABC):
    name: str  # used in results/<model>/<strategy>/...

    @abstractmethod
    def run(
        self,
        model,
        image,
        metadata: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Must return a dict with at least:
          - "model_output": str
          - "parsed_output": str
          - "is_correct": bool
          - "is_wrong": bool
          - "is_no_verdict": bool
        """
        ...
