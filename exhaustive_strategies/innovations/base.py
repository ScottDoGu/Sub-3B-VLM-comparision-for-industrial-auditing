import re
from abc import ABC, abstractmethod
from typing import Dict, Any

class Strategy(ABC):
    name: str

    def build_prompt(self, base_prompt: str, metadata: Dict[str, Any], model_name: str) -> str:
        return base_prompt

    def postprocess(self, model_output: str, parsed_output: str, metadata: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        return {"model_output": model_output, "parsed_output": parsed_output}

    @abstractmethod
    def run(self, engine, img, cat):
        ...

    def _result(self, text: str, elapsed: float = 0.0) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "label": label_from_text(text),
            "confidence": conf_from_text(text),
            "text": text,
            "elapsed_s": elapsed
        }

def base_prompt(category: str) -> str:
    return f"Category: {category}. Analyze the image and provide a PASS/FAIL verdict with confidence. "

def label_from_text(text: str) -> str:
    match = re.search(r'\b(PASS|FAIL)\b', text.upper())
    return match.group(1) if match else "UNKNOWN"

def conf_from_text(text: str) -> float:
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match:
        val = float(match.group(1))
        if "%" in text or val > 1.0:
            return val / 100.0 if val > 1.0 else val
        return val
    return 0.5

__all__ = ["Strategy", "base_prompt", "label_from_text", "conf_from_text"]
