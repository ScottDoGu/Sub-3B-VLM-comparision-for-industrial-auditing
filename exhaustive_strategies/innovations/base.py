import re
from abc import ABC, abstractmethod

PASS_KW = ["pass", "acceptable", "normal", "good", "safe", "within range", "no defect", "clean", "intact"]
FAIL_KW = ["fail", "defect", "abnormal", "danger", "unsafe", "out of range", "corroded", "corrosion", "rust", "damage", "leak", "crack"]

def label_from_text(text):
    t = text.lower()
    f = sum(1 for k in FAIL_KW if k in t)
    p = sum(1 for k in PASS_KW if k in t)
    if f > p: return "fail"
    if p > f: return "pass"
    return "uncertain"

def conf_from_text(text):
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    return min(float(m.group(1)) / 100.0, 1.0) if m else 0.5

GAUGE_PROMPT = "You are an industrial visual inspector. Examine this pressure gauge. Determine if the reading is within acceptable range (PASS) or indicates danger (FAIL). "
PIPE_PROMPT = "You are an industrial visual inspector. Examine this pipe. Determine if it shows corrosion or damage (FAIL) or is in acceptable condition (PASS). "

def base_prompt(cat): return GAUGE_PROMPT if "gauge" in cat else PIPE_PROMPT

class Strategy(ABC):
    name = "base"
    def run(self, engine, img, cat): ...
    def _result(self, text, **meta):
        return {"strategy": self.name, "label": label_from_text(text), "confidence": conf_from_text(text), "raw_text": text, "meta": meta}
