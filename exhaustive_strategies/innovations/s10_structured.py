import json, re
from exhaustive_strategies.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text
class StructuredOutput(Strategy):
    name = "structured_output"
    def run(self, engine, img, cat):
        r = engine.infer(base_prompt(cat) + "Respond ONLY with valid JSON schema: {'label':'PASS|FAIL', 'confidence_pct':'int'}", images=[img])
        return {"strategy": self.name, "label": label_from_text(r["text"]), "confidence": conf_from_text(r["text"])}
