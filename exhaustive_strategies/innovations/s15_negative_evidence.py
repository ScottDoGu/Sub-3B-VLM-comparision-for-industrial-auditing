# s15_negative_evidence.py
from exhaustive_strategies.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text

class NegativeEvidence(Strategy):
    name = "negative_evidence"

    def run(self, engine, img, cat):
        # Ask model to list failure conditions
        fail_prompt = (
            base_prompt(cat) +
            "Before giving a verdict, list what visual evidence would indicate FAIL. "
            "Then analyze the image and give PASS/FAIL with confidence %."
        )
        r = engine.infer(fail_prompt, images=[img])

        return {
            "strategy": self.name,
            "label": label_from_text(r["text"]),
            "confidence": conf_from_text(r["text"])
        }
