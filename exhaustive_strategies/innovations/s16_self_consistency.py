# s16_self_consistency.py
import random
from exhaustive_strategies.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text

class SelfConsistency(Strategy):
    name = "self_consistency"

    def run(self, engine, img, cat):
        labels = []
        confs = []

        for _ in range(3):
            jitter = random.choice([
                "Think step-by-step.",
                "Analyze carefully.",
                "Double-check before answering."
            ])
            prompt = base_prompt(cat) + jitter + " Give PASS/FAIL and confidence %."
            r = engine.infer(prompt, images=[img])
            labels.append(label_from_text(r["text"]))
            confs.append(conf_from_text(r["text"]))

        # Aggregate
        score = {"pass": 0.0, "fail": 0.0, "uncertain": 0.0}
        for l, c in zip(labels, confs):
            if l not in score:
                l = "uncertain"
            score[l] += c

        label = max(score, key=score.get)
        conf = round(score[label] / (sum(score.values()) or 1), 3)

        return {
            "strategy": self.name,
            "label": label,
            "confidence": conf
        }
