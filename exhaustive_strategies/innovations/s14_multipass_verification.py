# s14_multipass_verification.py
from exhaustive_strategies.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text

class MultiPassVerification(Strategy):
    name = "multi_pass_verification"

    def run(self, engine, img, cat):
        # Pass 1: initial verdict
        p1 = engine.infer(
            base_prompt(cat) + "Give PASS/FAIL and confidence %.",
            images=[img]
        )
        label1 = label_from_text(p1["text"])
        conf1 = conf_from_text(p1["text"])

        # Pass 2: ask model to critique its own answer
        critique_prompt = (
            f"The model previously answered: {p1['text']}. "
            "Re-evaluate carefully. If the previous answer seems wrong, correct it. "
            "Return PASS/FAIL and confidence %."
        )
        p2 = engine.infer(critique_prompt, images=[img])
        label2 = label_from_text(p2["text"])
        conf2 = conf_from_text(p2["text"])

        # Final decision: prefer second pass unless confidence collapses
        if abs(conf2 - conf1) > 0.25:
            label = label1
            conf = conf1
        else:
            label = label2
            conf = conf2

        return {
            "strategy": self.name,
            "label": label,
            "confidence": conf
        }
