from qwen_extended.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text
class DualPassVerification(Strategy):
    name = "dual_pass_verification"
    def run(self, engine, img, cat):
        r1 = engine.infer(base_prompt(cat) + "Verdict PASS/FAIL, confidence %.", images=[img])
        l1, c1 = label_from_text(r1["text"]), conf_from_text(r1["text"])
        p2 = "A previous inspector flagged this as FAILING. Re-examine. Verdict PASS/FAIL, confidence %." if "gauge" in cat else "A previous inspector said this looks GOOD. Agree? Verdict PASS/FAIL, confidence %."
        r2 = engine.infer(p2, images=[img])
        l2, c2 = label_from_text(r2["text"]), conf_from_text(r2["text"])
        return {"strategy": self.name, "label": l1 if l1==l2 else "uncertain", "confidence": max(c1, c2) if l1==l2 else min(c1, c2)*0.7}
