from exhaustive_strategies.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text
class ConfidenceCalibration(Strategy):
    name = "confidence_calibration"
    def run(self, engine, img, cat):
        r = engine.infer(base_prompt(cat) + "Verdict PASS/FAIL, confidence %, explain uncertainty.", images=[img])
        t = r["text"]
        pen = sum(v for k,v in {"might":.15, "maybe":.15, "possibly":.15, "appears":.10, "uncertain":.20}.items() if k in t.lower())
        return {"strategy": self.name, "label": label_from_text(t), "confidence": max(0.0, conf_from_text(t) - pen)}
