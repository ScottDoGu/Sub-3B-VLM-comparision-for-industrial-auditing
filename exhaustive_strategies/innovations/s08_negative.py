from exhaustive_strategies.innovations.base import Strategy, base_prompt
class NegativePrompting(Strategy):
    name = "negative_prompting"
    def run(self, engine, img, cat):
        neg = "GUARDRAILS: Do NOT confuse reflections for needle." if "gauge" in cat else "GUARDRAILS: Do NOT confuse paint for corrosion."
        r = engine.infer(base_prompt(cat) + neg + "Verdict PASS/FAIL, conf %.", images=[img])
        return self._result(r["text"])
