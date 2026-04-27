from qwen_extended.innovations.base import Strategy, base_prompt
class FewShotGrounding(Strategy):
    name = "few_shot_grounding"
    def run(self, engine, img, cat):
        ex = "Ex: Clean surface -> PASS 95%. Ex: Patches covering 40% -> FAIL 90%."
        r = engine.infer(base_prompt(cat) + ex + "Now analyze. Verdict PASS/FAIL, conf %.", images=[img])
        return self._result(r["text"])
