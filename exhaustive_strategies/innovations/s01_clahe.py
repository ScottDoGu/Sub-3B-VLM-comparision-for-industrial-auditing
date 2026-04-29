from exhaustive_strategies.innovations.base import Strategy, base_prompt
from exhaustive_strategies.preprocessing.image_backend import enhance
class CLAHEContrast(Strategy):
    name = "clahe_contrast"
    def run(self, engine, img, cat):
        r = engine.infer(base_prompt(cat) + "Verdict PASS/FAIL, confidence %.", images=[enhance(img)])
        return self._result(r["text"], elapsed=r["elapsed_s"])
