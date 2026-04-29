from exhaustive_strategies.innovations.base import Strategy, base_prompt
from exhaustive_strategies.preprocessing.image_backend import enhance
from exhaustive_strategies.innovations.s02_cot import COT_SUFFIX
class ContrastCoT(Strategy):
    name = "contrast_cot"
    def run(self, engine, img, cat):
        r = engine.infer(base_prompt(cat) + COT_SUFFIX, images=[enhance(img)])
        return self._result(r["text"], elapsed=r["elapsed_s"])
