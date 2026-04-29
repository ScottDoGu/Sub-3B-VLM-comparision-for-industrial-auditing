from qwen_extended.innovations.base import Strategy, base_prompt
from qwen_extended.preprocessing.image_backend import enhance
from qwen_extended.innovations.s02_cot import COT_SUFFIX
class ContrastCoT(Strategy):
    name = "contrast_cot"
    def run(self, engine, img, cat):
        r = engine.infer(base_prompt(cat) + COT_SUFFIX, images=[enhance(img)])
        return self._result(r["text"], elapsed=r["elapsed_s"])
