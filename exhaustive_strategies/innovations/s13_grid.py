import re
from qwen_extended.innovations.base import Strategy, base_prompt
from qwen_extended.preprocessing.image_backend import draw_grid
class GridOverlay(Strategy):
    name = "grid_overlay"
    def run(self, engine, img, cat):
        r = engine.infer(base_prompt(cat) + "Image has labeled 10x10 grid (A1-J10). Reference grid cells. Verdict PASS/FAIL, conf %.", images=[draw_grid(img)])
        return self._result(r["text"])
