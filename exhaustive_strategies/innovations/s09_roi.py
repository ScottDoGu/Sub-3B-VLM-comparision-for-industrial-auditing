from qwen_extended.innovations.base import Strategy, base_prompt
from qwen_extended.preprocessing.image_backend import extract_gauge_roi, extract_pipe_roi
class ROICropping(Strategy):
    name = "roi_cropping"
    def run(self, engine, img, cat):
        roi = extract_gauge_roi(img) if "gauge" in cat else extract_pipe_roi(img)
        r = engine.infer(base_prompt(cat) + "Cropped to ROI. Verdict PASS/FAIL, conf %.", images=[roi])
        return self._result(r["text"])
