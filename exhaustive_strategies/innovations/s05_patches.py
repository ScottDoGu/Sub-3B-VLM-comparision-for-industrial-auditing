from qwen_extended.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text
from qwen_extended.preprocessing.image_backend import top_patches
class MultiScalePatch(Strategy):
    name = "multi_scale_patch"
    def run(self, engine, img, cat):
        labels, confs = [], []
        for p in top_patches(img):
            r = engine.infer(base_prompt(cat) + "Verdict PASS/FAIL, confidence %.", images=[p])
            labels.append(label_from_text(r["text"]))
            confs.append(conf_from_text(r["text"]))
        final = "fail" if labels.count("fail") >= labels.count("pass") else "pass"
        return {"strategy": self.name, "label": final, "confidence": round(sum(confs)/len(confs), 3)}
