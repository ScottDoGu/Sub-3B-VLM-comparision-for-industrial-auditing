from qwen_extended.innovations.base import Strategy, base_prompt, label_from_text, conf_from_text
class TaskDecomposition(Strategy):
    name = "task_decomposition"
    def run(self, engine, img, cat):
        prev = ""
        for step in ["Identify material.", f"Given prior analysis: {prev} Quantify severity.", f"Given prior analysis: {prev} Final verdict PASS/FAIL. Confidence %."]:
            r = engine.infer(base_prompt(cat) + step, images=[img])
            prev = r["text"]
        return {"strategy": self.name, "label": label_from_text(prev), "confidence": conf_from_text(prev), "raw_text": prev}
