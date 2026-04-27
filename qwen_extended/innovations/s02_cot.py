from qwen_extended.innovations.base import Strategy, base_prompt
COT_SUFFIX = "Think step by step:\n1. Describe condition.\n2. Note anomalies.\n3. Final verdict: PASS or FAIL. Confidence %."
class ChainOfThought(Strategy):
    name = "chain_of_thought"
    def run(self, engine, img, cat):
        r = engine.infer(base_prompt(cat) + COT_SUFFIX, images=[img])
        return self._result(r["text"], elapsed=r["elapsed_s"])
