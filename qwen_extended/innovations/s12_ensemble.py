rom qwen_extended.config import STRATEGY_WEIGHTS
class AdaptiveEnsembleRouter:
    name = "adaptive_ensemble_router"
    def aggregate(self, results):
        scores = {"pass": 0.0, "fail": 0.0, "uncertain": 0.0}
        for r in results:
            w = STRATEGY_WEIGHTS.get(r["strategy"], 1.0)
            label = r.get("label", "uncertain")
            if label not in scores: label = "uncertain"
            scores[label] += w * r.get("confidence", 0.5)
        total = sum(scores.values()) or 1.0
        probs = {k: round(v / total, 4) for k, v in scores.items()}
        label = max(probs, key=probs.get)
        if abs(probs.get("pass", 0) - probs.get("fail", 0)) < 0.15: label = "uncertain"
        return {"strategy": self.name, "label": label, "confidence": round(probs.get(label, 0), 3)}
