from exhaustive_strategies.config import FN_WEIGHT, FP_WEIGHT
def score_single(predicted, ground_truth, confidence=0.5, category="gauge"):
    if ground_truth is None: return {"scored": False, "penalty": 0.0}
    pred, gt = predicted.lower().strip(), ground_truth.lower().strip()
    conf_amp = 1.0 + max(0, confidence - 0.5)
    if pred == gt: return {"scored": True, "correct": True, "penalty": 0.0}
    if gt == "fail" and pred in ("pass", "uncertain"):
        return {"scored": True, "correct": False, "type": "FN", "penalty": round(FN_WEIGHT * conf_amp, 2)}
    if gt == "pass" and pred == "fail":
        return {"scored": True, "correct": False, "type": "FP", "penalty": round(FP_WEIGHT * conf_amp, 2)}
    return {"scored": True, "correct": False, "type": "OTHER", "penalty": round(FP_WEIGHT * 0.5, 2)}

def score_batch(results, ground_truths):
    total_pen = n_scored = n_cor = n_fn = n_fp = 0
    for r, gt in zip(results, ground_truths):
        s = score_single(r.get("label", "uncertain"), gt, r.get("confidence", 0.5))
        if s["scored"]:
            n_scored += 1
            total_pen += s["penalty"]
            if s.get("correct"): n_cor += 1
            if s.get("type") == "FN": n_fn += 1
            if s.get("type") == "FP": n_fp += 1
    return {"accuracy": n_cor/n_scored if n_scored else 0.0, "cfr": n_fn/n_scored if n_scored else 0.0, "n_fn": n_fn, "n_fp": n_fp, "total_penalty": total_pen}

def confusion_matrix(results, ground_truths):
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r, gt in zip(results, ground_truths):
        if gt is None: continue
        pred, gt = r.get("label", "uncertain").lower(), gt.lower()
        if pred == "fail" and gt == "fail": cm["TP"] += 1
        elif pred == "pass" and gt == "pass": cm["TN"] += 1
        elif pred == "fail" and gt == "pass": cm["FP"] += 1
        elif pred == "pass" and gt == "fail": cm["FN"] += 1
    return cm
