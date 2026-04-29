import os, json, time, traceback
from exhaustive_strategies.config import SWEEP_STRATEGIES, OUTPUT_DIR
from exhaustive_strategies.innovations.registry import STRATEGY_REGISTRY
from exhaustive_strategies.innovations.s12_ensemble import AdaptiveEnsembleRouter
from exhaustive_strategies.pipeline.checkpoint import Checkpoint
from exhaustive_strategies.pipeline.image_loader import load_image, ground_truth
from exhaustive_strategies.evaluation.catastrophic_metric import score_batch, confusion_matrix

def run_sweep(engine, images, out_dir=None, resume=False):
    out = out_dir or OUTPUT_DIR
    ckpt = Checkpoint(out) if resume else None
    router = AdaptiveEnsembleRouter()
    all_results = []
    t_start = time.time()

    for img_i, entry in enumerate(images):
        img_key = entry["file"]
        print(f"\n[{img_i+1}/{len(images)}] {img_key}  ({entry['cat']})")
        img = load_image(entry["path"])
        strat_results = []

        for sname in SWEEP_STRATEGIES:
            if ckpt and ckpt.is_done(img_key, sname):
                print(f"  [{sname}] skipped (checkpoint)")
                strat_results.append(ckpt.state["done"][img_key][sname])
                continue
            try:
                result = STRATEGY_REGISTRY[sname]().run(engine, img, entry["cat"])
                result.update({"image": img_key, "category": entry["cat"]})
                strat_results.append(result)
                print(f"  [{sname}] {result['label']}  conf={result['confidence']}")
                if ckpt: ckpt.mark(img_key, sname, result)
            except Exception as e:
                err = {"strategy": sname, "label": "error", "confidence": 0.0, "raw_text": str(e), "image": img_key, "category": entry["cat"]}
                strat_results.append(err)
                print(f"  [{sname}] ERROR: {e}")

        valid = [r for r in strat_results if r["label"] != "error"]
        if valid:
            ensemble = router.aggregate(valid)
            strat_results.append(ensemble)
            print(f"  [ENSEMBLE] {ensemble['label']}  conf={ensemble['confidence']}")

        all_results.append({"image": img_key, "strategies": strat_results})

    elapsed = time.time() - t_start
    per_strategy = {}
    for res in all_results:
        gt_label = ground_truth(next(e for e in images if e["file"] == res["image"]))
        for sr in res["strategies"]:
            sn = sr["strategy"]
            per_strategy.setdefault(sn, {"results": [], "gts": []})
            per_strategy[sn]["results"].append(sr)
            per_strategy[sn]["gts"].append(gt_label)

    strat_eval = {}
    for sn, data in per_strategy.items():
        ev = score_batch(data["results"], data["gts"])
        strat_eval[sn] = ev

    ranking = sorted(strat_eval.items(), key=lambda x: (x[1]["cfr"], x[1]["total_penalty"]))
    print(f"\nSweep complete! Safety ranking: {[r[0] for r in ranking]}")
    return ranking
