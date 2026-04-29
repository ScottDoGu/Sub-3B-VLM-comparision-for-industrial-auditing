import argparse, sys
from exhaustive_strategies.pipeline.model_wrapper import Qwen2VLEngine
from exhaustive_strategies.pipeline.image_loader import load_subset
from exhaustive_strategies.pipeline.orchestrator import run_sweep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct-AWQ")
    ap.add_argument("--num-gauges", type=int, default=0)
    ap.add_argument("--num-pipes", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    engine = Qwen2VLEngine(model_id=args.model_id, dry_run=args.dry_run)
    images = load_subset(n_gauge=args.num_gauges, n_pipe=args.num_pipes)
    if not images: sys.exit("[ERROR] No images found.")
    print(f"Loaded {len(images)} images")
    if args.mode == "sweep": run_sweep(engine, images, resume=args.resume)

if __name__ == "__main__": main()