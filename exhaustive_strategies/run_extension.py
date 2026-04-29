import argparse
import os
import torch
from exhaustive_strategies.pipeline.orchestrator import StrategyOrchestrator
from exhaustive_strategies.config import MODEL_CONFIGS, STRATEGY_LIST
from exhaustive_strategies.evaluation.evaluate import evaluate_results

def parse_args():
    parser = argparse.ArgumentParser(description="Run exhaustive inference-time strategies.")
    parser.add_argument("--model", type=str, required=True, help="Model name or HF path.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder.")
    parser.add_argument("--results", type=str, required=True, help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.results, exist_ok=True)

    # Load model-specific settings (precision, quantization, etc.)
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Model {args.model} not found in MODEL_CONFIGS.")

    model_settings = MODEL_CONFIGS[args.model]

    # Enforce deterministic behavior
    torch.manual_seed(42)

    orchestrator = StrategyOrchestrator(
        model_name=args.model,
        dataset_path=args.dataset,
        results_path=args.results,
        strategy_list=STRATEGY_LIST,
        model_settings=model_settings,
        resume=args.resume
    )

    orchestrator.run_all_strategies()

    # Evaluate results
    evaluate_results(args.results)

if __name__ == "__main__":
    main()
