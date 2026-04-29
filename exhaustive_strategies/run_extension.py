import json
import os
from pathlib import Path
from typing import Dict, List, Any

from src.generation_baseline.inference_utils import (
    load_image_and_preprocess,
    run_model_inference,
    parse_model_output,
)
from src.generation_baseline.download_models import load_model  # whatever the actual name is
from src.ingestion.load_metadata import load_clean_metadata    # or direct json.load

from exhaustive_strategies.base_strategy import Strategy
from exhaustive_strategies.generation_cot.cot_simple import CoTSimple
from exhaustive_strategies.generation_contrast.contrast_dual_prompt import ContrastDual
# add more strategies here


STRATEGIES: List[Strategy] = [
    CoTSimple(),
    ContrastDual(),
    # ...
]

MODELS: List[str] = [
    "Qwen2-VL",
    "MiniCPM",
    "InternVL2",
    "Janus",
    "SmolVLM",
    "Gemma4-E2B",
]


def run_extension(
    results_root: str = "results",
    metadata_path: str = "src/ingestion/clean_metadata.json",
    num_runs: int = 1,
    config: Dict[str, Any] | None = None,
) -> None:
    config = config or {}
    metadata = json.load(open(metadata_path, "r", encoding="utf-8"))

    for model_name in MODELS:
        model = load_model(model_name)  # must use baseline loader

        for strategy in STRATEGIES:
            for run_idx in range(num_runs):
                out_dir = Path(results_root) / model_name / strategy.name / f"run_{run_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)

                predictions: List[Dict[str, Any]] = []

                for sample in metadata:
                    image_path = sample["processed_path"]
                    image = load_image_and_preprocess(image_path, model_name)

                    raw_output = run_model_inference(model, image, sample, config)
                    parsed = parse_model_output(raw_output)

                    result = {
                        "image_id": sample["image_id"],
                        "artifact_tag": sample["artifact_tag"],
                        "category": sample["category"],
                        "ground_truth": sample["expected_verdict"],
                        "model_output": raw_output,
                        "parsed_output": parsed,
                        # strategy-specific fields:
                        **strategy.run(model, image, sample, config),
                    }

                    # ensure correctness flags exist
                    assert all(
                        k in result
                        for k in ["is_correct", "is_wrong", "is_no_verdict"]
                    )

                    predictions.append(result)

                with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=2)

                # you can call the same metric computation used in baseline here
                # e.g. compute_metrics(predictions, out_dir)

    # Evaluate results
    evaluate_results(args.results)

if __name__ == "__main__":
    main()
