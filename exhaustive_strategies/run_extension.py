import json
from pathlib import Path
from typing import Dict, Any, List

from src.generation_baseline.inference_utils import (
    load_image_and_preprocess,
    run_model_inference,
    parse_model_output,
)
from src.generation_baseline.download_models import (
    load_qwen2_vl,
    load_internvl2,
    load_minicpm,
    load_janus,
    load_smolvlm,
    load_gemma4_e2b,
)
from src.ingestion.clean_metadata import load_clean_metadata  # adjust if different

from exhaustive_strategies.base_strategy import Strategy
from exhaustive_strategies.generation_cot.cot_verify import CoTVerify
from exhaustive_strategies.generation_contrast.contrast_dual import ContrastDual


MODEL_LOADERS = {
    "Qwen2-VL": load_qwen2_vl,
    "InternVL2": load_internvl2,
    "MiniCPM": load_minicpm,
    "Janus": load_janus,
    "SmolVLM": load_smolvlm,
    "Gemma4-E2B": load_gemma4_e2b,
}

STRATEGIES: List[Strategy] = [
    CoTVerify(),
    ContrastDual(),
    # add more here
]


def build_baseline_prompt(metadata: Dict[str, Any], model_name: str) -> str:
    """
    Reproduce the exact baseline prompt logic you have now.
    For now, you can literally copy the prompt construction
    from run_qwen2_vl.py / run_internvl2.py / etc. and branch
    on model_name.
    """
    # Pseudocode – you’ll paste your real templates here:
    if model_name == "Qwen2-VL":
        # build Qwen prompt exactly as in run_qwen2_vl.py
        ...
    elif model_name == "InternVL2":
        ...
    elif model_name == "Gemma4-E2B":
        ...
    else:
        ...
    return prompt


def run_extension(
    results_root: str = "results",
    metadata_path: str = "src/ingestion/clean_metadata.json",
    num_runs: int = 1,
    config: Dict[str, Any] | None = None,
) -> None:
    config = config or {}
    metadata = json.load(open(metadata_path, "r", encoding="utf-8"))

    for model_name, loader in MODEL_LOADERS.items():
        model, processor = loader()

        for strategy in STRATEGIES:
            for run_idx in range(num_runs):
                out_dir = Path(results_root) / model_name / strategy.name / f"run_{run_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)

                predictions: List[Dict[str, Any]] = []

                for sample in metadata:
                    image_path = sample["processed_path"]
                    image_inputs = load_image_and_preprocess(image_path, processor)

                    base_prompt = build_baseline_prompt(sample, model_name)
                    prompt = strategy.build_prompt(base_prompt, sample, model_name)

                    raw_output = run_model_inference(model, processor, image_inputs, prompt)
                    parsed = parse_model_output(raw_output)

                    post = strategy.postprocess(raw_output, parsed, sample, model_name)

                    result = {
                        "image_id": sample["image_id"],
                        "artifact_tag": sample["artifact_tag"],
                        "category": sample["category"],
                        "ground_truth": sample["expected_verdict"],
                        "model_output": post["model_output"],
                        "parsed_output": post["parsed_output"],
                        "is_correct": post["is_correct"],
                        "is_wrong": post["is_wrong"],
                        "is_no_verdict": post["is_no_verdict"],
                    }
                    # carry any extra fields (e.g., traces)
                    for k, v in post.items():
                        if k not in result:
                            result[k] = v

                    predictions.append(result)

                with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=2)

                # call your existing metric computation here if it’s a function
                # compute_metrics(predictions, out_dir)
