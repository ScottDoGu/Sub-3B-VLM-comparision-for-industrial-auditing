import os
import torch
import pandas as pd
import gc
import argparse
from tqdm import tqdm
from PIL import Image
from exhaustive_strategies.innovations.registry import get_all_strategies
from src.generation_baseline.inference_utils import load_preprocessed_metadata

class VLM_Engine:
    def __init__(self, model_id):
        from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText, AutoModelForMultimodalLM
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        mapping = {
            "HuggingFaceTB/SmolVLM-2.2B": "models/SmolVLM",
            "Qwen/Qwen2-VL-2B-Instruct": "models/Qwen2VL",
            "openbmb/MiniCPM-V-2_6": "models/MiniCPM",
            "deepseek-ai/Janus-Pro-1.6B": "models/Janus",
            "internvl/internvl2-2b": "models/InternVL2",
            "google/gemma-4-E2B-it": "models/Gemma4E2B"
        }
        local_path = mapping.get(model_id, f"models/{model_id.split('/')[-1]}")

        print(f"[ENGINE] Initializing {model_id} from {local_path}")
        self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)

        if "Gemma4E2B" in local_path:
            model_class = AutoModelForMultimodalLM
        elif any(x in local_path for x in ["SmolVLM", "InternVL", "Janus"]):
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModelForVision2Seq

        self.model = model_class.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        ).eval()

    def infer(self, prompt, images=None):
        if images and not isinstance(images, list): images = [images]

        if "Qwen2" in self.model_id:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

        generated_ids = out[0][inputs['input_ids'].shape[1]:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return {"text": text.strip(), "elapsed_s": 0.1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()

    engine = VLM_Engine(args.model)
    strategies = get_all_strategies()
    dataset = load_preprocessed_metadata()
    os.makedirs(args.results, exist_ok=True)

    torch.manual_seed(42 + args.run)
    final_results = []

    print(f"[INFO] Evaluating {args.model} | Run {args.run}")

    for i, item in enumerate(tqdm(dataset, desc="Processing Images")):
        img = Image.open(item['processed_path']).convert("RGB")
        row = item.copy()
        for name, strategy in strategies.items():
            try:
                res = strategy.run(engine, img, item.get('category', 'item'))
                row[f"{name}_label"] = res.get('label')
                row[f"{name}_text"] = res.get('text')
            except Exception as e:
                row[f"{name}_error"] = str(e)
        final_results.append(row)
        if (i+1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    pd.DataFrame(final_results).to_csv(os.path.join(args.results, f"results_run_{args.run}.csv"), index=False)
    print(f"[SUCCESS] Saved comprehensive evaluations to {args.results}")
