"""
Qwen2-VL-2B + LoRA Adapter + CLAHE (no decomposition)
Tests the LoRA fine-tuned vision encoder with dual-channel CLAHE preprocessing.
"""
import torch
from PIL import Image
import os
import sys
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# inference_utils lives in generation_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

# image_utils lives in generation_contrast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_contrast"))
from image_utils import apply_clahe_and_concatenate

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Base Model + LoRA Adapter
local_model_path = "models/Qwen2VL"
lora_adapter_path = "models/qwen2vl_gauge_lora"
print(f"Loading Qwen2-VL base from {local_model_path}...")
print(f"Loading LoRA adapter from {lora_adapter_path}...")

processor = AutoProcessor.from_pretrained(local_model_path)
base_model = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model = model.eval()
print(f"LoRA model loaded on {device}.")

# 3. Load Data
dataset = load_preprocessed_metadata()
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    torch.manual_seed(42 + run_i)

    print(f"\n--- Starting LoRA+CLAHE Run {run_i}/{N_RUNS} on {len(dataset)} images ---")

    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."

        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue

        # Load Image + Apply CLAHE
        original_image = Image.open(image_path).convert("RGB")
        image = apply_clahe_and_concatenate(original_image, max_dim=512)

        # Standard Prompt (same as baseline)
        standard_prompt = get_standard_prompt(constraint)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": standard_prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        result_entry = item.copy()
        result_entry["model_response"] = output_text
        result_entry["run_iteration"] = run_i
        results.append(result_entry)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")

    save_results(results, "qwen2_vl_lora_clahe", iteration=run_i, out_dir="results/innovation/lora")

# Cleanup
del model, base_model, processor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
