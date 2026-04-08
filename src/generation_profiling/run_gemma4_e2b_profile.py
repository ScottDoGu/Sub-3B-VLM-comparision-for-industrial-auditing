import torch
from PIL import Image
import os
import gc
import sys
import time

# inference_utils lives in generation_baseline; add it to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata

# Add the local directory for image_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_contrast"))
from image_utils import apply_clahe_and_concatenate

from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Model & Processor
local_model_path = "models/Gemma4E2B"
print(f"Loading Gemma-4-E2B-it model from {local_model_path}...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
) if device == "cuda" else None

processor = AutoProcessor.from_pretrained(local_model_path)

model = AutoModelForMultimodalLM.from_pretrained(
    local_model_path,
    quantization_config=quant_config,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map={"": 0} if device == "cuda" else None,
    trust_remote_code=True
).eval()

print(f"Model loaded on {device}.")

# 3. Load Data
dataset = load_preprocessed_metadata()[:50] # Profile on 50-row subset
results = []

print(f"Starting Gemma 4 Profiling inference on {len(dataset)} images...")

for i, item in enumerate(dataset):
    image_path = item.get("processed_path")
    constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
    
    if not image_path or not os.path.exists(image_path):
        print(f"WARNING: Skipping missing image: {image_path}")
        continue
    
    # Load Image
    original_image = Image.open(image_path).convert("RGB")
    
    # APPLY DUAL-CHANNEL CLAHE INNOVATION
    clahe_image = apply_clahe_and_concatenate(original_image, max_dim=512)

    MAX_EDGE = 384
    w, h = clahe_image.size
    scale = MAX_EDGE / max(w, h)
    if scale < 1.0:
        proc_image = clahe_image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
    else:
        proc_image = clahe_image

    decomp_prompt = (
        f"You are an AI industrial safety auditor.\n"
        f"Safety Rule: '{constraint}'\n\n"
        f"Identify the current reading or condition shown in the image, then evaluate it against the rule.\n"
        f"Answer the following three questions exactly in order:\n\n"
        f"Q1: Observation: What is the exact numeric reading or visible physical condition shown in the image?\n"
        f"Q2: Evaluation: Does the observation from Q1 violate the Safety Rule? (Answer Yes or No)\n"
        f"Q3: Final Verdict: If Q2 is Yes, output UNSAFE. If Q2 is No, output SAFE.\n\n"
        f"Format your response exactly as:\n"
        f"Q1: Observation: [Your observation here]\n"
        f"Q2: Evaluation: [Yes or No]\n"
        f"Q3: Final Verdict: [SAFE or UNSAFE]"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": decomp_prompt}
            ]
        }
    ]
    
    try:
        prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    prompt_with_nudge = prompt + "Q1: "
    
    inputs = processor(
        text=prompt_with_nudge,
        images=proc_image,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=384,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1
        )
        
    end_time = time.perf_counter()
    inference_time_sec = end_time - start_time
    
    if torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    else:
        peak_vram_gb = 0.0

    input_len = inputs["input_ids"].shape[-1]
    generated_ids_trimmed = [
        out_ids[input_len:] for out_ids in generate_ids
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    output_text = output_text.strip()
    output_text = "Q1: " + output_text
    
    # Store result
    result_entry = item.copy()
    result_entry["model_response"] = output_text
    result_entry["inference_time_sec"] = inference_time_sec
    result_entry["peak_vram_gb"] = peak_vram_gb
    results.append(result_entry)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...", flush=True)

# 5. Save Results
output_dir = "results/profiling"
os.makedirs(output_dir, exist_ok=True)

import csv
output_path = os.path.join(output_dir, "gemma4_e2b_profile_results.csv")
if results:
    keys = results[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
print(f"Profiling results saved to {output_path}")

# Cleanup
del model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Execution Complete.")
