import torch
from PIL import Image
import os
import gc
import sys

# inference_utils lives in generation_baseline; add it to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata, save_results
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
dataset = load_preprocessed_metadata()
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    torch.manual_seed(42 + run_i)
    
    print(f"\n--- Starting Decomp Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
        
        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue
        
        # Load Image
        image = Image.open(image_path).convert("RGB")
        MAX_EDGE = 384
        w, h = image.size
        scale = MAX_EDGE / max(w, h)
        if scale < 1.0:
            proc_image = image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
        else:
            proc_image = image

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
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=384,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.1
            )
            
        input_len = inputs["input_ids"].shape[-1]
        generated_ids_trimmed = [
            out_ids[input_len:] for out_ids in generated_ids
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        output_text = output_text.strip()
        output_text = "Q1: " + output_text
        
        # Store result
        result_entry = item.copy()
        result_entry["model_response"] = output_text
        result_entry["run_iteration"] = run_i  
        results.append(result_entry)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...", flush=True)

    # 5. Save Results
    save_results(results, "gemma4_e2b_decomp", iteration=run_i, out_dir="results/innovation/decomposition")

# Cleanup
del model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Execution Complete.")
