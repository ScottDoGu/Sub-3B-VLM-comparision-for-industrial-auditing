import torch
from PIL import Image
import os
import gc
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from inference_utils import load_preprocessed_metadata, get_standard_prompt, save_results

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

# Number of evaluation runs to capture generation variance
N_RUNS = 3

for run_i in range(1, N_RUNS + 1):
    results = []
    # Seed control for reproducibility across runs
    torch.manual_seed(42 + run_i)
    
    # 4. Inference Loop
    print(f"\n--- Starting Baseline Run {run_i}/{N_RUNS} on {len(dataset)} images ---")
    for i, item in enumerate(dataset):
        image_path = item.get("processed_path")
        constraint = item.get("logic_constraint") or item.get("constraint") or "Inspect this image for any safety concern."
        
        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Skipping missing image: {image_path}")
            continue
        
        # Load Image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare Prompt (Gemma 4 best practices: place image before text)
        standard_prompt = get_standard_prompt(constraint)
        
        MAX_EDGE = 384
        w, h = image.size
        scale = MAX_EDGE / max(w, h)
        if scale < 1.0:
            proc_image = image.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
        else:
            proc_image = image

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": standard_prompt}
                ]
            }
        ]
        
        # Apply Template with enable_thinking=False for strict baseline bounds
        try:
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            # Fallback if transformers version doesn't implement enable_thinking natively yet
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
        # Process Inputs
        inputs = processor(
            text=text,
            images=proc_image,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate Response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=False, # True zero-shot greedy decoding
                repetition_penalty=1.1 # Standard for comparison
            )
            
        # Trim prompt from output
        input_len = inputs["input_ids"].shape[-1]
        generated_ids_trimmed = [
            out_ids[input_len:] for out_ids in generated_ids
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Clean stray system labels if they leak
        output_text = output_text.strip()
        
        # Store result
        result_entry = item.copy()
        result_entry["model_response"] = output_text
        result_entry["run_iteration"] = run_i  
        results.append(result_entry)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...", flush=True)

    # 5. Save Results
    save_results(results, "gemma4_e2b", iteration=run_i)

# Cleanup Memory Hygiene
del model, processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Execution Complete.")
