"""
Qwen2-VL-2B — Agentic Foveation Pipeline
Three-stage inference: Localize → Crop & Upscale → Reason

This is a Qwen2-VL-only innovation that uses the model as its own
spatial grounding agent before performing the final audit judgement,
simulating how a human inspector would squint at a specific region.

Stage 1 (The Squint):  Ask the model to locate the region of interest
                        and output a bounding box in [0, 1000] coords.
Stage 2 (The Focus):   Crop and upscale the detected region to 1024x1024
                        to maximise high-resolution token allocation.
Stage 3 (The Verdict): Run spatially-grounded CoT on the upscaled crop
                        to read the value and apply the safety rule.
"""
import torch
import re
from PIL import Image, ImageDraw
import os
import sys
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

# inference_utils lives in generation_baseline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generation_baseline"))
from inference_utils import load_preprocessed_metadata, save_results

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CROP_PAD_RATIO = 0.15          # 15 % padding around detected bbox
UPSCALE_SIZE   = (1024, 1024)  # Force high-res token mapping
FALLBACK_BBOX  = (250, 250, 750, 750)  # Centre crop if parsing fails

# ─────────────────────────────────────────────────────────────
# Stage Prompts
# ─────────────────────────────────────────────────────────────

# Stage 1 – Localization (identical for gauges and pipelines)
PROMPT_STAGE1 = (
    "Task: Locate the primary region of interest in this industrial image.\n"
    "First, briefly describe where the main equipment or area of concern is "
    "located in the image (e.g., centre, upper-left, etc.).\n"
    "Then, output a tight bounding box around it in exactly this format: "
    "[xmin, ymin, xmax, ymax] where values are integers from 0 to 1000."
)

def build_stage3_prompt(constraint):
    return (
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


def run_vlm(model, processor, image, prompt, device, max_tokens=256):
    """Single-pass VLM inference helper. Returns decoded text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
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
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def parse_bbox(response_text):
    """Extract the first four integers from model output as [xmin, ymin, xmax, ymax]."""
    coords = re.findall(r"\d+", response_text)
    if len(coords) >= 4:
        xmin, ymin, xmax, ymax = map(int, coords[:4])
        # Clamp to valid range
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(1000, xmax), min(1000, ymax)
        # Sanity: ensure box has positive area
        if xmax > xmin and ymax > ymin:
            return xmin, ymin, xmax, ymax
    return FALLBACK_BBOX


def crop_and_upscale(image, bbox_1000):
    """Crop the image using normalised [0-1000] coords and upscale."""
    orig_w, orig_h = image.size
    xmin, ymin, xmax, ymax = bbox_1000

    left   = int((xmin / 1000.0) * orig_w)
    top    = int((ymin / 1000.0) * orig_h)
    right  = int((xmax / 1000.0) * orig_w)
    bottom = int((ymax / 1000.0) * orig_h)

    # Pad to avoid clipping dial edges
    pad_w = int((right - left) * CROP_PAD_RATIO)
    pad_h = int((bottom - top) * CROP_PAD_RATIO)
    left   = max(0, left - pad_w)
    top    = max(0, top - pad_h)
    right  = min(orig_w, right + pad_w)
    bottom = min(orig_h, bottom + pad_h)

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize(UPSCALE_SIZE, Image.LANCZOS)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model & Processor
    local_model_path = "models/Qwen2VL"
    print(f"Loading Qwen2-VL model from {local_model_path}...")

    processor = AutoProcessor.from_pretrained(local_model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()
    print(f"Model loaded on {device}.")

    # 2. Load Golden 100 dataset
    dataset = load_preprocessed_metadata()
    N_RUNS = 3

    for run_i in range(1, N_RUNS + 1):
        results = []
        torch.manual_seed(42 + run_i)

        print(f"\n--- Starting Foveation Run {run_i}/{N_RUNS} on {len(dataset)} rows ---")

        for i, item in enumerate(dataset):
            image_path = item.get("processed_path")
            constraint = (
                item.get("logic_constraint")
                or item.get("constraint")
                or "Inspect this image for any safety concern."
            )
            category = item.get("category", "guage")

            if not image_path or not os.path.exists(image_path):
                print(f"WARNING: Skipping missing image: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")

            # ── STAGE 1: Localize ──
            stage1_response = run_vlm(
                model, processor, image, PROMPT_STAGE1, device, max_tokens=80
            )
            bbox = parse_bbox(stage1_response)

            # ── STAGE 2: Crop & Upscale (no files saved) ──
            cropped_image = crop_and_upscale(image, bbox)

            # ── STAGE 3: Reason on high-res crop ──
            stage3_prompt = build_stage3_prompt(constraint)
            stage3_response = run_vlm(
                model, processor, cropped_image, stage3_prompt, device, max_tokens=384
            )

            # Store result — same schema as all other runners
            result_entry = item.copy()
            result_entry["model_response"] = stage3_response
            result_entry["stage1_bbox"] = str(bbox)
            result_entry["stage1_raw"] = stage1_response
            result_entry["run_iteration"] = run_i
            results.append(result_entry)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images...")

        # 3. Save Results
        save_results(
            results,
            "qwen2_vl_decomp_foveation",
            iteration=run_i,
            out_dir="results/innovation/decomp_foveation",
        )

    # Cleanup
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
