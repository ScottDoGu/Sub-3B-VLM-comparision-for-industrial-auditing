# %% [markdown]
# # Qwen2-VL-2B Vision Encoder LoRA Fine-Tuning
# 
# **Goal:** Fix the 92% perception failure bottleneck by applying LoRA to the last 2 vision encoder blocks.
# 
# **Hardware:** Colab Pro A100 (80GB) | **Dataset:** 400 gauge images, 800 constraint-paired rows
# 
# **Strategy:** QLoRA 4-bit base → LoRA on `visual.blocks.{22,23}` → Save adapter → Deploy at 8GB edge

# %% [markdown]
# ## Cell 0: Setup & Dependencies

# %%
!pip install -q "transformers>=4.45.0" "peft>=0.13.0" "trl>=0.12.0" "bitsandbytes>=0.44.0" "accelerate>=0.34.0" qwen-vl-utils datasets pillow tensorboard

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "No GPU")

# %% [markdown]
# ## Cell 1: Mount Google Drive & Load Data

# %%
from google.colab import drive
import json, os, shutil

drive.mount('/content/drive')

# === CONFIGURE THESE PATHS ===
# Upload your Dataset_FineTune folder to Google Drive first
DRIVE_DATA = "/content/drive/MyDrive/VLM_FineTune/Dataset_FineTune"
DRIVE_OUTPUT = "/content/drive/MyDrive/VLM_FineTune/lora_adapter"
LOCAL_DATA = "/content/finetune_data"

# Copy data locally for faster I/O
if not os.path.exists(LOCAL_DATA):
    print("Copying data from Drive to local...")
    shutil.copytree(DRIVE_DATA, LOCAL_DATA)
    print("Done!")

# Load metadata
with open(os.path.join(LOCAL_DATA, "finetune_metadata_augmented.json"), 'r') as f:
    metadata = json.load(f)

print(f"Loaded {len(metadata)} training rows")
print(f"Unique images: {len(set(r['image_id'] for r in metadata))}")

# Verify a few images exist
img_dir_selected = os.path.join(LOCAL_DATA, "Selected images")
img_dir_augmented = os.path.join(LOCAL_DATA, "Augmented")
sample = metadata[0]
sample_path = os.path.join(LOCAL_DATA, sample["full_path"].replace("Dataset_FineTune/", ""))
print(f"Sample image exists: {os.path.exists(sample_path)} → {sample_path}")

# %% [markdown]
# ## Cell 2: Load Qwen2-VL-2B with 4-bit Quantization

# %%
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model (4-bit)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# %% [markdown]
# ## Cell 3: Inspect Vision Encoder Architecture

# %%
# Print vision encoder structure to identify exact layer names for LoRA targeting
print("=== Vision Encoder Blocks ===")
for name, module in model.named_modules():
    if "visual" in name and ("attn" in name or "mlp" in name):
        if hasattr(module, 'weight') or any(hasattr(module, a) for a in ['in_features', 'out_features']):
            print(f"  {name}: {type(module).__name__}")

print("\n=== Last 3 Visual Blocks (LoRA targets) ===")
for name, param in model.named_parameters():
    if "visual" in name and any(f"blocks.{i}" in name for i in [21, 22, 23]):
        print(f"  {name}: {param.shape}")

# %% [markdown]
# ## Cell 4: LoRA Configuration
# 
# Target: Last 2 vision encoder transformer blocks (blocks 22-23 of 24).
# The exact module names will be confirmed from Cell 3 output.

# %%
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Freeze everything explicitly
for param in model.parameters():
    param.requires_grad = False

# LoRA config targeting vision encoder final blocks
# NOTE: Adjust target_modules based on Cell 3 output if names differ
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        # Last 2 vision encoder blocks - attention layers
        "visual.blocks.22.attn.qkv",
        "visual.blocks.22.attn.proj",
        "visual.blocks.22.mlp.fc1",
        "visual.blocks.22.mlp.fc2",
        "visual.blocks.23.attn.qkv",
        "visual.blocks.23.attn.proj",
        "visual.blocks.23.mlp.fc1",
        "visual.blocks.23.mlp.fc2",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~1-2M trainable / ~2B total = <0.1%

# %% [markdown]
# ## Cell 5: Build Training Dataset
# 
# Convert metadata JSON → HuggingFace Dataset with Qwen2-VL chat format.

# %%
from datasets import Dataset
from PIL import Image

def resolve_image_path(row):
    """Resolve image path from metadata to local filesystem."""
    rel_path = row["full_path"].replace("Dataset_FineTune/", "")
    return os.path.join(LOCAL_DATA, rel_path)

def build_conversation(row):
    """Build a single training conversation from a metadata row."""
    img_path = resolve_image_path(row)
    
    # Build the prompt matching our evaluation protocol
    constraint = row["logic_constraint"]
    prompt = (
        f"You are an industrial safety auditor inspecting field equipment. "
        f"Examine this gauge image carefully. "
        f"Read the numeric value shown on the gauge. "
        f"Then apply this safety rule: {constraint} "
        f"Provide your reading and a verdict: [SAFE] or [UNSAFE]."
    )
    
    # Build the expected response
    value = row["ground_truth_value"]
    unit = row.get("unit") or "units"
    verdict = row["expected_verdict"]
    reasoning = row["reasoning"]
    
    if isinstance(value, float) and value == int(value):
        value = int(value)
    
    response = f"The gauge reads {value} {unit}. {reasoning} [{verdict}]"
    
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ]
    }

# Filter out rows where images don't exist
valid_rows = []
missing = 0
for row in metadata:
    path = resolve_image_path(row)
    if os.path.exists(path):
        valid_rows.append(row)
    else:
        missing += 1

if missing > 0:
    print(f"WARNING: {missing} images not found, skipped")

# Build conversations
conversations = [build_conversation(r) for r in valid_rows]

# Split 90/10
import random
random.seed(42)
random.shuffle(conversations)
split_idx = int(len(conversations) * 0.9)
train_convos = conversations[:split_idx]
val_convos = conversations[split_idx:]

train_dataset = Dataset.from_list(train_convos)
val_dataset = Dataset.from_list(val_convos)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
print(f"\nSample conversation:")
print(json.dumps(train_convos[0], indent=2, default=str)[:500])

# %% [markdown]
# ## Cell 6: Training

# %%
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info

# Collator that handles Qwen2-VL multi-modal inputs
def collate_fn(examples):
    texts = []
    image_inputs = []
    
    for example in examples:
        messages = example["messages"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        
        # Extract image inputs
        imgs, _ = process_vision_info(messages)
        image_inputs.append(imgs)
    
    # Process all inputs together
    batch = processor(
        text=texts,
        images=[img for imgs in image_inputs for img in imgs] if any(image_inputs) else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    
    # Set labels = input_ids for causal LM training
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    return batch

# Training config
training_args = SFTConfig(
    output_dir="/content/qwen2vl_lora_output",
    
    # Core training
    num_train_epochs=5,
    per_device_train_batch_size=4,      # 80GB VRAM can handle this
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,       # Effective batch = 8
    
    # Optimizer
    learning_rate=1e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    
    # Precision
    bf16=True,
    
    # Memory
    gradient_checkpointing=True,
    
    # Logging
    logging_dir="/content/tb_logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    
    # Dataset
    max_seq_length=1024,
    dataset_text_field="",              # We use custom collator
    remove_unused_columns=False,
    
    # Misc
    report_to="tensorboard",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)

print(f"Training samples: {len(train_dataset)}")
print(f"Steps per epoch: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
print(f"Total steps: {trainer.args.max_steps if trainer.args.max_steps > 0 else 'auto'}")
print("\nStarting training...")

trainer.train()
print("Training complete!")

# %% [markdown]
# ## Cell 7: Loss Curves

# %%
# %load_ext tensorboard
# %tensorboard --logdir /content/tb_logs

# Alternative: plot from trainer log history
import matplotlib.pyplot as plt

log_history = trainer.state.log_history

train_losses = [(l["step"], l["loss"]) for l in log_history if "loss" in l]
eval_losses = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]

fig, ax = plt.subplots(figsize=(10, 5))
if train_losses:
    steps, losses = zip(*train_losses)
    ax.plot(steps, losses, label="Train Loss", color="#4A90D9", linewidth=2)
if eval_losses:
    steps, losses = zip(*eval_losses)
    ax.plot(steps, losses, label="Val Loss", color="#E74C3C", linewidth=2, linestyle="--")

ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Qwen2-VL-2B Vision Encoder LoRA Fine-Tuning")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/content/loss_curve.png", dpi=150)
plt.show()

print(f"Final train loss: {train_losses[-1][1]:.4f}" if train_losses else "No train loss")
print(f"Final val loss: {eval_losses[-1][1]:.4f}" if eval_losses else "No val loss")

# %% [markdown]
# ## Cell 8: Save LoRA Adapter to Google Drive

# %%
import os

# Save adapter locally first
LOCAL_ADAPTER = "/content/qwen2vl_gauge_lora"
model.save_pretrained(LOCAL_ADAPTER)
processor.save_pretrained(LOCAL_ADAPTER)

# Copy to Drive for persistence
os.makedirs(DRIVE_OUTPUT, exist_ok=True)
shutil.copytree(LOCAL_ADAPTER, DRIVE_OUTPUT, dirs_exist_ok=True)

# Also save loss curve
shutil.copy("/content/loss_curve.png", os.path.join(DRIVE_OUTPUT, "loss_curve.png"))

# Save training log
with open(os.path.join(DRIVE_OUTPUT, "training_log.json"), 'w') as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"Adapter saved to: {DRIVE_OUTPUT}")
adapter_size = sum(os.path.getsize(os.path.join(LOCAL_ADAPTER, f)) for f in os.listdir(LOCAL_ADAPTER) if f.endswith(('.bin', '.safetensors')))
print(f"Adapter size: {adapter_size / 1e6:.1f} MB")

# %% [markdown]
# ## Cell 9: Quick Sanity Check Inference

# %%
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load fresh base model for inference test
print("Loading base model for inference test...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load LoRA adapter
finetuned_model = PeftModel.from_pretrained(base_model, LOCAL_ADAPTER)
finetuned_model.eval()

test_processor = AutoProcessor.from_pretrained(MODEL_ID)

# Pick 5 random validation samples for sanity check
import random
random.seed(123)
test_samples = random.sample(val_convos, min(5, len(val_convos)))

print("\n" + "="*80)
print("SANITY CHECK: Fine-tuned Model Inference")
print("="*80)

for i, sample in enumerate(test_samples):
    user_msg = sample["messages"][0]
    expected = sample["messages"][1]["content"][0]["text"]
    
    # Get image path from user message
    img_path = None
    prompt_text = ""
    for content in user_msg["content"]:
        if content["type"] == "image":
            img_path = content["image"]
        elif content["type"] == "text":
            prompt_text = content["text"]
    
    # Build inference message (user only, no assistant)
    inference_msg = [{"role": "user", "content": user_msg["content"]}]
    text = test_processor.apply_chat_template(inference_msg, tokenize=False, add_generation_prompt=True)
    
    img = Image.open(img_path)
    inputs = test_processor(text=[text], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(finetuned_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = finetuned_model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    # Decode only generated tokens
    generated = test_processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    
    print(f"\n--- Sample {i+1} ---")
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Expected: {expected}")
    print(f"Generated: {generated}")
    match = "✅" if any(v in generated for v in ["SAFE", "UNSAFE"]) else "❌"
    print(f"Has verdict: {match}")

print("\n" + "="*80)
print("Done! Download adapter from Google Drive:")
print(f"  {DRIVE_OUTPUT}")
print("="*80)
