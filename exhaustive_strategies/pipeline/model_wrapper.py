import time
from PIL import Image
from exhaustive_strategies.config import MODEL_ID, MAX_TOKENS, MIN_PIXELS, MAX_PIXELS

class Qwen2VLEngine:
    def __init__(self, model_id=None, dry_run=False):
        self.model_id = model_id or MODEL_ID
        self.model = self.processor = None
        self._loaded = False
        self.dry_run = dry_run
        self.call_count = 0

    def load(self):
        if self._loaded or self.dry_run:
            self._loaded = True
            return
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        print(f"Loading {self.model_id} ...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto")
        self._loaded = True
        if torch.cuda.is_available():
            print(f"  VRAM allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    def infer(self, prompt: str, images=None, max_tokens=MAX_TOKENS) -> dict:
        if not self._loaded: self.load()
        self.call_count += 1
        if self.dry_run: return {"text": "[DRY-RUN] PASS 75%", "elapsed_s": 0.01}

        import torch
        from qwen_vl_utils import process_vision_info
        content = []
        for img in (images or []):
            content.append({"type": "image", "image": img, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text_in], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(self.model.device)

        t0 = time.time()
        with torch.no_grad(): gen_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        elapsed = time.time() - t0

        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
        text_out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return {"text": text_out, "elapsed_s": round(elapsed, 2)}
