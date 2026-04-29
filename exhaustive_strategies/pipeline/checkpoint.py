import os, json, time, hashlib
class Checkpoint:
    def __init__(self, out_dir, config=None):
        self.path = os.path.join(out_dir, ".checkpoint.json")
        self.state = self._load()
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f: s = json.load(f)
            print(f"[Checkpoint] Resuming \u2014 {len(s.get('done',{}))} images done")
            return s
        return {"done": {}, "t0": time.time()}
    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f: json.dump(self.state, f, indent=2, default=str)
        os.replace(tmp, self.path)
    def is_done(self, img_key, strategy=None):
        d = self.state["done"].get(img_key, {})
        return strategy in d if strategy else bool(d)
    def mark(self, img_key, strategy, result):
        self.state["done"].setdefault(img_key, {})[strategy] = result
        self.save()
