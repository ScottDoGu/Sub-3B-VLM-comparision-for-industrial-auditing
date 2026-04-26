import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
try:
    import cv2 as _cv2
    HAS_CV2 = True
except ImportError:
    _cv2 = None
    HAS_CV2 = False

def apply_clahe(img, clip=2.0, grid=8):
    if HAS_CV2:
        arr = np.array(img.convert("RGB"))
        lab = _cv2.cvtColor(arr, _cv2.COLOR_RGB2LAB)
        clahe = _cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(_cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB))
    return ImageOps.autocontrast(img.convert("RGB"), cutoff=1)

def sharpen(img, factor=1.5): return ImageEnhance.Sharpness(img).enhance(factor)
def denoise(img): return img.filter(ImageFilter.MedianFilter(3))
def enhance(img, do_clahe=True, sharpness=1.3, do_denoise=False):
    if do_denoise: img = denoise(img)
    if do_clahe: img = apply_clahe(img)
    if sharpness > 1.0: img = sharpen(img, sharpness)
    return img

def edge_density(img):
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    return float(np.array(edges).mean()) / 255.0

def extract_gauge_roi(img, pad_pct=0.1): return _center_crop(img, 0.7)
def extract_pipe_roi(img, pad=20): return _center_crop(img, 0.8)
def _center_crop(img, ratio):
    W, H = img.size
    nw, nh = int(W * ratio), int(H * ratio)
    left, top = (W - nw) // 2, (H - nh) // 2
    return img.crop((left, top, left + nw, top + nh))

def top_patches(img, grid=3, k=4):
    W, H = img.size
    pw, ph = W // grid, H // grid
    scored = []
    for r in range(grid):
        for c in range(grid):
            patch = img.crop((c*pw, r*ph, (c+1)*pw, (r+1)*ph))
            scored.append((edge_density(patch), patch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [img] + [p for _, p in scored[:k]]

def draw_grid(img, rows=10, cols=10, color="lime"):
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    W, H = out.size
    cw, ch = W / cols, H / rows
    for r in range(1, rows): draw.line([(0, int(r*ch)), (W, int(r*ch))], fill=color, width=1)
    for c in range(1, cols): draw.line([(int(c*cw), 0), (int(c*cw), H)], fill=color, width=1)
    for r in range(rows):
        for c in range(cols):
            draw.text((int(c*cw)+2, int(r*ch)+1), f"{chr(65+r)}{c+1}", fill=color)
    return Image.blend(img.convert("RGB"), out, 0.35)
