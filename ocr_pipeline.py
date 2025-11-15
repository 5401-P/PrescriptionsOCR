# ocr_pipeline.py
import os
import re
import csv
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from rapidfuzz import fuzz
from paddleocr import PaddleOCR

# -------------------------
# Load PaddleOCR
# -------------------------
ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    det_db_score_mode="slow",
    rec_algorithm="SVTR_LCNet"
)

# -------------------------
# Medications list
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MEDS_CSV = os.path.join(BASE_DIR, "meds.csv")

def load_meds(path=MEDS_CSV):
    meds = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("drug_name"):
                    meds.append(r["drug_name"].strip())
    return meds

MEDS = load_meds()

# -------------------------
# Image preprocessing (PIL + NumPy)
# -------------------------
def detect_handwriting_block(pil_img, pad=10):
    arr = np.array(pil_img.convert("L"))
    ys, xs = np.where(arr < 250)  # non-white pixels
    if len(xs) == 0 or len(ys) == 0:
        return pil_img
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(arr.shape[1], x_max + pad)
    y_max = min(arr.shape[0], y_max + pad)
    return pil_img.crop((x_min, y_min, x_max, y_max))

def enhance_handwriting(pil_img):
    img_gray = pil_img.convert("L")
    img_eq = ImageOps.autocontrast(img_gray)
    img_blur = img_eq.filter(ImageFilter.MedianFilter(size=3))
    return img_blur.convert("RGB")

# -------------------------
# OCR using PaddleOCR
# -------------------------
def run_ocr(pil_img):
    img = np.array(pil_img)
    results = ocr.ocr(img, cls=True)
    lines = []
    for block in results:
        for line in block:
            text = line[1][0]
            if text.strip():
                lines.append(text)
    return "\n".join(lines)

# -------------------------
# Medication matching
# -------------------------
def clean_for_fuzzy(text):
    return re.sub(r"[^A-Za-z]", "", text).lower()

def match_medication_ultra(candidate, meds=MEDS):
    c = clean_for_fuzzy(candidate)
    if not c:
        return None, 0
    best, best_score = None, 0
    for med in meds:
        m = med.lower()
        score = max(fuzz.partial_ratio(c, m), fuzz.token_sort_ratio(c, m))
        if score > best_score:
            best_score = score
            best = med
    return (best, best_score) if best_score >= 45 else (None, 0)

# -------------------------
# Dosage + Frequency extraction
# -------------------------
DOSAGE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(mg|mcg|g|ml|iu|units|tab|tablet|capsule))\b",
    re.IGNORECASE,
)

FREQ_RE = re.compile(
    r"\b(od|bd|tid|qid|qhs|hs|prn|daily|once daily|twice daily|three times daily)\b",
    re.IGNORECASE,
)

def extract_dosage(text):
    m = DOSAGE_RE.search(text)
    return m.group(1) if m else None

def extract_frequency(text):
    m = FREQ_RE.search(text)
    return m.group(1) if m else None

# -------------------------
# Structured output
# -------------------------
def extract_structured_from_text(raw_text):
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    results = []
    for line in lines:
        med, score = match_medication_ultra(line)
        if not med:
            continue
        dose = extract_dosage(line)
        freq = extract_frequency(line)
        results.append({
            "raw": line,
            "med_match": med,
            "match_score": score,
            "dose": dose,
            "frequency": freq,
        })
    return results

# -------------------------
# Line segmentation + OCR runner
# -------------------------
def _segment_lines_projection(pil_img, min_height=10, merge_gap=4):
    arr = np.array(pil_img.convert("L"))
    bw = (arr < 250).astype(np.uint8) * 255
    proj = np.sum(bw, axis=1)
    lines, in_line, start, last_end = [], False, 0, -999
    for i, v in enumerate(proj):
        if v > 0 and not in_line:
            in_line = True
            start = i
        elif v == 0 and in_line:
            in_line = False
            end = i
            if last_end >= 0 and start - last_end <= merge_gap:
                prev_start, prev_end = lines[-1][0], lines[-1][1]
                lines[-1] = (prev_start, end)
                last_end = end
            else:
                if end - start >= min_height:
                    lines.append((start, end))
                    last_end = end
    if in_line:
        end = len(proj)
        if end - start >= min_height:
            if last_end >= 0 and start - last_end <= merge_gap and lines:
                prev_start, prev_end = lines[-1][0], lines[-1][1]
                lines[-1] = (prev_start, end)
            else:
                lines.append((start, end))
    h, w = arr.shape
    line_imgs = [pil_img.crop((0, max(0, s-2), w, min(h, e+2))).convert("RGB") for (s, e) in lines]
    return line_imgs

def run_ocr_lines(pil_img, max_len=128, fallback_full_if_empty=True):
    try:
        line_images = _segment_lines_projection(pil_img)
    except Exception:
        line_images = []

    results = []
    for li in line_images:
        try:
            t = run_ocr(li)
            if t and t.strip():
                results.append(t.strip())
        except Exception:
            continue
    if not results and fallback_full_if_empty:
        try:
            full = run_ocr(pil_img)
            if full and full.strip():
                results.append(full.strip())
        except Exception:
            pass
    return "\n".join(results)
