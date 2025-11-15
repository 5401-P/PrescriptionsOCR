# ocr_pipeline.py
import os
import re
import csv
import cv2
import numpy as np
from PIL import Image
from rapidfuzz import fuzz
from paddleocr import PaddleOCR

# -------------------------
# Load PaddleOCR
# -------------------------
ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    det_db_score_mode="slow",
    rec_algorithm="SVTR_LCNet"  # best for handwriting
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
# Image preprocessing
# -------------------------
def detect_handwriting_block(pil_img, pad=10):
    img_gray = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(img_gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return pil_img

    x_min = min(cv2.boundingRect(c)[0] for c in contours)
    y_min = min(cv2.boundingRect(c)[1] for c in contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

    h, w = img_gray.shape
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)

    cropped = img_gray[y_min:y_max, x_min:x_max]
    return Image.fromarray(cropped)

def enhance_handwriting(pil_img):
    img_gray = np.array(pil_img.convert("L"))

    # Slight blur
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # CLAHE for faint handwriting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)

    return Image.fromarray(img_gray)

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

    best = None
    best_score = 0

    for med in meds:
        m = med.lower()
        score = max(
            fuzz.partial_ratio(c, m),
            fuzz.token_sort_ratio(c, m)
        )
        if score > best_score:
            best_score = score
            best = med

    return (best, best_score) if best_score >= 45 else (None, 0)

# -------------------------
# Dosage + Frequency
# -------------------------
DOSAGE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?\s*(mg|mcg|g|ml|iu|units|tab|tablet|capsule))\b",
    re.IGNORECASE,
)

FREQ_RE = re.compile(
    r"\b("
    r"od|bd|tid|qid|qhs|hs|prn|daily|once daily|twice daily|three times daily"
    r")\b",
    re.IGNORECASE,
)

def extract_dosage(text):
    m = DOSAGE_RE.search(text)
    return m.group(1) if m else None

def extract_frequency(text):
    m = FREQ_RE.search(text)
    return m.group(1) if m else None

# -------------------------
# Build structured output
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
# Line segmentation + line-wise OCR runner (helper)
# -------------------------
def _segment_lines_projection(pil_img, min_height=10, merge_gap=4):
    """
    Return list of PIL line images by horizontal projection.
    Conservative segmentation: keeps small gaps merged.
    """
    img = np.array(pil_img.convert("L"))
    # Binarize (invert text white)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    proj = np.sum(bw, axis=1)
    lines = []
    in_line = False
    start = 0
    last_end = -999

    for i, v in enumerate(proj):
        if v > 0 and not in_line:
            in_line = True
            start = i
        elif v == 0 and in_line:
            in_line = False
            end = i
            # merge with previous if gap small
            if last_end >= 0 and start - last_end <= merge_gap:
                # extend previous line by replacing last slice
                prev_start, prev_end = lines[-1][0], lines[-1][1]
                lines[-1] = (prev_start, end)
                last_end = end
            else:
                if end - start >= min_height:
                    lines.append((start, end))
                    last_end = end

    # tail
    if in_line:
        end = len(proj)
        if end - start >= min_height:
            if last_end >= 0 and start - last_end <= merge_gap and lines:
                prev_start, prev_end = lines[-1][0], lines[-1][1]
                lines[-1] = (prev_start, end)
            else:
                lines.append((start, end))

    # build PIL line images
    line_imgs = []
    h, w = img.shape
    for (s, e) in lines:
        line_crop = img[max(0, s-2):min(h, e+2), :]  # small vertical padding
        line_imgs.append(Image.fromarray(line_crop).convert("RGB"))

    return line_imgs


def run_ocr_lines(pil_img, max_len=128, fallback_full_if_empty=True):
    """
    Segment the preprocessed handwriting region into lines and run OCR on each.
    Returns joined string with newline-separated lines.
    """
    # first try segmentation
    try:
        line_images = _segment_lines_projection(pil_img)
    except Exception:
        line_images = []

    results = []
    for li in line_images:
        try:
            t = run_ocr(li, max_len=max_len)
            if t and t.strip():
                results.append(t.strip())
        except Exception:
            continue

    # fallback to whole-image OCR if segmentation produced nothing
    if not results and fallback_full_if_empty:
        try:
            full = run_ocr(pil_img, max_len=max_len)
            if full and full.strip():
                results.append(full.strip())
        except Exception:
            pass

    return "\n".join(results)
