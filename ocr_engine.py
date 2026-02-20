import os
import re
from functools import lru_cache

import cv2
import numpy as np
import pytesseract


def _ocr_pass(image, config: str) -> dict:
    text = pytesseract.image_to_string(image, config=config).strip()
    data = pytesseract.image_to_data(image, config=config,
                                     output_type=pytesseract.Output.DICT)
    confs = [float(c) for c in data.get("conf", [])
             if c is not None and str(c).lstrip("-").replace(".", "").isdigit() and float(c) >= 0]
    avg_conf   = round(sum(confs) / len(confs), 2) if confs else 0.0
    word_count = len([w for w in (text or "").split() if w])
    return {"text": text, "avg_confidence": avg_conf,
            "word_count": word_count, "score": avg_conf + word_count * 1.8}


def _postprocess(text: str) -> str:
    t = re.sub(r"[^\S\r\n]+", " ", str(text or ""))
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.replace("\x0c", "").strip()


@lru_cache(maxsize=1)
def _get_paddle():
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return None
    lang = (os.getenv("PADDLE_LANG") or "en").strip() or "en"
    try:
        return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_easyocr():
    """Load EasyOCR reader (downloaded once, cached in memory)."""
    try:
        import easyocr
        return easyocr.Reader(["en"], verbose=False)
    except Exception:
        return None


def _easyocr_extract(image_bgr) -> dict | None:
    """Run EasyOCR — works well on handwriting, fully local, no API needed."""
    reader = _get_easyocr()
    if reader is None:
        return None
    try:
        results = reader.readtext(image_bgr, detail=1, paragraph=False)
    except Exception:
        return None
    if not results:
        return None
    lines, confs = [], []
    for (_bbox, word, conf) in results:
        word = str(word or "").strip()
        if word:
            lines.append(word)
            confs.append(float(conf) * 100.0)
    if not lines:
        return None
    text       = _postprocess(" ".join(lines))
    word_count = len([w for w in text.split() if w])
    avg_conf   = round(sum(confs) / len(confs), 2) if confs else 0.0
    return {"text": text, "avg_confidence": avg_conf,
            "word_count": word_count, "engine": "easyocr",
            "score": avg_conf + word_count * 1.8}


def _paddle_extract(image_bgr) -> dict | None:
    ocr = _get_paddle()
    if ocr is None:
        return None
    lines, confs = [], []
    for block in ocr.ocr(image_bgr, cls=True) or []:
        for item in block or []:
            if len(item) < 2 or not item[1] or len(item[1]) < 2:
                continue
            word = str(item[1][0] or "").strip()
            if not word:
                continue
            score = float(item[1][1]) if item[1][1] is not None else 0.0
            lines.append(word)
            if score >= 0:
                confs.append(score * 100.0)
    if not lines:
        return None
    text       = _postprocess("\n".join(lines))
    word_count = len([w for w in text.split() if w])
    avg_conf   = round(sum(confs) / len(confs), 2) if confs else 0.0
    return {"text": text, "avg_confidence": avg_conf,
            "word_count": word_count, "engine": "paddleocr",
            "score": avg_conf + word_count * 1.8}


def extract_text(image_bytes: bytes, fast_mode: bool = True) -> dict:
    """Return OCR text with confidence and engine metadata."""
    nparr    = np.frombuffer(image_bytes, np.uint8)
    gray     = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode OCR image bytes.")
    color    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    upscaled = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    if color is not None:
        result = _paddle_extract(color)
        if result and result["word_count"] >= 4:
            return {k: result[k] for k in ("text", "avg_confidence", "word_count", "engine")}

    # EasyOCR — local, no API, better than Tesseract on handwriting.
    # Only used when PaddleOCR returns nothing (likely a handwritten page).
    easy = _easyocr_extract(color if color is not None else
                            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    if easy and easy["word_count"] >= 4:
        return {k: easy[k] for k in ("text", "avg_confidence", "word_count", "engine")}

    lang = os.getenv("TESS_LANG", "eng").strip() or "eng"
    if fast_mode:
        candidates = [
            _ocr_pass(upscaled, f"-l {lang} --oem 3 --psm 6"),
            _ocr_pass(gray,     f"-l {lang} --oem 3 --psm 6"),
        ]
    else:
        inverted   = cv2.bitwise_not(gray)
        candidates = [
            _ocr_pass(img, f"-l {lang} {cfg}")
            for img in (gray, upscaled, inverted)
            for cfg in ("--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 3 --psm 11")
        ]

    best      = max(candidates, key=lambda r: r["score"])
    best_text = _postprocess(best["text"])
    return {
        "text":           best_text,
        "avg_confidence": best["avg_confidence"],
        "word_count":     len([w for w in best_text.split() if w]),
        "engine":         "tesseract",
    }
