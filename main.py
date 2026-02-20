import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import pytesseract
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import uvicorn
from ai_engine import (
    _HEURISTIC_CONFIDENCE_THRESHOLD,
    _heuristic_classify,
    categorize_document,
    vision_ocr,
)
from ocr_engine import extract_text
from processor import clean_document
from storage import (
    ensure_scan_dir,
    get_output_base_dir,
    resolve_artifact_path,
    write_bytes,
    write_json,
    write_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Document Scanner API")


@app.on_event("startup")
async def _warmup():
    """Load OCR models in the background so the server starts immediately.
    The /health endpoint responds right away; models load behind the scenes.
    """
    import asyncio

    async def _load_models():
        loop = asyncio.get_event_loop()
        from ocr_engine import _get_paddle, _get_easyocr
        await loop.run_in_executor(None, _get_paddle)
        logger.info("PaddleOCR ready.")
        await loop.run_in_executor(None, _get_easyocr)
        logger.info("EasyOCR ready.")
        logger.info("All OCR engines warmed up.")

    asyncio.create_task(_load_models())

MIN_WORDS_FOR_AI       = 6
GOOD_OCR_WORDS         = 20
GOOD_OCR_CONFIDENCE    = 45
AI_TIMEOUT_SECONDS     = 8
HANDWRITING_FALLBACK_CONF = 60
VISION_OCR_MODE = (os.getenv("VISION_OCR_MODE") or "auto").strip().lower()

_cors_origins_env = (os.getenv("CORS_ALLOW_ORIGINS") or "*").strip()
_cors_origins = (
    ["*"]
    if "*" in _cors_origins_env
    else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _text_quality_score(text: str) -> float:
    txt = str(text or "").strip()
    if not txt:
        return 0.0
    words = [w for w in txt.split() if w]
    alpha = sum(ch.isalpha() for ch in txt)
    return float(len(words)) + alpha / 20.0


def _text_looks_garbled(text: str) -> bool:
    """Detect garbled PaddleOCR output on handwritten images.

    Uses only alphabetic word tokens so that legitimate OCR artefacts
    (bullet points read as ¢/©/«, brackets, currency symbols) on printed
    documents do not cause false positives.
    """
    import re as _re
    txt = str(text or "").strip()
    if not txt or len(txt) < 10:
        return False

    # High pipe-character density is a strong PaddleOCR-on-handwriting signal.
    if txt.count("|") / len(txt) > 0.02:
        return True

    # Work only on alphabetic word tokens — ignore symbols, digits, URLs.
    alpha_words = _re.findall(r"[a-zA-Z]{2,}", txt)
    all_tokens  = txt.split()
    if not all_tokens:
        return True

    # Too many non-alphabetic tokens relative to total (noise-heavy output).
    if len(alpha_words) < len(all_tokens) * 0.35:
        return True

    # Very short alphabetic words dominate — typical of garbled character runs.
    short = sum(1 for w in alpha_words if len(w) <= 2)
    if len(alpha_words) > 6 and short / len(alpha_words) > 0.55:
        return True

    return False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    api_key_set = bool((os.getenv("GEMINI_API_KEY") or "").strip())
    try:
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        tesseract_ok = False
    return {
        "ready": api_key_set and tesseract_ok,
        "gemini_api_key_configured": api_key_set,
        "tesseract_available": tesseract_ok,
        "output_dir_configured": get_output_base_dir() is not None,
    }


@app.post("/api/scan")
async def scan_document(
    file: UploadFile = File(...),
    force_vision: bool = Form(False),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    scan_id = uuid.uuid4().hex
    started = time.perf_counter()

    try:
        processed = clean_document(contents)
        cropped_image_bytes = processed["cropped_image_bytes"]
        ocr_image_bytes     = processed["ocr_ready_image_bytes"]
        document_detected   = processed["document_detected"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document preprocessing failed: {e}") from e

    preprocess_ms = round((time.perf_counter() - started) * 1000, 2)

    try:
        best_ocr   = None
        best_source = "unknown"
        for source_name, source_bytes in [("ocr_ready", ocr_image_bytes), ("original", contents)]:
            result = extract_text(source_bytes, fast_mode=True)
            if best_ocr is None or result["word_count"] > best_ocr["word_count"]:
                best_ocr    = result
                best_source = source_name
            if result["word_count"] >= GOOD_OCR_WORDS and result["avg_confidence"] >= GOOD_OCR_CONFIDENCE:
                break
        ocr_result     = best_ocr or {"text": "", "word_count": 0, "avg_confidence": 0.0}
        extracted_text = (ocr_result.get("text") or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}") from e

    ocr_engine_used = ocr_result.get("engine", "tesseract")
    text_garbled  = _text_looks_garbled(extracted_text)
    too_sparse    = ocr_result["word_count"] < MIN_WORDS_FOR_AI
    very_low_conf = ocr_result["avg_confidence"] < HANDWRITING_FALLBACK_CONF

    # If local heuristics can already classify the document with confidence,
    # Vision OCR is unnecessary — the type and headings are already known,
    # so we skip the Gemini call entirely.
    _, _, heuristic_conf = _heuristic_classify(extracted_text)
    already_classified = heuristic_conf >= _HEURISTIC_CONFIDENCE_THRESHOLD
    logger.info(
        "OCR summary — engine=%s words=%d conf=%.1f garbled=%s "
        "heuristic_conf=%.2f already_classified=%s force_vision=%s",
        ocr_engine_used, ocr_result["word_count"], ocr_result["avg_confidence"],
        text_garbled, heuristic_conf, already_classified, force_vision,
    )

    # Invoke Vision OCR only when text quality is poor AND the document cannot
    # already be classified locally.  The already_classified guard means known
    # document types (Resume, Invoice …) never pay the Vision OCR cost even if
    # their OCR confidence is low (e.g. fancy designed templates).
    #
    # force_vision=True means the user explicitly clicked "Improve with AI Vision"
    # — always honour it unconditionally.  Accidental clicks are prevented by the
    # frontend clearing the result the moment a new file is selected.
    should_try_vision = (
        VISION_OCR_MODE == "always"
        or force_vision
        or (text_garbled and not already_classified)
        or (very_low_conf and not already_classified)
    )
    vision_ocr_failed = False
    if should_try_vision:
        try:
            vision_text = vision_ocr(cropped_image_bytes, mime_type="image/jpeg")
            if vision_text:
                vision_word_count = len([w for w in vision_text.split() if w])
                if vision_word_count >= 3:
                    use_vision = (
                        VISION_OCR_MODE == "always"
                        or _text_looks_garbled(extracted_text)
                        or ocr_result["avg_confidence"] < HANDWRITING_FALLBACK_CONF
                        or _text_quality_score(vision_text) >= _text_quality_score(extracted_text) * 0.75
                    )
                    if use_vision:
                        extracted_text  = vision_text
                        ocr_result      = {
                            "text": vision_text,
                            "word_count": vision_word_count,
                            "avg_confidence": max(ocr_result["avg_confidence"], HANDWRITING_FALLBACK_CONF),
                        }
                        ocr_engine_used = "gemini_vision"
        except Exception as e:
            vision_ocr_failed = True
            logger.warning("vision_ocr failed: %s: %s", type(e).__name__, str(e)[:120])

    ocr_ms = round((time.perf_counter() - started) * 1000 - preprocess_ms, 2)

    # If Vision OCR was needed but failed (e.g. quota), the remaining text is
    # unreliable Tesseract output — skip AI categorization and return Other so
    # we don't waste a second API call on garbage input.
    # Exception: if the user explicitly forced AI, try categorization anyway.
    ocr_text_reliable = force_vision or not (vision_ocr_failed and very_low_conf)

    try:
        if ocr_result["word_count"] < MIN_WORDS_FOR_AI:
            preview = extracted_text.strip()
            document_data = {
                "category": "Other",
                "subcategory": "LowTextContent",
                "editable_text": preview,
                "summary": preview[:180] if preview else "Very little readable text found.",
                "key_information": [preview[:80]] if preview else [],
            }
        elif not ocr_text_reliable:
            preview = extracted_text.strip()
            document_data = {
                "category": "Other",
                "subcategory": "UnreadableContent",
                "editable_text": preview,
                "summary": "Could not read document clearly. Try a well-lit, straight-on photo.",
                "key_information": [],
                "_classification_method": "fallback_quota",
            }
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(categorize_document, extracted_text)
                try:
                    document_data = future.result(timeout=AI_TIMEOUT_SECONDS)
                except FuturesTimeoutError:
                    preview = extracted_text[:220]
                    document_data = {
                        "category": "Other",
                        "subcategory": "TimeoutFallback",
                        "editable_text": extracted_text,
                        "summary": preview + ("..." if len(extracted_text) > 220 else ""),
                        "key_information": [preview[:80]] if preview else [],
                    }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Categorization failed: {e}") from e

    categorize_ms = round((time.perf_counter() - started) * 1000 - preprocess_ms - ocr_ms, 2)
    total_ms      = round((time.perf_counter() - started) * 1000, 2)

    scan_dir = ensure_scan_dir(scan_id)
    if scan_dir is not None:
        try:
            write_bytes(scan_dir / "cropped.jpg",   cropped_image_bytes)
            write_bytes(scan_dir / "ocr_ready.png", ocr_image_bytes)
            write_text(scan_dir  / "ocr.txt",       extracted_text)
            write_json(scan_dir  / "result.json",   document_data)
        except Exception:
            pass

    return {
        "scan_id":  scan_id,
        "filename": file.filename,
        "results":  document_data,
        "artifacts": {
            "cropped_image":  f"/api/scans/{scan_id}/cropped.jpg"    if scan_dir else None,
            "ocr_ready_image": f"/api/scans/{scan_id}/ocr_ready.png" if scan_dir else None,
            "ocr_text":        f"/api/scans/{scan_id}/ocr.txt"       if scan_dir else None,
            "result_json":     f"/api/scans/{scan_id}/result.json"   if scan_dir else None,
        },
        "meta": {
            "document_detected":   document_detected,
            "ocr_word_count":      ocr_result["word_count"],
            "ocr_avg_confidence":  ocr_result["avg_confidence"],
            "ocr_source":          best_source,
            "ocr_engine":          ocr_engine_used,
            "classification_method": document_data.get("_classification_method", "unknown"),
            "used_ai_fallback":    document_data.get("_classification_method") != "ai",
            "durations_ms": {
                "preprocess": preprocess_ms,
                "ocr":        ocr_ms,
                "categorize": categorize_ms,
                "total":      total_ms,
            },
        },
    }


@app.get("/api/scans/{scan_id}/{artifact_name}")
async def get_scan_artifact(scan_id: str, artifact_name: str):
    allowed = {"cropped.jpg", "ocr_ready.png", "ocr.txt", "result.json"}
    if artifact_name not in allowed:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = resolve_artifact_path(scan_id, artifact_name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    media_types = {
        ".jpg":  "image/jpeg",
        ".png":  "image/png",
        ".txt":  "text/plain",
        ".json": "application/json",
    }
    ext = os.path.splitext(artifact_name)[1]
    return FileResponse(str(path), media_type=media_types.get(ext), filename=artifact_name)


if __name__ == "__main__":
    dev_mode = os.getenv("DEV_MODE", "true").lower() == "true"
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=dev_mode)
