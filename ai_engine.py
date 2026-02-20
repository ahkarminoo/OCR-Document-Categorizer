"""
AI engine — Gemini is used ONLY when local heuristics cannot confidently
categorise or summarise the document.  For common, well-structured documents
(resumes, invoices, receipts …) everything is done locally.
"""
import json
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_MODEL = "models/gemini-2.5-flash-lite"

ALLOWED_CATEGORIES = ["Invoice", "Resume", "Legal", "Medical", "Academic", "Receipt", "Other"]
MAX_OCR_CHARS_FOR_AI         = 3200
MAX_VISION_OCR_OUTPUT_TOKENS = 1400

# Heuristic confidence required to skip the AI categorisation call.
# 0.0–1.0 — lower means AI is called more often.
_HEURISTIC_CONFIDENCE_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _call_with_retry(fn, retries: int = 0, wait: float = 4.0):
    """Call fn(); on 429 wait and retry up to `retries` times.

    Default retries=0 — fail fast on quota errors rather than burning the
    remaining quota with repeated attempts.  Callers can pass retries=1
    for operations where a single retry is worth the cost.
    """
    import time
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < retries:
                time.sleep(wait * (attempt + 1))
                continue
            raise


# ---------------------------------------------------------------------------
# Local (no-AI) helpers
# ---------------------------------------------------------------------------

def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate(text: str) -> str:
    return _compact(text)[:MAX_OCR_CHARS_FOR_AI]


_CATEGORY_RULES = [
    ("Invoice",  ["invoice", "bill to", "due date", "subtotal", "amount due", "tax",
                  "payment terms", "total amount", "invoice no", "billing"]),
    ("Receipt",  ["receipt", "cashier", "change due", "total paid", "thank you for",
                  "amount paid", "transaction"]),
    ("Resume",   ["resume", "curriculum vitae", "work experience", "experience", "education",
                  "technical skills", "skills", "projects", "linkedin", "github", "references"]),
    ("Legal",    ["agreement", "contract", "terms and conditions", "party", "clause",
                  "whereas", "hereinafter"]),
    ("Medical",  ["patient", "diagnosis", "prescription", "dosage", "hospital", "physician",
                  "symptoms"]),
    ("Academic", ["university", "college", "course", "exam", "assignment", "grade",
                  "student id", "syllabus"]),
]


def _heuristic_classify(text: str) -> tuple[str, str, float]:
    """Return (category, subcategory, confidence 0-1) using keyword scoring."""
    low    = text.lower()
    best_cat, best_sub, best_conf = "Other", "GeneralDocument", 0.0

    _subs = {
        "Invoice": "BillingDocument", "Receipt": "PurchaseProof",
        "Resume":  "CandidateProfile", "Legal": "Agreement",
        "Medical": "ClinicalRecord",   "Academic": "StudyMaterial",
    }
    for cat, keywords in _CATEGORY_RULES:
        hits = sum(1 for kw in keywords if kw in low)
        conf = hits / len(keywords)
        if conf > best_conf:
            best_conf = conf
            best_cat  = cat
            best_sub  = _subs[cat]

    return best_cat, best_sub, best_conf


def _extract_headings(text: str) -> list[str]:
    """
    Pull explicit section headings from the text without any AI.
    Looks for:
      - ALL-CAPS short lines  (e.g.  EDUCATION, EXPERIENCE)
      - Lines ending with ':'  (e.g.  Skills:, Date:)
      - Short title-cased lines that don't end with punctuation
    """
    headings = []
    seen     = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 80:
            continue
        # ALL-CAPS heading
        if line.isupper() and 2 < len(line) < 60:
            h = line.title()
        # Ends with colon
        elif line.endswith(":") and len(line) < 60:
            h = line.rstrip(":")
        # Short title-case line with no trailing period (likely a heading)
        elif (len(line.split()) <= 6 and line[0].isupper()
              and not line[-1] in ".!?,;"):
            h = line
        else:
            continue
        key = h.lower()
        if key not in seen:
            seen.add(key)
            headings.append(h)
    return headings[:8]


def _local_summary(text: str, max_chars: int = 200) -> str:
    """Build a plain-text summary from the first sentences — no AI."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary   = ""
    for s in sentences:
        if len(summary) + len(s) > max_chars:
            break
        summary = (summary + " " + s).strip()
    return summary or text[:max_chars]


def _extract_key_info(text: str) -> list:
    values = []
    for pattern in [
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\$\s?\d+(?:\.\d{2})?\b",
        r"\b(?:INR|USD|EUR)\s?\d+(?:\.\d{2})?\b",
        r"\b\d{5,}\b",
    ]:
        values.extend(re.findall(pattern, text))
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:5]


# ---------------------------------------------------------------------------
# Gemini helpers (only called when heuristics are not confident enough)
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON in AI response.")
        return json.loads(match.group())


def _normalize_category(value: str) -> str:
    for cat in ALLOWED_CATEGORIES:
        if str(value or "").strip().lower() == cat.lower():
            return cat
    return "Other"


@lru_cache(maxsize=128)
def _ai_categorize(ocr_text: str) -> str:
    client = _get_client()
    prompt = (
        "Classify the following OCR text.\n"
        f"Allowed category values: {', '.join(ALLOWED_CATEGORIES)}.\n"
        "Provide a short dynamic subcategory and extract the main topic headings "
        "found in the document as key_information.\n"
        "Return valid JSON only with keys: "
        "category, subcategory, summary, key_information.\n"
        "summary <= 2 sentences. key_information max 8 items — prefer actual "
        "section headings or topics found in the text.\n"
        f"OCR_TEXT: {ocr_text}"
    )
    response = _call_with_retry(lambda: client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=280,
            response_mime_type="application/json",
        ),
    ))
    return response.text.strip()


@lru_cache(maxsize=128)
def _ai_categorize_retry(ocr_text: str) -> str:
    client = _get_client()
    prompt = (
        "Return ONLY JSON. No markdown.\n"
        f"Allowed category: {', '.join(ALLOWED_CATEGORIES)}.\n"
        'Schema: {"category":"...","subcategory":"...","summary":"...","key_information":["..."]}\n'
        "key_information should list the main topic headings found in the text (max 8).\n"
        f"OCR_TEXT: {ocr_text}"
    )
    response = _call_with_retry(lambda: client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=280),
    ))
    return response.text.strip()


# ---------------------------------------------------------------------------
# Gemini Vision OCR (handwriting fallback only)
# ---------------------------------------------------------------------------

def vision_ocr(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Transcribe an image — used only for handwriting / low-confidence cases."""
    if not (os.getenv("GEMINI_API_KEY") or "").strip():
        return ""
    client = _get_client()
    prompt = (
        "You are an expert OCR engine specialising in handwritten text.\n"
        "Carefully read every word, including cursive or messy handwriting.\n"
        "Preserve original line structure. If a word is ambiguous, best-guess it.\n"
        "Return ONLY the transcribed text — no commentary."
    )
    response = _call_with_retry(lambda: client.models.generate_content(
        model=_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=MAX_VISION_OCR_OUTPUT_TOKENS,
        ),
    ))
    return (response.text or "").strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def categorize_document(ocr_text: str) -> dict:
    """
    Categorise OCR text into a heading + subheading with topic headings.

    Strategy (cheapest first):
      1. Extract structural headings directly from the text (no AI).
      2. Classify document type with keyword scoring (no AI).
      3. If heuristic confidence >= threshold → return without any API call.
      4. Otherwise call Gemini for classification + heading extraction.
    """
    trimmed = _compact(ocr_text)
    if not trimmed:
        return {
            "category": "Other", "subcategory": "Uncategorized",
            "editable_text": "", "summary": "No readable text found.",
            "key_information": [],
        }

    # Step 1 & 2 — local, zero API cost
    headings = _extract_headings(trimmed)
    category, subcategory, confidence = _heuristic_classify(trimmed)

    if confidence >= _HEURISTIC_CONFIDENCE_THRESHOLD:
        # Good enough — skip AI entirely
        key_info = headings if headings else _extract_key_info(trimmed)
        return {
            "category":      category,
            "subcategory":   subcategory,
            "editable_text": trimmed,
            "summary":       _local_summary(trimmed),
            "key_information": key_info,
            "_classification_method": "heuristic",
        }

    # Step 3 — call Gemini only for ambiguous / unstructured documents
    ai_text = _truncate(trimmed)
    try:
        raw = _ai_categorize(ai_text)
        try:
            parsed = _extract_json(raw)
        except Exception:
            raw    = _ai_categorize_retry(ai_text[:1200])
            parsed = _extract_json(raw)

        ai_category    = _normalize_category(parsed.get("category", "Other"))
        ai_subcategory = (parsed.get("subcategory") or subcategory).strip() or subcategory
        ai_summary     = (parsed.get("summary") or _local_summary(trimmed)).strip()
        ai_key_info    = parsed.get("key_information")
        if not isinstance(ai_key_info, list):
            ai_key_info = headings or _extract_key_info(trimmed)

        return {
            "category":        ai_category,
            "subcategory":     ai_subcategory,
            "editable_text":   trimmed,
            "summary":         ai_summary,
            "key_information": ai_key_info,
            "_classification_method": "ai",
        }

    except Exception:
        key_info = headings if headings else _extract_key_info(trimmed)
        return {
            "category":        category,
            "subcategory":     subcategory,
            "editable_text":   trimmed,
            "summary":         _local_summary(trimmed),
            "key_information": key_info,
            "_classification_method": "heuristic",
        }
