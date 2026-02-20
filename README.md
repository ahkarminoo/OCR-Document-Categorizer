# OCR Document Categorizer

## Backend Setup

1. Create and activate a virtual environment.
2. Install Python dependencies:

```bash
pip install -r requirement.txt
```

3. Install Tesseract OCR engine on your OS:
   - Windows: install from UB Mannheim package or official Tesseract builds.
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

4. Ensure `tesseract` is available in your system `PATH`.
5. Add your Gemini key in `.env`:

```env
GEMINI_API_KEY=your_key_here
TESS_LANG=eng
PADDLE_LANG=en
VISION_OCR_MODE=auto
# Optional: persist artifacts (cropped image, OCR text, JSON) per scan
OUTPUT_DIR=outputs
CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

6. Run API:

```bash
python main.py
```

### Deployment-ready endpoints

- `GET /health`: basic liveness check
- `GET /ready`: checks whether Gemini key is set, Tesseract is available, and output dir configured
- `POST /api/scan`: returns structured OCR + classification results
- `GET /api/scans/{scan_id}/{artifact}`: download saved artifacts (when `OUTPUT_DIR` is configured)
  - artifacts: `cropped.jpg`, `ocr_ready.png`, `ocr.txt`, `result.json`

### OCR engine notes

- The app now prefers `PaddleOCR` for better real-world photo accuracy.
- If PaddleOCR is unavailable, it falls back to Tesseract automatically.
- For very low-confidence text, it can also use Gemini vision OCR fallback.
- Set `VISION_OCR_MODE=always` to force Gemini vision OCR for handwritten pages.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend expects backend at `http://127.0.0.1:8000`.

## Local smoke test (no browser needed)

```bash
python smoke_test.py
```

## Docker (deployment-style local run)

1. Create an `.env` in the repo root containing at least:

```env
GEMINI_API_KEY=your_key_here
```

2. Run:

```bash
docker compose up --build
```

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs`
