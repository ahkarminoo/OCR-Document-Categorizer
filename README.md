# OCR Document Categorizer

Production-ready OCR pipeline that:
- crops document pages from wide-angle/noisy photos,
- extracts editable text,
- categorizes content into topic headings,
- uses AI only when local confidence is low.

## Key Features

- **Document crop + perspective correction** using OpenCV.
- **Editable OCR output** from local OCR engines.
- **Category + heading extraction** for common document types.
- **Local-first architecture** with AI fallback (not an AI-only wrapper).
- **Deployment-ready API** with health/ready checks and artifact endpoints.
- **Mobile-friendly frontend** with camera upload support.

## Processing Pipeline

1. **Preprocess image**  
   Detect document boundaries, crop page region, and correct perspective.

2. **Run OCR locally first**  
   Prefer local OCR output for speed/cost control.

3. **Classify with heuristics first**  
   If document type is obvious (e.g., invoice/resume/receipt), classify locally.

4. **AI fallback only when needed**  
   Gemini is called only for low-confidence or ambiguous cases.

## Fallback Strategy (Cost and Reliability)

- **OCR fallback chain:** local OCR -> alternate local OCR -> Gemini Vision OCR (conditional).
- **Categorization fallback:** heuristic classification first, AI categorization only when heuristic confidence is low.
- **Manual override:** frontend includes an "Improve with AI Vision" action for hard cases.
- **Quota-aware behavior:** when AI quota is exhausted, pipeline still returns local OCR results and structured fallback output.

## API Flow

### `POST /api/scan`

Input:
- `file` (multipart image)
- `force_vision` (optional, boolean)

Output:
- `scan_id`
- `results`:
  - `category`
  - `subcategory`
  - `summary`
  - `editable_text`
  - `key_information`
- `artifacts` paths (cropped image, OCR text, JSON)
- `meta` (document detected, OCR confidence, timings, classification method)

### Other endpoints

- `GET /health` - liveness check
- `GET /ready` - dependency/config readiness check
- `GET /api/scans/{scan_id}/{artifact}` - fetch scan artifacts  
  Supported artifacts: `cropped.jpg`, `ocr_ready.png`, `ocr.txt`, `result.json`

## Environment Variables

```env
GEMINI_API_KEY=your_key_here
VISION_OCR_MODE=auto
OUTPUT_DIR=outputs
CORS_ALLOW_ORIGINS=*
DEV_MODE=true
CLOUD_MODE=false
PADDLE_LANG=en
```

Notes:
- `VISION_OCR_MODE=auto` keeps AI usage minimal.
- `CLOUD_MODE=true` is recommended on lower-memory cloud instances.

## Local Setup

### Backend

```bash
pip install -r requirement.txt
python main.py
```

Install Tesseract on your OS and ensure `tesseract` is available in `PATH`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL` when pointing frontend to a deployed backend.

## Docker / Deployment

```bash
docker compose up --build
```

- Frontend: `http://localhost:8080`
- Backend docs: `http://localhost:8000/docs`

For lightweight cloud deployments, use `requirements-cloud.txt` and enable `CLOUD_MODE=true`.
