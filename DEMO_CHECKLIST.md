# Demo Script and Validation Checklist

## 2-Minute Demo Script

1. Open backend at `http://127.0.0.1:8000/docs`.
2. Open frontend with `npm run dev`.
3. Upload a wide-angle document photo that includes background clutter.
4. Show output sections:
   - `Category` (fixed heading),
   - `Subcategory` (dynamic heading),
   - `Editable Extracted Text`,
   - `Summary` and `Key Information`.
5. Edit text directly in the textarea and use `Copy Extracted Text`.
6. Highlight metadata (`document_detected`, OCR word count, confidence).

## Validation Cases (Run Manually)

- Invoice photo with perspective tilt.
- Receipt with shadows and non-document background.
- Resume screenshot/photo.
- Academic notes or textbook page.
- Printed legal form.
- Medical report page.
- Document with very little text.
- Blurry/noisy photo to verify OCR error handling.

## Pass Criteria

- Document area is cropped in most cases.
- Extracted text is editable and copyable.
- Category is one of: `Invoice`, `Resume`, `Legal`, `Medical`, `Academic`, `Receipt`, `Other`.
- Subcategory is non-empty and meaningful.
- API returns stable JSON fields without frontend crashes.
