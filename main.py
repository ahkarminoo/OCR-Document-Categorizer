from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from processor import clean_document
from ai_engine import categorize_document # Import the new AI logic

app = FastAPI(title="OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/scan")
async def scan_document(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    contents = await file.read()
    
    # 1. Clean and flatten the image (OpenCV)
    processed_image_bytes = clean_document(contents)
    
    # 2. Extract and categorize text (Gemini API)
    # Try passing 'contents' instead of 'processed_image_bytes' if results stay empty
    document_data = categorize_document(processed_image_bytes)
    
    # DEBUG: See what the AI is actually sending back in your terminal
    print(f"--- DEBUG START ---")
    print(f"Filename: {file.filename}")
    print(f"AI Output: {document_data}")
    print(f"--- DEBUG END ---")
    
    # 3. Return the AI insights
    return {
        "filename": file.filename,
        "results": document_data
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)