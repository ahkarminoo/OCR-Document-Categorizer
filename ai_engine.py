import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def categorize_document(image_bytes):
    """Sends the image to Gemini for OCR and Categorization with robust JSON parsing."""
    
    # Using gemini-1.5-flash for high-speed multimodal analysis
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = """
    Analyze the provided image and extract meaningful data.
    
    Instructions:
    1. Identify the document type (e.g., Financial, Code, News, Receipt).
    2. Summarize the content in 2 clear sentences.
    3. Extract key details (names, dates, values, identifiers).

    YOU MUST RESPOND ONLY WITH A VALID JSON OBJECT:
    {
      "category": "String",
      "summary": "String",
      "key_information": ["List", "of", "strings"]
    }
    """

    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }
    
    try:
        response = model.generate_content([prompt, image_part])
        raw_text = response.text.strip()

        # 1. Try a direct JSON load first
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # 2. If it fails, use Regex to extract JSON from inside markdown blocks
            # This looks for everything between the first '{' and the last '}'
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON structure found in AI response.")

    except Exception as e:
        # Return a structured error so the frontend doesn't crash
        return {
            "category": "Error",
            "summary": f"AI Processing failed: {str(e)}",
            "key_information": []
        }