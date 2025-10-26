import cv2
import pytesseract
import re
import json
import io
import os
import tempfile
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# Load env vars
load_dotenv()

app = FastAPI(title="MUTCD Sign OCR API", description="Extract and correct sign data from images")

# Add CORS middleware (update origins to match your Next.js domain, e.g., http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://mutcd-ai.vercel.app"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageUrlRequest(BaseModel):
    url: str

def analyze_with_grok(ocr_results: List[Dict]) -> List[Dict]:
    api_key = os.getenv('XAI_API_KEY')
    if not api_key:
        raise HTTPException(status_code=500, detail="XAI_API_KEY not set - Grok analysis skipped.")
    url = 'https://api.x.ai/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'grok-4-fast',  # Or 'grok-4-fast' if available
        'messages': [
            {
                'role': 'system',
                'content': '''You are a MUTCD sign expert. Analyze and correct this OCR-extracted JSON for accuracy: Fix codes (e.g., "Ma-8" to "M4-8"), remove artifacts (e.g., "|", commas), add full descriptions from MUTCD standards, infer quantities if possible. Return ONLY the corrected JSON array - no extra text.
                - Match the MUTCD code to the description because the description is easier for the OCR to read. the MUTCD code should always be matched to the description it should never be the description right and mutcd code wrong. always check the mutcd code youre outputting against the description.'''
            },
            {
                'role': 'user',
                'content': json.dumps(ocr_results)
            }
        ],
        'max_tokens': 1000,
        'temperature': 0
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        grok_response = response.json()['choices'][0]['message']['content'].strip()
        # Strip markdown if present
        if grok_response.startswith('```json'):
            grok_response = grok_response.replace('```json', '').replace('```', '').strip()
        corrected_results = json.loads(grok_response)
        print('Grok Analysis Complete:', corrected_results)
        return corrected_results
    except Exception as e:
        print(f'Grok API failed: {e} - falling back to raw OCR.')
        return ocr_results

def process_image_bytes(contents: bytes) -> List[Dict]:
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    _, thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    
    # Temp file for Tesseract
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        cv2.imwrite(tmp.name, thresh)
        text = pytesseract.image_to_string(tmp.name, lang='eng', config='--psm 4')
        os.unlink(tmp.name)
    
    print('OCR Result:\n', text)
    
    # Parse
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    results = []
    for line in lines:
        if any(word in line.upper() for word in ['STD. NO.', 'SIZE', 'DESCRIPTION', 'QUANTITY', 'TABULATION', 'INCLUDED', 'CHANNEL', 'TYPE']):
            continue
        words = re.split(r'\s+', line)
        if len(words) >= 4:
            code = words[0].strip()
            size = ' '.join(words[1:4]).strip()
            desc_words = words[4:]
            if desc_words and desc_words[-1].isdigit():
                quantity = desc_words[-1].strip()
                description = ' '.join(desc_words[:-1]).strip()
            else:
                quantity = ''
                description = ' '.join(desc_words).strip()
            results.append({
                'code': code,
                'size': size,
                'description': description,
                'quantity': quantity
            })
    
    # Grok
    return analyze_with_grok(results)

@app.post("/process-image", response_model=List[Dict])
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    return JSONResponse(content=process_image_bytes(contents))

@app.post("/process-url", response_model=List[Dict])
async def process_url(req: ImageUrlRequest):
    try:
        # Download image bytes from public URL
        response = requests.get(req.url, timeout=30)
        response.raise_for_status()
        contents = response.content
        return JSONResponse(content=process_image_bytes(contents))
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "MUTCD OCR API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
