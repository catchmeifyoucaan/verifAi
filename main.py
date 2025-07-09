# FIX: Ensure all necessary imports are at the top of the file.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import base64
import io
from PIL import Image
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# --- Add CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- Helper function to decode base64 image ---
def decode_image(base64_string: str):
    try:
        # Remove data:image/jpeg;base64, prefix if present
        header, encoded = base64_string.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")






# --- API Data Models ---
class VerificationRequest(BaseModel):
    image: str
    object_class: str

class VerificationResponse(BaseModel):
    status: str
    title: str
    confidence: float
    summary: str
    details: list

# --- API Endpoint ---
@app.post("/verify", response_model=VerificationResponse)
async def verify_item(request: VerificationRequest):
    print(f"--- Received API request for: {request.object_class} ---")

    try:
        # Decode the image
        image = decode_image(request.image)

        # Construct prompt for Gemini
        prompt = [
            f"Analyze this image of a {request.object_class}. Determine if it appears authentic or counterfeit. "
            "Provide a detailed explanation of your reasoning, highlighting specific features that support your conclusion. "
            "Consider aspects like logos, stitching, material quality, and any visible serial numbers or tags. "
            "If you cannot determine authenticity, state why. "
            "Format your response as a JSON object with the following keys: "
            "\"status\": (\"verified\", \"warning\", or \"danger\"), "
            "\"title\": (a concise title), "
            "\"confidence\": (a float between 0.0 and 100.0), "
            "\"summary\": (a brief summary of the finding), "
            "\"details\": (a list of objects, each with \"agent\", \"finding\", and \"status\": (\"success\" or \"fail\"))."
        ]

        # Call Gemini Vision API
        response = gemini_model.generate_content(prompt + [image])
        
        # Extract JSON from Gemini's response
        gemini_output = response.text.strip()
        
        # Attempt to parse the JSON. Gemini might sometimes include markdown.
        if gemini_output.startswith("```json"):
            gemini_output = gemini_output[len("```json"):].strip()
        if gemini_output.endswith("```"):
            gemini_output = gemini_output[:-len("```")].strip()

        verification_result = json.loads(gemini_output)

        # Validate the structure of Gemini's response
        if not all(k in verification_result for k in ["status", "title", "confidence", "summary", "details"]):
            raise ValueError("Gemini response missing required fields.")

        print(f"--- Gemini verification successful. Status: {verification_result['status']} ---")

        return VerificationResponse(**verification_result)

    except json.JSONDecodeError as e:
        print(f"[ERROR] Gemini response was not valid JSON: {e}\nResponse: {gemini_output}")
        raise HTTPException(status_code=500, detail="AI analysis returned malformed data. Please try again.")
    except ValueError as e:
        print(f"[ERROR] Invalid Gemini response structure: {e}")
        raise HTTPException(status_code=500, detail="AI analysis returned unexpected data structure. Please try again.")
    except Exception as e:
        print(f"[ERROR] An exception occurred during verification: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during AI analysis.")

@app.get("/")
def read_root():
    return {"message": "VerifAi Backend v1.2 (scikit-learn compatible) is running."}
