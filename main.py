from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import base64
import io
from PIL import Image
import json
import PyPDF2
import pandas as pd
from fastapi.concurrency import run_in_threadpool
import httpx # For making HTTP requests to microservices
import uuid # For generating unique IDs

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# --- Add CORS Middleware ---
origins = [
    "http://localhost:3000",
    "https://verifai-app.com",  # Replace with your production domain
]
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
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# --- API Data Models ---
class VerificationRequest(BaseModel):
    object_class: str
    file_type: str | None = None
    image_data_url: str | None = None
    media_data_url: str | None = None
    text_content: str | None = None
    agent_id: str | None = None

class VerificationResponse(BaseModel):
    verification_id: str # Unique ID for this verification instance
    status: str
    title: str
    confidence: float
    summary: str
    details: list
    explanation: str | None = None
    remediation: str | None = None
    liveness_score: float | None = None
    spoof_detection_result: str | None = None
    recommended_for_human_review: bool = False

class FeedbackRequest(BaseModel):
    verification_id: str
    is_helpful: bool
    user_id: str | None = None

# --- Agent Configurations ---
AGENTS = {
    "general_purpose": {
        "description": "A general-purpose agent for broad authenticity checks.",
        "prompt_template": (
            "Analyze this {input_type} of a {object_class}. "
            "Determine if it appears authentic or counterfeit. "
            "Provide a detailed explanation of your reasoning, highlighting specific features that support your conclusion. "
            "Consider aspects like logos, stitching, material quality, and any visible serial numbers or tags. "
            "If this is a document, perform OCR and analyze for signs of fraud or tampering. "
            "If you cannot determine authenticity, state why. "
        )
    },
    "id_document_verifier": {
        "description": "Specialized agent for verifying identity documents.",
        "prompt_template": (
            "Analyze this image of an ID document ({object_class}). "
            "Perform OCR to extract all text. Verify the authenticity of the document, checking for signs of tampering, "
            "forgery, or inconsistencies in fonts, holograms, and security features. "
            "Crucially, also analyze for signs of spoofing or presentation attacks (e.g., image of a screen, printed photo, mask). "
            "Provide a detailed explanation of any anomalies found, including spoofing attempts. "
            "If you cannot determine authenticity, state why. "
        )
    },
    "product_authenticator": {
        "description": "Specialized agent for authenticating physical products.",
        "prompt_template": (
            "Analyze this {input_type} of a {object_class} (product). "
            "Focus on brand logos, serial numbers, material quality, stitching, and packaging. "
            "Determine if it is an authentic product or a counterfeit. "
            "Provide a detailed explanation of your findings and reasoning. "
            "If you cannot determine authenticity, state why. "
        )
    },
    "text_analyzer": {
        "description": "Specialized agent for analyzing textual content.",
        "prompt_template": (
            "Analyze this text content related to a {object_class}. "
            "Determine if it appears authentic, contains anomalies, or exhibits signs of AI generation/plagiarism. "
            "Provide a detailed explanation of your reasoning, highlighting specific phrases or patterns. "
            "If you cannot determine authenticity, state why. "
        )
    },
    "pharmaceutical_authenticator": {
        "description": "Specialized agent for authenticating pharmaceuticals.",
        "prompt_template": (
            "Analyze this {input_type} of a {object_class} (pharmaceutical product). "
            "Focus on packaging, seals, batch numbers, expiry dates, holograms, and any visible security features. "
            "Determine if it is an authentic product or a counterfeit. "
            "Provide a detailed explanation of your findings and reasoning. "
            "If you cannot determine authenticity, state why. "
            "Crucially, highlight any potential health risks if deemed counterfeit or tampered. "
            "Consider past verification results for similar items (if available for reinforcement learning)."
        )
    },
    "drink_authenticator": {
        "description": "Specialized agent for authenticating drinks, including alcohol and cognacs.",
        "prompt_template": (
            "Analyze this {input_type} of a {object_class} (drink/beverage). "
            "Focus on bottle/can design, labels, seals, caps, liquid clarity, fill levels, and any unique identifiers. "
            "Determine if it is an authentic product or a counterfeit. "
            "Provide a detailed explanation of your findings and reasoning. "
            "If you cannot determine authenticity, state why. "
            "For alcoholic beverages, consider specific brand characteristics and regional authenticity markers. "
            "Consider past verification results for similar items (if available for reinforcement learning)."
        )
    },
    "food_authenticator": {
        "description": "Specialized agent for authenticating food products.",
        "prompt_template": (
            "Analyze this {input_type} of a {object_class} (food product). "
            "Focus on packaging integrity, seals, nutritional labels, expiry dates, ingredients list, and any visible signs of spoilage or tampering. "
            "Determine if it is an authentic product, safe for consumption, or potentially counterfeit/compromised. "
            "Provide a detailed explanation of your findings and reasoning. "
            "If you cannot determine authenticity, state why. "
            "Prioritize food safety and highlight any risks. "
            "Consider past verification results for similar items (if available for reinforcement learning)."
        )
    },
    "water_authenticator": {
        "description": "Specialized agent for authenticating bottled water and other non-alcoholic beverages.",
        "prompt_template": (
            "Analyze this {input_type} of a {object_class} (bottled water or non-alcoholic beverage). "
            "Focus on bottle design, cap seal, label authenticity, water clarity, and any unique identifiers. "
            "Determine if it is an authentic product or a counterfeit. "
            "Provide a detailed explanation of your findings and reasoning. "
            "If you cannot determine authenticity, state why. "
            "Highlight any signs of contamination or tampering. "
            "Consider past verification results for similar items (if available for reinforcement learning)."
        )
    }
}

# --- API Endpoint ---
@app.post("/verify", response_model=VerificationResponse)
async def verify_item(request: VerificationRequest):
    print(f"--- Received API request for: {request.object_class} (Type: {request.file_type}, Agent: {request.agent_id}) ---")

    try:
        agent_id = request.agent_id if request.agent_id in AGENTS else "general_purpose"
        selected_agent = AGENTS[agent_id]

        prompt_template = selected_agent["prompt_template"]
        
        prompt_parts = []
        input_content = None
        input_type_desc = "data" # Default description for prompt formatting

        if request.file_type and request.file_type.startswith("image/") and request.image_data_url:
            try:
                header, encoded = request.image_data_url.split(",", 1)
                image_data = base64.b64decode(encoded)
                input_content = Image.open(io.BytesIO(image_data))
                input_type_desc = "image"
            except (PIL.UnidentifiedImageError, IOError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
            
        elif request.file_type and (request.file_type.startswith("video/") or request.file_type.startswith("audio/")) and request.media_data_url:
            input_content = {
                "mime_type": request.file_type,
                "data": request.media_data_url.split(",", 1)[1] # Get base64 data part
            }
            input_type_desc = request.file_type.split("/")[0] # "video" or "audio"

        elif request.file_type and request.file_type.startswith("text/") and request.text_content:
            input_content = request.text_content
            input_type_desc = "text content"
        elif request.file_type == "application/pdf" and request.media_data_url:
            try:
                header, encoded = request.media_data_url.split(",", 1)
                pdf_data = base64.b64decode(encoded)
                pdf_file = io.BytesIO(pdf_data)
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
                input_content = text
                input_type_desc = "PDF document"
            except PyPDF2.errors.PdfReadError as e:
                raise HTTPException(status_code=400, detail=f"Invalid PDF data: {e}")
            except Exception as e: # Catch other potential errors during PDF processing
                raise HTTPException(status_code=400, detail=f"Error processing PDF data: {e}")
        elif request.file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"] and request.media_data_url:
            try:
                header, encoded = request.media_data_url.split(",", 1)
                excel_data = base64.b64decode(encoded)
                excel_file = io.BytesIO(excel_data)
                df = pd.read_excel(excel_file, sheet_name=None) # Read all sheets
                text = ""
                for sheet_name, sheet_df in df.items():
                    text += f"\n--- Sheet: {sheet_name} ---\n"
                    text += sheet_df.to_string(index=False) + "\n"
                input_content = text
                input_type_desc = "spreadsheet document"
            except ValueError as e: # Catch errors related to pandas reading excel
                raise HTTPException(status_code=400, detail=f"Invalid Excel data: {e}")
            except Exception as e: # Catch other potential errors during Excel processing
                raise HTTPException(status_code=400, detail=f"Error processing Excel data: {e}")
        elif request.file_type == "text/csv" and request.media_data_url:
            try:
                header, encoded = request.media_data_url.split(",", 1)
                csv_data = base64.b64decode(encoded)
                csv_file = io.BytesIO(csv_data)
                df = pd.read_csv(csv_file)
                input_content = df.to_string(index=False)
                input_type_desc = "CSV document"
            except (pd.errors.EmptyDataError, ValueError) as e: # Catch errors related to pandas reading csv
                raise HTTPException(status_code=400, detail=f"Invalid CSV data: {e}")
            except Exception as e: # Catch other potential errors during CSV processing
                raise HTTPException(status_code=400, detail=f"Error processing CSV data: {e}")
        else:
            raise HTTPException(status_code=400, detail="No valid input data or file type provided.")

        # Format the prompt using the selected agent's template
        formatted_prompt = prompt_template.format(input_type=input_type_desc, object_class=request.object_class)
        prompt_parts.append(formatted_prompt)
        prompt_parts.append(input_content)

        # Add the common JSON formatting instruction
        prompt_parts.append(
            "Format your response as a JSON object with the following keys: "
            "\"status\": (\"verified\", \"warning\", or \"danger\"), "
            "\"title\": (a concise title), "
            "\"confidence\": (a float between 0.0 and 100.0), "
            "\"summary\": (a brief summary of the finding), "
            "\"details\": (a list of objects, each with \"agent\", \"finding\", and \"status\": (\"success\" or \"fail\")), "
            "\"explanation\": (a detailed explanation of the anomaly if status is 'warning' or 'danger', otherwise null), "
            "\"remediation\": (suggested steps for remediation if status is 'warning' or 'danger', otherwise null), "
            "\"liveness_score\": (a float between 0.0 and 1.0 indicating liveness, or null), "
            "\"spoof_detection_result\": (a string indicating spoofing detection outcome, e.g., \"live\", \"spoof\", or null)."
        )

        # Call Gemini API
        response = await run_in_threadpool(gemini_model.generate_content, prompt_parts)
        
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

        print(f"--- Gemini verification successful. Status: {verification_result['status']} (Agent: {agent_id}) ---")

        # Generate a unique verification ID
        verification_id = str(uuid.uuid4())
        verification_result["verification_id"] = verification_id

        # Determine if human review is recommended
        if verification_result["status"] in ["warning", "danger"] or verification_result["confidence"] < 70.0:
            verification_result["recommended_for_human_review"] = True
        else:
            verification_result["recommended_for_human_review"] = False

        return VerificationResponse(**verification_result)

    except json.JSONDecodeError as e:
        print(f"[ERROR] Gateway - Gemini response was not valid JSON: {e}\nResponse: {gemini_output}")
        raise HTTPException(status_code=500, detail="AI analysis returned malformed data. Please try again.")
    except ValueError as e:
        print(f"[ERROR] Gateway - Invalid Gemini response structure: {e}")
        raise HTTPException(status_code=500, detail="AI analysis returned unexpected data structure. Please try again.")
    except Exception as e:
        print(f"[ERROR] Gateway - An unexpected error occurred: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during AI analysis.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the VerifAi Backend API. Access the interactive API documentation at /docs for details on available endpoints and models."}

@app.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest):
    print(f"--- Received feedback for verification {feedback.verification_id}: Helpful={feedback.is_helpful}, User={feedback.user_id} ---")
    # In a real application, you would store this feedback in a database
    # and use it for model retraining or analysis.
    return {"message": "Feedback received successfully!"}

@app.get("/agents")
def get_agents():
    return AGENTS
