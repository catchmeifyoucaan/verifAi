from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

# --- Add CORS Middleware ---
# This allows your frontend to reliably communicate with this backend.
origins = ["*"] # Allows all origins for simplicity

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- A simple, pure Python mock model ---
# This removes the dependency on heavy libraries like scikit-learn.
class SimpleMockModel:
    def predict(self, object_class: str) -> str:
        print(f"--- Model: Predicting for class '{object_class}' ---")
        if "bottle" in object_class or "cup" in object_class:
            return "verified"
        else:
            return "warning"

ai_model = None

@app.on_event("startup")
def load_model():
    """Instantiates our simple model when the server starts."""
    global ai_model
    print("--- Server starting up: Instantiating AI model. ---")
    ai_model = SimpleMockModel()
    print("--- AI model is ready. ---")


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
def verify_item(request: VerificationRequest):
    print(f"--- Received API request for: {request.object_class} ---")
    if not ai_model:
        print("[ERROR] AI Model is not loaded.")
        raise HTTPException(status_code=500, detail="AI model is not available.")

    try:
        # Get a prediction from our loaded model
        predicted_label = ai_model.predict(request.object_class)
        print(f"--- Prediction successful. Result: {predicted_label} ---")

        # Create a response based on the prediction
        return {
            "status": predicted_label,
            "title": f"Live Verification for {request.object_class.title()}",
            "confidence": 0.95,
            "summary": f"Self-hosted AI model returned a '{predicted_label}' status.",
            "details": [
                {"agent": "Render-Hosted Model v2", "finding": f"Prediction output: {predicted_label}", "status": "success"}
            ]
        }
    except Exception as e:
        print(f"[ERROR] An exception occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during AI analysis.")

@app.get("/")
def read_root():
    return {"message": "VerifAi Backend v1.3 (Lightweight) is running."}
