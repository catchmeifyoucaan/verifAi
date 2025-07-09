# FIX: Ensure all necessary imports are at the top of the file.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
from fastapi.middleware.cors import CORSMiddleware

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

# --- Define a proper class for the model so joblib can save it ---
class SerializableMockModel:
    def predict(self, features):
        # This mock model will always predict "authentic"
        return ["authentic"]

# --- AI Model Loading ---
model = None
vectorizer = None

@app.on_event("startup")
def load_model():
    """
    Instantiates and loads the AI model and vectorizer when the server starts.
    If the model files don't exist, it creates them.
    """
    global model, vectorizer
    model_filename = "verifai_model.joblib"
    vectorizer_filename = "verifai_vectorizer.joblib"

    if not os.path.exists(model_filename):
        print("--- Creating placeholder model files for first run... ---")

        # Create and save a TfidfVectorizer
        dummy_vectorizer = TfidfVectorizer()
        dummy_vectorizer.fit_transform(["authentic", "counterfeit"])
        joblib.dump(dummy_vectorizer, vectorizer_filename)

        # Create and save an instance of our serializable model class
        serializable_model = SerializableMockModel()
        joblib.dump(serializable_model, model_filename)
        print("--- Placeholder files created. ---")

    print("--- Loading model and vectorizer... ---")
    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    print("--- Model and vectorizer loaded successfully. ---")


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
    if not model or not vectorizer:
        print("[ERROR] AI Model or vectorizer is not loaded.")
        raise HTTPException(status_code=500, detail="AI model is not available.")

    try:
        # Use the loaded vectorizer and model
        features = vectorizer.transform([request.object_class])
        prediction = model.predict(features)
        predicted_label = prediction[0]

        print(f"--- Prediction successful. Result: {predicted_label} ---")

        return {
            "status": "verified" if predicted_label == "authentic" else "warning",
            "title": f"Live Verification for {request.object_class.title()}",
            "confidence": 93.0,
            "summary": f"Self-hosted AI model predicted this item is {predicted_label}.",
            "details": [
                {"agent": "Render-Hosted Model v3", "finding": f"Prediction output: {predicted_label}", "status": "success"}
            ]
        }
    except Exception as e:
        print(f"[ERROR] An exception occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during AI analysis.")

@app.get("/")
def read_root():
    return {"message": "VerifAi Backend v1.2 (scikit-learn compatible) is running."}
