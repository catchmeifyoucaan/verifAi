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

# --- UPGRADED AI LOGIC ---
# The model now has a dictionary of predefined responses for different objects.
class SmartMockModel:
    MOCK_RESPONSES = {
        "person": {
            "status": "verified",
            "confidence": 98.6,
            "summary": "Biometric scan matches. Identity verified.",
            "agent": "Identity Verification Agent"
        },
        "bottle": {
            "status": "verified",
            "confidence": 99.2,
            "summary": "Seal is intact. Batch number is valid.",
            "agent": "Consumable Safety Agent"
        },
        "cup": {
            "status": "verified",
            "confidence": 99.5,
            "summary": "Material analysis complete. Safe for use.",
            "agent": "Consumable Safety Agent"
        },
        "tv": {
            "status": "warning",
            "confidence": 85.1,
            "summary": "Serial number does not match manufacturer records for this region.",
            "agent": "Electronics Authenticity Agent"
        },
        "default": {
            "status": "warning",
            "confidence": 75.0,
            "summary": "Object class not recognized in our high-confidence database.",
            "agent": "General Object Agent"
        }
    }

    def predict(self, object_class: str) -> dict:
        print(f"--- Model: Predicting for class '{object_class}' ---")
        # Return the specific response for the class, or the default one
        return self.MOCK_RESPONSES.get(object_class, self.MOCK_RESPONSES["default"])

ai_model = None

@app.on_event("startup")
def load_model():
    """Instantiates our smart model when the server starts."""
    global ai_model
    print("--- Server starting up: Instantiating Smart AI model. ---")
    ai_model = SmartMockModel()
    print("--- Smart AI model is ready. ---")


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
        raise HTTPException(status_code=500, detail="AI model is not available.")

    try:
        # Get a full prediction dictionary from our model
        prediction = ai_model.predict(request.object_class)
        print(f"--- Prediction successful. Result: {prediction} ---")

        # Create a response using the data from the prediction
        return {
            "status": prediction["status"],
            "title": f"Live Verification for {request.object_class.title()}",
            "confidence": prediction["confidence"], # FIX: Confidence is now correct
            "summary": prediction["summary"],
            "details": [
                {"agent": prediction["agent"], "finding": f"Prediction status: {prediction['status']}", "status": "success"}
            ]
        }
    except Exception as e:
        print(f"[ERROR] An exception occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during AI analysis.")

@app.get("/")
def read_root():
    return {"message": "VerifAi Backend v1.4 (Smarter Logic) is running."}
