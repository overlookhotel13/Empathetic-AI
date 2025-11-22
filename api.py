from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from emotion_classifier import EmotionClassifier
from prompt_builder import build_prompt
from response_generator import generate_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
classifier = None
try:
    classifier = EmotionClassifier("OVERLOOKHOTEL13/emotion-detector-final")
except Exception as e:
    print("Error loading model:", e)

class Request(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok" if classifier else "initializing"}

@app.post("/predict")
def predict(req: Request):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = classifier.predict(req.text)[0]  
    prompt = build_prompt(req.text, result["labels"])
    ai_reply = generate_response(prompt)

    return {
        "text": req.text,
        "emotions": result["labels"],
        "prompt": prompt,
        "llm_response": ai_reply
    }
