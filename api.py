# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from emotion_classifier import EmotionClassifier
from prompt_builder import build_prompt
from response_generator import generate_response

app = FastAPI()

# Load classifier at startup
classifier = EmotionClassifier()

class MessageRequest(BaseModel):
    text: str


class MessageResponse(BaseModel):
    text: str
    emotions: list
    prompt: str
    llm_response: str


@app.post("/predict", response_model=MessageResponse)
def predict(request: MessageRequest):

    # 1) Emotion detection
    result = classifier.predict(request.text)
    emotions = result[0]["labels"]

    # 2) Build prompt for the LLM
    prompt = build_prompt(request.text, emotions)

    # 3) Generate an empathetic LLM response
    llm_answer = generate_response(prompt)

    return MessageResponse(
        text=request.text,
        emotions=emotions,
        prompt=prompt,
        llm_response=llm_answer
    )
