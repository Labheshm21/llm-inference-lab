import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="LLM Inference Lab")

model = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
    dtype=torch.float32,
)


class ChatRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [{"role": "user", "content": request.prompt}]
    result = model(messages, max_new_tokens=40, do_sample=False)
    answer = result[0]["generated_text"][-1]["content"]
    return {"answer": answer}