import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# ================== CLIENT ==================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================== APP =====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODELS ==================
PRIMARY_MODEL = "openai/gpt-oss-120b"
FAST_MODEL = "llama-3.3-70b-versatile"
REASON_MODEL = "qwen/qwen3-32b"
GUARD_MODEL = "meta-llama/llama-guard-4-12b"

# ================== SCHEMA ==================
class ChatRequest(BaseModel):
    message: str
    mode: str | None = "auto"
    system: str | None = "You are a helpful AI assistant."

# ================== SAFETY ==================
def safety_check(text: str) -> bool:
    result = client.chat.completions.create(
        model=GUARD_MODEL,
        messages=[{"role": "user", "content": text}]
    )
    output = result.choices[0].message.content.lower()
    return "unsafe" not in output

# ================== ROUTER ==================
def choose_model(msg: str, mode: str) -> str:
    if mode == "fast":
        return FAST_MODEL
    if mode == "reason":
        return REASON_MODEL
    if mode == "power":
        return PRIMARY_MODEL

    # AUTO MODE
    if len(msg) > 400:
        return PRIMARY_MODEL
    if any(k in msg.lower() for k in ["math", "calculate", "prove", "logic"]):
        return REASON_MODEL
    return FAST_MODEL

# ================== CHAT ====================
@app.post("/chat")
def chat(req: ChatRequest):

    if not safety_check(req.message):
        return {"reply": "⚠️ Message blocked by safety system."}

    model = choose_model(req.message, req.mode)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.message}
        ],
        temperature=0.7
    )

    reply = completion.choices[0].message.content

    return {
        "reply": reply,
        "model_used": model
    }

