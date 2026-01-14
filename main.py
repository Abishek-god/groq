import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# ================= CLIENT =================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= APP ====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= MODEL REGISTRY =========
MODEL_NICKNAMES = {
    "jarvis": "openai/gpt-oss-120b",
    "friday": "llama-3.3-70b-versatile",
    "vision": "qwen/qwen3-32b",
    "atlas": "openai/gpt-oss-20b",
    "kimi": "moonshotai/kimi-k2-instruct",
    "kimi-pro": "moonshotai/kimi-k2-instruct-0905",
    "maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "swift": "llama-3.1-8b-instant",
    "allam": "allam-2-7b",
    "compound": "groq/compound-mini"
}

SAFETY_MODELS = {
    "guardian": "meta-llama/llama-guard-4-12b",
    "safeguard": "openai/gpt-oss-safeguard-20b"
}

# ================= SCHEMA =================
class ChatRequest(BaseModel):
    message: str
    model: str = "friday"   # nickname
    system: str | None = "You are a helpful AI assistant."
    guard: bool | None = False

# ================= SAFETY =================
def safety_check(text: str) -> bool:
    result = client.chat.completions.create(
        model=SAFETY_MODELS["guardian"],
        messages=[{"role": "user", "content": text}]
    )
    output = result.choices[0].message.content.lower()
    return "unsafe" not in output

# ================= CHAT ===================
@app.post("/chat")
def chat(req: ChatRequest):

    if req.model not in MODEL_NICKNAMES:
        return {
            "reply": f"❌ Unknown model nickname: {req.model}",
            "available_models": list(MODEL_NICKNAMES.keys())
        }

    if req.guard:
        if not safety_check(req.message):
            return {"reply": "⚠️ Blocked by safety system."}

    model_name = MODEL_NICKNAMES[req.model]

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.message}
        ],
        temperature=0.7
    )

    return {
        "reply": completion.choices[0].message.content,
        "model_used": req.model
    }

