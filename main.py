import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# =====================================================
# CLIENT
# =====================================================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Quantum Forge AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODEL REGISTRY (NICKNAME → MODEL ID)
# =====================================================
CHAT_MODELS = {
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

# =====================================================
# REQUEST SCHEMA
# =====================================================
class ChatRequest(BaseModel):
    message: str
    model: str = "jarvis"
    task: str | None = "chat"     # chat | code | research
    guard: bool | None = False

# =====================================================
# IDENTITY PROMPT (ALL MODELS)
# =====================================================
def build_identity_prompt(model_key: str) -> str:
    name = model_key.capitalize()
    return (
        f"You are {name}, an AI assistant created by the Quantum Forge team. "
        f"If asked about your name or creator, reply exactly: "
        f"'I am {name}, created by the Quantum Forge team.' "
        f"Do not mention OpenAI, Meta, Groq, or any other organization."
    )

# =====================================================
# TASK PROMPTS (CHAT / CODE / RESEARCH)
# =====================================================
def task_prompt(task: str) -> str:
    if task == "code":
        return (
            "You are a senior software engineer. "
            "Write correct, production-ready code. "
            "Follow best practices and keep explanations concise."
        )

    if task == "research":
        return (
            "You are a research assistant. "
            "Respond with structured sections using markdown: "
            "## Overview, ## Key Concepts, ## Analysis, "
            "## Limitations, ## Conclusion."
        )

    return "Respond clearly and concisely."

# =====================================================
# HARD IDENTITY OVERRIDE (100% GUARANTEE)
# =====================================================
def identity_override(model_key: str, msg: str) -> str | None:
    triggers = [
        "your name",
        "who are you",
        "what are you",
        "who created you",
        "who made you",
        "your creator"
    ]
    if any(t in msg.lower() for t in triggers):
        return f"I am {model_key.capitalize()}, created by the Quantum Forge team."
    return None

# =====================================================
# SAFETY CHECK (OPTIONAL)
# =====================================================
def safety_check(text: str) -> bool:
    result = client.chat.completions.create(
        model=SAFETY_MODELS["guardian"],
        messages=[{"role": "user", "content": text}]
    )
    return "unsafe" not in result.choices[0].message.content.lower()

# =====================================================
# CHAT ENDPOINT
# =====================================================
@app.post("/chat")
def chat(req: ChatRequest):

    if req.model not in CHAT_MODELS:
        return {
            "reply": "Unknown model",
            "model_used": req.model
        }

    # Absolute identity guarantee
    override = identity_override(req.model, req.message)
    if override:
        return {
            "reply": override,
            "model_used": req.model
        }

    # Optional safety
    if req.guard:
        if not safety_check(req.message):
            return {
                "reply": "⚠️ Blocked by safety system.",
                "model_used": req.model
            }

    # Input size guard
    if len(req.message) > 6000:
        return {
            "reply": "Input too long. Please split your request.",
            "model_used": req.model
        }

    system_prompt = (
        build_identity_prompt(req.model)
        + " "
        + task_prompt(req.task)
    )

    completion = client.chat.completions.create(
        model=CHAT_MODELS[req.model],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        temperature=0.7
    )

    return {
        "reply": completion.choices[0].message.content,
        "model_used": req.model
    }

# =====================================================
# MODELS LIST ENDPOINT (ONLY MODEL NAMES)
# =====================================================
@app.get("/models")
def list_models():
    return list(CHAT_MODELS.keys())

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/")
def health():
    return {"status": "Quantum Forge backend running"}
