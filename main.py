import os
from datetime import datetime
from uuid import uuid4
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq
from supabase import create_client, Client

# =====================================================
# ENV
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise RuntimeError("Missing environment variables")

# =====================================================
# MODEL REGISTRY (NICKNAME → GROQ MODEL)
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
    "compound": "groq/compound-mini",
}

# =====================================================
# FASTAPI APP (SINGLE INSTANCE)
# =====================================================
app = FastAPI(title="Quantum Forge AI – Enterprise")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LIFECYCLE (CORRECT)
# =====================================================
@app.on_event("startup")
async def startup():
    app.state.groq = AsyncGroq(api_key=GROQ_API_KEY)
    app.state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================
# SCHEMAS
# =====================================================
class ChatRequest(BaseModel):
    user_id: str
    message: str
    model: str = "jarvis"
    conversation_id: Optional[str] = None

class RegenerateRequest(BaseModel):
    user_id: str
    conversation_id: str
    model: str = "jarvis"

# =====================================================
# SYSTEM IDENTITY (HARD GUARANTEE)
# =====================================================
def identity_prompt(name: str) -> str:
    return (
        f"You are {name.capitalize()}, an AI assistant created by the Quantum Forge team. "
        f"If asked about your name or creator, reply exactly: "
        f"'I am {name.capitalize()}, created by the Quantum Forge team.' "
        f"Never mention OpenAI, Groq, Meta, or any other organization."
    )

# =====================================================
# MEMORY (CHATGPT STYLE)
# =====================================================
def fetch_history(supabase: Client, user_id: str, conversation_id: str):
    res = (
        supabase
        .table("chats")
        .select("role,message")
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )
    return res.data or []

def store_message(
    supabase: Client,
    user_id: str,
    conversation_id: str,
    role: str,
    message: str,
    model: str,
):
    supabase.table("chats").insert({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "role": role,
        "message": message,
        "model_used": model,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

# =====================================================
# CHAT ENDPOINT
# =====================================================
@app.post("/chat")
async def chat(req: ChatRequest):

    if req.model not in CHAT_MODELS:
        raise HTTPException(400, "Unknown model")

    conversation_id = req.conversation_id or str(uuid4())
    supabase: Client = app.state.supabase
    groq: AsyncGroq = app.state.groq

    history = fetch_history(supabase, req.user_id, conversation_id)

    messages = [
        {"role": "system", "content": identity_prompt(req.model)}
    ]

    for h in history:
        messages.append({"role": h["role"], "content": h["message"]})

    messages.append({"role": "user", "content": req.message})

    completion = await groq.chat.completions.create(
        model=CHAT_MODELS[req.model],
        messages=messages,
        temperature=0.7
    )

    reply = completion.choices[0].message.content

    store_message(supabase, req.user_id, conversation_id, "user", req.message, req.model)
    store_message(supabase, req.user_id, conversation_id, "assistant", reply, req.model)

    return {
        "reply": reply,
        "conversation_id": conversation_id,
        "model_used": req.model
    }

# =====================================================
# REGENERATE RESPONSE (GPT-STYLE)
# =====================================================
@app.post("/regenerate")
async def regenerate(req: RegenerateRequest):

    if req.model not in CHAT_MODELS:
        raise HTTPException(400, "Unknown model")

    supabase: Client = app.state.supabase
    groq: AsyncGroq = app.state.groq

    history = fetch_history(supabase, req.user_id, req.conversation_id)

    if not history:
        raise HTTPException(400, "No conversation found")

    # Remove last assistant reply
    trimmed = history[:-1]

    messages = [
        {"role": "system", "content": identity_prompt(req.model)}
    ]

    for h in trimmed:
        messages.append({"role": h["role"], "content": h["message"]})

    completion = await groq.chat.completions.create(
        model=CHAT_MODELS[req.model],
        messages=messages,
        temperature=0.9
    )

    reply = completion.choices[0].message.content

    store_message(
        supabase,
        req.user_id,
        req.conversation_id,
        "assistant",
        reply,
        req.model
    )

    return {
        "reply": reply,
        "conversation_id": req.conversation_id,
        "model_used": req.model
    }

# =====================================================
# MODELS LIST
# =====================================================
@app.get("/models")
def list_models():
    return list(CHAT_MODELS.keys())

# =====================================================
# HEALTH
# =====================================================
@app.get("/")
def health():
    return {"status": "Quantum Forge backend running"}
