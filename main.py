import os
from datetime import datetime, timezone
from uuid import uuid4
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq
from supabase import create_client, Client
from dotenv import load_dotenv

# =====================================================
# LOAD ENV (LOCAL DEV SUPPORT)
# =====================================================
load_dotenv()

# =====================================================
# ENV CONFIG
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not all([GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise RuntimeError("Missing env vars: GROQ_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY")

# =====================================================
# MODEL REGISTRY (FULL GROQ ENTERPRISE SET)
# =====================================================
CHAT_MODELS = {
    # 🔥 Premium / Research
    "jarvis": "openai/gpt-oss-120b",

    # ⚖️ Standard / General
    "friday": "llama-3.3-70b-versatile",
    "maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",

    # 🧠 Logic / Reasoning
    "vision": "qwen/qwen3-32b",
    "atlas": "openai/gpt-oss-20b",

    # ✍️ Creative
    "kimi": "moonshotai/kimi-k2-instruct",
    "kimi-pro": "moonshotai/kimi-k2-instruct-0905",

    # ⚡ Fast / Cheap
    "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "swift": "llama-3.1-8b-instant",

    # 🌍 Regional
    "allam": "allam-2-7b",

    # 🛟 Always-Available Fallback
    "compound": "groq/compound-mini",
}

# =====================================================
# APP LIFECYCLE
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.groq = AsyncGroq(api_key=GROQ_API_KEY)
    app.state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    yield
    try:
        await app.state.groq.close()
    except Exception:
        pass

app = FastAPI(title="Quantum Forge AI – Backend", lifespan=lifespan)

# =====================================================
# CORS (LOCK THIS TO YOUR FRONTEND IN PROD)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500",
        "https://quantumforge-studio.onrender.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# SCHEMAS
# =====================================================
class ChatRequest(BaseModel):
    user_id: str
    message: str
    model: str = "friday"
    conversation_id: Optional[str] = None

class RegenerateRequest(BaseModel):
    user_id: str
    conversation_id: str
    model: str = "friday"

# =====================================================
# CORE FUNCTIONS
# =====================================================
def identity_prompt(name: str) -> str:
    return (
        f"You are {name.capitalize()}, an AI assistant created by the Quantum Forge team. "
        f"If asked about your name or creator, reply exactly: "
        f"'I am {name.capitalize()}, created by the Quantum Forge team.'"
    )

def fetch_history(
    supabase: Client,
    user_id: str,
    conversation_id: str,
    limit: int = 10
):
    res = (
        supabase
        .table("chats")
        .select("role,message")
        .eq("user_id", user_id)
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    data = res.data or []
    return list(reversed(data))

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
        "created_at": datetime.now(timezone.utc).isoformat()
    }).execute()

# =====================================================
# ROUTES
# =====================================================
@app.post("/chat")
async def chat(req: ChatRequest):
    if req.model not in CHAT_MODELS:
        raise HTTPException(
            400,
            f"Unknown model. Choose from: {list(CHAT_MODELS.keys())}"
        )

    conversation_id = req.conversation_id or str(uuid4())
    supabase: Client = app.state.supabase
    groq: AsyncGroq = app.state.groq

    # Get history (last N messages)
    history = fetch_history(supabase, req.user_id, conversation_id)

    # Build prompt chain
    messages = [{"role": "system", "content": identity_prompt(req.model)}]
    for h in history:
        messages.append({"role": h["role"], "content": h["message"]})
    messages.append({"role": "user", "content": req.message})

    # Call AI with fallback
    try:
        completion = await groq.chat.completions.create(
            model=CHAT_MODELS[req.model],
            messages=messages,
            temperature=0.7
        )
    except Exception:
        completion = await groq.chat.completions.create(
            model=CHAT_MODELS["compound"],
            messages=messages,
            temperature=0.7
        )

    reply = completion.choices[0].message.content

    # Store conversation
    store_message(
        supabase, req.user_id, conversation_id,
        "user", req.message, req.model
    )
    store_message(
        supabase, req.user_id, conversation_id,
        "assistant", reply, req.model
    )

    return {
        "reply": reply,
        "conversation_id": conversation_id,
        "model_used": req.model
    }

@app.post("/regenerate")
async def regenerate(req: RegenerateRequest):
    if req.model not in CHAT_MODELS:
        raise HTTPException(
            400,
            f"Unknown model. Choose from: {list(CHAT_MODELS.keys())}"
        )

    supabase: Client = app.state.supabase
    groq: AsyncGroq = app.state.groq

    history = fetch_history(
        supabase,
        req.user_id,
        req.conversation_id,
        limit=20
    )

    if not history:
        raise HTTPException(400, "No conversation found")

    # Remove previous assistant messages
    trimmed = [h for h in history if h["role"] != "assistant"]

    messages = [{"role": "system", "content": identity_prompt(req.model)}]
    for h in trimmed[-10:]:
        messages.append({"role": h["role"], "content": h["message"]})

    # Call AI with fallback
    try:
        completion = await groq.chat.completions.create(
            model=CHAT_MODELS[req.model],
            messages=messages,
            temperature=0.9
        )
    except Exception:
        completion = await groq.chat.completions.create(
            model=CHAT_MODELS["compound"],
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

@app.get("/models")
def list_models():
    return list(CHAT_MODELS.keys())

@app.get("/")
def health():
    return {
        "status": "Quantum Forge backend running",
        "version": "4.0.0"
    }

        "status": "Quantum Forge backend running",
        "version": "4.0.0"
    }

