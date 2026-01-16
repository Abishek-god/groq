import os
import traceback
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
# LOAD ENV (LOCAL + RENDER)
# =====================================================
load_dotenv()

# =====================================================
# ENV CONFIG
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # MUST be service role key

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")

# =====================================================
# MODEL REGISTRY (GROQ SUPPORTED ONLY)
# =====================================================
CHAT_MODELS = {
    "jarvis": "llama-3.3-70b-versatile",
    "friday": "llama-3.1-8b-instant",
    "vision": "qwen-2.5-32b",
    "swift": "llama-3.1-8b-instant",
    "compound": "groq/compound-mini"  # fallback
}

# =====================================================
# APP LIFECYCLE
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.groq = AsyncGroq(api_key=GROQ_API_KEY)
    app.state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    yield

app = FastAPI(title="Quantum Forge AI Backend", lifespan=lifespan)

# =====================================================
# CORS (LOCK TO YOUR FRONTEND DOMAINS)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500",
        "https://quantumforge-studio.onrender.com"
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
# HELPERS
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
    try:
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
        return list(reversed(res.data or []))
    except Exception as e:
        print("History fetch error:", e)
        return []

def store_message(
    supabase: Client,
    user_id: str,
    conversation_id: str,
    role: str,
    message: str,
    model: str
):
    try:
        supabase.table("chats").insert({
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": role,
            "message": message,
            "model_used": model,
            "created_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        print("DB insert error:", e)

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def health():
    return {
        "status": "Quantum Forge backend running",
        "version": "4.0.0"
    }

@app.get("/models")
def list_models():
    return list(CHAT_MODELS.keys())

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        if not GROQ_API_KEY:
            raise HTTPException(500, "GROQ_API_KEY not configured")

        if req.model not in CHAT_MODELS:
            raise HTTPException(
                400,
                f"Unknown model. Choose from: {list(CHAT_MODELS.keys())}"
            )

        conversation_id = req.conversation_id or str(uuid4())

        supabase: Client = app.state.supabase
        groq: AsyncGroq = app.state.groq

        # Fetch chat history
        history = fetch_history(supabase, req.user_id, conversation_id)

        # Build prompt chain
        messages = [{"role": "system", "content": identity_prompt(req.model)}]
        for h in history:
            messages.append({"role": h["role"], "content": h["message"]})
        messages.append({"role": "user", "content": req.message})

        # Call Groq (with fallback)
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

        if not completion.choices:
            raise HTTPException(502, "Empty AI response")

        reply = completion.choices[0].message.content or "No response generated"

        # Store messages (non-blocking)
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

    except HTTPException:
        raise
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(500, "Backend crash — check Render logs")

@app.post("/regenerate")
async def regenerate(req: RegenerateRequest):
    try:
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

        # Remove assistant messages
        trimmed = [h for h in history if h["role"] != "assistant"]

        messages = [{"role": "system", "content": identity_prompt(req.model)}]
        for h in trimmed[-10:]:
            messages.append({"role": h["role"], "content": h["message"]})

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

        if not completion.choices:
            raise HTTPException(502, "Empty AI response")

        reply = completion.choices[0].message.content or "No response generated"

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

    except HTTPException:
        raise
    except Exception:
        print(traceback.format_exc())
        raise HTTPException(500, "Backend crash — check Render logs")
