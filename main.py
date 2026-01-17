import os
import time
import asyncio
import httpx
from uuid import uuid4
from collections import defaultdict
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

# =====================================================
# ENVIRONMENT
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
RENDER_URL = os.getenv("RENDER_URL")
PORT = int(os.getenv("PORT", 10000))

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")

# =====================================================
# MODEL REGISTRY (GROQ)
# =====================================================
CHAT_MODELS = {
    "jarvis": "openai/gpt-oss-120b",
    "friday": "llama-3.3-70b-versatile",
    "vision": "llama-3.1-8b-instant",  # safe default
    "swift": "openai/gpt-oss-20b",
    "compound": "groq/compound-mini"
}

# =====================================================
# CONFIG
# =====================================================
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_REQUESTS_PER_MIN = 30
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2
MEMORY_LIMIT = 12  # how many past messages to inject

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Quantum Forge AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# =====================================================
# SUPABASE
# =====================================================
sb = None
if SUPABASE_URL and SUPABASE_KEY:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# =====================================================
# RATE LIMITING
# =====================================================
rate_limit = defaultdict(list)

def check_rate_limit(ip: str):
    now = time.time()
    window = 60
    rate_limit[ip] = [t for t in rate_limit[ip] if now - t < window]

    if len(rate_limit[ip]) >= MAX_REQUESTS_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    rate_limit[ip].append(now)

# =====================================================
# REQUEST MODEL
# =====================================================
class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    model: str
    conversation_id: Optional[str] = None
    stream: bool = False

# =====================================================
# HEALTH
# =====================================================
@app.get("/health")
async def health():
    return {
        "status": "online",
        "engine": "groq",
        "time": time.time()
    }

# =====================================================
# SAFE SUPABASE LOGGING
# =====================================================
def safe_log(data: dict):
    try:
        if sb:
            sb.table("chats").insert(data).execute()
    except Exception as e:
        print("Supabase log error:", e)

# =====================================================
# PROFILE FETCH
# =====================================================
def get_user_name(user_id: Optional[str]) -> str:
    if not sb or not user_id:
        return "User"

    try:
        result = sb.table("profiles") \
            .select("name") \
            .eq("id", user_id) \
            .single() \
            .execute()

        if result.data and result.data.get("name"):
            return result.data["name"]
    except Exception as e:
        print("Profile fetch error:", e)

    return "User"

# =====================================================
# CONVERSATION MEMORY
# =====================================================
async def get_conversation_context(conversation_id: Optional[str], limit=MEMORY_LIMIT) -> List[dict]:
    if not sb or not conversation_id:
        return []

    try:
        result = sb.table("chats") \
            .select("role,message") \
            .eq("conversation_id", conversation_id) \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()

        return result.data or []
    except Exception as e:
        print("Context fetch error:", e)
        return []

# =====================================================
# GROQ CALL (RETRY)
# =====================================================
async def call_groq(payload):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                res = await client.post(GROQ_URL, headers=headers, json=payload)
                res.raise_for_status()
                return res.json()
        except Exception as e:
            if attempt == RETRY_ATTEMPTS - 1:
                print("GROQ error:", e)
                raise HTTPException(status_code=502, detail="GROQ backend unavailable")
            await asyncio.sleep(RETRY_BACKOFF ** attempt)

# =====================================================
# GROQ STREAM
# =====================================================
async def stream_groq(payload):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", GROQ_URL, headers=headers, json=payload) as r:
            async for line in r.aiter_lines():
                if line:
                    yield f"data: {line}\n\n"

# =====================================================
# CHAT ENDPOINT
# =====================================================
@app.post("/chat")
async def chat(req: Request, body: ChatRequest):
    ip = req.client.host
    check_rate_limit(ip)

    conversation_id = body.conversation_id or str(uuid4())
    user_id = body.user_id or "anonymous"

    model_id = CHAT_MODELS.get(body.model, CHAT_MODELS["compound"])

    # Fetch user name
    user_name = get_user_name(user_id)

    # Log user message
    safe_log({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "role": "user",
        "message": body.message
    })

    # Load memory
    history = await get_conversation_context(conversation_id)

    # Build messages
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful AI assistant. The user's name is {user_name}. "
                       "You can remember and reference earlier messages in this conversation."
        }
    ]

    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["message"]
        })

    messages.append({
        "role": "user",
        "content": body.message
    })

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
        "stream": body.stream
    }

    # STREAM MODE
    if body.stream:
        return StreamingResponse(
            stream_groq(payload),
            media_type="text/event-stream"
        )

    # NORMAL MODE
    result = await call_groq(payload)
    reply = result["choices"][0]["message"]["content"]

    # Log AI reply
    safe_log({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "role": "ai",
        "message": reply
    })

    return JSONResponse({
        "response": reply,
        "conversation_id": conversation_id,
        "model": model_id
    })

# =====================================================
# RENDER UPTIME PROTECTION
# =====================================================
@app.on_event("startup")
async def keep_alive():
    async def ping():
        if not RENDER_URL:
            return
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    await client.get(f"{RENDER_URL}/health")
            except:
                pass
            await asyncio.sleep(300)  # every 5 minutes

    asyncio.create_task(ping())
