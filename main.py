import os
import time
import asyncio
import httpx
from uuid import uuid4
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

# =====================================================
# ENV
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
# MODEL REGISTRY
# =====================================================
CHAT_MODELS = {
    "jarvis": "openai/gpt-oss-120b",
    "friday": "llama-3.3-70b-versatile",
    "vision": "qwen-2.5-32b",
    "swift": "llama-3.1-8b-instant",
    "compound": "groq/compound-mini"
}

# =====================================================
# CONFIG
# =====================================================
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_REQUESTS_PER_MIN = 30
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2

# =====================================================
# APP SETUP
# =====================================================
app = FastAPI(title="Quantum Forge AI Backend (GROQ)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
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
# MODELS
# =====================================================
class ChatRequest(BaseModel):
    user_id: str
    message: str
    model: str
    conversation_id: Optional[str] = None
    stream: bool = False

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
async def health():
    return {
        "status": "online",
        "engine": "groq",
        "time": time.time()
    }

# =====================================================
# GROQ CALL WITH RETRY
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
        except Exception:
            if attempt == RETRY_ATTEMPTS - 1:
                raise HTTPException(status_code=502, detail="GROQ backend unavailable")
            await asyncio.sleep(RETRY_BACKOFF ** attempt)

# =====================================================
# GROQ STREAM HANDLER
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

    model_id = CHAT_MODELS.get(body.model, CHAT_MODELS["compound"])

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": f"You are {body.model}, an AI assistant."},
            {"role": "user", "content": body.message}
        ],
        "temperature": 0.7,
        "stream": body.stream
    }

    conversation_id = body.conversation_id or str(uuid4())

    # Log user message
    if sb:
        sb.table("chats").insert({
            "user_id": body.user_id,
            "conversation_id": conversation_id,
            "role": "user",
            "message": body.message
        }).execute()

    # STREAM MODE
    if body.stream:
        return StreamingResponse(
            stream_groq(payload),
            media_type="text/event-stream"
        )

    # NORMAL MODE
    result = await call_groq(payload)
    reply = result["choices"][0]["message"]["content"]

    # Log AI message
    if sb:
        sb.table("chats").insert({
            "user_id": body.user_id,
            "conversation_id": conversation_id,
            "role": "ai",
            "message": reply
        }).execute()

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
