# backend/main.py

import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from backend.retrieval import initialize_retrieval
from backend.langchain_pipeline import get_chain  # ✅ Use LangChain pipeline now

# ─── Config ─────────────────────────────────────────────────────────────
MEMORY_DIR = Path(__file__).parent / "chat_memory"
MEMORY_DIR.mkdir(exist_ok=True)

app = FastAPI()

# ─── CORS for Frontend ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request Schema ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    model: str
    prompt: str
    session_id: str

# ─── Startup Event: Load FAISS ──────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    print("🚀 Starting backend and initializing semantic retrieval...")
    try:
        initialize_retrieval()
        print("✅ Retrieval system ready.")
    except Exception as e:
        print(f"❌ Retrieval failed to initialize: {e}")

# ─── POST /chat — LangChain Chain ───────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    print(f"🟡 Incoming → model: {req.model}, session: {req.session_id}")
    print(f"📝 Prompt: {req.prompt}")

    try:
        chain = get_chain(session_id=req.session_id, model=req.model)
        result = chain.invoke({"question": req.prompt})
        print(f"✅ Chain result: {result['answer']}")
        return {"response": result["answer"]}
    except Exception as e:
        print("❌ Error in /chat:", e)
        return {"error": str(e)}

# ─── Serve Frontend ─────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if not frontend_path.exists():
        raise RuntimeError(f"File at path {frontend_path} does not exist.")
    return FileResponse(frontend_path)
