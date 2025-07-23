import json
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.ollama_client import query_ollama

# ─── Config ─────────────────────────────────────────────────────────────
CHATLOG_DIR = Path("backend/chatlogs")
CHATLOG_DIR.mkdir(exist_ok=True)

CHAT_DATA_FILE = Path("data/parsed_conversations.json")
parsed_messages = []

# ─── Load Cleaned ChatGPT History ───────────────────────────────────────
if CHAT_DATA_FILE.exists():
    with open(CHAT_DATA_FILE, encoding="utf-8") as f:
        try:
            all_convs = json.load(f)
            for conv in all_convs:
                parsed_messages.extend(conv.get("history", []))
        except Exception as e:
            print(f"⚠️ Failed to load parsed conversations: {e}")
else:
    print("⚠️ parsed_conversations.json not found — running without memory context.")

# ─── Save Chat Session ──────────────────────────────────────────────────
def save_chatlog(model: str, conversation: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CHATLOG_DIR / f"{model}_chat_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

# ─── Retrieve Context Messages ──────────────────────────────────────────
def retrieve_similar(user_prompt: str, k=4):
    if not isinstance(user_prompt, str):
        print(f"⚠️ Invalid prompt passed: {user_prompt}")
        return []

    scored = []
    for msg in parsed_messages:
        content = msg.get("content")
        if msg.get("role") == "user" and isinstance(content, str):
            score = SequenceMatcher(None, user_prompt.lower(), content.lower()).ratio()
            scored.append((score, msg))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]

# ─── FastAPI Setup ──────────────────────────────────────────────────────
app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    prompt: str

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print(f"🟡 Incoming request → model: {req.model}, prompt: {req.prompt}")

        similar = retrieve_similar(req.prompt)
        print(f"🔍 Retrieved {len(similar)} similar messages.")

        memory = "\n".join([f"User: {m['content']}" for m in similar])
        prompt_with_memory = f"{memory}\nUser: {req.prompt}\nAssistant:"

        response = await query_ollama(req.model, prompt_with_memory)

        convo = [{"role": "user", "content": req.prompt}, {"role": "assistant", "content": response}]
        save_chatlog(req.model, convo)

        return {"response": response}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ─── Static Frontend ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
