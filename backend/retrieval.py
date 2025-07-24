import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

INDEX_PATH = "data/faiss_index.index"
MESSAGES_PATH = "data/faiss_user_messages.json"
RAW_DATA_PATH = "data/parsed_conversations.json"

index = None
encoder = None
user_messages = None  # List of dicts with "content", "topic", etc.


def initialize_retrieval():
    global index, encoder, user_messages

    print("🔄 Loading user messages and building FAISS index...")

    if os.path.exists(INDEX_PATH) and os.path.exists(MESSAGES_PATH):
        print("📦 Loading existing FAISS index and messages...")
        index = faiss.read_index(INDEX_PATH)
        with open(MESSAGES_PATH, "r", encoding="utf-8") as f:
            user_messages = json.load(f)
        encoder = SentenceTransformer("hkunlp/instructor-base")
        print(f"✅ Loaded {len(user_messages)} messages.")
        return

    # Build from scratch
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    encoder = SentenceTransformer("hkunlp/instructor-base")
    user_messages = []

    def store_msg(text, topic=None, source=None):
        user_messages.append({
            "content": text,
            "topic": topic or "general",
            "source": source or "raw"
        })

    # ─── Ingest formats ───────────────────────────────
    for entry in raw_data:
        if isinstance(entry, dict) and "history" in entry:
            for msg in entry["history"]:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    store_msg(msg["content"], source="chatlog")
        elif isinstance(entry, dict) and "content" in entry:
            store_msg(entry["content"], topic=entry.get("topic"), source=entry.get("source"))
        elif isinstance(entry, str):
            store_msg(entry)

    if not user_messages:
        raise ValueError("❌ No messages found to index.")

    embeddings = encoder.encode([msg["content"] for msg in user_messages])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    with open(MESSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(user_messages, f, indent=2)

    print(f"✅ Indexed {len(user_messages)} messages.")


def retrieve_similar(prompt, top_k=4, topic_filter=None, max_length=400):
    if index is None:
        raise RuntimeError("⚠️ Retrieval not initialized.")

    query_vec = encoder.encode([prompt])
    D, I = index.search(np.array(query_vec), top_k * 3)

    results = []
    for i in I[0]:
        msg = user_messages[i]
        content = msg["content"]
        if topic_filter and msg.get("topic") != topic_filter:
            continue
        if len(content) > max_length:
            continue
        results.append(content)
        if len(results) >= top_k:
            break

    return results


def add_new_messages(new_texts, topic="general", source="appended"):
    global index, user_messages
    if index is None or encoder is None:
        raise RuntimeError("⚠️ Retrieval not initialized.")

    new_records = [{
        "content": text,
        "topic": topic,
        "source": source
    } for text in new_texts]

    new_embeddings = encoder.encode([r["content"] for r in new_records])
    index.add(np.array(new_embeddings))
    user_messages.extend(new_records)

    faiss.write_index(index, INDEX_PATH)
    with open(MESSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(user_messages, f, indent=2)

    print(f"➕ Added {len(new_texts)} new messages.")


# ─── LangChain-compatible Retriever ─────────────────────────────
class StaticFAISSRetriever:
    def similarity_search(self, query, k=4):
        results = retrieve_similar(query, top_k=k)
        return [Document(page_content=txt) for txt in results]


loaded_faiss_index = StaticFAISSRetriever()
