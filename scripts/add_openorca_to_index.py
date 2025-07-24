import os
import json
import faiss
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

# ─── Paths ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ← project root
INDEX_PATH = os.path.join(ROOT_DIR, "data", "faiss_index.index")
MESSAGES_PATH = os.path.join(ROOT_DIR, "data", "faiss_user_messages.json")
ORCA_PATH = os.path.join(ROOT_DIR, "data", "openorca")  # ← FIXED absolute path

# ─── Load OpenOrca Dataset ─────────────────────────────
print("📚 Loading OpenOrca from:", ORCA_PATH)
if not os.path.isdir(ORCA_PATH):
    raise RuntimeError(f"❌ Dataset path does not exist: {ORCA_PATH}")

dataset = load_from_disk(ORCA_PATH)
print("✅ Dataset loaded:", dataset)

# ─── Format: Combine prompt + response ─────────────────
all_messages = []
for example in dataset:  # ✅ FIXED — flat Dataset, not DatasetDict
    prompt = example.get("question", "").strip()
    answer = example.get("response", "").strip()
    if prompt and answer:
        all_messages.append(f"### Question:\n{prompt}\n\n### Answer:\n{answer}")

print(f"📝 Found {len(all_messages)} usable examples.")

# ─── Load Existing Index + Messages ────────────────────
if not os.path.exists(INDEX_PATH) or not os.path.exists(MESSAGES_PATH):
    raise RuntimeError("❌ FAISS index or messages not initialized. Run retrieval init first.")

index = faiss.read_index(INDEX_PATH)
with open(MESSAGES_PATH, "r", encoding="utf-8") as f:
    user_messages = json.load(f)

# ─── Embed and Add in Chunks ───────────────────────────
encoder = SentenceTransformer("hkunlp/instructor-base", device="cuda")
SAVE_EVERY = 100_000

for i in range(0, len(all_messages), SAVE_EVERY):
    batch = all_messages[i:i + SAVE_EVERY]
    print(f"🚀 Embedding batch {i // SAVE_EVERY + 1} of {len(all_messages) // SAVE_EVERY + 1}")

    embeddings = encoder.encode(
        batch,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    index.add(np.array(embeddings))
    user_messages.extend(batch)

    # Save after each batch
    faiss.write_index(index, INDEX_PATH)
    with open(MESSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(user_messages, f, indent=2)

    print(f"✅ Saved batch {i // SAVE_EVERY + 1} with {len(batch)} entries.")

print(f"✅ All {len(all_messages)} new examples added to index.")
