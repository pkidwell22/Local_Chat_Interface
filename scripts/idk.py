# scripts/idk.py
import json
from pathlib import Path

# Dynamically resolve the absolute path to parsed_conversations.json
data_path = Path(__file__).resolve().parent.parent / "data" / "parsed_conversations.json"

# Load and validate the structure
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, convo in enumerate(data):
    if not isinstance(convo, dict):
        print(f"❌ Conversation #{i} is not a dict: {type(convo)}")
    elif "history" not in convo:
        print(f"❌ Conversation #{i} missing 'history'")
    elif not isinstance(convo["history"], list):
        print(f"❌ Conversation #{i} 'history' is not a list")
