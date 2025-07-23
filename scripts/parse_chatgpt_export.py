import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXPORT_PATH = BASE_DIR / "data" / "conversations.json"
PARSED_OUTPUT_PATH = BASE_DIR / "data" / "parsed_conversations.json"


def parse_conversations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = []

    for convo in data:
        title = convo.get("title", "Untitled")
        mapping = convo.get("mapping", {})
        history = []

        for node in mapping.values():
            msg = node.get("message")
            if not msg:
                continue
            role = msg.get("author", {}).get("role")
            content_parts = msg.get("content", {}).get("parts", [])
            content = content_parts[0] if content_parts else ""
            if role and content:
                history.append({"role": role, "content": content})

        if history:
            parsed.append({"title": title, "history": history})

    return parsed


if __name__ == "__main__":
    all_conversations = parse_conversations(EXPORT_PATH)

    with open(PARSED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"✅ Parsed {len(all_conversations)} conversations.")
    print(f"💾 Saved to: {PARSED_OUTPUT_PATH}")
    print(json.dumps(all_conversations[0], indent=2))
