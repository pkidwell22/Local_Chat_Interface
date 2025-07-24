import httpx
import asyncio
import json

async def query_ollama(model: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", "http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": True
        }) as response:
            response.raise_for_status()
            full_output = ""
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        piece = data.get("response", "")
                        print(f"🧩 Stream chunk: {piece}")
                        full_output += piece
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e} on line: {line}")
            print(f"✅ Returning to client: {full_output}")
            return full_output

