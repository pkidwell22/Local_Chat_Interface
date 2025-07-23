import httpx
import asyncio
import json  # <-- Add this

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
                if line.strip():  # skip blank lines
                    data = json.loads(line)  # <-- Fix here
                    full_output += data.get("response", "")
            return full_output
