import os
import sys

import numpy as np
import requests
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI

app = FastAPI()
templates = Jinja2Templates(directory="templates")

LOCAL_EMBEDDER_URL = os.environ.get("LOCAL_EMBEDDER_URL", "http://localhost:8080")
REMOTE_EMBEDDER_URL = os.environ.get("REMOTE_EMBEDDER_URL", "http://localhost:4141")

print(f"Local embedder:  {LOCAL_EMBEDDER_URL}")
print(f"Remote embedder: {REMOTE_EMBEDDER_URL}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_local_embedding(url: str, text: str) -> list[float] | None:
    try:
        resp = requests.post(
            f"{url}/v1/embeddings",
            data='{"input": "' + text + '"}',
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Local embed error: {e}")
        return None


client = OpenAI(api_key="", base_url=REMOTE_EMBEDDER_URL)


def get_remote_embedding(text: str) -> list[float] | None:
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"Remote embed error: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/compare")
async def compare(text: str = Form(...)):
    local_emb = get_local_embedding(LOCAL_EMBEDDER_URL, text)
    remote_emb = get_remote_embedding(text)

    if local_emb is None:
        return {
            "error": f"Failed to get embedding from local server ({LOCAL_EMBEDDER_URL})"
        }
    if remote_emb is None:
        return {
            "error": f"Failed to get embedding from remote server ({REMOTE_EMBEDDER_URL})"
        }

    similarity = cosine_similarity(local_emb, remote_emb)

    return {
        "text": text,
        "local_embedding": local_emb[:10],
        "remote_embedding": remote_emb[:10],
        "similarity": similarity,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
