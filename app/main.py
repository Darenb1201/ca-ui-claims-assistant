"""FastAPI server: POST /api/chat streams Server-Sent Events; / serves the chat UI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag import UIClaimsAssistant
from .retriever import BM25Retriever

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("KB_DIR", ROOT / "knowledge_base"))

app = FastAPI(title="California UI Claims Assistant")
retriever = BM25Retriever.from_directory(KB_DIR)
_assistant: UIClaimsAssistant | None = None


def assistant() -> UIClaimsAssistant:
    global _assistant
    if _assistant is None:  # create lazily so the server starts without a key
        _assistant = UIClaimsAssistant(retriever)
    return _assistant


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


@app.post("/api/chat")
def chat(req: ChatRequest):
    def events():
        for event in assistant().stream_answer(req.message, [t.model_dump() for t in req.history]):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/search")
def search(q: str, k: int = 5):
    """Debug endpoint: see what the retriever returns for a query."""
    return [
        {"id": c.id, "title": c.title, "section": c.section, "score": round(s, 3), "text": c.text}
        for c, s in retriever.search(q, k=min(k, 20))
    ]


@app.get("/healthz")
def healthz():
    return {"ok": True, "chunks": len(retriever.chunks)}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
