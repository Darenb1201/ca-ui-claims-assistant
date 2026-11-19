"""Retrieval-augmented answering with Claude."""
from __future__ import annotations

import os
import re
from collections.abc import Iterator

import anthropic

from .retriever import BM25Retriever, Chunk

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
TOP_K = int(os.getenv("RAG_TOP_K", "6"))

SYSTEM_PROMPT = """\
You are the California UI Claims Assistant, a friendly helper for people in California \
who have lost their job or had their hours cut and want to file for Unemployment \
Insurance (UI) with the Employment Development Department (EDD).

How to answer:
- Base your answer on the numbered EDD reference excerpts provided with each question. \
Cite them inline like [1] or [2]. If the excerpts do not cover the question, say you \
are not sure and point the person to edd.ca.gov or the EDD UI line (1-800-300-5616) \
instead of guessing. Never invent dollar amounts, deadlines, form numbers, or phone numbers.
- Use plain, warm language at about an 8th-grade reading level. Many people asking are \
stressed; be encouraging and practical.
- Prefer short numbered steps for "how do I" questions, and end with the single most \
useful next step.
- Reply in the same language the person writes in (for example, Spanish, Chinese, \
Vietnamese, Tagalog, Korean).
- You are not EDD and cannot see or change anyone's claim. You cannot decide whether \
someone is eligible; you can explain the rules and what EDD usually considers.
- Never ask for, or repeat back, a Social Security number, password, ID.me login, or \
bank details. If someone shares them, tell them not to share sensitive data in chat.
- If a question is about a complex dispute, fraud accusation, or appeal hearing, \
suggest free legal aid in addition to the general information.
- Stay on topic: California unemployment, related EDD programs (SDI, PFL), job search \
help, and safety-net resources for people out of work."""

SSN_RE = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")


def redact(text: str) -> str:
    """Strip SSN-shaped numbers before anything leaves the server."""
    return SSN_RE.sub("[REDACTED]", text)


def format_context(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (c, _) in enumerate(results, 1):
        blocks.append(
            f"[{i}] {c.title} — {c.section}\nSource: {c.source}\n{c.text}"
        )
    return "\n\n".join(blocks)


class UIClaimsAssistant:
    def __init__(self, retriever: BM25Retriever, client: anthropic.Anthropic | None = None):
        self.retriever = retriever
        self.client = client or anthropic.Anthropic()

    def retrieve(self, question: str, history: list[dict]) -> list[tuple[Chunk, float]]:
        # Include the previous user turn so follow-ups like "how long do I have?" still
        # retrieve the right topic.
        prev_user = [m["content"] for m in history if m["role"] == "user"][-1:]
        return self.retriever.search(" ".join(prev_user + [question]), k=TOP_K)

    def build_messages(self, question: str, history: list[dict], results) -> list[dict]:
        messages = [
            {"role": m["role"], "content": redact(m["content"])}
            for m in history[-10:]
            if m["role"] in ("user", "assistant") and m["content"].strip()
        ]
        context = format_context(results) or "(No matching reference excerpts found.)"
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<edd_reference_excerpts>\n{context}\n</edd_reference_excerpts>\n\n"
                    f"Question: {redact(question)}"
                ),
            }
        )
        return messages

    def stream_answer(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Yield {"type": "sources"|"text"|"done"|"error", ...} events."""
        history = history or []
        results = self.retrieve(question, history)
        yield {
            "type": "sources",
            "sources": [
                {"n": i, "title": c.title, "section": c.section, "url": c.source}
                for i, (c, _) in enumerate(results, 1)
            ],
        }
        try:
            with self.client.beta.messages.stream(
                model=MODEL,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                cache_control={"type": "ephemeral"},
                output_config={"effort": "medium"},
                # Retry on another model if a safety classifier declines the request.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=self.build_messages(question, history, results),
            ) as stream:
                for text in stream.text_stream:
                    yield {"type": "text", "text": text}
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield {
                    "type": "text",
                    "text": "\n\nI can't help with that one. For UI questions, visit edd.ca.gov or call 1-800-300-5616.",
                }
            yield {"type": "done"}
        except anthropic.RateLimitError:
            yield {"type": "error", "message": "The assistant is busy. Please try again in a minute."}
        except anthropic.APIStatusError as e:
            yield {"type": "error", "message": f"Upstream error ({e.status_code}). Please try again."}
        except anthropic.APIConnectionError:
            yield {"type": "error", "message": "Could not reach the AI service. Please try again."}
        except TypeError as e:
            if "authentication" not in str(e):
                raise
            yield {"type": "error", "message": "Server is missing ANTHROPIC_API_KEY. See README setup."}
