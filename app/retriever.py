"""Lightweight BM25 retriever over the Markdown knowledge base.

No external vector DB or embedding service is required: documents are split
into heading-aware chunks and ranked with Okapi BM25. Swap this class out for a
vector store later without touching the rest of the app.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

STOPWORDS = set(
    "a an and are as at be by for from has have how i if in is it its my of on or "
    "that the this to was were what when where which who will with you your can do "
    "does me we our they them".split()
)
TOKEN_RE = re.compile(r"[a-z0-9$]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    title: str
    section: str
    source: str
    text: str
    tokens: list[str] = field(default_factory=list, repr=False)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        for line in header.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        return meta, body
    return meta, raw


def chunk_markdown(path: Path, max_chars: int = 1200) -> list[Chunk]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    title = meta.get("title", path.stem)
    source = meta.get("source", "")
    chunks: list[Chunk] = []
    section, buf = title, []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        # Split oversized sections on paragraph boundaries.
        parts, cur = [], ""
        for para in text.split("\n\n"):
            if cur and len(cur) + len(para) > max_chars:
                parts.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}" if cur else para
        parts.append(cur)
        for p in parts:
            cid = f"{path.stem}#{len(chunks)}"
            chunks.append(Chunk(cid, title, section, source, p.strip()))

    for line in body.splitlines():
        if line.startswith("#"):
            flush()
            buf = []
            section = line.lstrip("#").strip()
        else:
            buf.append(line)
    flush()
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        for c in chunks:
            # Index title + section heading too, so "appeal" matches the appeals doc.
            c.tokens = tokenize(f"{c.title} {c.section} {c.section} {c.text}")
        self.doc_freq: Counter[str] = Counter()
        for c in chunks:
            self.doc_freq.update(set(c.tokens))
        self.avg_len = sum(len(c.tokens) for c in chunks) / max(len(chunks), 1)
        self.n = len(chunks)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "BM25Retriever":
        chunks: list[Chunk] = []
        for path in sorted(Path(directory).glob("**/*.md")):
            chunks.extend(chunk_markdown(path))
        if not chunks:
            raise RuntimeError(f"No Markdown documents found in {directory}")
        return cls(chunks)

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q_terms = tokenize(query)
        scored = []
        for c in self.chunks:
            tf = Counter(c.tokens)
            length_norm = self.k1 * (1 - self.b + self.b * len(c.tokens) / self.avg_len)
            score = sum(
                self._idf(t) * tf[t] * (self.k1 + 1) / (tf[t] + length_norm)
                for t in q_terms
                if t in tf
            )
            if score > 0:
                scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
