# California UI Claims Assistant

A retrieval-augmented (RAG) chatbot that helps people in California understand and file **Unemployment Insurance (UI)** claims with the EDD. It answers questions about eligibility, filing, what to have ready, benefit amounts, biweekly certification, payments, appeals, and related programs, and it cites the EDD sources it used.

> **Disclaimer:** This project is not affiliated with the California Employment Development Department (EDD). It gives general information, not legal advice. Users should confirm details at [edd.ca.gov](https://edd.ca.gov).

## How it works

```
question ──► BM25 retriever ──► top-k EDD excerpts ──► Claude (claude-opus-5-5) ──► streamed answer + [n] citations
               (knowledge_base/*.md)                  with grounding system prompt
```

- **Knowledge base:** `knowledge_base/*.md`. These are curated summaries of EDD guidance. Each file has a `source:` URL in its front matter.
- **Retriever:** `app/retriever.py`. It splits documents on headings and ranks chunks with BM25. It has no external dependencies and no vector database. Replace it with embeddings or a vector store if the corpus grows.
- **Generation:** `app/rag.py`. Claude answers only from the retrieved excerpts and cites them inline. It replies in the user's language and says when it isn't sure instead of guessing.
- **Safety:** SSN-shaped numbers are redacted on the server before anything is sent to the model. The bot never asks for SSNs, passwords, or bank details. Server-side refusal fallbacks are enabled.
- **API and UI:** `app/main.py` (FastAPI, Server-Sent Events) and `static/index.html` (a single-file chat UI with mobile and dark-mode support).

## Quick start

```bash
pip install -r requirements.txt
```

```bash
export ANTHROPIC_API_KEY=your-key
```

```bash
uvicorn app.main:app --reload
```

Then open http://localhost:8000.

### Docker

```bash
docker build -t ca-ui-assistant .
```

```bash
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key ca-ui-assistant
```

## Keeping the knowledge current

EDD rules change, so refresh the corpus from official pages:

```bash
python scripts/ingest_edd.py
```

This fetches the pages listed in `DEFAULT_URLS` (only `.ca.gov` URLs are accepted) into `knowledge_base/edd_live/`. That folder is git-ignored, so review the files before you commit or deploy them. The server indexes every `.md` file under `knowledge_base/` when it starts.

## API

| Endpoint | Description |
|---|---|
| `POST /api/chat` | `{"message": "...", "history": [{"role","content"}]}` → SSE events: `sources`, `text`, `done`, `error` |
| `GET /api/search?q=...` | Debug view of the chunks the retriever returns for a query |
| `GET /healthz` | Health check and chunk count |

## Configuration

| Env var | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | Required |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used for answers |
| `RAG_TOP_K` | `6` | Number of excerpts sent to the model |
| `KB_DIR` | `knowledge_base/` | Corpus directory |

## Tests

```bash
pytest -q
```

The tests cover retrieval quality, SSN redaction, and the streaming endpoint (with a mocked Claude client).

## Roadmap ideas

- Add the Spanish, Chinese, and Vietnamese EDD pages to the corpus.
- Add an eval set of real claimant questions with expected answers.
- Add rate limiting and abuse protection before public deployment.
- Add a hand-off to live help (EDD phone lines, legal aid by county).
