from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main
from app.rag import UIClaimsAssistant, redact
from app.retriever import BM25Retriever


def test_retrieval_finds_relevant_docs():
    r = BM25Retriever.from_directory("knowledge_base")
    assert r.search("how do I appeal a denied claim")[0][0].id.startswith("07_appeals")
    assert r.search("weekly benefit amount maximum")[0][0].id.startswith("04_benefit")
    assert r.search("documents needed to apply")[0][0].id.startswith("03_what_you_need")


def test_redacts_ssn():
    assert "123-45-6789" not in redact("my ssn is 123-45-6789")


class FakeStream:
    def __init__(self, captured): self.captured = captured
    def __enter__(self): return self
    def __exit__(self, *a): return False
    text_stream = iter(["You can ", "appeal [1]."])
    def get_final_message(self): return SimpleNamespace(stop_reason="end_turn")


class FakeClient:
    def __init__(self):
        self.captured = {}
        def stream(**kwargs):
            self.captured.update(kwargs)
            return FakeStream(self.captured)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))


def test_chat_endpoint_streams_sources_and_text():
    fake = FakeClient()
    main._assistant = UIClaimsAssistant(main.retriever, client=fake)
    client = TestClient(main.app)
    res = client.post("/api/chat", json={"message": "How do I appeal? SSN 123-45-6789", "history": []})
    body = res.text
    assert '"type": "sources"' in body and "appeal [1]." in body and '"type": "done"' in body
    prompt = fake.captured["messages"][-1]["content"]
    assert "<edd_reference_excerpts>" in prompt and "123-45-6789" not in prompt
    assert fake.captured["model"] == "claude-opus-5-5"
