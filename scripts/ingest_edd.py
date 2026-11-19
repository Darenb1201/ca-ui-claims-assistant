"""Fetch official EDD pages and save them as Markdown in knowledge_base/edd_live/.

Run periodically to keep the knowledge base current:
    python scripts/ingest_edd.py
    python scripts/ingest_edd.py --urls https://edd.ca.gov/en/unemployment/ ...

Only official .ca.gov pages are accepted. Review the output before deploying.
"""
from __future__ import annotations

import argparse
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_URLS = [
    "https://edd.ca.gov/en/unemployment/eligibility/",
    "https://edd.ca.gov/en/unemployment/filing-a-claim/",
    "https://edd.ca.gov/en/unemployment/certify-for-benefits/",
    "https://edd.ca.gov/en/unemployment/appeals/",
    "https://edd.ca.gov/en/unemployment/faqs/",
]
OUT_DIR = Path(__file__).resolve().parent.parent / "knowledge_base" / "edd_live"
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript", "svg", "form"}
BLOCK_TAGS = {"p", "li", "div", "section", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6"}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif not self.skip and tag in {"h1", "h2", "h3", "h4"}:
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif not self.skip and tag == "li":
            self.out.append("\n- ")
        elif not self.skip and tag in BLOCK_TAGS:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(re.sub(r"\s+", " ", data))

    def markdown(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch(url: str) -> str:
    host = urlparse(url).hostname or ""
    if not host.endswith(".ca.gov"):
        raise ValueError(f"Refusing non-.ca.gov URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ca-ui-claims-assistant-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="*", default=DEFAULT_URLS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for url in args.urls:
        try:
            p = TextExtractor()
            p.feed(fetch(url))
        except Exception as e:  # keep going on individual failures
            print(f"FAILED {url}: {e}")
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", urlparse(url).path.lower()).strip("_") or "index"
        title = (p.title or slug).split("|")[0].strip()
        doc = f"---\ntitle: {title}\nsource: {url}\n---\n# {title}\n\n{p.markdown()}\n"
        (OUT_DIR / f"{slug}.md").write_text(doc, encoding="utf-8")
        print(f"saved {slug}.md ({len(doc)} chars)")


if __name__ == "__main__":
    main()
