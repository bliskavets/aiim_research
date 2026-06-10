"""
retriever.py — pluggable retrievers for Search-R1 rollouts.

Two implementations:
  - HTTPRetriever  — talks to the Search-R1 retrieval_server (FastAPI on
    http://127.0.0.1:8000/retrieve by default). Wire this in once the
    real E5+wiki-18 index is built.
  - StubRetriever — returns canned mock documents per query. For smoke-
    testing the multi-turn rollout pipeline without any retrieval infra.
"""
from __future__ import annotations

import json
from typing import List, Sequence

try:
    import requests  # type: ignore
except Exception:  # requests may not be in the slim image
    requests = None


class Retriever:
    """Abstract retriever. Subclasses return a list of `topk` doc strings."""

    def retrieve_batch(self, queries: Sequence[str], topk: int = 3) -> List[List[str]]:
        raise NotImplementedError

    def retrieve(self, query: str, topk: int = 3) -> List[str]:
        return self.retrieve_batch([query], topk=topk)[0]


class StubRetriever(Retriever):
    """Returns deterministic mock documents per query.

    Useful to validate the multi-turn rollout + reward + shaping pipeline
    end-to-end without standing up the real retrieval infrastructure.
    """

    def __init__(self, seed_text: str = "stub-wiki-passage"):
        self.seed_text = seed_text

    def retrieve_batch(self, queries: Sequence[str], topk: int = 3) -> List[List[str]]:
        out = []
        for q in queries:
            docs = [
                f"Doc {i+1}(Title: \"{self.seed_text}-{hash(q) % 10000:04d}-{i}\") "
                f"Stub passage related to: {q[:80]}"
                for i in range(topk)
            ]
            out.append(docs)
        return out


class HTTPRetriever(Retriever):
    """Talks to the Search-R1 retrieval_server FastAPI.

    Default endpoint mirrors the official `retrieval_launch.sh`:
        POST http://127.0.0.1:8000/retrieve
        body: {"queries": [...], "topk": 3, "return_scores": false}
        response: {"result": [[{"contents": "..."}, ...], ...]}
    """

    def __init__(self, url: str = "http://127.0.0.1:8000/retrieve",
                 timeout_s: float = 30.0):
        if requests is None:
            raise RuntimeError("The `requests` package is required for HTTPRetriever")
        self.url = url
        self.timeout_s = timeout_s

    def retrieve_batch(self, queries: Sequence[str], topk: int = 3) -> List[List[str]]:
        resp = requests.post(
            self.url,
            json={"queries": list(queries), "topk": int(topk), "return_scores": False},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
        # Expected shape: {"result": [[{"contents": "...", "title": "...", ...}, ...]]}
        out: List[List[str]] = []
        for per_q in payload["result"]:
            docs: List[str] = []
            for i, item in enumerate(per_q):
                if isinstance(item, dict):
                    title = item.get("title") or item.get("id") or ""
                    contents = item.get("contents") or item.get("text") or ""
                else:
                    title, contents = "", str(item)
                docs.append(f"Doc {i+1}(Title: \"{title}\") {contents}")
            out.append(docs)
        return out


def format_information_block(docs: Sequence[str]) -> str:
    """Format a list of doc strings into a Search-R1 <information> block.

    Mirrors the official format from search_r1/llm_agent/generation.py:
        <information>Doc 1(Title: ...) ...
        Doc 2(Title: ...) ...
        Doc 3(Title: ...) ...</information>
    """
    body = "\n".join(docs)
    return f"\n\n<information>{body}</information>\n\n"
