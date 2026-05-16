from __future__ import annotations

from collections import defaultdict
from rank_bm25 import BM25Okapi
from duckduckgo_search import DDGS
from ackknow_a1.types import Source


class HybridSearcher:
    def __init__(self, web_enabled: bool = True, max_web_results: int = 5) -> None:
        self.web_enabled = web_enabled
        self.max_web_results = max_web_results
        self.local_docs = [
            "Ackknow A1 has native neural llm, iqA, confic, searcher, guider, store modules",
            "Ackknow supports India-focused multilingual UX including Hindi Hinglish Tamil",
            "Ackknow uses safety checks, memory, tools, and retrieval for robust responses",
            "Ackknow can run self-hosted with FastAPI and Docker",
        ]
        self.bm25 = BM25Okapi([d.lower().split() for d in self.local_docs])

    def _local(self, query: str) -> list[Source]:
        scores = self.bm25.get_scores(query.lower().split())
        rows = [Source(title=f"LocalDoc-{i}", score=float(s), snippet=self.local_docs[i], kind="bm25") for i, s in enumerate(scores)]
        return sorted(rows, key=lambda x: x.score, reverse=True)[:4]

    def _web(self, query: str) -> list[Source]:
        if not self.web_enabled:
            return []
        out: list[Source] = []
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=self.max_web_results):
                out.append(Source(title=row.get("title", "web"), url=row.get("href"), score=0.45, snippet=row.get("body", ""), kind="web"))
        return out

    def _rerank(self, query: str, items: list[Source]) -> list[Source]:
        qset = set(query.lower().split())
        rescored: list[Source] = []
        for s in items:
            overlap = len(qset.intersection(set((s.title + " " + s.snippet).lower().split())))
            s.score = float(s.score + overlap * 0.02)
            rescored.append(s)
        return sorted(rescored, key=lambda x: x.score, reverse=True)

    def search(self, query: str) -> list[Source]:
        merged = self._local(query) + self._web(query)
        reranked = self._rerank(query, merged)
        dedup: dict[str, Source] = {}
        for s in reranked:
            key = s.url or s.title
            if key not in dedup:
                dedup[key] = s
        return list(dedup.values())[:8]
