from rank_bm25 import BM25Okapi
from duckduckgo_search import DDGS
from ackknow_a1.types import Source


class HybridSearcher:
    def __init__(self, web_enabled: bool = True, max_web_results: int = 5) -> None:
        self.web_enabled = web_enabled
        self.max_web_results = max_web_results
        self.local_docs = [
            "Ackknow A1 uses modular architecture with Unip IQA Confic Searcher Guider Store",
            "Ackknow supports multilingual Indian language routing and personalization",
            "Ackknow integrates Ollama for self-hosted inference and model routing"
        ]
        self.bm25 = BM25Okapi([d.lower().split() for d in self.local_docs])

    def search(self, query: str) -> list[Source]:
        scores = self.bm25.get_scores(query.lower().split())
        local = [Source(title=f"LocalDoc-{i}", score=float(s), snippet=self.local_docs[i]) for i, s in enumerate(scores)]
        local = sorted(local, key=lambda x: x.score, reverse=True)[:3]
        if not self.web_enabled:
            return local
        web_sources: list[Source] = []
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=self.max_web_results):
                web_sources.append(Source(title=row.get("title", "web"), url=row.get("href"), score=0.5, snippet=row.get("body", "")))
        return local + web_sources
