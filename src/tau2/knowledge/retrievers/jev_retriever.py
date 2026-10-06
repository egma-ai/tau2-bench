"""Exhaustive relevance-classification retriever backed by TypeSafe's Jev model.

Instead of ranking a precomputed index, every document is checked by Jev with an
independent yes/no ("noul") relevance question. Documents are sent in batches of
``batch_size``: one request carries the query plus ``batch_size`` documents in its
state and asks one question per document; batches run in parallel. Documents whose
probability of being relevant clears ``threshold`` are returned, highest probability
first (optionally capped at ``top_k``). API reference: https://docs.typesafe.ai/api
"""

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import httpx

from tau2.knowledge.registry import register_retriever
from tau2.knowledge.retrievers.base import BaseRetriever

# Override with TYPESAFE_BASE_URL, e.g. https://openrouter.ai/api for OpenRouter.
DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-1.13.0"
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}


def relevance_question(ref: str) -> Dict[str, Any]:
    """Noul question about the document at state path ``ref`` (e.g. ``documents[3]``).

    Same relevance definition as the pointwise LLM reranker's prompt.
    """
    return {
        "type": "noul",
        "instructions": (
            f"Does `{ref}` contain information that helps answer or address `query`?"
        ),
        "criteria": {
            "true": f"`{ref}` contains information that helps answer or address the query.",
            "false": (
                f"`{ref}` does not contain information that helps answer the query, "
                "even if it mentions similar topics."
            ),
        },
    }


class _RequestPacer:
    """Spaces out requests process-wide (tau2 runs simulations as threads).

    Each request waits for a slot sized by whichever limit it would hit first:
    requests per second or (estimated) input tokens per second.
    """

    def __init__(self, requests_per_s: float, tokens_per_s: float):
        self.requests_per_s = requests_per_s
        self.tokens_per_s = tokens_per_s
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def wait(self, est_tokens: float) -> None:
        cost = max(1.0 / self.requests_per_s, est_tokens / self.tokens_per_s)
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + cost
        time.sleep(slot - now)


# Jev's documented limits are 80 requests/s and 100K tokens/s per account; stay under.
_PACER = _RequestPacer(
    float(os.getenv("TYPESAFE_MAX_RPS", "75")),
    float(os.getenv("TYPESAFE_MAX_TPS", "90000")),
)


def _retry_delay(attempt: int, response: Optional[httpx.Response]) -> float:
    """Honor the server's retry-after headers, else use jittered exponential backoff."""
    if response is not None:
        for header, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
            try:
                return float(response.headers[header]) / scale
            except (KeyError, ValueError):
                pass
    return min(30.0, 2.0**attempt) * random.uniform(0.5, 1.0)


@register_retriever("jev")
class JevRetriever(BaseRetriever):
    def __init__(
        self,
        query_key: str = "query",
        content_state_key: str = "doc_content_map",
        top_k: Optional[int] = 10,
        threshold: float = 0.5,
        batch_size: int = 50,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrency: int = 32,
        max_retries: int = 8,
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(
            query_key=query_key,
            content_state_key=content_state_key,
            top_k=top_k,
            threshold=threshold,
            batch_size=batch_size,
            model=model,
            **kwargs,
        )
        self.query_key = query_key
        self.content_state_key = content_state_key
        self.top_k = top_k
        self.threshold = threshold
        self.batch_size = batch_size
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL", DEFAULT_JEV_MODEL)
        base_url = base_url or os.getenv("TYPESAFE_BASE_URL", DEFAULT_JEV_BASE_URL)
        self.api_url = f"{base_url.rstrip('/')}/v1/systemone"
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.client = httpx.Client(
            timeout=timeout, limits=httpx.Limits(max_connections=max_concurrency)
        )
        # (query, doc_id) -> P(relevant); repeated identical searches are free.
        self._cache: Dict[Tuple[str, str], float] = {}

    def _classify_batch(self, query: str, docs: List[Tuple[str, str]]) -> List[float]:
        """Return Jev's P(relevant) for each (title, content) document, in one request."""
        payload = {
            "model": self.model,
            "state": {
                "query": query,
                "documents": [{"title": t, "content": c} for t, c in docs],
            },
            "questions": {
                f"doc_{i}": relevance_question(f"documents[{i}]")
                for i in range(len(docs))
            },
        }
        est_tokens = len(json.dumps(payload, ensure_ascii=False)) / 3
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = None
        for attempt in range(self.max_retries + 1):
            response = None
            try:
                _PACER.wait(est_tokens)
                response = self.client.post(self.api_url, json=payload, headers=headers)
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    answers = response.json()["answers"]
                    return [
                        float(answers[f"doc_{i}"]["noul"]) for i in range(len(docs))
                    ]
            except httpx.TransportError:
                pass
            if attempt < self.max_retries:
                time.sleep(_retry_delay(attempt, response))
        status = response.status_code if response is not None else "transport error"
        raise RuntimeError(
            f"Jev request failed after {self.max_retries + 1} attempts ({status})"
        )

    def retrieve(
        self, input_data: Dict[str, Any], state: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        query = input_data.get(self.query_key, "")
        if not query.strip():
            return []

        contents = state[self.content_state_key]
        titles = state.get("doc_title_map", {})
        # Sorted so batch composition doesn't depend on filesystem order.
        pending = sorted(d for d in contents if (query, d) not in self._cache)
        batches = [
            pending[i : i + self.batch_size]
            for i in range(0, len(pending), self.batch_size)
        ]

        def classify(batch: List[str]) -> List[float]:
            docs = [(titles.get(doc_id, doc_id), contents[doc_id]) for doc_id in batch]
            return self._classify_batch(query, docs)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for batch, probs in zip(batches, executor.map(classify, batches)):
                for doc_id, prob in zip(batch, probs):
                    self._cache[(query, doc_id)] = prob

        results = [
            (doc_id, self._cache[(query, doc_id)])
            for doc_id in contents
            if self._cache[(query, doc_id)] >= self.threshold
        ]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results if self.top_k is None else results[: self.top_k]
