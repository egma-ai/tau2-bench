"""Unit tests for the Jev relevance-classification retriever (HTTP mocked)."""

from __future__ import annotations

import json

import httpx
import pytest

from tau2.knowledge.retrievers.jev_retriever import JevRetriever

STATE = {
    "doc_content_map": {
        "doc_fee": "The Gold card has a $95 annual fee.",
        "doc_apy": "Savings accounts earn 4.25% APY.",
        "doc_wire": "Wire transfers cost $25.",
    },
    "doc_title_map": {
        "doc_fee": "Gold Card Fees",
        "doc_apy": "Savings Rates",
        "doc_wire": "Wire Transfers",
    },
}
# P(relevant) the mock server returns, keyed by document title.
PROBS = {"Gold Card Fees": 0.97, "Savings Rates": 0.62, "Wire Transfers": 0.03}


def _retriever(handler, **kwargs) -> JevRetriever:
    retriever = JevRetriever(api_key="test-key", **kwargs)
    retriever.client = httpx.Client(transport=httpx.MockTransport(handler))
    return retriever


def _answer(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    prob = PROBS[body["state"]["document"]["title"]]
    return httpx.Response(
        200, json={"answers": {"relevant": {"type": "noul", "noul": prob}}}
    )


def test_returns_docs_above_threshold_ranked_by_probability():
    results = _retriever(_answer).retrieve({"query": "annual fee"}, STATE)
    assert results == [("doc_fee", 0.97), ("doc_apy", 0.62)]


def test_top_k_and_threshold_limit_results():
    assert _retriever(_answer, top_k=1).retrieve({"query": "fee"}, STATE) == [
        ("doc_fee", 0.97)
    ]
    assert _retriever(_answer, threshold=0.99).retrieve({"query": "fee"}, STATE) == []


def test_request_payload_shape():
    requests = []

    def handler(request):
        requests.append(request)
        return _answer(request)

    _retriever(handler).retrieve({"query": "annual fee"}, STATE)
    assert len(requests) == len(STATE["doc_content_map"])
    body = json.loads(requests[0].content)
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "jev-1.13.0"
    assert body["state"]["query"] == "annual fee"
    assert set(body["state"]["document"]) == {"title", "content"}
    assert body["questions"]["relevant"]["type"] == "noul"


def test_retries_rate_limited_requests():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after-ms": "0"})
        return _answer(request)

    results = _retriever(handler, max_concurrency=1).retrieve({"query": "fee"}, STATE)
    assert results[0] == ("doc_fee", 0.97)
    assert calls["n"] == len(STATE["doc_content_map"]) + 1


def test_non_retryable_error_raises():
    retriever = _retriever(lambda request: httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        retriever.retrieve({"query": "fee"}, STATE)


def test_repeated_query_is_served_from_cache():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _answer(request)

    retriever = _retriever(handler)
    first = retriever.retrieve({"query": "fee"}, STATE)
    assert retriever.retrieve({"query": "fee"}, STATE) == first
    assert calls["n"] == len(STATE["doc_content_map"])


def test_empty_query_makes_no_requests():
    retriever = _retriever(lambda request: pytest.fail("unexpected request"))
    assert retriever.retrieve({"query": "  "}, STATE) == []
