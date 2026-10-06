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


def test_alltools_jev_swaps_only_the_dense_tool():
    from unittest.mock import MagicMock, patch

    from tau2.domains.banking_knowledge.data_model import TransactionalDB
    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    from tau2.domains.banking_knowledge.retrieval import (
        build_policy,
        build_tools,
        resolve_variant,
    )

    docs = [
        {"id": d, "title": STATE["doc_title_map"][d], "text": text}
        for d, text in STATE["doc_content_map"].items()
    ]
    with (
        patch(
            "tau2.domains.banking_knowledge.retrieval.get_or_create_docs",
            return_value=docs,
        ),
        patch("tau2.domains.banking_knowledge.retrieval._create_sandbox"),
    ):
        tools = build_tools(
            resolve_variant("alltools-jev"), MagicMock(spec=TransactionalDB), None
        )
    assert {"KB_search_bm25", "KB_search_jev", "shell"} <= set(tools.get_tools())
    assert not tools.has_tool("KB_search_dense")

    retriever = tools._kb_jev_pipeline.retrievers[0]
    retriever.api_key = "test-key"
    retriever.client = httpx.Client(transport=httpx.MockTransport(_answer))
    output = tools.KB_search_jev(query="annual fee", k=1)
    assert output.startswith("1. Gold Card Fees") and "Savings Rates" not in output

    kb = get_knowledge_base()
    alltools = build_policy(resolve_variant("alltools"), kb).splitlines()
    alltools_jev = build_policy(resolve_variant("alltools-jev"), kb).splitlines()
    changed = [(a, b) for a, b in zip(alltools, alltools_jev) if a != b]
    assert len(alltools) == len(alltools_jev) and len(changed) == 2
    assert all("KB_search_dense" in a and "KB_search_jev" in b for a, b in changed)
