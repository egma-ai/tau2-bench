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


def test_no_cap_returns_every_doc_above_threshold():
    retriever = _retriever(_answer, top_k=None, threshold=0.5)
    assert retriever.retrieve({"query": "fee"}, STATE) == [
        ("doc_fee", 0.97),
        ("doc_apy", 0.62),
    ]


KB_SEARCH_DESCRIPTION = (
    "Search the knowledge base by having a zero-shot classifier model look at every "
    "document to check if it's relevant to the given question."
)


def _build(variant_name: str):
    from unittest.mock import MagicMock, patch

    from tau2.domains.banking_knowledge.data_model import TransactionalDB
    from tau2.domains.banking_knowledge.retrieval import build_tools, resolve_variant

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
            resolve_variant(variant_name), MagicMock(spec=TransactionalDB), None
        )
    retriever = tools._kb_pipeline.retrievers[0]
    retriever.api_key = "test-key"
    retriever.client = httpx.Client(transport=httpx.MockTransport(_answer))
    return tools


def test_jev_shell_has_classifier_search_and_shell_only():
    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    from tau2.domains.banking_knowledge.retrieval import build_policy, resolve_variant

    tools = _build("jev-shell")
    names = set(tools.get_tools())
    assert {"KB_search", "shell"} <= names
    assert not names & {"KB_search_bm25", "KB_search_dense", "grep"}
    schema = tools.get_tools()["KB_search"].openai_schema["function"]
    assert schema["description"] == KB_SEARCH_DESCRIPTION
    assert set(schema["parameters"]["properties"]) == {"query"}

    output = tools.KB_search(query="annual fee")
    assert output.startswith("1. Gold Card Fees") and "2. Savings Rates" in output
    assert "Wire Transfers" not in output

    policy = build_policy(resolve_variant("jev-shell"), get_knowledge_base())
    assert "You have two complementary ways" in policy
    assert KB_SEARCH_DESCRIPTION in policy and "### `shell`" in policy


def test_jev_has_classifier_search_without_shell():
    tools = _build("jev")
    assert tools.has_tool("KB_search") and not tools.has_tool("shell")
    assert (
        tools.get_tools()["KB_search"].openai_schema["function"]["description"]
        == KB_SEARCH_DESCRIPTION
    )
