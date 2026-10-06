"""Phase 0: replay real agent search queries from a tau2 banking results file through Jev.

No agent LLM is involved. For a sample of simulations we take every KB_search /
KB_search_bm25 / KB_search_dense query the agent issued, score every KB document with Jev,
and compare required-document recall of:
  - what the baseline agent actually got back from its search tools, vs.
  - Jev (P >= threshold, top_k cap) on the very same queries.
All Jev probabilities are written to a JSONL cache, so thresholds/caps can be
re-analysed afterwards without new API calls.

Usage (needs TYPESAFE_API_KEY):
  uv run python src/experiments/jev_retrieval/phase0_jev_offline.py RESULTS.json --sims 30
"""

import argparse
import collections
import json
import random
import re
import statistics
import time
from pathlib import Path

from tau2.domains.banking_knowledge.environment import get_knowledge_base, get_tasks
from tau2.knowledge.retrievers.jev_retriever import JevRetriever

SEARCH_TOOLS = ("KB_search", "KB_search_bm25", "KB_search_dense")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectories")
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--out", default="phase0.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=None, help="cap (default: none)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kb = get_knowledge_base()
    state = {
        "doc_content_map": {d.id: d.content for d in kb.get_all_documents()},
        "doc_title_map": {d.id: d.title for d in kb.get_all_documents()},
    }
    required = {t.id: set(t.required_documents or []) for t in get_tasks()}
    sims = json.load(open(args.trajectories))["simulations"]
    random.Random(args.seed).shuffle(sims)
    sims = [s for s in sims if required.get(s["task_id"])][: args.sims]

    # Cached probabilities from previous (partial) runs.
    out = Path(args.out)
    probs = {}
    if out.exists():
        for line in out.open():
            rec = json.loads(line)
            probs[rec["query"]] = rec["probs"]

    retriever = JevRetriever(threshold=0.0, top_k=10**6)
    rows, latencies = [], []
    for s in sims:
        queries, baseline_seen = [], set()
        tool_out = {
            m["id"]: str(m.get("content") or "")
            for m in s["messages"]
            if m.get("role") == "tool" and m.get("id")
        }
        for m in s["messages"]:
            for tc in m.get("tool_calls") or []:
                if tc["name"] in SEARCH_TOOLS:
                    queries.append(tc["arguments"].get("query", ""))
                    baseline_seen |= set(
                        re.findall(r"ID: (\S+)", tool_out.get(tc["id"], ""))
                    )
        jev_seen = set()
        for q in queries:
            if q not in probs:
                start = time.perf_counter()
                retriever.retrieve({"query": q}, state)
                latencies.append(time.perf_counter() - start)
                probs[q] = {
                    d: retriever._cache[(q, d)] for d in state["doc_content_map"]
                }
                with out.open("a") as f:
                    f.write(json.dumps({"query": q, "probs": probs[q]}) + "\n")
            ranked = sorted(probs[q].items(), key=lambda x: x[1], reverse=True)
            jev_seen |= {d for d, p in ranked[: args.top_k] if p >= args.threshold}
        req = required[s["task_id"]]
        rows.append(
            dict(
                task=s["task_id"],
                n_queries=len(queries),
                baseline_recall=len(req & baseline_seen) / len(req),
                jev_recall=len(req & jev_seen) / len(req),
            )
        )
        print(
            f"{s['task_id']}: {len(queries)} queries | required-doc recall "
            f"baseline search tools {rows[-1]['baseline_recall']:.2f} vs Jev {rows[-1]['jev_recall']:.2f}"
        )

    n_above = [sum(p >= args.threshold for p in v.values()) for v in probs.values()]
    print("\n=== Phase 0 summary ===")
    print(f"sims={len(rows)} unique queries scored={len(probs)}")
    print(
        f"mean required-doc recall from search tools: baseline {statistics.mean(r['baseline_recall'] for r in rows):.3f}"
        f" | Jev {statistics.mean(r['jev_recall'] for r in rows):.3f}"
    )
    print(
        f"docs with P>={args.threshold} per query: median {statistics.median(n_above)}, "
        f"p90 {sorted(n_above)[int(0.9 * len(n_above))]}, max {max(n_above)}"
    )
    if latencies:
        print(f"seconds per 698-doc search: median {statistics.median(latencies):.1f}")
    print(collections.Counter(round(p, 1) for v in probs.values() for p in v.values()))


if __name__ == "__main__":
    main()
