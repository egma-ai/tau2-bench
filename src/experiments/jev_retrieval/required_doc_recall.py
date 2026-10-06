"""Did each task's required documents actually reach the agent? Diagnostic for tau2 banking results.

A required document counts as "in context" for a simulation when at least 90% of its
distinctive lines (lines that appear in no other KB document) occur somewhere in that
conversation's tool outputs, whichever tool produced them (KB_search*, shell, grep...).

Reports, per results file: pass^1, average tool calls, how often all required documents
were in context, how many failures happened with every required document in context, and
a within-task comparison (same task, trials with vs. without all required documents).

Usage: python required_doc_recall.py RESULTS.json [RESULTS.json ...]
"""

import collections
import json
import statistics
import sys
from pathlib import Path

DOMAIN_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "tau2"
    / "domains"
    / "banking_knowledge"
)
MIN_LINE_CHARS = 25
IN_CONTEXT_FRACTION = 0.9


def load_kb_lines() -> dict[str, list[str]]:
    """Distinctive lines per document (lines shared with another document are dropped)."""
    lines = {}
    for path in (DOMAIN_DIR / "documents").glob("*.json"):
        doc = json.loads(path.read_text())
        cleaned = [
            ln.strip().lstrip("-*#0123456789. ").strip()
            for ln in doc["content"].splitlines()
        ]
        lines[doc["id"]] = [ln for ln in cleaned if len(ln) >= MIN_LINE_CHARS] or [
            doc["content"].strip()[:200]
        ]
    owners = collections.Counter(
        ln for doc_lines in lines.values() for ln in set(doc_lines)
    )
    return {
        doc_id: [ln for ln in doc_lines if owners[ln] == 1] or doc_lines
        for doc_id, doc_lines in lines.items()
    }


def load_required_docs() -> dict[str, list[str]]:
    required = {}
    for path in (DOMAIN_DIR / "tasks").glob("task_*.json"):
        task = json.loads(path.read_text())
        required[task["id"]] = task.get("required_documents") or []
    return required


def analyze(results_path: str, kb_lines, required) -> None:
    sims = json.loads(Path(results_path).read_text())["simulations"]
    rows, tool_calls = [], collections.Counter()
    for sim in sims:
        tool_text = "\n".join(
            str(m.get("content") or "")
            for m in sim["messages"]
            if m.get("role") == "tool"
        )
        for m in sim["messages"]:
            tool_calls.update(tc["name"] for tc in m.get("tool_calls") or [])
        req = [d for d in required.get(sim["task_id"], []) if d in kb_lines]
        in_context = sum(
            sum(ln in tool_text for ln in kb_lines[d]) / len(kb_lines[d])
            >= IN_CONTEXT_FRACTION
            for d in req
        )
        rows.append(
            dict(
                task=sim["task_id"],
                success=(sim.get("reward_info") or {}).get("reward", 0.0) >= 1.0,
                all_in_context=in_context == len(req),
                recall=in_context / len(req) if req else 1.0,
            )
        )

    def rate(rs):
        return (
            f"{100 * sum(r['success'] for r in rs) / max(1, len(rs)):.1f}% of {len(rs)}"
        )

    n = len(rows)
    complete = [r for r in rows if r["all_in_context"]]
    incomplete = [r for r in rows if not r["all_in_context"]]
    failures = [r for r in rows if not r["success"]]
    print(f"== {results_path}: {n} simulations, pass^1 {rate(rows)}")
    print(
        "   tool calls per simulation:",
        {k: round(v / n, 1) for k, v in tool_calls.most_common(6)},
    )
    print(
        f"   mean required-doc recall {statistics.mean(r['recall'] for r in rows):.3f}; "
        f"all required docs in context in {len(complete)}/{n} simulations"
    )
    print(f"   success | all in context: {rate(complete)}")
    print(f"   success | some missing:   {rate(incomplete)}")
    print(
        f"   failures with every required doc in context: "
        f"{sum(r['all_in_context'] for r in failures)}/{len(failures)}"
    )
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    mixed = [
        rs for rs in by_task.values() if len({r["all_in_context"] for r in rs}) == 2
    ]
    print(
        f"   within-task ({len(mixed)} tasks with mixed recall): "
        f"success | all in context {rate([r for rs in mixed for r in rs if r['all_in_context']])}"
        f" vs some missing {rate([r for rs in mixed for r in rs if not r['all_in_context']])}"
    )


if __name__ == "__main__":
    kb_lines, required = load_kb_lines(), load_required_docs()
    for results_path in sys.argv[1:]:
        analyze(results_path, kb_lines, required)
