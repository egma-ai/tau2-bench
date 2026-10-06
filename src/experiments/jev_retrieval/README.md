# Jev retrieval experiment (`banking_knowledge`)

Does exhaustive relevance classification fix knowledge retrieval? In the `jev-shell` config
the agent keeps the leaderboard's read-only `shell` but its other search tools (BM25 and
OpenAI-embedding search) are replaced by one `KB_search(query)` tool: every KB document is
sent to TypeSafe's Jev model in parallel as a yes/no "is this relevant to the question?"
call, and every document with P(relevant) >= 0.5 is returned, most confident first, with no
cap (see `src/tau2/knowledge/retrievers/jev_retriever.py`). `jev` is the same search tool
without `shell`.

## Setup

```bash
uv sync --extra knowledge --extra dev
uv pip install websockets   # `import tau2` currently needs it even without the voice extra
```

Environment variables: `TYPESAFE_API_KEY` (Jev), `OPENAI_API_KEY` (user simulator, embeddings
control, NL-assertion judge) and the agent model's provider key. Optional:
`TYPESAFE_BASE_URL` / `TYPESAFE_DEFAULT_MODEL` to reach Jev through a gateway.

## 1. Baseline diagnostic (free)

Do the required documents reach the agent in published leaderboard runs? Download a
submission's banking results (directory names are under `web/leaderboard/public/submissions/`):

```bash
curl -o baseline.json "https://sierra-tau-bench-public.s3.amazonaws.com/submissions/<submission_dir>/trajectories/banking_knowledge_results.json"
python src/experiments/jev_retrieval/required_doc_recall.py baseline.json
```

## 2. Phase 0: replay real search queries through Jev (a few dollars, no agent cost)

```bash
uv run python src/experiments/jev_retrieval/phase0_jev_offline.py baseline.json --sims 30 --out phase0.jsonl
```

Compares required-document recall of Jev against what the baseline's search tools returned
for the same queries, and reports how many documents clear the threshold per query. Jev
probabilities are cached in `phase0.jsonl`, so re-runs and other thresholds are free. Use it
as a go/no-go check and to sanity-check the cap; do not tune settings on agent pass rates.

## 3. Agent runs

Leaderboard settings (user simulator gpt-5.2 at low reasoning, seed 300, all 97 tasks); only
`--retrieval-config` changes between arms. First run: GPT-6.1 Sol at xhigh effort (agent
arguments mirror the published GPT-5.6 Sol leaderboard run), one trial:

```bash
tau2 run --domain banking_knowledge --retrieval-config jev-shell \
  --agent-llm openai/responses/gpt-6.1-sol \
  --agent-llm-args '{"extra_body": {"reasoning_effort": "xhigh"}, "allowed_openai_params": ["tool_choice"]}' \
  --user-llm gpt-5.2 --user-llm-args '{"reasoning_effort": "low"}' \
  --num-trials 1 --seed 300 --max-concurrency 10 --save-to jev_shell_gpt-6.1-sol
```

- `jev-shell`: treatment (Jev `KB_search` + `shell`).
- `alltools`: baseline, the leaderboard config (BM25 + OpenAI-embedding search + `shell`);
  needed for models without a published leaderboard run.
- `jev`: Jev `KB_search` only.
- Jev calls are paced process-wide at `TYPESAFE_MAX_RPS` (default 75 requests/s).
- `shell` needs sandbox-runtime (see `src/tau2/knowledge/README.md`).

One trial detects differences of roughly 11+ points of pass^1; four trials roughly 5-6.
Run `required_doc_recall.py` on every arm's `results.json` as well.
