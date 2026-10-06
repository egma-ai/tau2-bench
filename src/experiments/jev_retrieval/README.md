# Jev retrieval experiment (`banking_knowledge`)

Controlled comparison of the `jev` retrieval config against existing configs. In `jev`, the
agent has the same single `KB_search(query)` tool as `openai_embeddings` / `bm25`, but each
search asks TypeSafe's Jev model, for every KB document in parallel, whether that document
helps answer the query. Documents with P(relevant) >= 0.5 are returned, best first, at most
10 (see `src/tau2/knowledge/retrievers/jev_retriever.py`). Relative to `openai_embeddings`,
the agent-visible difference is one sentence of the system prompt.

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

Same settings for every arm (leaderboard defaults: user simulator gpt-5.2 at low reasoning,
seed 300, all 97 tasks, 4 trials); only `--retrieval-config` changes:

```bash
for config in jev openai_embeddings golden_retrieval; do
  tau2 run --domain banking_knowledge --retrieval-config $config \
    --agent-llm <agent_model> --agent-llm-args '<agent_args_json>' \
    --user-llm gpt-5.2 --user-llm-args '{"reasoning_effort": "low"}' \
    --num-trials 4 --seed 300 --max-concurrency 20 --save-to jev_exp_$config
done
```

- `jev`: treatment.
- `openai_embeddings`: matched control (same tool, embedding backend).
- `alltools`: leaderboard reference; reuse published trajectories when the agent model has a
  leaderboard submission, else run it too (needs sandbox-runtime, see
  `src/tau2/knowledge/README.md`).
- `golden_retrieval`: optional ceiling (required documents given in the prompt).

With 4 trials, differences of roughly 5–6 points of pass^1 are detectable; analyse pass^1
paired by task. Run `required_doc_recall.py` on every arm's `results.json` as well.
