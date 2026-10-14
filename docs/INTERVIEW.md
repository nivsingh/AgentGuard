# Talking points and known weaknesses

For a line-by-line map to the job posting, with what is built and run versus written but not run, see [JD_COVERAGE.md](JD_COVERAGE.md).

## What this shows

| Skill | Where it lives |
| --- | --- |
| Agent design with LangGraph | `agent/graph.py`: planner, data check, SQL analyst, document researcher, analyst, evidence gate, with routing that can stop early |
| Tool use with real constraints | `agent/sql_guard.py`: read-only connection, SQLite authorizer allowlist, row limit, timeout |
| Retrieval | `agent/retrieval.py`: Chroma, stemmed TF-IDF, sibling version fetch, temporal selection by effective date |
| Grounding and abstention | `agent/evidence.py`, `agent/tools.py`: every claim re-verified in code, abstain on missing data, tool failure or unverified claims |
| Prompt injection and governance | structural separation of untrusted text, canary token, PII redaction (`agent/pii.py`) |
| Evaluation | `guard/`: 100 cases, independent ground truth, deterministic evaluators, named failure classes, dev and held-out split |
| Evaluating the evaluators | `guard/calibration.py`, `guard/regression.py`: planted defects and broken safeguards |
| Observability | `tracing/`: spans with latency, tokens and cost, SQLite store, OpenTelemetry export |
| Multiple LLM providers | `agent/llm.py` (Anthropic, OpenAI, Azure OpenAI) and `python -m guard compare` |
| Big data | `warehouse/`: PySpark marts over a Parquet export, verified three ways against ground truth |
| SQL and NoSQL | SQLite for the agent's data, MongoDB for traces (`tracing/mongo_store.py`) |
| MLOps hygiene | seeded reproducible data, experiment log with decision rule, golden baseline in CI (`python -m guard check`), lint, Docker, runbook, 214 tests |

## Decisions worth explaining

- Ground truth before AI: planted facts declared once, answers recomputed with plain SQL.
- Structural safety over prompt-based safety: retrieved text cannot reach the step that can run queries, so the property holds even if a model is fooled.
- The gate checks claims, the evaluator checks truth. They catch different things, and the stale index shows why both exist.
- A change is kept only if dev improves, held out does not get worse, and nothing regresses. Experiment B passed that rule and was still not adopted, with the reason written down.
- The evaluator re-verifies claims itself instead of trusting the agent's "verified" flag. A regression check found that gap.

## Known weaknesses (say these first)

- The model in every result here is simulated. Nothing says how Claude performs as this agent until a key is used.
- The 100 cases were written by the same person who built the corpus. They are a strong development set and a modest held-out set. Held out is already at 40 / 40, so it cannot show gains.
- The summary number check might reject correct summaries from a real model. That rate is unmeasured.
- Retrieval is TF-IDF over 28 short documents with a crude stemmer. It proves the pipeline, not retrieval quality at scale.
- The synthetic dataset is simple. Real hospital data has missingness, coding drift and many more tables.
- Docker was verified by configuration check and a clean Python 3.11 install, not by a built image, because the build sandbox could not pull the base image.
- The OpenAI and Azure OpenAI adapters have only run against fake clients. The MongoDB store has only run against mongomock. The Terraform, Jenkinsfile, Azure DevOps pipeline and deploy workflow have never run.
- Spark ran in local mode on a laptop-sized dataset. It shows the job is correct and the code is Spark-shaped, not that I have operated a cluster.
