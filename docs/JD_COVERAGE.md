# How this project maps to the job description

Machine Learning Engineer III, AI Agent Engineer (Mount Sinai, requisition 3039692). Each line of the posting is marked one of three ways, and the marking is meant to be read literally.

- **Built and run:** exercised here, with tests or a command you can rerun.
- **Written, not run:** the files exist and are syntactically valid, but nothing here executed them against the real service.
- **Not covered:** not in the project. The last column says how to talk about it.

## Responsibilities

| Posting | Status | Where | Notes |
| --- | --- | --- | --- |
| Own design, development, deployment, governance and evolution of agent systems | Built and run (deployment: Docker run on a clean install, Terraform written, not run) | whole repo | Governance: PII redaction, canary token, read-only SQL guard, governed metric definitions |
| Agentic AI with LLMs, multi-agent systems, RAG, orchestration | Built and run, with a caveat on "multi-agent" | `agent/graph.py` | LangGraph workflow of specialized roles: planner (supervisor), SQL analyst, document researcher, analyst, evidence gate (verifier). The roles run in sequence. There is no peer-to-peer or parallel agent negotiation, and no CrewAI or AutoGen |
| Enterprise integrations | Not covered | | A synthetic SQLite database and document library stand in for a warehouse and a policy library. No EHR or Epic integration. Talk about the SQL guard and the governed metric layer as the integration pattern |
| Monitor and update deployed models | Built and run | `tracing/`, `guard/drift.py`, `guard/baseline.py`, Monitor tab, `docs/RUNBOOK.md` | Traces, drift simulator, golden baseline in CI, a runbook for triage |
| Technical leadership, standards, reviews, documentation | Built and run (the artifacts) | `CONTRIBUTING.md`, PR template, `ruff` in CI, `docs/RUNBOOK.md`, `experiments.md` | Mentorship itself cannot be shown by a repository. Bring a real example |
| Design agents that plan, reason, use tools | Built and run | `agent/` | Planning with abstain and clarify exits, tool use through a guarded SQL tool, retrieval, verified claims |
| Root cause of hallucinations, drift, workflow failures | Built and run | `guard/evaluators.py`, `docs/RUNBOOK.md` | Ten named failure classes with a first action for each |
| Research and development | Built and run | `experiments.md` | Hypothesis, dev and held-out counts, decision, including rejected changes |
| Documentation of methods, decisions and caveats | Built and run | README, `docs/`, `experiments.md` | |
| Query design, data integrity, integrated data warehouse | Built and run (small scale) | `agent/sql_guard.py`, `warehouse/` | A Spark mart verified against plain SQL and planted truths, 618 month and filter combinations |
| Process improvement | Built and run | the experiment decision rule in `guard/experiments.py` | |

## Qualifications

| Posting | Status | Where | Notes |
| --- | --- | --- | --- |
| LLM platforms: Azure AI Foundry, Azure OpenAI, OpenAI, Anthropic | Built, tested with fake clients, never called a real API | `agent/llm.py` | One interface, three providers, `python -m guard compare` runs identical cases against each. No API key was used. Azure AI Foundry is not covered separately |
| LangGraph | Built and run | `agent/graph.py` | |
| LangChain, Semantic Kernel, CrewAI, AutoGen | Not covered | | "Or similar" in the posting. Be ready to say what LangGraph's state graph does that they also do, and why the provider adapter is a thin custom layer |
| RAG with a vector database (Chroma) | Built and run | `agent/retrieval.py` | Chroma, stemmed TF-IDF, sibling version fetch, temporal selection. Pinecone, Azure AI Search, Elasticsearch, Weaviate not covered. The corpus is 28 short documents |
| End-to-end ML system and SDLC | Built and run | tests, CI, Docker, runbook, experiment log | 214 tests |
| Multiple languages and ML frameworks | Partly | Python, SQL, plus HCL and Groovy files that were not run | Frameworks used: scikit-learn, SciPy, PySpark. The project is Python-centric, and only that part is exercised |
| SQL and NoSQL | Built and run (SQL), tested against a stand-in (NoSQL) | `agent/sql_guard.py`, `tracing/mongo_store.py` | The MongoDB store is tested with mongomock, an in-memory fake, never a real server. Chroma is also a document store |
| Big Data, Apache Spark | Built and run, at small scale | `warehouse/` | PySpark in local mode: a metric cube over the visits table, and a synthetic scale check (5 million generated rows aggregated in about 4 seconds on this machine, a throughput check of the job, not hospital data). "Extensive experience" with big data is not something this project can show |
| Unix | Built and run | shell commands, Docker, CI | |
| Cloud: AWS, Azure or Google Cloud | Written, not run | `infra/terraform/main.tf`, `.github/workflows/deploy.yml` | Terraform for ECR, App Runner and Secrets Manager. It was not validated or applied. Do not claim cloud experience from this. Run `terraform validate` and `plan` first |
| Jenkins, Git, Azure DevOps, Terraform | Partly | `.github/workflows/ci.yml` (run locally as the same commands), `Jenkinsfile`, `azure-pipelines.yml`, `infra/terraform/` | Only the commands CI runs were executed here. The pipeline files themselves have never run on their platforms. The folder is not yet a Git repository: `git init` and push to see CI run |
| Proven history operating ML systems in production | Not covered | | A project cannot show this. Talk about real experience, and use the runbook as a way to describe how you operate |
| Leadership, communication, mentorship | Not covered | | Same. `CONTRIBUTING.md` and the documentation are evidence of how you write for a team, not of leading one |

## What would close the remaining gaps cheaply

1. Push to GitHub so the CI workflow runs once on a clean machine.
2. Run `terraform validate` and `plan`, and apply to a throwaway AWS account if you have one.
3. Run `python -m guard compare` with real keys for at least one provider.
4. Run `docker compose -f docker-compose.yml -f docker-compose.mongo.yml up` to exercise a real MongoDB.
