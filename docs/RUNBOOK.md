# Runbook: operating and updating the agent

How to keep the agent reliable after it ships. Everything here uses commands that exist in this repository.

## Routine

| When | Command | What you are looking for |
| --- | --- | --- |
| Every change | `python -m pytest -q` and `python -m guard check` | No test fails and no case that passed now fails (CI runs both) |
| Before changing retrieval, prompts or the gate | `python -m guard experiments` | Dev improves, held out does not get worse, nothing regresses. Log it in `experiments.md` |
| After switching or upgrading a model or provider | `python -m guard compare --providers anthropic,openai` | Same 100 cases, same agent: compare passed counts, wrong answers delivered, failure classes, latency, tokens |
| After any change to documents or policies | `python -m guard drift` | A stale index shows up as failures the evidence gate cannot see |
| Weekly | Monitor tab, or `TraceStore.status_counts()` | Share of answered, abstained and clarify runs; median latency; slowest step |
| After a metric definition changes | `python -m warehouse verify` | The Spark marts still match plain SQL and the planted truths |

## Triage by failure class

Start from the class AgentGuard assigned, not from the symptom. "Wrong answer delivered" means it reached the user; fix those first.

| Class | What it means | First thing to check | Typical fix |
| --- | --- | --- | --- |
| `retrieval_miss` | The right document never reached the agent | Show my work, documents section: what was retrieved? Does the query share vocabulary with the document? | Embedder or tokenization (stemming fixed one), top K, hybrid search. Measure on dev, confirm on held out |
| `stale_knowledge` | The answer used a superseded policy | Index contents against document versions. Was a newer version indexed? | Re-index. Sibling version fetch makes temporal selection robust. The gate cannot see this, only ground truth can |
| `sql_error` | The query was invalid or rejected | The query and error in Show my work | Schema description in the prompt, metric glossary, query rules |
| `sql_semantic` | The query ran and computed the wrong thing | Compare the SQL with the governed definition. Off by one date boundary? Missing filter? | Prompt rules and examples; add a case that pins the behavior |
| `wrong_denominator` | The answer used the naive metric | The glossary document that defines the metric: was it in the SQL step's context? | Governed standards in the SQL prompt; add paraphrases to the denominator cases |
| `hallucination` | A number or fact the evidence does not support | Was the number in a claim? Did the gate check it? Is the summary check on? | Keep the gate's summary check on; tighten analyst prompt |
| `injection` | Retrieved text steered the agent or a secret leaked | Did retrieved text reach a step that can call tools? Was the canary in the output? | Structure, not wording: keep untrusted text away from tool-calling steps |
| `wrong_abstain` | The agent refused or asked when it could have answered | The gate's reason string | Usually an over-strict check. Measure false rejects before loosening |
| `data_integrity` | The database changed during a run | Stop. Check the SQL guard configuration | Restore the database. This should never happen with the read-only guard |
| `run_error` | The run crashed (network, quota, bug) | The error string in the outcome | Infrastructure, not model behavior. Rerun before drawing conclusions |

## Model drift

1. Run `python -m guard run --model <provider>` on the current model and `python -m guard check --update` only after you have read and accepted every difference.
2. If passed counts fall, `python -m guard compare` against the previous model or configuration, and read the failures by class.
3. If the data changed rather than the model, rerun `python -m warehouse verify` and `python -m data.generate` checks to see which planted truth no longer holds.

## What this does not cover

Production alerting, paging, and access control are not built. The Monitor tab and trace store are the data those would be built on.
