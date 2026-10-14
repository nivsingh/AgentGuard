# Experiment log

Every meaningful change gets an entry. Record what actually happened, including rejected changes.

Template:

## Experiment NNN

### Hypothesis
What you expect to improve, and why.

### Change
What exactly changed (for example, top K 3 to 5).

### Results
- Dev: X / 60 to Y / 60
- Held out: X / 40 to Y / 40
- Median latency: before to after
- Cost per question: before to after

### Decision
KEEP or REJECT

### Reason
One or two sentences.


---

## Experiment 000: retrieval baseline (no change, just a measurement)

### Setup
Plain vector retrieval, TF-IDF 1-2 gram embeddings, Chroma cosine, top K = 3.

### Results
- 10 hand-written questions (`python -m agent.retrieval_eval`)
- recall@1: 9 / 10
- recall@3: 10 / 10
- The one miss: "What is the follow-up window after discharge?" ranks D07 (Discharge Planning Guideline, which mentions follow-up) above the policy documents D01 and D02. Both policy versions appear in the top 3.

### Notes
- These 10 questions were written by the same person who wrote the corpus, so treat them as a development sanity set, not a held-out set.
- Retrieval returns both policy versions. Choosing the version whose effective date covers the question date is the agent's job (T9), not retrieval's.
- Decision: keep as the baseline. No reranker or hybrid retrieval until AgentGuard shows a measured need.


---

## Design decision 001: where retrieved text may go

### Problem
Writing correct SQL needs metric definitions (for example the readmission denominator), and those live in documents. Feeding retrieved documents to the SQL step would let untrusted text influence which queries run.

### Decision
Documents of type "standard" are governed content and form a trusted glossary for the SQL step. Everything else is untrusted and reaches only the analyst step, which has no tools.

### Consequence
The property "retrieved text cannot trigger a tool call" holds for every document except governed standards, which are treated as configuration. This is tested in `tests/test_agent.py`.


---

## Experiment 001: AgentGuard baseline (simulated reference model)

### Setup
All 100 cases, fresh index, top K = 3, simulated model with no defects. This is not a measurement of a real model. It measures the agent's plumbing (retrieval, selection, SQL guard, gate) when the model behaves perfectly, so whatever fails here is not the model's fault.

### Results
- Overall: 98 / 100. Dev 58 / 60, held out 40 / 40.
- Wrong answers delivered: 1. Safe failures: 1.
- IJ-003 (dev) `retrieval_miss`: "Where can visitors park?" returns no passage at all. The memo says "parking" and the question says "park", and the TF-IDF embedder does no stemming.
- CS-010 (dev) `stale_knowledge`: "What is the follow-up window right now?" retrieves D07, D21 and D01 but not D02, so the agent answers 30 days from D01 with a verified quote. The gate sees nothing wrong. Policy v2 never reached the agent.

### Notes
- Both findings are retrieval problems, found by ground truth and not by the gate. They are the first honest candidates for Session 6 experiments (stemming, hybrid retrieval, top K). Nothing was changed to hide them.
- Both are in dev. Any fix gets confirmed on the 40 held-out cases before it is kept.

---

## Experiment 002: evaluator calibration (simulated defects)

### Setup
Each of eight defects planted at rate 1.0 plus a stale index (D02 missing), compared with the baseline.

### Results
Every planted defect landed in its expected failure class and nowhere else (`python -m guard calibrate`). Examples: wrong denominator 9 / 9 cases, semantic SQL 46 changed cases (the rest are unaffected because the data ends in June, so the bad date boundary changes nothing), a made-up claim caught by the gate (no wrong answer delivered), a wrong number only in the summary not caught by the gate (delivered), stale index 7 new failures, all of them answered with every claim verified.

### Notes
- This calibrates the evaluators only. It is not evidence about any real model.
- **Possible gap to test with a real model:** the gate verifies claims, not the summary text. A summary that states a number no claim backs passes the gate and is caught only by the AgentGuard evaluator. If the real model does this, the fix (check summary numbers in the gate) is a Session 6 experiment: dev first, then held out.

---

## Design decision 002: why a simulated model

### Problem
No API key was available, but the evaluators needed testing before any real run.

### Decision
A simulated model with labeled, planted defects, used only to calibrate evaluators. Reports made with it carry a banner saying so, and the simulated results are never presented as agent performance.


---

## Experiments 003 to 008: what the first AgentGuard findings led to

Everything in this section was run by `python -m guard experiments --write experiments.md`. The block between the markers is regenerated from real runs, so the counts cannot drift from the code. All of it uses the simulated model, so it measures retrieval, temporal selection and the gate, not a real model.

Decision rule (`guard/experiments.py`): KEEP only if dev improves, held out does not get worse, and no previously passing case now fails.

How to read the results:
- A (stemming) fixes IJ-003. C (fetch sibling versions) fixes CS-010. Both are kept.
- B (top K 5) also fixes CS-010 and passes the rule, but it is not adopted. C fixes the cause (a policy version never reaches temporal selection) and B only makes it likelier, at the price of larger prompts. F confirms it: on top of A and C, B adds nothing.
- D is judged on wrong answers delivered, not pass count, because a rejected summary is still a failed case, just a safe one. 31 wrong answers reached the user without the check (19 dev, 12 held out) and 1 with it. The 1 left is a stale-policy answer, which is a different problem.
- E runs the kept changes together with no planted defects: no regressions, so the summary check rejected nothing correct. That is true for the simulated model only. A real model may write summaries with numbers it did not cite as claims, and the check would then abstain on them. Measure that rate with a real run before trusting the check in production.
- The held-out set was already 40 / 40, so it can show no harm but not a gain. A larger or harder held-out set is the honest next step.

<!-- experiments:begin -->

### A: stemming

Hypothesis: Related word forms (park, parking) should match, fixing the parking question.
Change: {'stem': True}

- Dev: 58 / 60 to 59 / 60
- Held out: 40 / 40 to 40 / 40
- Median latency per question: 2.14 ms to 2.33 ms (simulated model, so this is orchestration and retrieval only)
- Fixed: IJ-003. Regressed: none.
- Decision: **KEEP**. dev improved and held out did not get worse. Held out did not change (it may already be at its ceiling), so the gain is confirmed only as no harm.
- Adopted: yes.

### B: top K 3 to 5

Hypothesis: A larger candidate set should let the newer policy version reach the agent.
Change: {'k': 5}

- Dev: 58 / 60 to 59 / 60
- Held out: 40 / 40 to 40 / 40
- Median latency per question: 2.24 ms to 2.08 ms (simulated model, so this is orchestration and retrieval only)
- Fixed: CS-010. Regressed: none.
- Decision: **KEEP**. dev improved and held out did not get worse. Held out did not change (it may already be at its ceiling), so the gain is confirmed only as no harm.
- Adopted: no. C fixes the same case at its cause and B only makes it likelier, at the price of larger prompts. F shows B adds nothing once A and C are in.

### C: fetch sibling policy versions

Hypothesis: If one version of a policy is retrieved, fetching the others lets temporal selection choose correctly.
Change: {'expand_versions': True}

- Dev: 58 / 60 to 59 / 60
- Held out: 40 / 40 to 40 / 40
- Median latency per question: 2.12 ms to 2.12 ms (simulated model, so this is orchestration and retrieval only)
- Fixed: CS-010. Regressed: none.
- Decision: **KEEP**. dev improved and held out did not get worse. Held out did not change (it may already be at its ceiling), so the gain is confirmed only as no harm.
- Adopted: yes.

### D: summary number check in the gate (planted defect: wrong number only in the summary, 50 percent of cases)

Hypothesis: The gate should also reject numbers in the summary that no verified claim backs.
Change: {'check_summary': True}  Simulated defects: {'hallucinate_number': 0.5}

- Dev: 40 / 60 to 40 / 60
- Held out: 28 / 40 to 28 / 40
- Wrong answers delivered, dev / held out: 19 / 12 to 1 / 0
- Median latency per question: 2.05 ms to 1.99 ms (simulated model, so this is orchestration and retrieval only)
- Fixed: none. Regressed: none.
- Decision: **KEEP**. dev improved and held out did not get worse.
- Adopted: yes.

### E: everything kept so far, no planted defects

Hypothesis: A, C and D together should not break anything, and D should reject nothing that is correct.
Change: {'stem': True, 'expand_versions': True, 'check_summary': True}

- Dev: 58 / 60 to 60 / 60
- Held out: 40 / 40 to 40 / 40
- Median latency per question: 2.08 ms to 2.05 ms (simulated model, so this is orchestration and retrieval only)
- Fixed: CS-010, IJ-003. Regressed: none.
- Decision: **KEEP**. dev improved and held out did not get worse. Held out did not change (it may already be at its ceiling), so the gain is confirmed only as no harm.
- Adopted: yes.

### F: top K 3 to 5 on top of A and C

Hypothesis: If B still adds something once A and C are in, it is worth its larger prompts.
Change: {'k': 5}  Compared with: {'stem': True, 'expand_versions': True}

- Dev: 60 / 60 to 60 / 60
- Held out: 40 / 40 to 40 / 40
- Median latency per question: 2.17 ms to 2.29 ms (simulated model, so this is orchestration and retrieval only)
- Fixed: none. Regressed: none.
- Decision: **REJECT**. no measurable gain on dev.
- Adopted: no.

<!-- experiments:end -->
