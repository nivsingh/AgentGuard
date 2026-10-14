# Contributing

Standards for changes to this repository. They are enforced by CI where they can be.

## Before you open a pull request

```bash
pip install -r requirements-dev.txt
ruff check .                  # style and likely bugs (enforced in CI)
python -m pytest -q           # every test (enforced in CI)
python -m guard check         # no case that passed may now fail (enforced in CI)
```

## Rules

1. **Ground truth before AI.** A new agent behavior needs a case whose expected answer is computed independently, in `guard/cases.py`.
2. **Every behavior change is an experiment.** Use `guard/experiments.py` and the rule there: dev improves, held out does not get worse, nothing regresses. Dev cases may be used to tune. Held-out cases may only confirm. Record the result in `experiments.md`, including rejections.
3. **Safeguards need a test that breaks them.** If you add a check, add a case to `guard/regression.py` or `guard/calibration.py` that shows AgentGuard noticing when it is missing.
4. **Retrieved text is untrusted.** It may only reach a step that has no tools. `tests/test_agent.py` and the injection evaluator enforce this.
5. **Report raw counts.** Write 36 / 40, not 90 percent. Never present simulated-model results as agent performance.
6. **Tests describe behavior.** Name tests as sentences, assert one idea, and prefer a mutation check over a coverage number: break the code on purpose and confirm a test fails.
7. **Document decisions, not just code.** Anything with a trade-off goes in `experiments.md` (measured) or as a short design decision (reasoned).

## Reviewing

Check that: the change has a case or an experiment; held-out results were not used to pick the change; no test was loosened to pass; the baseline in `guard/baselines/` changed only if you read and accepted every difference (`python -m guard check --update`); and the README still tells the truth about what is real and what is simulated.
