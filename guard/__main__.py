"""python -m guard [run | calibrate | drift] ... See --help."""

import argparse
import sys
from pathlib import Path

from guard import calibration, drift, report
from guard.metrics import summarize
from guard.runner import load_world, run_suite
from guard.simulated_model import DEFECTS, SimulatedModel
from guard.store import ResultStore


def _world(data_dir):
    data_dir = Path(data_dir)
    if not (data_dir / "agentguard.db").exists():
        from data import generate

        generate.main(data_dir)
    return load_world(data_dir)


def _llm(args, cases):
    if args.model == "simulated":
        defects = dict(item.split("=") for item in args.defects.split(",")) if args.defects else {}
        return SimulatedModel(cases, {k: float(v) for k, v in defects.items()})
    from agent.llm import get_llm

    return get_llm(cache_dir=".llm_cache", provider=args.model)


BEGIN, END = "<!-- experiments:begin -->", "<!-- experiments:end -->"


def _experiments(cases, docs, db, args):
    from guard.experiments import EXPERIMENTS, format_result, run_experiment

    text = "\n".join(format_result(run_experiment(e, cases, docs, db)) for e in EXPERIMENTS)
    if args.write:
        path = Path(args.write)
        body = path.read_text()
        head, rest = body.split(BEGIN)
        tail = rest.split(END)[1]
        path.write_text(f"{head}{BEGIN}\n\n{text}\n{END}{tail}")
    print(text)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="guard", description="AgentGuard: try to break the agent and report how.")
    p.add_argument("command", choices=["run", "calibrate", "drift", "experiments", "regress", "check", "compare"])
    p.add_argument("--data", default="data/generated")
    p.add_argument("--model", choices=["simulated", "anthropic", "openai", "azure"], default="simulated")
    p.add_argument("--providers", default="", help="compare: comma list of simulated, anthropic, openai, azure (default: every provider with credentials)")
    p.add_argument("--defects", default="", help="simulated only, e.g. wrong_denominator=0.5,hallucinate_number=0.2. Known: " + ", ".join(DEFECTS))
    p.add_argument("--split", choices=["dev", "heldout", "all"], default="all")
    p.add_argument("--results", default="results.db")
    p.add_argument("--report", default="")
    p.add_argument("--write", default="", help="experiments: rewrite the generated block of this markdown file")
    p.add_argument("--update", action="store_true", help="check: accept the current results as the new baseline")
    args = p.parse_args(argv)

    cases, docs, db = _world(args.data)
    if args.command == "compare":
        from agent.llm import available_providers, get_llm
        from guard import compare

        names = [n for n in args.providers.split(",") if n] or available_providers()
        if not names:
            print("No provider credentials found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY or AZURE_OPENAI_API_KEY (with "
                  "AZURE_OPENAI_ENDPOINT), or try --providers simulated.")
            return 2
        models = {n: (SimulatedModel(cases) if n == "simulated" else get_llm(cache_dir=".llm_cache", provider=n)) for n in names}
        only = None if args.split == "all" else {args.split}
        text = compare.render(compare.compare_models(cases, docs, db, models, only=only), cases)
        if args.report:
            Path(args.report).write_text(text)
        print(text)
        return 0
    if args.command == "regress":
        import tempfile

        from guard import regression

        with tempfile.TemporaryDirectory() as tmp:
            text = regression.render(regression.regression_matrix(cases, docs, db, Path(tmp)))
        print(text)
        return 0
    if args.command == "check":
        from guard import baseline

        llm = _llm(args, cases)
        outcomes = run_suite(cases, llm, docs, db)
        model = getattr(llm, "model", "unknown")
        if args.update:
            baseline.save(model, outcomes)
            print(f"baseline for {model} updated: {sum(o.passed for o in outcomes)} / {len(outcomes)} passing")
            return 0
        regressions, fixes, problems = baseline.check(model, outcomes)
        print(f"{sum(o.passed for o in outcomes)} / {len(outcomes)} passing. Regressions: {regressions or 'none'}. Fixes: {fixes or 'none'}.")
        for pr in problems:
            print("problem:", pr)
        return 1 if regressions or problems else 0
    if args.command == "experiments":
        return _experiments(cases, docs, db, args)
    if args.command == "calibrate":
        _, rows = calibration.matrix(cases, docs, db)
        text = report.render_calibration(rows)
    elif args.command == "drift":
        llm = _llm(args, cases)
        result = drift.stale_index_scenario(cases, llm, docs, db)
        fresh, stale = summarize(result["fresh"])["overall"], summarize(result["stale"])["overall"]
        text = (f"# Drift: stale index\n\nDocuments missing from the index: {', '.join(result['missing_docs'])}.\n\n"
                f"Fresh index: {fresh['passed']} / {fresh['total']}. Stale index: {stale['passed']} / {stale['total']}.\n"
                f"New failures: {', '.join(result['regressions']) or 'none'}.\n"
                f"Of those, answered with every claim verified (invisible to the evidence gate): {len(result['gate_saw_nothing'])} / {len(result['regressions'])}.\n")
    else:
        llm = _llm(args, cases)
        only = None if args.split == "all" else {args.split}
        outcomes = run_suite(cases, llm, docs, db, only=only)
        ResultStore(args.results).save_run("cli", getattr(llm, "model", "unknown"), "fresh index", outcomes)
        text = report.render(outcomes, cases, getattr(llm, "model", "unknown"))
    if args.report:
        Path(args.report).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
