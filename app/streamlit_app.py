"""AgentGuard: Healthcare Operations Agent plus an evaluation system that tries to break it.

    streamlit run app/streamlit_app.py

Three tabs: Ask (with "Show my work"), Test Lab, Monitor. Synthetic data only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.services import EXAMPLES, Settings, Workspace, parse_experiments  # noqa: E402
from guard import report  # noqa: E402
from guard.metrics import summarize  # noqa: E402
from guard.simulated_model import DEFECTS  # noqa: E402

st.set_page_config(page_title="AgentGuard", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Preparing the synthetic dataset...")
def workspace():
    return Workspace()


ws = workspace()
STATUS_STYLE = {"answered": ("Answered", "green"), "abstained": ("Abstained", "orange"), "clarify": ("Asked a question", "blue")}


# ------------------------------------------------------------------------------------------------ sidebar
def sidebar():
    st.sidebar.title("AgentGuard")
    providers = ws.providers()
    labels = {"simulated": "Simulated (no API key)", "anthropic": "Anthropic", "openai": "OpenAI", "azure": "Azure OpenAI"}
    options = [*providers, "simulated"]
    mode_key = st.sidebar.radio("Model", options, index=0, format_func=labels.get,
                                help="Simulated is a stand-in that exists to test the evaluators. Real providers appear when their API keys are set.")
    defect = "none"
    if mode_key == "simulated":
        defect = st.sidebar.selectbox("Plant a defect", ["none", *DEFECTS], format_func=lambda d: "None (behaves correctly)" if d == "none" else f"{d}: {DEFECTS[d]}",
                                      help="Makes the simulated model misbehave in one known way, to show AgentGuard catching it.")
    stale = st.sidebar.checkbox("Stale document index (policy v2 missing)", help="The index was never refreshed after the 2026-07-01 policy change.")
    with st.sidebar.expander("Agent options"):
        from agent.config import RECOMMENDED

        opts = {
            "stem": st.checkbox("Stemming in retrieval", RECOMMENDED["stem"]),
            "expand_versions": st.checkbox("Fetch sibling policy versions", RECOMMENDED["expand_versions"]),
            "check_summary": st.checkbox("Gate checks numbers in the summary", RECOMMENDED["check_summary"]),
            "k": st.number_input("Documents retrieved (top K)", 1, 8, RECOMMENDED["k"]),
        }
    if mode_key == "simulated":
        st.sidebar.warning("Simulated model: results show whether the evaluators catch planted defects.")
    return Settings(mode_key, defect, stale, opts)


# ------------------------------------------------------------------------------------------------- Ask tab
def show_work(result):
    ev = result.evidence
    st.markdown("**1. Plan**")
    st.json(ev.get("plan", {}), expanded=False)
    if ev.get("redactions"):
        st.info(f"Removed identifiers from the question before any model saw it: {ev['redactions']}")
    st.markdown("**2. Database queries**")
    if not ev.get("queries"):
        st.caption("No query was run.")
    for i, q in enumerate(ev.get("queries", [])):
        st.caption(f"Query {i}: {q.get('purpose', '')}")
        st.code(q["sql"], language="sql")
        if q.get("error"):
            st.error(q["error"])
        else:
            st.dataframe(pd.DataFrame(q["rows"], columns=q["columns"]), hide_index=True)
    st.markdown("**3. Documents** (retrieved text is untrusted data and never reaches the step that writes queries)")
    if ev.get("documents"):
        st.dataframe(pd.DataFrame(ev["documents"]), hide_index=True)
    for d in ev.get("dropped_documents", []):
        st.caption(f"Dropped {d['doc_id']}: {d['reason']}")
    if not ev.get("documents") and not ev.get("dropped_documents"):
        st.caption("No documents were used.")
    st.markdown("**4. Claims, each re-checked by code**")
    if result.claims:
        st.dataframe(pd.DataFrame([{"": "✅" if c["verified"] else "❌", "claim": c.get("text", ""), "kind": c.get("kind"),
                                    "check": c.get("check", "")} for c in result.claims]), hide_index=True)
    else:
        st.caption("No claims.")
    st.markdown("**5. Evidence gate**")
    st.write(result.reason or "")
    if result.limitations:
        st.caption("Limitations: " + "; ".join(result.limitations))
    if result.confidence is not None:
        st.caption(f"Model confidence {result.confidence} (reported only; it never decides whether the agent answers).")
    st.markdown("**6. Trace**")
    st.dataframe(pd.DataFrame(result.trace)[["name", "latency_ms", "input_tokens", "output_tokens", "error"]], hide_index=True)


def ask_tab(settings):
    st.subheader("Ask the operations agent")
    label = {f"{cat}: {q}": q for cat, q in EXAMPLES}
    choice = st.selectbox("Try a question", list(label), help="Examples chosen to probe what the agent does when things get tricky.")
    typed = st.text_input("...or type your own", placeholder="Free-form questions need a real model.")
    question = typed.strip() or label[choice]
    if st.button("Ask", type="primary"):
        with st.spinner("Working..."):
            st.session_state["last"] = (question, *ws.ask(question, settings))
    if "last" not in st.session_state:
        return
    q, result, verdict = st.session_state["last"]
    name, color = STATUS_STYLE[result.status]
    st.markdown(f"**{q}**")
    st.markdown(f":{color}-badge[{name}]")
    st.write(result.answer)
    verified = sum(c["verified"] for c in result.claims)
    c1, c2, c3 = st.columns(3)
    c1.metric("Claims verified", f"{verified} / {len(result.claims)}")
    c2.metric("Latency", f"{result.totals['latency_ms']:.0f} ms")
    c3.metric("Tokens in / out", f"{result.totals['input_tokens']} / {result.totals['output_tokens']}")
    if typed.strip() and settings.mode == "simulated":
        st.caption("The simulated model only knows the built-in test questions. Choose one above, or set an API key for free-form questions.")
    if verdict is not None:
        if verdict.passed:
            st.success("AgentGuard check against ground truth: PASS")
        else:
            st.error(f"AgentGuard check against ground truth: FAIL ({verdict.failure_class}). {verdict.detail.rstrip('.')}. "
                     + ("A wrong answer reached the user." if verdict.unsafe else "The agent failed safely."))
    with st.expander("Show my work", expanded=False):
        show_work(result)


# --------------------------------------------------------------------------------------------- Test Lab tab
def render_summary(outcomes, model):
    s = summarize(outcomes)
    c = st.columns(5)
    c[0].metric("Passed", f"{s['overall']['passed']} / {s['overall']['total']}")
    for col, split in zip(c[1:3], ("dev", "heldout")):
        t = s["by_split"].get(split)
        col.metric("Dev" if split == "dev" else "Held out", f"{t['passed']} / {t['total']}" if t else "not run")
    c[3].metric("Wrong answers delivered", s["unsafe"], help="Failures where a wrong answer reached the user instead of an abstention.")
    c[4].metric("Paraphrase groups stable", f"{s['consistency']['stable']} / {s['consistency']['groups']}")
    left, right = st.columns(2)
    left.markdown("**By category**")
    left.dataframe(pd.DataFrame([{"category": k, "passed": f"{v['passed']} / {v['total']}"} for k, v in s["by_category"].items()]), hide_index=True)
    right.markdown("**Failure classes**")
    if s["failure_classes"]:
        right.bar_chart(pd.Series(s["failure_classes"]))
    else:
        right.success("No failures.")
    failed = [o for o in outcomes if not o.passed]
    if failed:
        by_id = {c.id: c for c in ws.cases}
        st.markdown("**Every failure, with its reason**")
        st.dataframe(pd.DataFrame([{"case": o.case_id, "split": o.split, "class": o.failure_class,
                                    "wrong answer delivered": "yes" if o.unsafe else "no", "question": by_id[o.case_id].question,
                                    "why": o.detail} for o in failed]), hide_index=True)
    st.download_button("Download report (Markdown)", report.render(outcomes, ws.cases, model), file_name="agentguard-report.md")


def lab_tab(settings):
    st.subheader("Test Lab: try to break the agent")
    st.caption("100 cases with independently computed answers: 60 for development, 40 held out for confirming a change.")
    if settings.mode == "simulated":
        st.info("Simulated model selected. Plant a defect in the sidebar and run the suite to watch AgentGuard find and name it.")
    if settings.defect == "hallucinate_number":
        st.caption("Tip: the gate's summary check is on, so this defect becomes safe abstentions. Turn it off under Agent options and run again to see the wrong answers get through.")
    split = st.radio("Cases to run", ["all", "dev", "heldout"], horizontal=True, format_func={"all": "All 100", "dev": "Dev (60)", "heldout": "Held out (40)"}.get)
    if settings.mode != "simulated":
        st.warning(f"This calls the {settings.mode} API about three times per case. Running all 100 cases is roughly 300 calls.")
    if st.button("Run AgentGuard", type="primary"):
        bar = st.progress(0.0, "Running cases...")
        run_id, outcomes, model = ws.run_lab(settings, split, progress=lambda i, n, o: bar.progress(i / n, f"{i} / {n} cases"))
        bar.empty()
        st.session_state["lab"] = (run_id, outcomes, model)
    if "lab" in st.session_state:
        run_id, outcomes, model = st.session_state["lab"]
        st.caption(f"Run {run_id}, model `{model}`. Saved: it appears in the Monitor tab.")
        render_summary(outcomes, model)
    st.divider()
    st.markdown("**Does AgentGuard itself work?**")
    st.caption("Both checks use the simulated model. They test the evaluators and the agent's safeguards, not a real model.")
    a, b = st.columns(2)
    if a.button("Evaluator calibration"):
        with st.spinner("Planting each known defect..."):
            st.session_state["calib"] = ws.calibration()
    if b.button("Break a safeguard (regression checks)"):
        with st.spinner("Breaking each safeguard in turn..."):
            st.session_state["regress"] = ws.regression()
    if "calib" in st.session_state:
        st.markdown("Each planted defect should land in its expected failure class and nowhere else.")
        st.dataframe(pd.DataFrame(st.session_state["calib"]).drop(columns=["description"]), hide_index=True)
    if "regress" in st.session_state:
        st.markdown("Each broken safeguard should be flagged.")
        st.dataframe(pd.DataFrame(st.session_state["regress"]), hide_index=True)


# ------------------------------------------------------------------------------------------------ Monitor tab
def monitor_tab(settings):
    st.subheader("Monitor")
    st.markdown(f"**Questions asked in this app** (trace store: {ws.traces.kind})")
    counts = ws.traces.status_counts()
    if counts:
        c = st.columns(4)
        c[0].metric("Questions", sum(counts.values()))
        c[1].metric("Answered", counts.get("answered", 0))
        c[2].metric("Abstained or asked", counts.get("abstained", 0) + counts.get("clarify", 0))
        c[3].metric("Median latency", f"{ws.traces.median_latency_ms():.0f} ms")
        st.caption("Slowest steps, averaged over all runs")
        st.dataframe(pd.DataFrame(ws.traces.slowest_steps(), columns=["step", "avg latency (ms)", "runs"]), hide_index=True)
    else:
        st.caption("Ask a question in the Ask tab and its trace shows up here.")

    st.markdown("**AgentGuard runs over time**")
    hist = ws.history()
    if hist:
        df = pd.DataFrame(hist)
        st.dataframe(df.drop(columns=["created"]), hide_index=True)
        if len(df) > 1:
            st.line_chart(df.set_index("run")[["passed", "wrong answers delivered"]])
        diff = ws.compare_last_two()
        if diff is not None:
            regressions, fixes = diff
            st.caption(f"Latest run against the one before it. New failures: {', '.join(regressions) or 'none'}. Fixed: {', '.join(fixes) or 'none'}.")
    else:
        st.caption("Run AgentGuard in the Test Lab and the results are kept here, so runs can be compared.")

    st.markdown("**Experiments**")
    log = (Path(__file__).resolve().parent.parent / "experiments.md").read_text()
    rows = parse_experiments(log)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander("Full experiment log"):
        st.markdown(log)

    st.markdown("**Drift simulator**")
    st.caption("A policy changed on 2026-07-01 but the document index was never refreshed. Does anything notice? "
               "Runs the 24 policy and consistency cases twice, with a fresh and a stale index.")
    if st.button("Simulate a stale index"):
        with st.spinner("Running both indexes..."):
            st.session_state["drift"] = ws.drift(settings)
    if "drift" in st.session_state:
        d = st.session_state["drift"]
        f, s = summarize(d["fresh"])["overall"], summarize(d["stale"])["overall"]
        c = st.columns(3)
        c[0].metric("Fresh index", f"{f['passed']} / {f['total']}")
        c[1].metric("Stale index", f"{s['passed']} / {s['total']}")
        c[2].metric("Wrong answers the evidence gate could not see", f"{len(d['gate_saw_nothing'])} / {len(d['regressions'])}",
                    help="They answered with every claim verified against a real (but out-of-date) document.")
        st.write("New failures: " + (", ".join(d["regressions"]) or "none"))
        if d["regressions"]:
            st.warning("The agent answered confidently from the old policy. Every claim checked out against a real document, so only a comparison with ground truth catches it.")


settings = sidebar()
tab_ask, tab_lab, tab_monitor = st.tabs(["Ask", "Test Lab", "Monitor"])
with tab_ask:
    ask_tab(settings)
with tab_lab:
    lab_tab(settings)
with tab_monitor:
    monitor_tab(settings)
