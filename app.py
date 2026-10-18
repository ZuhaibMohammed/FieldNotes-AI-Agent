"""Streamlit interface for the AI Research Agent."""

from __future__ import annotations

import html
import os
import queue
import threading
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.db import delete_run, get_run, list_runs, save_run
from agent.graph import build_graph
from agent.llm import DEFAULT_MODEL, FALLBACK_MODEL, MODEL_ALIASES, QUICK_FALLBACK_MODEL, MissingAPIKeyError, normalize_citations
from agent.quick import fallback_comparison, fallback_limitations
from agent.telemetry import RunTelemetry, activate_telemetry, reset_telemetry

load_dotenv()
st.set_page_config(page_title="Fieldnotes · Research Agent", page_icon="✳", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#172d35; --muted:#63757a; --teal:#087f75; --mint:#e7f3ef; --line:#e4ece8; }
    .stApp { background: #f7f9f7; color:var(--ink); }
    [data-testid="stMainBlockContainer"] { max-width: 1220px; padding-top: 1.15rem; padding-bottom: 4rem; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Manrope','DM Sans',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
    h1 { font-weight:800 !important; }
    [data-testid="stSidebar"] { background: #eef3f0; border-right:1px solid #e2eae5; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#50656a; }
    .brand { font:800 1.18rem 'Manrope',sans-serif; letter-spacing:-.04em; color:#173840; padding:.3rem 0 1.4rem; }
    .brand-mark { display:inline-grid; place-items:center; width:30px; height:30px; border-radius:10px; margin-right:.5rem; background:#d6ebe4; color:#087f75; }
    .hero { position:relative; overflow:hidden; border-radius:22px; padding:1.45rem 2.1rem; margin:.1rem 0 1rem; color:#f7fffc;
      background:linear-gradient(115deg,#123d42 0%,#12655e 55%,#18887b 100%); box-shadow:0 18px 45px rgba(17,65,64,.13); }
    .hero:after { content:''; position:absolute; width:250px; height:250px; right:-55px; top:-160px; border:1px solid rgba(255,255,255,.18); border-radius:50%; box-shadow:0 0 0 30px rgba(255,255,255,.035),0 0 0 62px rgba(255,255,255,.025); }
    .hero-kicker { color:#c7f1e4; text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:700; margin-bottom:.3rem; }
    .hero h1 { color:white; font-size:clamp(1.65rem,3vw,2.35rem); line-height:1.12; margin:0 0 .38rem; }
    .hero p { color:#e1f2ed; max-width:760px; font-size:.96rem; line-height:1.5; margin:0; }
    .section-label { color:#405b5e; text-transform:uppercase; letter-spacing:.13em; font-size:.73rem; font-weight:700; margin:.7rem 0 .35rem; }
    .topic-help { color:#40575b; font-size:.92rem; margin:.1rem 0 .7rem; }
    div[data-testid="stForm"] { border:1px solid var(--line); background:#fff; border-radius:20px; padding:1.3rem 1.45rem 1.1rem; box-shadow:0 8px 24px rgba(24,59,55,.045); }
    div[data-testid="stTextArea"] textarea { background:#fbfdfb; border-color:#dce7e1; border-radius:12px; line-height:1.5; }
    div[data-testid="stButton"] button[kind="primary"], div[data-testid="stFormSubmitButton"] button { border-radius:11px; font-weight:700; min-height:2.8rem; padding:0 1.2rem; box-shadow:0 5px 12px rgba(8,127,117,.17); }
    div[data-testid="stButton"] button { border-radius:10px; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:15px; padding:.8rem 1rem; }
    [data-testid="stMetricLabel"] { color:#687d7d; }
    [data-testid="stMetricValue"] { color:#173d42; font-family:'Manrope',sans-serif; }
    [data-testid="stStatusWidget"] { border-radius:15px; border:1px solid var(--line); }
    [data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:#fff; }
    .welcome-card { border:1px solid var(--line); border-radius:20px; background:linear-gradient(135deg,#fff,#f1f7f3); padding:1.5rem 1.7rem; margin-top:1.5rem; }
    .welcome-title { font:700 1.12rem 'Manrope',sans-serif; color:#21454a; margin-bottom:.35rem; }
    .welcome-copy { color:#40575b; line-height:1.6; margin:0; }
    .report-heading { font:800 1.75rem 'Manrope',sans-serif; letter-spacing:-.04em; color:#173d42; margin:.3rem 0 .2rem; }
    .report-meta { color:#728180; font-size:.88rem; margin-bottom:1rem; }
    .source-card { padding:.78rem 0; border-bottom:1px solid #edf1ee; }
    .source-card:last-child { border-bottom:0; }
    .source-domain { color:#748784; font-size:.78rem; margin-top:.12rem; }
    .source-badge { display:inline-block; border-radius:999px; padding:.15rem .55rem; margin-left:.35rem; background:#e7f3ef; color:#155e55; font-size:.72rem; font-weight:700; }
    .pipeline { display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0; }
    .pipeline-step { padding:.42rem .72rem; border:1px solid #dce7e1; border-radius:999px; background:#fff; color:#657773; font-size:.82rem; }
    .pipeline-step.done { background:#e3f2ec; border-color:#b8ded0; color:#155e55; }
    .pipeline-step.running { background:#fff3d9; border-color:#eed08b; color:#765213; }
    [data-testid="stCaptionContainer"] { color:#40575b !important; }
    [data-testid="stSidebar"] [data-testid="stButton"] button { text-align:left; justify-content:flex-start; min-height:2.55rem; border-color:#dce7e1; background:#f8fbf9; color:#24494b; }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover { border-color:#087f75; color:#075f59; background:#e4f2ed; }
    /* Keep Streamlit's sidebar open/close control usable in minimal-toolbar mode. */
    [data-testid="stToolbar"] { visibility:visible !important; height:auto !important; }
    [data-testid="stAppDeployButton"], #MainMenu { visibility:hidden !important; height:0 !important; }
    [data-testid="stSidebarCollapsedControl"], button[data-testid="collapsedControl"] { visibility:visible !important; display:flex !important; opacity:1 !important; }
    header[data-testid="stHeader"] { background:transparent; }
    a { color:#087f75 !important; }
    hr { border-color:#e5ece7 !important; }
    @media (max-width:700px) { .hero { padding:1.2rem 1.1rem; border-radius:18px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_report(report: str, topic: str, sources: list[dict[str, Any]], saved_at: str | None = None) -> None:
    """Show a report with quick context, markdown and a source drawer."""
    report = normalize_citations(report, sources)
    safe_topic = html.escape(topic)
    st.markdown(f"<div class='report-heading'>{safe_topic}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='report-meta'>{'Saved ' + html.escape(saved_at) + ' · ' if saved_at else ''}Research report</div>",
        unsafe_allow_html=True,
    )
    st.markdown(report)
    if sources:
        with st.expander(f"Sources · {len(sources)} retrieved", expanded=False):
            for source in sources:
                source_type = str(source.get("type", "other")).replace("_", " ").title()
                score = source.get("relevance_score", "—")
                title = html.escape(str(source.get("title", "Source")))
                url = html.escape(str(source.get("url", "")), quote=True)
                domain = html.escape(str(source.get("domain", "")))
                st.markdown(
                    f"<div class='source-card'><a href='{url}' target='_blank'>{title}</a>"
                    f"<span class='source-badge'>{html.escape(source_type)}</span>"
                    f"<span class='source-badge'>Relevance {html.escape(str(score))}/3</span>"
                    f"<div class='source-domain'>{domain}</div></div>", unsafe_allow_html=True,
                )
    st.download_button(
        "↓  Download report as Markdown",
        report,
        file_name="research-report.md",
        mime="text/markdown",
        key=f"download_{abs(hash((topic, report[:80])))}",
    )


def open_history_report(run_id: int) -> None:
    """Select a saved report from the sidebar."""
    st.session_state["selected_history_id"] = run_id


def request_delete_report(run_id: int) -> None:
    """Ask for confirmation before removing a history entry."""
    st.session_state["pending_delete_id"] = run_id


def cancel_delete_report() -> None:
    """Close the history deletion confirmation."""
    st.session_state["pending_delete_id"] = None


def confirm_delete_report(run_id: int) -> None:
    """Remove one report and return to the form if it was open."""
    if delete_run(run_id):
        st.session_state["history_notice"] = "Report deleted."
    if st.session_state.get("selected_history_id") == run_id:
        st.session_state["selected_history_id"] = None
    st.session_state["pending_delete_id"] = None


def render_history_item(item: dict[str, Any], older: bool = False) -> None:
    """Render a saved report with open and delete actions."""
    run_id = int(item["id"])
    selected = st.session_state.get("selected_history_id") == run_id
    left, right = st.columns([0.82, 0.18], gap="small")
    with left:
        st.button(
            f"{'✓  ' if selected else ''}{item['topic'][:33]}",
            key=f"{'history_older' if older else 'history'}_{run_id}",
            help=f"{item['created_at'][:10]} · {item.get('snippet', 'Open saved report')[:180]}",
            on_click=open_history_report,
            args=(run_id,),
            use_container_width=True,
        )
    with right:
        st.button(
            "🗑",
            key=f"delete_history_{run_id}",
            help="Delete this saved report",
            on_click=request_delete_report,
            args=(run_id,),
            use_container_width=True,
        )
    st.caption(f"{item['created_at'][:10]} · {item.get('snippet', '')[:105]}")
    if st.session_state.get("pending_delete_id") == run_id:
        st.warning(f"Delete ‘{item['topic'][:80]}’? This cannot be undone.")
        confirm, cancel = st.columns(2)
        with confirm:
            st.button("Delete report", key=f"confirm_delete_{run_id}", type="primary", on_click=confirm_delete_report, args=(run_id,), use_container_width=True)
        with cancel:
            st.button("Cancel", key=f"cancel_delete_{run_id}", on_click=cancel_delete_report, use_container_width=True)


def start_new_report() -> None:
    """Return from a saved report to the new research form."""
    st.session_state["selected_history_id"] = None


def use_example_prompt(prompt: str) -> None:
    """Place an example topic in the research input."""
    st.session_state["topic_input"] = prompt


with st.sidebar:
    st.markdown("<div class='brand'><span class='brand-mark'>✳</span>Fieldnotes</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Research settings</div>", unsafe_allow_html=True)
    default_model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    default_model = MODEL_ALIASES.get(default_model, default_model)
    model_options = list(dict.fromkeys([DEFAULT_MODEL, FALLBACK_MODEL, default_model]))
    model_labels = {
        DEFAULT_MODEL: "GPT-OSS 120B · Recommended",
        FALLBACK_MODEL: "GPT-OSS 20B · Fast helpers/fallback",
    }
    model = st.selectbox(
        "Language model",
        model_options,
        index=model_options.index(DEFAULT_MODEL),
        key="language_model_v2",
        format_func=lambda value: model_labels.get(value, value),
        help=f"The selected model writes the plan, comparison, and report. Quick uses {QUICK_FALLBACK_MODEL} for source extraction and as a rate-limit fallback; Deep keeps its configured helper model.",
    )
    st.caption(f"Model ID: `{model}`")
    mode = st.radio("Research depth", ["Quick", "Deep"], horizontal=True, help="Quick is tuned for a short first answer. Deep allows more searches and one self-check round.")
    max_iterations = st.slider("Self-check passes", 0, 2, 1 if mode == "Deep" else 0, help="Deep mode may search a specific, unresolved evidence gap once more.", disabled=mode == "Quick")
    results_per_query = st.slider("Search results per question", 3, 8, 3 if mode == "Quick" else 5)
    deep_question_count = st.slider("Questions to explore", 4, 5, 4, disabled=mode == "Quick", help="Deep mode explores four or five focused questions.")
    st.divider()
    st.markdown("<div class='section-label'>Recent reports</div>", unsafe_allow_html=True)
    history = list_runs()
    st.button("＋  Start a new report", key="new_research", on_click=start_new_report, use_container_width=True)
    if notice := st.session_state.pop("history_notice", None):
        st.success(notice)
    selected_id = st.session_state.get("selected_history_id")
    for item in history[:6]:
        render_history_item(item)
    if len(history) > 6:
        with st.expander("Earlier reports"):
            for item in history[6:]:
                render_history_item(item, older=True)
    if not history:
        st.caption("Your completed research will show up here.")
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.caption("A workspace for thoughtful research.")

st.markdown(
    "<div class='hero'><div class='hero-kicker'>AI research workspace</div>"
    "<h1>Make sense of complex questions.</h1>"
    "<p>Explore a topic, find useful perspectives, and turn your research into a clear report.</p></div>",
    unsafe_allow_html=True,
)

selected_id = st.session_state.get("selected_history_id")
if selected_id is None:
    st.markdown("<div class='section-label'>Start with a question</div>", unsafe_allow_html=True)
    st.markdown("<div class='topic-help'>Choose an example to get started, or write your own research question.</div>", unsafe_allow_html=True)
    examples = [
        ("EV vs petrol costs", "Compare electric and petrol car life-cycle costs, emissions, and ownership costs since 2020."),
        ("RAG vs fine-tuning", "When should a team use retrieval-augmented generation versus fine-tuning for domain-specific question answering?"),
        ("Remote work since 2020", "How has remote work affected employee productivity, job satisfaction, and collaboration since 2020? Compare findings across different types of studies and note where the evidence is mixed"),
    ]
    prompt_cols = st.columns(len(examples))
    for column, (label, prompt) in zip(prompt_cols, examples):
        with column:
            st.button(label, key=f"example_{label}", on_click=use_example_prompt, args=(prompt,), use_container_width=True)
    with st.form("research_form", clear_on_submit=False):
        topic = st.text_area(
            "Research topic",
            placeholder="For example: What are the environmental and economic trade-offs of electric vehicles compared with petrol cars over their full life cycle?",
            height=112,
            label_visibility="collapsed",
            key="topic_input",
        )
        submitted = st.form_submit_button("✳  Build my research report", type="primary", use_container_width=True)
    if not submitted:
        st.markdown(
            "<div class='welcome-card'><div class='welcome-title'>A little more than a search</div>"
            "<p class='welcome-copy'>Quick mode explores three focused questions with two pages each. Deep mode allows more searches and a self-check pass. Your finished report and source links are saved in Your library.</p></div>",
            unsafe_allow_html=True,
        )
    if submitted:
        topic = topic.strip()
        if not topic:
            st.warning("Add a research question to get started.")
        else:
            budget_seconds = 90 if mode == "Quick" else 360
            initial_state = {
                "topic": topic,
                "findings": [],
                "sources": [],
                "gaps": [],
                "iteration": 0,
                "max_iterations": 0 if mode == "Quick" else max_iterations,
                "model": model,
                "helper_model": QUICK_FALLBACK_MODEL if mode == "Quick" else FALLBACK_MODEL,
                "mode": mode,
                "time_budget": budget_seconds,
                "deep_question_count": deep_question_count,
                "queries_per_question": 1 if mode == "Quick" else 2,
                "pages_per_question": 2 if mode == "Quick" else 3,
                "results_per_query": results_per_query,
            }
            progress = st.status("Preparing your research…", expanded=True)
            pipeline_slot = st.empty()
            activity_slot = st.empty()
            quick_sections_slot = st.empty()
            timer_slot = st.empty()
            stages = (["Plan", "Search · read", "Extract", "Write", "Assemble"] if mode == "Quick" else ["Plan", "Search · grade · read", "Self-check", "Compare", "Write"])
            stage_for_node = ({"quick_plan":"Plan", "quick_research":"Search · read", "quick_extract":"Extract", "quick_write_findings":"Write", "quick_write_comparison":"Write", "quick_write_limitations":"Write", "quick_assemble":"Assemble"} if mode == "Quick" else {"planner": "Plan", "researcher": "Search · grade · read", "reflector": "Self-check", "more_research": "Search · grade · read", "comparer": "Compare", "writer": "Write"})
            completed_stages: set[str] = set()
            active_stage: str | None = None
            telemetry = RunTelemetry(mode, budget_seconds)
            updates: queue.Queue[tuple[str, Any]] = queue.Queue()

            def show_pipeline() -> None:
                """Render the live workflow progress row."""
                chips = []
                for index, label in enumerate(stages):
                    css = "done" if label in completed_stages else "running" if label == active_stage else ""
                    icon = "✓" if css == "done" else "●" if css == "running" else "○"
                    chips.append(f"<span class='pipeline-step {css}'>{icon} {label}</span>")
                loop = final_state.get("iteration", 0)
                pipeline_slot.markdown(f"<div class='pipeline'>{''.join(chips)}</div><div class='topic-help'>Self-check pass {loop} of {initial_state['max_iterations']} · {mode} budget {budget_seconds}s</div>", unsafe_allow_html=True)

            final_state = {**initial_state, "iteration": 0, "query_log": [], "dropped_results": [], "gaps": []}
            show_pipeline()
            def run_graph() -> None:
                """Run the graph off the UI thread and stream node updates through a queue."""
                token = activate_telemetry(telemetry)
                try:
                    for item in build_graph(mode).stream(initial_state, stream_mode="updates"):
                        for name, delta in item.items():
                            updates.put(("update", (name, delta)))
                    updates.put(("done", None))
                except BaseException as exc:
                    updates.put(("error", exc))
                finally:
                    reset_telemetry(token)

            try:
                worker = threading.Thread(target=run_graph, daemon=True)
                worker.start()
                outcome: tuple[str, Any] = ("pending", None)
                while outcome[0] == "pending":
                    try:
                        outcome = updates.get(timeout=0.25)
                    except queue.Empty:
                        elapsed = time.monotonic() - telemetry.started_at
                        timer_slot.markdown(f"**Elapsed:** {elapsed:.0f}s / {budget_seconds}s budget · {max(0, budget_seconds-elapsed):.0f}s remaining")
                        activity = [event["name"] for event in telemetry.events if event.get("kind") == "activity"][-5:]
                        if activity:
                            activity_slot.markdown("**Live research activity**\n\n" + "\n".join(f"- {item}" for item in activity))
                        continue
                    if outcome[0] == "update":
                        node, delta = outcome[1]
                        final_state.update(delta)
                        next_stage = stage_for_node.get(node)
                        if active_stage and active_stage != next_stage and active_stage != "Search · grade · read":
                            completed_stages.add(active_stage)
                        active_stage = next_stage or active_stage
                        show_pipeline()
                        message = {"planner":"Planning focused questions", "researcher":delta.get("current_step", "Searching, grading and reading in parallel"), "reflector":delta.get("current_step", "Self-check complete"), "more_research":"Searching specific new gaps", "comparer":"Comparing source perspectives", "writer":"Writing report", "quick_plan":"Planning three focused questions", "quick_research":delta.get("current_step", "Searching and reading sources"), "quick_extract":delta.get("current_step", "Extracting claims in parallel"), "quick_write_findings":"Executive summary and findings ready", "quick_write_comparison":"Source comparison ready", "quick_write_limitations":"Limitations and open questions ready", "quick_assemble":"Report assembled"}.get(node, node)
                        progress.write(str(message))
                        if node == "planner":
                            for q in delta.get("sub_questions", []):
                                progress.write(f"Question · {q['question']}")
                        if node == "reflector" and delta.get("gaps"):
                            for gap in delta["gaps"]:
                                progress.write(f"New gap · {gap['question']} — {gap['reason']}")
                        if node == "quick_extract" and delta.get("quick_extract_warnings"):
                            progress.warning("Extraction failed after one retry for: " + "; ".join(delta["quick_extract_warnings"]))
                        if mode == "Quick" and node.startswith("quick_write_"):
                            quick_sections = []
                            if "quick_findings" in final_state:
                                value = final_state["quick_findings"]
                                quick_sections.append("## Executive Summary\n\n" + value.get("executive_summary", "") + "\n\n## Key Findings\n" + "\n".join(f"- {item.get('text','')} **Confidence: {item.get('confidence','Medium')}.** " + " ".join(f"[{i}]" for i in item.get("source_ids", [])) for item in value.get("findings", [])))
                            if "quick_comparison" in final_state or "quick_comparison_failed" in final_state:
                                value = final_state.get("quick_comparison") or fallback_comparison(final_state)
                                agreements = "\n".join(f"- {x.get('text','')} " + " ".join(f"[{i}]" for i in x.get("source_ids", [])) for x in value.get("agreements", [])) or "- The extracted claims do not establish a shared conclusion."
                                disagreements = "\n".join(f"- {x.get('text','')} " + " ".join(f"[{i}]" for i in x.get("source_ids", [])) for x in value.get("disagreements", [])) or "- The extracted claims contain no directly conflicting results."
                                quick_sections.append(f"## Agreements vs Disagreements\n\n**Agreements**\n{agreements}\n\n**Disagreements**\n{disagreements}")
                            if "quick_limitations" in final_state or "quick_limitations_failed" in final_state:
                                value = final_state.get("quick_limitations") or fallback_limitations(final_state)
                                limit_items = [f"- {x}" for key in ("limitations", "open_questions", "not_established") for x in value.get(key, [])]
                                quick_sections.append("## Open Questions/Limitations\n\n" + "\n".join(limit_items or ["- No distinct limitation was identified in the supplied claims."]))
                            quick_sections_slot.markdown("\n\n---\n\n".join(quick_sections))
                            failure_key = {"quick_write_findings":"quick_findings_failed", "quick_write_comparison":"quick_comparison_failed", "quick_write_limitations":"quick_limitations_failed"}.get(node)
                            if failure_key and delta.get(failure_key):
                                progress.warning(f"{node.replace('quick_write_', '').replace('_', ' ').title()} generation failed after one retry. The report will use extracted source claims for that section; see Quick run diagnostics.")
                        # Keep draining every streamed node update until the worker
                        # sends its explicit completion or error message.
                        outcome = ("pending", None)
                    elif outcome[0] == "error":
                        raise outcome[1]
                worker.join(timeout=0.1)
                if active_stage:
                    completed_stages.add(active_stage)
                show_pipeline()
                counts = telemetry.counts()
                timer_slot.markdown(f"**Finished in {counts['total_seconds']:.1f}s** · {counts['llm_calls']} LLM calls · {counts['fetches']} fetches · {counts['429s']} rate limits · {counts['fallbacks']} model fallbacks · {counts['warnings']} warnings")
                activity_slot.markdown("### Performance by node")
                if telemetry.node_table():
                    activity_slot.dataframe(telemetry.node_table(), hide_index=True, use_container_width=True)
                st.caption(f"Run totals · {counts['llm_calls']} LLM calls · {counts['searches']} searches · {counts['fetches']} fetches · {counts['429s']} HTTP 429s · {counts['timeouts']} timeouts · {counts['sleep_seconds']}s per-call wait · {counts['fallbacks']} fallbacks · {counts['warnings']} warnings · log: logs/research_agent.jsonl")
                if mode == "Quick" and counts["warnings"]:
                    st.warning(f"{counts['warnings']} search or writing warnings occurred. Evidence-based assembly was used where possible; see Quick run diagnostics for details.")
                with st.expander("Quick run diagnostics", expanded=False):
                    st.dataframe([{k:v for k,v in event.items() if k not in {"run_id","mode"}} for event in telemetry.events], hide_index=True, use_container_width=True)
                report = final_state.get("report", "")
                if not report:
                    raise RuntimeError("The research workflow finished without a report.")
                save_run(topic, report, final_state.get("sources", []))
                progress.update(label="Research complete", state="complete", expanded=False)
                st.divider()
                render_report(report, topic, final_state.get("sources", []))
            except MissingAPIKeyError as exc:
                progress.update(label="Configuration needed", state="error")
                st.error(str(exc) + " Copy .env.example to .env and add your key.")
            except Exception as exc:
                progress.update(label="Research failed", state="error")
                st.error(f"Research could not be completed: {exc}")
else:
    saved = get_run(selected_id)
    if saved:
        st.markdown("<div class='section-label'>From your library</div>", unsafe_allow_html=True)
        render_report(saved["report"], saved["topic"], saved["sources"], saved["created_at"])
    else:
        st.info("That saved report is no longer available. Choose Start a new report to continue.")
