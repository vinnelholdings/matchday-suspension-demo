"""Matchday suspension demo -- Streamlit frontend.

Rebuilt to look like the Matchday assignor suspensions page
(frontend/app/assignor/suspensions/page.tsx), not like a default
Streamlit page: light-gray page background, white bordered cards,
the #0C55AC blue theme, the "Approve Suspensions and Fines" title
block, a table for the recommendation, and a "Final Decision" editor
card for the human decision.

Deliberate deviations from that page, because this is a standalone
demo, not the portal:
  - No Matchday logo in the header bar (not available in Streamlit).
    The header bar keeps the same position and border instead.
  - No "Back to assignor dashboard" link: there is no dashboard in
    a standalone demo to go back to.
  - No From / To date inputs: demo reports carry no kickoff date,
    so the table has no Date column rather than a faked one.
  - No fines card: fines were removed from the demo on purpose.
  - No bulk Approve checkbox / Approve All: the demo runs one
    report at a time, so approval lives in the per-case Final
    Decision card below the table, instead of a bulk footer.
  - In Matchday the Final Decision editor opens on Edit. Here it is
    open after a run, because there is one report in front of the
    user and the decision is the point of the demo.
  - Demo additions, kept on purpose: the report text box, the
    Run Pipeline button, the DETERMINISTIC / LLM badge, retrieved
    examples, and the session eval counts. Stage / final / audit
    JSON is kept, but tucked into an expander so the main page
    still reads like the Matchday page.

Decision support only: the pipeline recommends, a human decides.
Session-only: user-approved cases help retrieval in this tab only.
A new tab resets to the preloaded cases.
"""

import html
import json

import streamlit as st

from pipeline import (
    PipelineError,
    add_to_session_eval,
    approve,
    load_eval_cases,
    run_pipeline,
)

st.set_page_config(page_title="Matchday Suspension Demo", layout="wide")

# ---------------------------------------------------------------------------
# Matchday visual language, as CSS over Streamlit's widgets.
# Colours and sizes are taken from the suspensions page:
#   page bg #F9FAFB (gray-50), card border #D1D5DB (gray-300),
#   table header #F3F4F6 (gray-100), primary blue #0C55AC,
#   primary hover #094892, selected row #F7FAFF, editor border #D7E4F7.
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp { background-color: #F9FAFB; }
    .block-container { max-width: 80rem; padding-top: 0; padding-bottom: 4rem; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }

    .md-header {
        background: #FFFFFF;
        border-bottom: 1px solid #E5E7EB;
        display: flex; justify-content: space-between; align-items: center;
        height: 64px; padding: 0 24px; margin: 0 -24px 0 -24px;
    }
    .md-header-left { font-size: 18px; font-weight: 600; color: #0C55AC; }
    .md-header-right { font-size: 13px; color: #6B7280; }

    .md-title-block { border-bottom: 1px solid #D1D5DB; padding: 40px 0 16px 0; }
    .md-title-row {
        display: flex; justify-content: space-between; align-items: center; gap: 16px;
    }
    .md-title-block h1 {
        font-size: 18px; font-weight: 600; color: #0C55AC; margin: 0;
    }
    .md-title-block p { font-size: 13px; color: #6B7280; margin: 6px 0 0 0; }
    /* Streamlit tabs default to red; Matchday's accent is blue. */
    [data-testid="stTabs"] button[role="tab"] { color: #6B7280; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #0C55AC !important; border-bottom-color: #0C55AC !important;
    }
    [data-baseweb="tab"] { color: #6B7280; }
    [data-baseweb="tab"][aria-selected="true"] { color: #0C55AC !important; }
    .md-count-pill {
        display: inline-flex; align-items: center; white-space: nowrap;
        border: 1px solid #D1D5DB; background: #FFFFFF; border-radius: 999px;
        padding: 4px 12px; font-size: 12px; font-weight: 500; color: #4B5563;
    }

    .md-card {
        background: #FFFFFF; border: 1px solid #D1D5DB; border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-top: 16px; overflow: hidden;
    }
    .md-card-header {
        background: rgba(249,250,251,0.8); border-bottom: 1px solid #E5E7EB;
        padding: 12px 16px;
    }
    .md-card-label {
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.18em; color: #6B7280;
    }
    .md-card-subtitle { font-size: 14px; color: #4B5563; margin-top: 4px; }
    .md-card-body { padding: 16px; }
    .md-card-footer {
        background: rgba(249,250,251,0.8); border-top: 1px solid #E5E7EB;
        padding: 12px 16px; font-size: 14px; color: #4B5563;
    }

    .md-example {
        background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;
        padding: 12px 16px; font-size: 14px; color: #374151; line-height: 1.5;
    }
    .md-example-label {
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.12em; color: #6B7280; margin-bottom: 6px;
    }

    table.md-table { width: 100%; border-collapse: collapse; font-size: 12px; color: #000000; }
    table.md-table thead { background: rgba(243,244,246,0.9); }
    table.md-table th {
        padding: 8px 12px; text-align: left; font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em; color: #374151;
    }
    table.md-table td {
        padding: 8px 12px; vertical-align: top; border-bottom: 1px solid #E5E7EB;
        line-height: 1.4; color: #374151;
    }
    table.md-table td.md-team { font-weight: 600; color: #000000; min-width: 124px; }
    table.md-table td.md-player { font-weight: 500; color: #000000; min-width: 124px; }
    .md-code-badge {
        display: inline-flex; min-width: 44px; align-items: center; justify-content: center;
        border: 1px solid #D1D5DB; background: #F3F4F6; border-radius: 2px;
        padding: 2px 6px; font-size: 10px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em; color: #374151;
    }
    .md-source-badge {
        display: inline-block; border: 1px solid #D7E4F7; background: #F7FAFF;
        border-radius: 999px; padding: 1px 8px; font-size: 10px; font-weight: 600;
        letter-spacing: 0.08em; color: #0C55AC; margin-bottom: 4px;
    }
    .md-recommended { font-weight: 600; color: #000000; }
    .md-handbook-appeal { border-top: 1px solid #E5E7EB; margin-top: 12px; padding-top: 12px; }
    .md-handbook-appeal-title { font-weight: 600; color: #000000; }
    .md-final-card {
        background: #FFFFFF; border: 1px solid #D7E4F7; border-radius: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05); padding: 16px; margin-top: 16px;
    }

    .stButton > button { border-radius: 6px; font-weight: 600; font-size: 14px; }
    .stButton > button[kind="primary"] {
        background-color: #0C55AC !important; border-color: #0C55AC !important; color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #094892 !important; border-color: #094892 !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important; border-color: #0C55AC !important; color: #0C55AC !important;
    }
    .stButton > button[kind="secondary"]:hover { background-color: #EAF2FF !important; }
    textarea:focus, input:focus {
        border-color: #0C55AC !important;
        box-shadow: 0 0 0 2px rgba(12,85,172,0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "session_cases" not in st.session_state:
    st.session_state.session_cases = []
if "report_input" not in st.session_state:
    st.session_state.report_input = ""
if "run_id" not in st.session_state:
    st.session_state.run_id = 0

preloaded_cases = load_eval_cases()
preloaded_count = len(preloaded_cases)
added_count = len(st.session_state.session_cases)
total_count = preloaded_count + added_count

# load_eval_cases() silently returns [] when the file is missing or
# invalid, which is why "0 preloaded" gave no clue. Surface the reason
# on the page instead, so a Cloud deployment that cannot see the
# file says so plainly instead of silently running with no examples.
if preloaded_count == 0:
    try:
        with open("eval_cases.json", "r", encoding="utf-8") as _f:
            _data = json.load(_f)
        _reason = "file loaded but it contains no cases"
    except FileNotFoundError:
        _reason = "eval_cases.json was not found in the app folder"
    except json.JSONDecodeError as _e:
        _reason = f"eval_cases.json is not valid JSON: {_e}"
    st.warning(
        "Preloaded eval cases: 0. "
        f"{_reason}. Check that eval_cases.json is in the GitHub repo root, "
        "next to app.py, on the branch the Cloud app points to, then Reboot."
    )

# ---------------------------------------------------------------------------
# Header bar (no logo) + title block, as on the suspensions page
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="md-header">
        <div class="md-header-left">Matchday Suspension Demo</div>
        <div class="md-header-right">Demo &middot; Synthetic cases only</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="md-title-block">
        <div class="md-title-row">
            <h1>Approve Suspensions and Fines</h1>
            <span class="md-count-pill">{total_count} cases available for retrieval
            &middot; {preloaded_count} preloaded &middot; {added_count} added this session</span>
        </div>
        <p>Adapted from the Matchday interface I built, rebuilt in Streamlit
        for this demo. Fines removed. Decision support only &mdash; the
        pipeline recommends, a human decides.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Reset is demo-only, so keep it small and out of the title block.
_, reset_col = st.columns([5, 1])
with reset_col:
    if st.button("Reset session", type="secondary"):
        st.session_state.session_cases = []
        st.session_state.last_result = None
        st.rerun()

# Preloaded cases, visible instead of just a count, collapsed by
# default so the main page stays clean. This is the same list
# retrieval searches, so 0 here means retrieval has nothing to search.
if preloaded_count > 0:
    with st.expander(f"Preloaded cases ({preloaded_count})"):
        for case in preloaded_cases:
            st.write(
                f"{case.get('case_id')}: {case.get('misconduct_code')} -- "
                f"{case.get('expected_games')} games"
            )

# ---------------------------------------------------------------------------
# Examples. One written example on the main tab, the rest on a second tab.
# Written out as text on purpose: no example buttons, the user copies
# an example into the report box or writes their own.
# ---------------------------------------------------------------------------

# Examples include a player and a team on purpose: the pipeline
# now fails a report that names neither, instead of rendering a
# table with "-" for Teams / Player. Names below are the synthetic
# players/teams from the matching eval cases.
MAIN_EXAMPLE = (
    "In the 20th minute, Alex Carter of Riverside United received a first "
    "caution for dissent, then a second caution for delaying the restart, "
    "and was shown a red card for a second caution in the same match. Sent off."
)

MORE_EXAMPLES = {
    "SYN-009 (S)": (
        "In the 76th minute, Riley Ford of Northgate Athletic spat at an "
        "opponent from close range during a stoppage in play. Sent off."
    ),
    "SYN-011 (VC)": (
        "In the 25th minute, Alexis Grant of Riverbend SC pushed an opponent "
        "in the chest off the ball after a throw-in, with no strike and no "
        "injury described. Sent off for violent conduct."
    ),
    "SYN-027 (SFP)": (
        "In the 72nd minute, Diego Ramos of Riverbend SC made a late tackle "
        "from behind, contact to the lower leg, opponent stayed down briefly "
        "then continued, no injury described. Sent off for serious foul play."
    ),
    "SYN-041 (OL)": (
        'In the 40th minute, Ryan Holt of Northgate Athletic shouted "fuck you" '
        "at an opponent after a foul, one outburst, then walked away. No threat, "
        "no repetition described. Sent off for offensive language."
    ),
    "SYN-046 (OL)": (
        'In the 75th minute, Zoe Hart of Eastside Athletic threatened an '
        'opponent after the whistle, saying "I\'ll hurt you after the game, '
        'just wait", directed and threatening. Sent off for offensive language.'
    ),
}


def build_recommendation_table(final_rows):
    """Render the recommendation as a Matchday-style HTML table.

    Pure function (no Streamlit calls) so it can be checked without
    running the app. All pipeline / LLM text is escaped: a report
    must never be able to break the table markup.
    """
    appeal_title = "15.7: An appeal is allowed against any disciplinary action."
    appeal_lines = [
        "Any appeal must be submitted in writing by email with supporting evidence",
        "within 7 days of receipt of the disciplinary report.",
        "A $250 filing fee applies and is refunded if the appeal is successful,",
        "less a $50 administrative fee.",
    ]
    appeal_html = "".join(
        f"<div>{html.escape(line)}</div>" for line in appeal_lines
    )

    rows_html = []
    for row in final_rows:
        # Plain language on screen, technical source values stay in
        # the JSON. Option C, chosen by Emmy: "Set by the rulebook"
        # and "Suggested by AI". Fallback cases are NOT suggested by
        # AI: the AI failed and the number came from the rulebook
        # minimum, so the badge is based on source, not on case type.
        if row.get("source") == "Automatic":
            badge = "Set by the rulebook"
        elif row.get("source") == "fallback_minimum":
            badge = "Rulebook minimum"
        else:
            badge = "Suggested by AI"
        handbook = (
            f"{html.escape(str(row.get('handbook_section') or ''))}: "
            f"{html.escape(str(row.get('handbook_text') or ''))}"
        )
        rows_html.append(
            "<tr>"
            f"<td class=\"md-team\">{html.escape(str(row.get('team_name') or '-'))}</td>"
            f"<td class=\"md-player\">{html.escape(str(row.get('player_name') or '-'))}</td>"
            f"<td><span class=\"md-code-badge\">{html.escape(str(row.get('misconduct_code') or '-'))}</span></td>"
            f"<td>{html.escape(str(row.get('action') or '-'))}</td>"
            "<td>"
            f"{handbook}"
            "<div class=\"md-handbook-appeal\">"
            f"<div class=\"md-handbook-appeal-title\">{html.escape(appeal_title)}</div>"
            f"{appeal_html}"
            "</div>"
            "</td>"
            "<td>"
            f"<div><span class=\"md-source-badge\">{badge}</span></div>"
            f"<span class=\"md-recommended\">{html.escape(str(row.get('recommended_games')))} "
            f"{'game' if row.get('recommended_games') == 1 else 'games'}</span>"
            f" <span style=\"color:#6B7280;\">-</span> "
            f"{html.escape(str(row.get('recommended_reason') or ''))}"
            "</td>"
            "</tr>"
        )
    return (
        "<table class=\"md-table\">"
        "<thead><tr>"
        "<th>Teams</th><th>Player</th><th>Code</th>"
        "<th>Supplemental Text</th><th>Handbook Text</th><th>Recommended</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows_html) + "</tbody></table>"
    )


tab_run, tab_examples, tab_logs = st.tabs(
    ["Run a report", "More examples", "Logs"]
)

with tab_examples:
    st.markdown(
        """
        <div class="md-card">
            <div class="md-card-header">
                <div class="md-card-label">More examples</div>
                <div class="md-card-subtitle">Copy any example into the report box
                on the Run a report tab, or write your own report.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for label, text in MORE_EXAMPLES.items():
        st.markdown(
            f"""
            <div class="md-card"><div class="md-card-body">
                <div class="md-example-label">{html.escape(label)}</div>
                <div>{html.escape(text)}</div>
            </div></div>
            """,
            unsafe_allow_html=True,
        )

with tab_logs:
    # Temporary debug tab, can be removed later. No more hunting
    # Cloud logs: everything logged for the last run is shown here
    # with st.info, in plain lines, in one place.
    st.markdown(
        """
        <div class="md-card">
            <div class="md-card-header">
                <div class="md-card-label">Logs</div>
                <div class="md-card-subtitle">Temporary debug tab, can be removed
                later. Shows the last run only.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _log_result = st.session_state.get("last_result")
    if not _log_result:
        st.info("No run yet. Run a report on the Run a report tab, then logs will appear here.")
    else:
        _timings = _log_result.get("timings", {}) or {}
        st.info(
            f"Run Pipeline took {_timings.get('total_seconds', '?')} seconds: "
            f"Call 1 {_timings.get('call1_seconds', '?')}s, "
            f"retrieval {_timings.get('retrieval_seconds', '?')}s, "
            f"Call 2 {_timings.get('call2_seconds', '?')}s."
        )
        _audit = _log_result.get("audit", {}) or {}
        st.info(
            f"Retrieval enabled: {_audit.get('retrieval_enabled')}, "
            f"eval cases used: {_audit.get('eval_cases_used')}, "
            f"sources: {', '.join(_audit.get('stage_sources', []))}."
        )
        for _warning in _log_result.get("warnings", []):
            st.info(f"Warning: {_warning}")
        for _case_id, _log in (_log_result.get("retrieval_log", {}) or {}).items():
            st.info(f"Retrieval for {_case_id}: code {_log.get('code')}, "
                    f"candidates {_log.get('candidates_considered')}, "
                    f"model loaded {_log.get('model_loaded')}.")
            if _log.get("embedding"):
                _emb = _log["embedding"]
                st.info(
                    f"Embedding for {_case_id}: model load {_emb.get('model_load_seconds')}s "
                    f"(cached: {_emb.get('model_was_cached')}), query encode {_emb.get('query_encode_seconds')}s, "
                    f"docs encode {_emb.get('docs_encode_seconds')}s for {_emb.get('docs_encoded')} docs, "
                    f"similarity {_emb.get('similarity_seconds')}s."
                )
            for _example in _log.get("top_examples", []):
                _boost = _example.get("session_boost", 0.0) or 0.0
                _boost_text = (
                    f" (raw {_example.get('raw_score')} + session boost {_boost})"
                    if _boost
                    else ""
                )
                st.info(
                    f"Retrieved {_example.get('case_id')}: score {_example.get('score')}{_boost_text}, "
                    f"{_example.get('source', '')}, {_example.get('expected_games')} games."
                )
        st.json(_log_result.get("audit", {}))

with tab_run:
    # --- Input card -------------------------------------------------------
    st.markdown(
        """
        <div class="md-card">
            <div class="md-card-header">
                <div class="md-card-label">Review Suspensions</div>
                <div class="md-card-subtitle">Enter a referee report. The pipeline
                classifies it, applies the handbook, and recommends a suspension
                for human approval.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="md-example" style="margin-top: 16px;">
            <div class="md-example-label">Example &mdash; second caution, sent off</div>
            <div>{html.escape(MAIN_EXAMPLE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key-only: no value= argument, so the text area keeps what the
    # user typed across reruns.
    report_text = st.text_area("Referee report", key="report_input", height=150)

    if st.button("Run pipeline", type="primary"):
        if not report_text.strip():
            st.warning("Enter a report first, or copy the example above.")
        else:
            try:
                result = run_pipeline(
                    report_text, session_cases=st.session_state.session_cases
                )
            except PipelineError as e:
                # Plain-English pipeline error -- no traceback.
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(
                    "Something went wrong running the pipeline. "
                    "Try again, or rephrase the report."
                )
                with st.expander("Technical detail"):
                    st.code(str(e))
                st.stop()
            st.session_state.last_result = result
            st.session_state.run_id += 1
            st.rerun()

    # --- Results ----------------------------------------------------------
    result = st.session_state.get("last_result")
    if not result:
        st.stop()

    for warning in result.get("warnings", []):
        st.warning(warning)

    facts = result.get("facts", [])
    stage_rows = result.get("stage", [])
    final_rows = result.get("final", [])

    if not final_rows:
        st.error("The pipeline returned no cases. Try rephrasing the report.")
        st.stop()

    stage_by_case = {row["case_id"]: row for row in stage_rows}

    # One card, as in Matchday: header, table, and footer together,
    # not three separate blocks with gaps between them.
    st.markdown(
        f"""
        <div class="md-card">
            <div class="md-card-header">
                <div class="md-card-label">Review Suspensions</div>
                <div class="md-card-subtitle">Compare the incident report, rulebook
                guidance, and recommendation before approving the final action.</div>
            </div>
            {build_recommendation_table(final_rows)}
            <div class="md-card-footer">{len(final_rows)} suspension
            {'case' if len(final_rows) == 1 else 'cases'} for approval</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrieved examples, per case (demo addition, under the table).
    for final_row in final_rows:
        stage_row = stage_by_case.get(final_row["case_id"], {})
        examples = stage_row.get("retrieved_examples", [])
        if examples:
            st.markdown(
                f"<div style=\"font-size: 12px; color: #6B7280; margin-top: 12px;\">"
                f"Retrieved examples used for {html.escape(str(final_row.get('case_id')))}:</div>",
                unsafe_allow_html=True,
            )
            for example in examples:
                st.write(
                    f"- {example.get('case_id')}: {example.get('expected_games')} games -- "
                    f"{example.get('expected_reason')}"
                )

    # Retrieval log, collapsed by default in an expander, NOT a
    # checkbox. The checkbox version hit the same Cloud chunk error
    # as NumberInput and Selectbox before it: "Failed to fetch
    # dynamically imported module: .../Checkbox....js". Expander was
    # already in the original app (Stage/Final/Audit), so its chunk
    # is already loaded, and no new widget chunk is requested at all.
    # The log shows how retrieval actually worked for each range
    # case: query action, whether the embedding model loaded, how
    # many same-code candidates were considered, and top examples
    # with similarity scores (0-1, higher is closer). Automatic
    # cases have no retrieval log.
    retrieval_log = result.get("retrieval_log", {})
    if retrieval_log:
        for case_id, log in retrieval_log.items():
            with st.expander(f"Retrieval detail: {case_id}"):
                st.write(f"Query action: {log.get('query_action')}")
                st.write(f"Code searched: {log.get('code')}")
                st.write(f"Embedding model loaded: {log.get('model_loaded')}")
                st.write(
                    f"Candidates considered (same code): "
                    f"{log.get('candidates_considered')}"
                )
                if log.get("error"):
                    st.write(f"Retrieval error: {log.get('error')}")
                if log.get("embedding"):
                    _emb = log["embedding"]
                    st.write(
                        f"Embedding timing (seconds): model load {_emb.get('model_load_seconds')} "
                        f"(cached: {_emb.get('model_was_cached')}), query encode {_emb.get('query_encode_seconds')}, "
                        f"docs encode {_emb.get('docs_encode_seconds')} for {_emb.get('docs_encoded')} docs, "
                        f"similarity {_emb.get('similarity_seconds')}"
                    )
                for example in log.get("top_examples", []):
                    _boost = example.get("session_boost", 0.0) or 0.0
                    _boost_text = (
                        f" (raw {example.get('raw_score')} + session boost {_boost})"
                        if _boost
                        else ""
                    )
                    st.write(
                        f"- {example.get('case_id')}: score {example.get('score')}{_boost_text} -- "
                        f"{example.get('source', '')} -- "
                        f"{example.get('expected_games')} games -- "
                        f"{example.get('expected_reason')}"
                    )
                if not log.get("top_examples"):
                    st.write("No examples retrieved for this case.")

    # Human decision, per case: the Matchday Final Decision editor,
    # open after a run. Keys include run_id so a new run starts fresh
    # instead of reusing the previous run's edited values.
    run_id = st.session_state.run_id
    for i, final_row in enumerate(final_rows):
        stage_row = stage_by_case.get(final_row["case_id"], {})
        # Bordered container, so Games / Reason / Approve sit INSIDE
        # the Final Decision card, as in Matchday's editor, instead
        # of floating below a separate header box.
        with st.container(border=True):
            st.markdown(
                """
                <div class="md-card-label">Final Decision</div>
                <div class="md-card-subtitle">Capture the final suspension outcome
                before approval.</div>
                """,
                unsafe_allow_html=True,
            )
            games_key = f"final_games_{run_id}_{i}"
            reason_key = f"final_reason_{run_id}_{i}"
            min_games = int(final_row["min_games"])
            max_games = int(final_row["max_games"])
            recommended = int(final_row["recommended_games"])
            recommended = max(min_games, min(max_games, recommended))
            col_games, col_reason = st.columns([1, 4])
            with col_games:
                # Selectbox, not number_input: the deployed app hit
                # "Failed to fetch dynamically imported module: .../NumberInput....js"
                # on the number_input widget, and a games range is only
                # 1-4 options anyway, so a dropdown is the better control.
                final_games = st.selectbox(
                    "Games",
                    options=list(range(min_games, max_games + 1)),
                    index=recommended - min_games,
                    key=games_key,
                )
            with col_reason:
                final_reason = st.text_area(
                    "Reason",
                    value=final_row.get("recommended_reason", ""),
                    key=reason_key,
                )
            if st.button(
                "Approve / Override + Add to session eval", key=f"approve_{run_id}_{i}"
            ):
                stage_copy = dict(stage_row)
                stage_copy["final_games"] = int(final_games)
                stage_copy["final_reason"] = final_reason
                add_to_session_eval(
                    st.session_state.session_cases, stage_copy, facts[i]
                )
                # Write the decision back into last_result. Without
                # this, Final JSON kept showing final_games: null and
                # status: "pending" after an approval, because the
                # expander was reading the pre-approval rows.
                st.session_state.last_result["final"][i] = approve(
                    final_row,
                    final_games=int(final_games),
                    final_reason=final_reason,
                )
                st.success(
                    "Added to session eval. "
                    f"Session total is now {preloaded_count + len(st.session_state.session_cases)} cases."
                )
                st.rerun()

    # Demo detail, tucked away so the main page reads like Matchday.
    with st.expander("Stage, final and audit (demo detail)"):
        st.subheader("Stage JSON")
        st.json(stage_rows)
        st.subheader("Final JSON")
        st.json(final_rows)
        st.download_button(
            "Download final JSON",
            data=json.dumps(final_rows, indent=2),
            file_name="final.json",
            mime="application/json",
        )
        st.subheader("Audit")
        st.json(result.get("audit", {}))
        # Timing is the point of this log: Call 1, retrieval, and
        # Call 2 are timed separately, so we can see which stage
        # makes Run Pipeline slow, before fixing any of them.
        if result.get("timings"):
            st.subheader("Timings (seconds, Run Pipeline only)")
            st.json(result["timings"])
