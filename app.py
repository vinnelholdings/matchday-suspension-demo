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

    .md-title-block {
        border-bottom: 1px solid #D1D5DB; padding: 40px 0 16px 0;
        display: flex; justify-content: space-between; align-items: flex-end; gap: 16px;
    }
    .md-title-block h1 {
        font-size: 18px; font-weight: 600; color: #0C55AC; margin: 0;
    }
    .md-title-block p { font-size: 13px; color: #6B7280; margin: 6px 0 0 0; }
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
        <div>
            <h1>Approve Suspensions and Fines</h1>
            <p>Adapted from the Matchday interface I built, rebuilt in Streamlit
            for this demo. Fines removed. Decision support only &mdash; the
            pipeline recommends, a human decides.</p>
        </div>
        <div><span class="md-count-pill">{total_count} cases available for retrieval
        &middot; {preloaded_count} preloaded &middot; {added_count} added this session</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Reset session", type="secondary"):
    st.session_state.session_cases = []
    st.session_state.last_result = None
    st.rerun()

# ---------------------------------------------------------------------------
# Examples. One written example on the main tab, the rest on a second tab.
# Written out as text on purpose: no example buttons, the user copies
# an example into the report box or writes their own.
# ---------------------------------------------------------------------------

MAIN_EXAMPLE = (
    "Received a first caution for dissent, then a second caution for "
    "delaying the restart, and was shown a red card for a second "
    "caution in the same match. Sent off."
)

MORE_EXAMPLES = {
    "SYN-009 (S)": (
        "Spat at an opponent from close range during a stoppage in play. "
        "Sent off."
    ),
    "SYN-011 (VC)": (
        "Pushed an opponent in the chest off the ball after a throw-in, "
        "with no strike and no injury described. Sent off for violent conduct."
    ),
    "SYN-027 (SFP)": (
        "Made a late tackle from behind, contact to the lower leg, opponent "
        "stayed down briefly then continued, no injury described. "
        "Sent off for serious foul play."
    ),
    "SYN-041 (OL)": (
        'Shouted "fuck you" at an opponent after a foul, one outburst, '
        "then walked away. No threat, no repetition described. "
        "Sent off for offensive language."
    ),
    "SYN-046 (OL)": (
        'Threatened an opponent after the whistle, saying "I\'ll hurt you '
        'after the game, just wait", directed and threatening. '
        "Sent off for offensive language."
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
        badge = (
            "DETERMINISTIC"
            if row.get("suspension_type") == "automatic"
            else "LLM"
        )
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


tab_run, tab_examples = st.tabs(["Run a report", "More examples"])

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

    st.markdown(
        """
        <div class="md-card">
            <div class="md-card-header">
                <div class="md-card-label">Review Suspensions</div>
                <div class="md-card-subtitle">Compare the incident report, rulebook
                guidance, and recommendation before approving the final action.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(build_recommendation_table(final_rows), unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="md-card-footer">{len(final_rows)} suspension
        {'case' if len(final_rows) == 1 else 'cases'} for approval</div>
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

    # Human decision, per case: the Matchday Final Decision editor,
    # open after a run. Keys include run_id so a new run starts fresh
    # instead of reusing the previous run's edited values.
    run_id = st.session_state.run_id
    for i, final_row in enumerate(final_rows):
        stage_row = stage_by_case.get(final_row["case_id"], {})
        st.markdown(
            """
            <div class="md-final-card">
                <div class="md-card-label">Final Decision</div>
                <div class="md-card-subtitle">Capture the final suspension outcome
                before approval.</div>
            </div>
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
            final_games = st.number_input(
                "Games",
                min_value=min_games,
                max_value=max_games,
                value=recommended,
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
            add_to_session_eval(st.session_state.session_cases, stage_copy, facts[i])
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
