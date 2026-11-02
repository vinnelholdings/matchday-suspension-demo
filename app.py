"""Matchday suspension demo -- Streamlit frontend.

Adapted from the Matchday interface I built, rebuilt in Streamlit
for this demo. Fines removed. Synthetic cases only.

Decision support only: the pipeline recommends, a human decides.
Session-only: user-approved cases help retrieval in this tab only.
A new tab resets to the preloaded cases.
"""

import json

import streamlit as st

from pipeline import (
    PipelineError,
    add_to_session_eval,
    load_eval_cases,
    run_pipeline,
)

st.set_page_config(page_title="Matchday Suspension Demo", layout="centered")
st.title("Matchday Suspension Demo")
st.caption(
    "Decision support only -- the pipeline recommends, a human decides. "
    "Adapted from the Matchday interface I built, rebuilt in Streamlit. "
    "Fines removed. Synthetic cases only."
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

# ---------------------------------------------------------------------------
# Sidebar: session eval store
# ---------------------------------------------------------------------------

st.sidebar.header("Session eval store")
st.sidebar.write(f"Preloaded cases: {preloaded_count}")
st.sidebar.write(f"Added this session: {added_count}")
st.sidebar.write(f"Total available for retrieval: {preloaded_count + added_count}")
if st.sidebar.button("Reset session"):
    st.session_state.session_cases = []
    st.session_state.last_result = None
    st.rerun()

# ---------------------------------------------------------------------------
# Example reports
# ---------------------------------------------------------------------------

EXAMPLES = {
    "SYN-001 (2CT)": (
        "Received a first caution for dissent, then a second caution for "
        "delaying the restart, and was shown a red card for a second "
        "caution in the same match. Sent off."
    ),
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

st.subheader("Try an example report")
example_labels = list(EXAMPLES.keys())
for row_start in range(0, len(example_labels), 3):
    cols = st.columns(3)
    for col, label in zip(cols, example_labels[row_start : row_start + 3]):
        if col.button(label):
            # Key-only text area below reads this on rerun. Do NOT pass
            # value= to that text area, or this update gets ignored.
            st.session_state.report_input = EXAMPLES[label]
            st.rerun()

# Key-only: no value= argument, so example chips can update the text.
report_text = st.text_area("Referee report", key="report_input", height=150)

if st.button("Run pipeline", type="primary"):
    if not report_text.strip():
        st.warning("Enter a report first, or pick an example above.")
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

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

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

st.subheader("Recommendation")
for i, final_row in enumerate(final_rows):
    badge = (
        "DETERMINISTIC"
        if final_row.get("suspension_type") == "automatic"
        else "LLM"
    )
    st.markdown(
        f"**{final_row.get('team_name')} - {final_row.get('player_name')}** "
        f"[{badge}]"
    )
    st.write(
        f"Code: {final_row.get('misconduct_code')} | "
        f"Handbook: {final_row.get('handbook_section')} | "
        f"Range: {final_row.get('min_games')}-{final_row.get('max_games')} games"
    )
    st.write(
        f"Recommended: {final_row.get('recommended_games')} games -- "
        f"{final_row.get('recommended_reason')}"
    )

    stage_row = stage_by_case.get(final_row["case_id"], {})
    examples = stage_row.get("retrieved_examples", [])
    if examples:
        st.write("Retrieved examples used:")
        for example in examples:
            st.write(
                f"- {example.get('case_id')}: {example.get('expected_games')} games -- "
                f"{example.get('expected_reason')}"
            )

    st.write("Handbook + appeal")
    st.write(final_row.get("handbook_section", ""))
    st.write(
        "Appeal (section 15.7): appeal in writing by email with evidence "
        "within 7 days. $250 filing fee, refunded if successful less a "
        "$50 admin fee."
    )

    # Human decision. Keys include run_id so a new run starts fresh
    # instead of reusing the previous run's edited values.
    run_id = st.session_state.run_id
    games_key = f"final_games_{run_id}_{i}"
    reason_key = f"final_reason_{run_id}_{i}"
    final_games = st.number_input(
        "Final games",
        min_value=int(final_row["min_games"]),
        max_value=int(final_row["max_games"]),
        value=int(final_row["recommended_games"]),
        key=games_key,
    )
    final_reason = st.text_area(
        "Final reason",
        value=final_row.get("recommended_reason", ""),
        key=reason_key,
    )
    if st.button("Approve / Override + Add to session eval", key=f"approve_{run_id}_{i}"):
        stage_copy = dict(stage_row)
        stage_copy["final_games"] = int(final_games)
        stage_copy["final_reason"] = final_reason
        add_to_session_eval(st.session_state.session_cases, stage_copy, facts[i])
        st.success(
            "Added to session eval. "
            f"Session total is now {preloaded_count + len(st.session_state.session_cases)} cases."
        )
        st.rerun()

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
