import json
import os

import streamlit as st

from pipeline import add_to_session_eval, approve, load_eval_cases, run_pipeline

# --- OpenAI key from Streamlit secrets, never from the repo ---
# Locally: .streamlit/secrets.toml with OPENAI_API_KEY = "..."
# Cloud: Manage app -> Settings -> Secrets, same line
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Matchday Suspension Demo", layout="wide")
st.title("Matchday Suspension Demo")
st.caption(
    "Decision support only — a human decides. "
    "Adapted from the Matchday assignor suspensions page, rebuilt in Streamlit. "
    "Fines removed. Synthetic data only, no real cases."
)

# --- session eval store: 50 preloaded + session additions, new tab resets ---
if "session_cases" not in st.session_state:
    st.session_state.session_cases = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

preloaded = load_eval_cases("eval_cases.json")
total_eval = len(preloaded) + len(st.session_state.session_cases)

with st.sidebar:
    st.header("Session eval store")
    st.write(f"Preloaded: {len(preloaded)}")
    st.write(f"Added this session: {len(st.session_state.session_cases)}")
    st.write(f"Total for retrieval: {total_eval}")
    st.caption("New tab resets to the 50 preloaded cases.")
    if st.button("Reset session"):
        st.session_state.session_cases = []
        st.session_state.last_result = None
        st.rerun()

# --- example chips: copy a case, edit it, run it ---
st.subheader("Try an example, or enter your own suspension")
examples = [
    c for c in preloaded
    if c["case_id"] in ("SYN-011", "SYN-027", "SYN-041", "SYN-046", "SYN-001", "SYN-009")
]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    with col:
        if st.button(f"{ex['misconduct_code']} · {ex['case_id']}", key=ex["case_id"]):
            st.session_state.report_text = ex["report"]
            st.rerun()

report = st.text_area(
    "Red card report",
    value=st.session_state.report_text,
    height=160,
    placeholder="Enter a red card report here. Call 1 will extract player, team, code, and action.",
    key="report_input",
)
st.session_state.report_text = report

if st.button("Run pipeline", type="primary", disabled=not report.strip()):
    with st.spinner("Running Call 1, branch, Call 2..."):
        st.session_state.last_result = run_pipeline(
            reports=[{"case_id": "USER-001", "report": report}],
            session_cases=st.session_state.session_cases,
        )

result = st.session_state.last_result
if not result:
    st.info("Enter a report above, or pick an example, and run the pipeline.")
    st.stop()

# --- step timeline + discipline card (adapted from assignor page) ---
for final_row in result["final"]:
    facts_by_id = {f["case_id"]: f for f in result.get("facts", [])}
    fact = facts_by_id.get(final_row["case_id"], {})

    st.divider()
    st.subheader(final_row.get("final_summary", final_row["case_id"]))

    # Step timeline, badged DETERMINISTIC / LLM
    st.markdown("**Steps**")
    if final_row["recommendation_source"] == "Automatic":
        st.write("✅ DETERMINISTIC — automatic, no LLM call")
    elif final_row["recommendation_source"] == "fallback_missing":
        st.write("⚠️ DETERMINISTIC — missing facts, no guess, returned for more information")
    elif final_row["recommendation_source"] == "fallback_minimum":
        st.write("⚠️ LLM — Call 2 failed, fell back to handbook minimum")
    else:
        st.write("🤖 LLM — Call 1 facts → branch (range) → Call 2 action only")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Handbook**")
        st.write(f"{final_row['handbook_section']}: {final_row['handbook_text']}")
        st.markdown("**Appeal — 15.7**")
        st.write(
            "An appeal is allowed against any disciplinary action. "
            "Any appeal must be submitted in writing by email with supporting evidence "
            "within 7 days of receipt of the disciplinary report. "
            "A $250 filing fee applies and is refunded if the appeal is successful, "
            "less a $50 administrative fee."
        )
        st.markdown("**Retrieved examples (top 3, same code)**")
        examples_used = final_row.get("retrieved_examples", [])
        if not examples_used:
            st.write("None — no embedding model, or no similar cases.")
        for ex in examples_used:
            st.write(
                f"- {ex['case_id']} · similarity {ex['similarity']} · "
                f"{ex['expected_games']} games — {ex['expected_reason']}"
            )
    with col2:
        st.markdown("**Recommendation**")
        st.write(f"Recommended: **{final_row['recommended_games']} games**")
        st.write(final_row["recommended_reason"])
        st.write(f"Source: {final_row['recommendation_source']}")
        st.markdown("**Stage JSON**")
        st.json({k: v for k, v in final_row.items() if k != "retrieved_examples"})

    # --- human approval: gold standard, adds to session eval ---
    st.markdown("**Human decision**")
    final_games = st.number_input(
        "Final games",
        min_value=final_row["min_games"],
        max_value=final_row["max_games"],
        value=final_row["recommended_games"] or final_row["min_games"],
        key=f"final_games_{final_row['case_id']}",
    )
    final_reason = st.text_area(
        "Final reason (if you override, this replaces the LLM reason)",
        value=final_row["recommended_reason"] or "",
        key=f"final_reason_{final_row['case_id']}",
    )
    if st.button("Approve / Override + Add to session eval", key=f"add_{final_row['case_id']}"):
        if final_row["recommended_games"] is None:
            st.warning("Cannot add a missing-facts case to eval — it has no recommendation yet.")
        else:
            new_case = add_to_session_eval(
                st.session_state.session_cases,
                final_row,
                fact,
                final_games=int(final_games),
                final_reason=final_reason,
            )
            st.success(
                f"Added {new_case['case_id']} to this session's eval store. "
                f"Total now: {len(preloaded) + len(st.session_state.session_cases)}. "
                "It will be available for retrieval on the next run in this tab."
            )

st.divider()
st.markdown("**Audit**")
st.json(result["audit"])
st.download_button(
    "Download stage + final JSON",
    data=json.dumps(result, indent=2),
    file_name="suspension_result.json",
    mime="application/json",
)
