"""Matchday suspension demo pipeline.

Deterministic workflow with LLM calls inside, nothing else:
  Call 1 -> branch (automatic / range) -> Call 2 (range only)
  -> clamp / fallback -> stage JSON -> final JSON (after human approval)

Handbook ranges come from discipline.json, not a database.
Retrieval examples come from eval_cases.json (50 synthetic cases)
plus any session cases added in this tab. Session-only: new tab resets.
No real cases, no real names — demo input is synthetic only.
"""

import json
import os
import time
from pathlib import Path

from prompts import CALL_1_SYSTEM, CALL_1_USER, CALL_2_SYSTEM, CALL_2_USER

def _get_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Streamlit only: secrets are in st.secrets, not os.environ,
        # so pick them up here if env is empty.
        # Wrapped in try/except, so local runs without Streamlit still work.
        try:
            import streamlit as st

            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)



AUTOMATIC_CODES = {"2CT", "DH", "DF", "S"}


# ---------- handbook ----------

def load_handbook(path="discipline.json"):
    data = json.loads(Path(path).read_text())
    return {rule["misconduct_code"]: rule for rule in data["rules"]}


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------- LLM helper ----------

def _call_llm_json(client, system, user):
    """One LLM call, JSON only. Returns a dict, or None on any failure."""
    try:
        response = client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-5"),
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.output_text
        return json.loads(text)
    except Exception:
        return None

# ---------- retrieval from eval store ----------

def load_eval_cases(path="eval_cases.json"):
    """Load preloaded synthetic eval cases. Returns [] if missing."""
    try:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            return data.get("cases", [])
        return data
    except Exception:
        return []


def _get_embedding_model():
    """Lazy load, so the pipeline still works without sentence-transformers.
    Returns (model, util) or (None, None) if not installed."""
    try:
        from sentence_transformers import SentenceTransformer, util

        return SentenceTransformer("all-MiniLM-L6-v2"), util
    except Exception:
        return None, None


def add_to_session_eval(session_cases, stage_row, fact, final_games=None, final_reason=None, approved_by=None):
    """Add a human-approved case to the session eval store.
    This is the training loop for v1, session-only.
    Returns the new eval case dict, and appends it to session_cases in place.
    No database — final games + final reason are the gold standard."""
    final = approve(
        stage_row, fact,
        final_games=final_games,
        final_reason=final_reason,
        approved_by=approved_by,
    )
    new_case = {
        "case_id": f"SESSION-{len(session_cases) + 1:03d}",
        "section": "Session - Human approved",
        "misconduct_code": stage_row["misconduct_code"],
        "min_games": stage_row["min_games"],
        "max_games": stage_row["max_games"],
        "player_name": fact.get("player_name"),
        "team_name": fact.get("team_name"),
        "action": fact.get("action"),
        "expected_games": final["final_games"],
        "expected_reason": final["final_reason"],
    }
    session_cases.append(new_case)
    return new_case


def retrieve_top_k(new_action, cases, code, k=3, exclude_case_id=None, model=None, util=None):
    """Top k most similar actions, same code only, never the case itself.
    Similarity is semantic (embedding + cosine), not keyword matching.
    Returns [] if no model or no cases — Call 2 still works without examples."""
    filtered = [
        c for c in cases
        if c.get("misconduct_code") == code
        and c.get("case_id") != exclude_case_id
    ]
    if not filtered or model is None or util is None:
        return []

    actions = [c["action"] for c in filtered]
    doc_embeddings = model.encode(actions, convert_to_tensor=True)
    query_embedding = model.encode(new_action, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    ranked = sorted(
        zip(filtered, scores), key=lambda x: float(x[1]), reverse=True
    )
    return [
        {
            "case_id": c["case_id"],
            "action": c["action"],
            "expected_games": c.get("expected_games"),
            "expected_reason": c.get("expected_reason"),
            "similarity": round(float(score), 4),
        }
        for c, score in ranked[:k]
    ]


# ---------- Call 1: classify + extract ----------

def _build_call1_cases_json(reports):
    # reports: [{"case_id": "...", "report": "..."}]
    # The LLM sees the report text only. No handbook, no DB.
    return json.dumps(
        [{"case_id": r["case_id"], "report": r["report"]} for r in reports],
        indent=2,
    )


def get_facts(reports, client):
    """Call 1. Returns facts, one per case_id, in the Call 1 schema."""
    user_prompt = CALL_1_USER.replace(
        "{cases_json}", _build_call1_cases_json(reports)
    )
    data = _call_llm_json(client, CALL_1_SYSTEM, user_prompt)
    if not data or "facts" not in data:
        return []
    return data["facts"]


def _strip_player_name(action, player_name):
    """Code-side guard: if the LLM left the player name in the action,
    take it out. The action blob must not include the player name."""
    if not action or not player_name:
        return action
    return action.replace(player_name, "").strip()


def _enrich_facts(facts, handbook):
    """Attach the handbook rule, and derive suspension_type from it.
    suspension_type is automatic when min == max, otherwise range.
    The handbook wins if the LLM disagreed."""
    enriched = []
    for fact in facts:
        rule = handbook.get(fact.get("misconduct_code"))
        if not rule:
            continue
        fact = dict(fact)
        fact["action"] = _strip_player_name(
            fact.get("action"), fact.get("player_name")
        )
        fact["min_games"] = rule["min_games"]
        fact["max_games"] = rule["max_games"]
        fact["handbook_section"] = rule["handbook_section"]
        fact["handbook_text"] = rule["handbook_text"]
        fact["suspension_type"] = (
            "automatic" if rule["min_games"] == rule["max_games"] else "range"
        )
        enriched.append(fact)
    return enriched


# ---------- automatic path (no LLM) ----------

def _build_automatic_recommendation_reason(code):
    # Kept exactly as in production.
    if code in AUTOMATIC_CODES:
        return "Automatic suspension per handbook."
    return "Automatic recommendation based on handbook minimum."


# ---------- Call 2: range reasoning, action only + retrieved examples ----------

def _build_call2_cases_json(facts, eval_cases=None, embedding_model=None, util=None):
    # Call 2 sees action only, plus top-3 retrieved examples for that case.
    # No team, no player, no full report. Examples are examples, not rules —
    # the handbook range in the same case still controls, and code clamps after.
    cases = []
    for fact in facts:
        examples = retrieve_top_k(
            fact["action"],
            eval_cases or [],
            fact["misconduct_code"],
            k=3,
            exclude_case_id=fact["case_id"],
            model=embedding_model,
            util=util,
        )
        cases.append(
            {
                "case_id": fact["case_id"],
                "action": fact["action"],
                "misconduct_code": fact["misconduct_code"],
                "handbook_section": fact["handbook_section"],
                "handbook_text": fact["handbook_text"],
                "min_games": fact["min_games"],
                "max_games": fact["max_games"],
                "examples": examples,
            }
        )
    return json.dumps(cases, indent=2), cases


def get_ai_recommendations(facts, client, eval_cases=None, embedding_model=None, util=None):
    """Call 2, batched. Returns {case_id: stage-dict}.
    Clamped to [min, max]. Missing case or any failure
    falls back to the handbook minimum."""
    cases_json, cases_with_examples = _build_call2_cases_json(
        facts, eval_cases, embedding_model, util
    )
    user_prompt = CALL_2_USER.replace("{cases_json}", cases_json)
    data = _call_llm_json(client, CALL_2_SYSTEM, user_prompt)
    by_id = {}
    if data and "recommendations" in data:
        for rec in data["recommendations"]:
            by_id[rec.get("case_id")] = rec

    examples_by_id = {
        c["case_id"]: c.get("examples", []) for c in cases_with_examples
    }

    stage = {}
    for fact in facts:
        rec = by_id.get(fact["case_id"])
        if rec is None:
            games = fact["min_games"]
            reason = "Defaulted to handbook minimum because AI recommendation failed."
            source = "fallback_minimum"
        else:
            games = _safe_int(rec.get("recommended_games"))
            if games is None:
                games = fact["min_games"]
                reason = "Defaulted to handbook minimum because AI recommendation failed."
                source = "fallback_minimum"
            else:
                games = max(fact["min_games"], min(fact["max_games"], games))
                reason = rec.get("recommended_reason", "")
                source = "AI"
        stage[fact["case_id"]] = {
            "case_id": fact["case_id"],
            "misconduct_code": fact["misconduct_code"],
            "min_games": fact["min_games"],
            "max_games": fact["max_games"],
            "handbook_section": fact["handbook_section"],
            "handbook_text": fact["handbook_text"],
            "recommended_games": games,
            "recommended_reason": reason,
            "recommendation_source": source,
            "retrieved_examples": examples_by_id.get(fact["case_id"], []),
        }
    return stage


# ---------- final summary, stitched in code ----------

def build_final_summary(fact, stage_row):
    # Team + player + LLM statement. The LLM never saw team/player;
    # code adds identity back here, after the decision.
    team = fact.get("team_name") or ""
    player = fact.get("player_name") or ""
    return f"{team} - {player}: {stage_row['recommended_reason']}".strip(" -:")


# ---------- human approval ----------

def approve(stage_row, fact, final_games=None, final_reason=None, approved_by=None):
    """Human approval. If a human overrides, the human reason replaces
    the LLM reason as the official record. Deterministic cases need
    no human approval; range cases do."""
    if final_games is None:
        final_games = stage_row["recommended_games"]
        status = "approved"
    elif final_games != stage_row["recommended_games"]:
        status = "overridden"
    else:
        status = "approved"
    return {
        **stage_row,
        "final_games": final_games,
        "final_reason": final_reason or stage_row["recommended_reason"],
        "approved_by": approved_by,
        "status": status,
        "final_summary": build_final_summary(fact, stage_row),
    }


# ---------- main pipeline ----------

def run_pipeline(reports, handbook_path="discipline.json", client=None,
                 eval_cases_path="eval_cases.json", session_cases=None):
    """reports: [{"case_id": "...", "report": "..."}]
    session_cases: user-added cases in this tab only, session-only.
    Returns {"stage": [...], "final": [...], "audit": {...}}.
    Final rows are pending until approve() is called."""
    start = time.time()
    handbook = load_handbook(handbook_path)
    if client is None:
        client = _get_client()

    # Eval store = preloaded 50 + session additions, session-only.
    eval_cases = load_eval_cases(eval_cases_path)
    if session_cases:
        eval_cases = eval_cases + session_cases
    embedding_model, util = _get_embedding_model()

    facts = _enrich_facts(get_facts(reports, client), handbook)
    facts_by_id = {f["case_id"]: f for f in facts}

    stage = []
    range_facts = []
    for fact in facts:
        if fact.get("missing_facts"):
            # Missing required facts: return noting what's missing, no guess.
            stage.append(
                {
                    "case_id": fact["case_id"],
                    "misconduct_code": fact["misconduct_code"],
                    "min_games": fact["min_games"],
                    "max_games": fact["max_games"],
                    "handbook_section": fact["handbook_section"],
                    "handbook_text": fact["handbook_text"],
                    "recommended_games": None,
                    "recommended_reason": (
                        "Missing required facts: "
                        + ", ".join(fact["missing_facts"])
                    ),
                    "recommendation_source": "fallback_missing",
                    "retrieved_examples": [],
                }
            )
        elif fact["suspension_type"] == "automatic":
            stage.append(
                {
                    "case_id": fact["case_id"],
                    "misconduct_code": fact["misconduct_code"],
                    "min_games": fact["min_games"],
                    "max_games": fact["max_games"],
                    "handbook_section": fact["handbook_section"],
                    "handbook_text": fact["handbook_text"],
                    "recommended_games": fact["min_games"],
                    "recommended_reason": _build_automatic_recommendation_reason(
                        fact["misconduct_code"]
                    ),
                    "recommendation_source": "Automatic",
                    "retrieved_examples": [],
                }
            )
        else:
            range_facts.append(fact)

    if range_facts:
        ai_stage = get_ai_recommendations(
            range_facts, client, eval_cases, embedding_model, util
        )
        stage.extend(ai_stage[f["case_id"]] for f in range_facts)

    # Keep stage in input order.
    order = {f["case_id"]: i for i, f in enumerate(facts)}
    stage.sort(key=lambda row: order.get(row["case_id"], 0))

    final = []
    for row in stage:
        fact = facts_by_id.get(row["case_id"], {})
        final.append(
            {
                **row,
                "final_games": None,
                "final_reason": None,
                "approved_by": None,
                "status": "pending",
                "final_summary": build_final_summary(fact, row)
                if row["recommended_games"] is not None
                else row["recommended_reason"],
            }
        )

    audit = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-5"),
        "rule_version": 1,
        "latency_seconds": round(time.time() - start, 2),
        "stage_sources": [row["recommendation_source"] for row in stage],
        "retrieval_enabled": embedding_model is not None,
        "eval_cases_used": len(eval_cases),
    }
    return {"stage": stage, "final": final, "audit": audit, "facts": facts}
