"""Matchday suspension demo pipeline.

Deterministic workflow with LLM calls inside, nothing else:
  Call 1 -> branch (automatic / range) -> Call 2 (range only)
  -> clamp / fallback -> stage JSON -> final JSON (after human approval)

Handbook ranges come from discipline.json, not a database.
No real cases, no real names — demo input is synthetic only.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from prompts import CALL_1_SYSTEM, CALL_1_USER, CALL_2_SYSTEM, CALL_2_USER

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


def _get_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


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


# ---------- Call 2: range reasoning, action only ----------

def _build_call2_cases_json(facts):
    # Call 2 sees action only. No team, no player, no full report.
    cases = []
    for fact in facts:
        cases.append(
            {
                "case_id": fact["case_id"],
                "action": fact["action"],
                "misconduct_code": fact["misconduct_code"],
                "handbook_section": fact["handbook_section"],
                "handbook_text": fact["handbook_text"],
                "min_games": fact["min_games"],
                "max_games": fact["max_games"],
            }
        )
    return json.dumps(cases, indent=2)


def get_ai_recommendations(facts, client):
    """Call 2, batched. Returns {case_id: stage-dict}.
    Clamped to [min, max]. Missing case or any failure
    falls back to the handbook minimum."""
    user_prompt = CALL_2_USER.replace(
        "{cases_json}", _build_call2_cases_json(facts)
    )
    data = _call_llm_json(client, CALL_2_SYSTEM, user_prompt)
    by_id = {}
    if data and "recommendations" in data:
        for rec in data["recommendations"]:
            by_id[rec.get("case_id")] = rec

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

def run_pipeline(reports, handbook_path="discipline.json", client=None):
    """reports: [{"case_id": "...", "report": "..."}]
    Returns {"stage": [...], "final": [...], "audit": {...}}.
    Final rows are pending until approve() is called."""
    start = time.time()
    handbook = load_handbook(handbook_path)
    if client is None:
        client = _get_client()

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
                }
            )
        else:
            range_facts.append(fact)

    if range_facts:
        ai_stage = get_ai_recommendations(range_facts, client)
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
        "input_hash": hashlib.sha256(
            json.dumps(reports, sort_keys=True).encode()
        ).hexdigest(),
        "model": os.environ.get("OPENAI_MODEL", "gpt-5"),
        "rule_version": 1,
        "latency_seconds": round(time.time() - start, 2),
        "stage_sources": [row["recommendation_source"] for row in stage],
    }
    return {"stage": stage, "final": final, "audit": audit}
