"""Matchday suspension demo pipeline.

Deterministic workflow with LLM calls inside -- no agents.

Flow:
  1. Call 1 (classify + extract) -> facts per case
  2. Branch: automatic (min == max) -> deterministic handbook reason
              range    (min != max) -> Call 2 (LLM recommends a number)
  3. Stage rows -> final rows (pending human approval)
  4. Human approves / overrides in the app

Error handling in this version:
  - Missing API key, missing/corrupt handbook, or a Call 1 failure that
    yields no usable facts are UNRECOVERABLE and raise PipelineError
    with a plain-English message. The app should catch PipelineError
    and show it with st.error, not a traceback.
  - Call 2 failures are RECOVERABLE: that case falls back to the
    handbook minimum, with the reason and a warning recorded.
  - Unknown misconduct codes are NOT silently dropped: they are
    recorded in the warnings list returned by run_pipeline.
  - Retrieval failures are recoverable: retrieval is disabled for
    that run and a warning is recorded; the pipeline still runs.
"""

import json
import os
import time

from prompts import CALL_1_SYSTEM, CALL_1_USER, CALL_2_SYSTEM, CALL_2_USER


class PipelineError(Exception):
    """Unrecoverable pipeline failure, safe to show to the user."""


# ---------------------------------------------------------------------------
# Handbook
# ---------------------------------------------------------------------------

def load_handbook(path="discipline.json"):
    """Load the handbook rules. Raises PipelineError if missing/corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise PipelineError(
            f"Handbook file not found: {path}. "
            "Make sure discipline.json is in the same folder as pipeline.py."
        )
    except json.JSONDecodeError as e:
        raise PipelineError(f"Handbook file {path} is not valid JSON: {e}")

    rules = data.get("rules") if isinstance(data, dict) else None
    if not rules:
        raise PipelineError(
            f"Handbook file {path} has no 'rules' list. "
            "Expected the discipline.json format with a 'rules' key."
        )

    handbook = {}
    for rule in rules:
        # One spelling everywhere: misconduct_code. discipline.json,
        # eval cases, Call 1 output, and the rest of the pipeline all
        # already use it; "code" was the one outlier, and it emptied
        # the handbook and raised "contains no usable rules".
        code = rule.get("misconduct_code")
        if not code:
            continue
        handbook[code] = rule
    if not handbook:
        raise PipelineError(f"Handbook file {path} contains no usable rules.")
    return handbook


def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _get_client():
    """Return an OpenAI client.

    Key order: environment variable first, then Streamlit secrets.
    Streamlit secrets live in Streamlit Cloud Settings, never in the repo.
    Raises PipelineError (not a raw auth traceback) if no key is found.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise PipelineError(
            "OPENAI_API_KEY is missing. Add it in Streamlit Cloud: "
            "app Settings > Secrets, as OPENAI_API_KEY = \"sk-...\", "
            "then reboot the app."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise PipelineError(
            "The 'openai' package is not installed. "
            "Check requirements.txt includes openai."
        )

    return OpenAI(api_key=api_key)


def _call_llm_json(client, model, system_prompt, user_prompt):
    """One LLM call. Returns (parsed_json_or_None, error_or_None).

    Never raises for an API/parse failure -- the caller decides whether
    that failure is recoverable (Call 2 fallback) or fatal (Call 1).
    """
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.output_text
    except Exception as e:
        return None, f"LLM call failed: {e}"

    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"LLM returned invalid JSON: {e}"


# ---------------------------------------------------------------------------
# Call 1: classify + extract
# ---------------------------------------------------------------------------

def _strip_player_name(action, player_name):
    """Remove the player name from the action text, in code.

    The action must not carry the player name (identity is stitched
    back in code at the end). Simple string replace is enough here:
    the name came from Call 1 itself, so it matches exactly.
    """
    if not action or not player_name:
        return action
    return action.replace(player_name, "").strip()


def get_facts(client, model, report_text):
    """Call 1. Returns (facts_list, error_or_None).

    Facts are NOT trusted blindly: suspension_type is re-derived
    from the handbook in _enrich_facts, because Call 1 never sees
    the handbook table and therefore cannot know min == max.
    """
    if not report_text or not report_text.strip():
        return [], "No report text was provided."

    # Do NOT use .format() here: the real CALL_1_USER contains a JSON
    # schema with single braces, so .format() raises KeyError: 'facts'
    # before the LLM is ever called. Replace the placeholder instead,
    # which leaves every other brace in the prompt untouched.
    # The real prompt uses {cases_json}; older stubs used {report}.
    user_prompt = CALL_1_USER.replace("{cases_json}", report_text).replace(
        "{report}", report_text
    )
    if report_text not in user_prompt:
        user_prompt += f"\n\nReport:\n{report_text}"
    data, error = _call_llm_json(client, model, CALL_1_SYSTEM, user_prompt)
    if error:
        return [], error
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        return [], "LLM response did not contain a 'facts' list."
    facts = data["facts"]
    # A free-text report has no case_id to copy, so the LLM may omit
    # it. Assign one in code rather than letting stage/final rows
    # (and the app's stage lookup) key on None.
    for i, fact in enumerate(facts):
        if isinstance(fact, dict) and not fact.get("case_id"):
            fact["case_id"] = f"CASE-{i + 1}"
    return facts, None


def _enrich_facts(facts, handbook):
    """Attach handbook rules, derive suspension_type, strip names.

    Returns (enriched_facts, warnings). Facts with an unknown
    misconduct code are dropped BUT recorded as warnings -- never
    silently.
    """
    enriched = []
    warnings = []
    for fact in facts:
        if not isinstance(fact, dict):
            warnings.append("Skipped a malformed fact (not an object).")
            continue
        code = fact.get("misconduct_code")
        rule = handbook.get(code)
        if rule is None:
            warnings.append(
                f"Skipped case {fact.get('case_id', '?')}: "
                f"misconduct code '{code}' is not in discipline.json."
            )
            continue
        min_games = _safe_int(rule.get("min_games"))
        max_games = _safe_int(rule.get("max_games"))
        if min_games is None or max_games is None:
            warnings.append(
                f"Skipped case {fact.get('case_id', '?')}: "
                f"rule for '{code}' has no valid min/max games."
            )
            continue
        player_name = fact.get("player_name")
        enriched.append(
            {
                "case_id": fact.get("case_id"),
                "player_name": player_name,
                "team_name": fact.get("team_name"),
                "misconduct_code": code,
                "suspension_type": "automatic" if min_games == max_games else "range",
                "action": _strip_player_name(fact.get("action"), player_name),
                "missing_facts": fact.get("missing_facts") or [],
                "min_games": min_games,
                "max_games": max_games,
                "handbook_section": rule.get("handbook_section"),
                "handbook_text": rule.get("handbook_text"),
            }
        )
    return enriched, warnings


# ---------------------------------------------------------------------------
# Automatic path (deterministic, same as production)
# ---------------------------------------------------------------------------

def _automatic_reason(code):
    if code in ("2CT", "DH", "DF", "S"):
        return "Automatic suspension per handbook."
    return "Automatic recommendation based on handbook minimum."


# ---------------------------------------------------------------------------
# Eval cases + retrieval (optional, failure is recoverable)
# ---------------------------------------------------------------------------

def load_eval_cases(path="eval_cases.json"):
    """Load preloaded eval cases. Returns [] if the file is missing.

    A missing eval file must not kill the pipeline -- retrieval is
    an enhancement, not a requirement.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        return data.get("cases", [])
    return data if isinstance(data, list) else []


# Embedding model cache: loaded once per process, not once per run.
# Streamlit reruns app.py top-to-bottom on every interaction, but imported
# modules persist, so this singleton survives reruns. Without it, torch +
# the model reload on every Run click -- slow, and repeated allocation
# on a ~1GB Streamlit Cloud free instance.
# Note: this fixes reload cost, not peak memory. The model still loads
# in full on the first range case; if that alone OOMs, retrieval fails
# gracefully (see retrieve_top_k) rather than crashing the pipeline.
_MODEL_CACHE = {"model": None, "util": None, "failed": False}


def _get_embedding_model():
    """Lazy-load sentence-transformers, cached. Returns (model, util) or (None, None)."""
    if _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["util"]
    if _MODEL_CACHE["failed"]:
        return None, None
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["util"] = util
        return model, util
    except Exception:
        # Remember the failure so we don't retry (and re-pay) every run.
        _MODEL_CACHE["failed"] = True
        return None, None


def retrieve_top_k(query_action, eval_cases, code, k=3):
    """Top-k same-code eval cases by whole-action semantic similarity.

    Returns (examples_list, error_or_None). Excludes a case identical
    to the query so eval never retrieves itself. Never raises.
    """
    if not query_action or not eval_cases:
        return [], None
    model, util = _get_embedding_model()
    if model is None:
        return [], "sentence-transformers not available; retrieval disabled."

    candidates = [c for c in eval_cases if c.get("misconduct_code") == code]
    candidates = [c for c in candidates if c.get("action") != query_action]
    if not candidates:
        return [], None

    try:
        query_emb = model.encode(query_action, convert_to_tensor=True)
        docs = [c.get("action", "") for c in candidates]
        doc_emb = model.encode(docs, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, doc_emb)[0]
        ranked = sorted(
            zip(candidates, scores.tolist()), key=lambda pair: pair[1], reverse=True
        )
        examples = []
        for case, _score in ranked[:k]:
            examples.append(
                {
                    "case_id": case.get("case_id"),
                    "action": case.get("action"),
                    "min_games": case.get("min_games"),
                    "max_games": case.get("max_games"),
                    "expected_games": case.get("expected_games"),
                    "expected_reason": case.get("expected_reason"),
                }
            )
        return examples, None
    except Exception as e:
        return [], f"Retrieval failed: {e}"


# ---------------------------------------------------------------------------
# Call 2: range recommendation (action only)
# ---------------------------------------------------------------------------

def get_ai_recommendations(client, model, range_facts, eval_cases):
    """Call 2, batched. Returns (recommendations_dict, warnings, retrieval_enabled, examples_by_case).

    recommendations_dict: case_id -> {recommended_games, recommended_reason, source}
    A per-case failure falls back to the handbook minimum, with a
    warning -- it never drops the case.
    """
    warnings = []
    retrieval_enabled = False
    examples_by_case = {}
    cases_json = []
    for fact in range_facts:
        examples, ret_error = retrieve_top_k(
            fact["action"], eval_cases, fact["misconduct_code"], k=3
        )
        examples_by_case[fact["case_id"]] = examples
        if examples:
            retrieval_enabled = True
        elif ret_error:
            warnings.append(f"Retrieval for {fact['case_id']}: {ret_error}")
        cases_json.append(
            {
                "case_id": fact["case_id"],
                "misconduct_code": fact["misconduct_code"],
                "min_games": fact["min_games"],
                "max_games": fact["max_games"],
                "action": fact["action"],
                "examples": examples,
            }
        )

    data, error = _call_llm_json(
        client, model, CALL_2_SYSTEM, CALL_2_USER.format(cases_json=json.dumps(cases_json))
    )

    by_case = {}
    if error:
        warnings.append(f"Call 2 failed ({error}); all range cases fell back to minimum.")
    elif not isinstance(data, dict) or not isinstance(data.get("recommendations"), list):
        warnings.append("Call 2 returned no 'recommendations' list; all range cases fell back to minimum.")
    else:
        for rec in data["recommendations"]:
            if not isinstance(rec, dict) or "case_id" not in rec:
                continue
            by_case[rec["case_id"]] = rec

    recommendations = {}
    for fact in range_facts:
        rec = by_case.get(fact["case_id"])
        games = _safe_int(rec.get("recommended_games")) if rec else None
        reason = rec.get("recommended_reason") if rec else None
        if games is None or reason is None:
            warnings.append(
                f"Case {fact['case_id']}: no valid Call 2 recommendation; "
                "fell back to handbook minimum."
            )
            recommendations[fact["case_id"]] = {
                "recommended_games": fact["min_games"],
                "recommended_reason": "Defaulted to handbook minimum because AI recommendation failed.",
                "source": "fallback_minimum",
            }
        else:
            # Clamp to the handbook range -- the LLM must never exceed it.
            clamped = max(fact["min_games"], min(fact["max_games"], games))
            if clamped != games:
                warnings.append(
                    f"Case {fact['case_id']}: LLM suggested {games} games, "
                    f"clamped to handbook range {fact['min_games']}-{fact['max_games']}."
                )
            recommendations[fact["case_id"]] = {
                "recommended_games": clamped,
                "recommended_reason": reason,
                "source": "AI",
            }
    return recommendations, warnings, retrieval_enabled, examples_by_case


# ---------------------------------------------------------------------------
# Stage / final / approval
# ---------------------------------------------------------------------------

def build_final_summary(final_row):
    """Assemble identity + decision in code. The LLM never sees names."""
    team = final_row.get("team_name") or "Unknown team"
    player = final_row.get("player_name") or "Unknown player"
    reason = final_row.get("final_reason") or final_row.get("recommended_reason") or ""
    return f"{team} - {player}: {reason}".strip()


def approve(final_row, final_games=None, final_reason=None):
    """Apply a human decision. Returns an updated copy of the row.

    Human reason replaces the LLM reason as the official record.
    Status is 'overridden' only if the human changed games or reason.
    """
    updated = dict(final_row)
    recommended_games = final_row.get("recommended_games")
    recommended_reason = final_row.get("recommended_reason")
    if final_games is None:
        final_games = recommended_games
    if final_reason is None:
        final_reason = recommended_reason
    updated["final_games"] = final_games
    updated["final_reason"] = final_reason
    updated["status"] = (
        "overridden"
        if (final_games != recommended_games or final_reason != recommended_reason)
        else "approved"
    )
    updated["final_summary"] = build_final_summary(updated)
    return updated


def add_to_session_eval(session_cases, stage_row, fact):
    """Add a human-approved case to the session eval store (in place).

    Session-only: it helps retrieval in this tab, and is gone on a
    new tab. Returns the new session case.
    """
    final = approve(
        stage_row,
        final_games=stage_row.get("final_games"),
        final_reason=stage_row.get("final_reason"),
    )
    case_id = f"SESSION-{len(session_cases) + 1:03d}"
    session_case = {
        "case_id": case_id,
        "section": "Session - Human approved",
        "misconduct_code": fact.get("misconduct_code"),
        "min_games": fact.get("min_games"),
        "max_games": fact.get("max_games"),
        "player_name": fact.get("player_name"),
        "team_name": fact.get("team_name"),
        "action": fact.get("action"),
        "expected_games": final.get("final_games"),
        "expected_reason": final.get("final_reason"),
    }
    session_cases.append(session_case)
    return session_case


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(report_text, session_cases=None):
    """Run one report end to end.

    Returns a dict with: facts, stage, final, audit, warnings, errors.
    Raises PipelineError only for unrecoverable failures (missing key,
    missing handbook, Call 1 producing no usable facts). Everything
    else is a fallback + warning, so a run never silently returns empty.
    """
    start = time.time()
    warnings = []
    errors = []

    handbook = load_handbook()
    client = _get_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-5")

    # --- Call 1 ---
    raw_facts, call1_error = get_facts(client, model, report_text)
    if call1_error:
        errors.append(call1_error)
    if not raw_facts:
        raise PipelineError(
            "Could not understand that report. "
            f"Detail: {call1_error or 'Call 1 returned no facts.'} "
            "Try rephrasing the report with the misconduct and what happened."
        )

    facts, enrich_warnings = _enrich_facts(raw_facts, handbook)
    warnings.extend(enrich_warnings)
    if not facts:
        raise PipelineError(
            "No usable cases from that report: every misconduct code was "
            "unknown to discipline.json. Check the code in the report."
        )

    # Force a fail on missing identity: a suspension cannot be
    # approved for "-" team / "-" player, so do not render a table
    # with blanks. The report must name the player and the team.
    missing_identity = []
    for fact in facts:
        missing = []
        if not (fact.get("player_name") or "").strip():
            missing.append("player name")
        if not (fact.get("team_name") or "").strip():
            missing.append("team name")
        if missing:
            missing_identity.append(
                f"case {fact.get('case_id', '?')}: missing {' and '.join(missing)}"
            )
    if missing_identity:
        raise PipelineError(
            "Cannot run this report: it is missing required information. "
            + "; ".join(missing_identity)
            + ". Add the player and team names to the report and try again."
        )

    # --- Branch ---
    automatic = [f for f in facts if f["suspension_type"] == "automatic"]
    range_facts = [f for f in facts if f["suspension_type"] == "range"]

    eval_cases = load_eval_cases()
    if session_cases:
        eval_cases = eval_cases + list(session_cases)

    recommendations = {}
    retrieval_enabled = False
    examples_by_case = {}
    if range_facts:
        (
            recommendations,
            call2_warnings,
            retrieval_enabled,
            examples_by_case,
        ) = get_ai_recommendations(client, model, range_facts, eval_cases)
        warnings.extend(call2_warnings)

    # --- Stage + final ---
    stage = []
    final = []
    for fact in facts:
        if fact["suspension_type"] == "automatic":
            rec_games = fact["min_games"]
            rec_reason = _automatic_reason(fact["misconduct_code"])
            source = "Automatic"
        else:
            rec = recommendations[fact["case_id"]]
            rec_games = rec["recommended_games"]
            rec_reason = rec["recommended_reason"]
            source = rec["source"]

        stage_row = {
            "case_id": fact["case_id"],
            "misconduct_code": fact["misconduct_code"],
            "min_games": fact["min_games"],
            "max_games": fact["max_games"],
            "handbook_section": fact["handbook_section"],
            # Without this, the app's Handbook Text column showed only
            # the section number, not the handbook text itself.
            "handbook_text": fact["handbook_text"],
            "recommended_games": rec_games,
            "recommended_reason": rec_reason,
            "source": source,
            "retrieved_examples": examples_by_case.get(fact["case_id"], []),
        }
        stage.append(stage_row)

        final_row = {
            **stage_row,
            "player_name": fact["player_name"],
            "team_name": fact["team_name"],
            "suspension_type": fact["suspension_type"],
            "action": fact["action"],
            "final_games": None,
            "final_reason": None,
            "approved_by": None,
            "status": "pending",
        }
        final.append(final_row)

    audit = {
        "model": model,
        "rule_version": 1,
        "latency_seconds": round(time.time() - start, 2),
        "retrieval_enabled": retrieval_enabled,
        "eval_cases_used": len(eval_cases),
        "stage_sources": sorted({row["source"] for row in stage}),
    }

    return {
        "facts": facts,
        "stage": stage,
        "final": final,
        "audit": audit,
        "warnings": warnings,
        "errors": errors,
    }
