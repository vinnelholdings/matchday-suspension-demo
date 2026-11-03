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
import logging
import os
import re
import sys
import time

from prompts import CALL_1_SYSTEM, CALL_1_USER, CALL_2_SYSTEM, CALL_2_USER

# Standard Python logging, not print(). Print output can sit
# buffered or in the wrong stream on Streamlit Cloud, which is why
# the timing print was hard to find in Cloud logs. A logger with a
# StreamHandler to stdout, level INFO, shows up in Manage app > Logs
# as a normal log line, with a timestamp and level.
logger = logging.getLogger("matchday_suspension_demo")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
# Do not propagate to root, otherwise the same line appears twice
# when the host (Streamlit / pytest) also configures root logging.
logger.propagate = False


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


_MINUTE_PATTERNS = [
    # "20th minute", "20 minute", "minute 20"
    re.compile(r"(\d{1,3})(?:st|nd|rd|th)?\s*(?:minute|min)\b", re.IGNORECASE),
    re.compile(r"\bminute\s*(\d{1,3})\b", re.IGNORECASE),
    # "20'" as in a match minute
    re.compile(r"\b(\d{1,3})'\b"),
]


def extract_minute(report_text, fact=None):
    """Minute a report happened, if the report or the fact gives one.

    Checked against the report text first in run_pipeline: the LLM
    may drop the minute from its fact even when the report has it,
    and that must not count as missing. A fact minute is used only
    when the report text has no minute pattern.
    """
    if fact is not None:
        minute = _safe_int(fact.get("minute"))
        if minute is not None:
            return minute
    if not report_text:
        return None
    for pattern in _MINUTE_PATTERNS:
        match = pattern.search(report_text)
        if match:
            return _safe_int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _get_config(key, default=None):
    """Read a config value from environment first, then Streamlit secrets.

    This matters: Streamlit Secrets are NOT environment variables
    on their own -- st.secrets.get() is required, otherwise a Secret
    set in Cloud would be silently ignored and the default used.
    """
    value = os.environ.get(key)
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(key)
    except Exception:
        value = None
    return value or default


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


# Last LLM usage, read by the caller so the Log tab can show
# what the API actually returned: input, output, reasoning, and
# total tokens. Kept as a module global so _call_llm_json keeps its
# (data, error) signature, which tests mock. Reasoning tokens are
# the key question for Call 1 slowness: a simple fact extraction
# should not spend many reasoning tokens.
_LAST_LLM_USAGE = {}


def _extract_usage(response):
    """Usage dict from an OpenAI Responses API response, safely.

    Handles attribute and dict access, and missing fields, because
    usage shape varies by model and by mock in tests.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    details = _get(usage, "output_tokens_details")
    return {
        "input_tokens": _get(usage, "input_tokens"),
        "output_tokens": _get(usage, "output_tokens"),
        "reasoning_tokens": _get(details, "reasoning_tokens"),
        "total_tokens": _get(usage, "total_tokens"),
    }


def _call_llm_json(client, model, system_prompt, user_prompt, reasoning_effort=None):
    """One LLM call. Returns (parsed_json_or_None, error_or_None).

    Never raises for an API/parse failure -- the caller decides whether
    that failure is recoverable (Call 2 fallback) or fatal (Call 1).
    Usage from the last call is stored in _LAST_LLM_USAGE.
    reasoning_effort is None by default, meaning do not pass the
    reasoning parameter at all, so the API uses its default medium.
    Pass "low" only for Call 1, the light_model extraction call.
    """
    try:
        _kwargs = {}
        if reasoning_effort:
            _kwargs["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **_kwargs,
        )
        text = response.output_text
        _LAST_LLM_USAGE.clear()
        _LAST_LLM_USAGE.update(_extract_usage(response))
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
    # The prompt template now uses {report} in place, so the report
    # is no longer appended after the return instruction by fallback.
    user_prompt = CALL_1_USER.replace("{cases_json}", report_text).replace(
        "{report}", report_text
    )
    if report_text not in user_prompt:
        user_prompt += f"\n\nReport:\n{report_text}"
    # Low reasoning for light_model only: nano at default medium used
    # 1408 reasoning tokens for Call 1 (Nov 3 run), versus 512-576 on
    # gpt-5, so the model change alone did not speed Call 1 up.
    # Call 2 keeps the default medium reasoning, unchanged.
    data, error = _call_llm_json(
        client, model, CALL_1_SYSTEM, user_prompt, reasoning_effort="minimal"
    )
    # Usage is stored globally by _call_llm_json; get_facts itself
    # keeps its (facts, error) signature, run_pipeline reads the
    # global after this call to log Call 1 usage.
    if error:
        return [], error
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        return [], "LLM response did not contain a 'facts' list."
    facts = data["facts"]
    # Case_id is autoincremented in code, not taken from the LLM.
    # Call 1 no longer outputs case_id at all: a free-text report
    # has no case_id to copy, and letting the LLM invent one caused
    # duplicates, non-sequential IDs, and wasted reasoning. Assign
    # CASE-1, CASE-2, ... in the order facts are returned, per run.
    # Session eval cases get their own SESSION-xxx IDs separately.
    for i, fact in enumerate(facts):
        if isinstance(fact, dict):
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
                "minute": _safe_int(fact.get("minute")),
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


# Last embedding timing, read by get_ai_recommendations so the
# app can log it next to the embeddings themselves. This is timing
# only, not a speed fix: it splits retrieval into model load, query
# encode, docs encode, and similarity, so we can see if embeddings
# are overkill before replacing them with something faster.
_LAST_EMBEDDING_TIMING = {}


def retrieve_top_k(query_action, eval_cases, code, k=3):
    """Top-k same-code eval cases by whole-action semantic similarity.

    Returns (examples_list, error_or_None). Never raises.
    Each example carries a "score" (cosine similarity, 0-1, higher is
    closer) so the app can show, not just claim, why those examples
    were picked. The full log (candidates, scores, model state) is
    built in get_ai_recommendations, which calls this function.
    NOTE: identical actions are NOT excluded here. An earlier version
    excluded an identical action so an eval case could not retrieve
    itself, but that broke the live demo: a session case a human just
    approved, run again with identical wording, was excluded and
    never retrieved, which is exactly what the user tested.
    """
    if not query_action or not eval_cases:
        return [], None
    _emb_start = time.time()
    _model_was_cached = _MODEL_CACHE["model"] is not None
    model, util = _get_embedding_model()
    model_load_seconds = round(time.time() - _emb_start, 2)
    if model is None:
        _LAST_EMBEDDING_TIMING.clear()
        _LAST_EMBEDDING_TIMING.update(
            {
                "model_was_cached": _model_was_cached,
                "model_load_seconds": model_load_seconds,
                "query_encode_seconds": 0.0,
                "docs_encode_seconds": 0.0,
                "similarity_seconds": 0.0,
                "docs_encoded": 0,
                "error": "sentence-transformers not available; retrieval disabled.",
            }
        )
        return [], "sentence-transformers not available; retrieval disabled."

    candidates = [c for c in eval_cases if c.get("misconduct_code") == code]
    if not candidates:
        return [], None

    try:
        _t = time.time()
        query_emb = model.encode(query_action, convert_to_tensor=True)
        query_encode_seconds = round(time.time() - _t, 2)
        docs = [c.get("action", "") for c in candidates]
        _t = time.time()
        doc_emb = model.encode(docs, convert_to_tensor=True)
        docs_encode_seconds = round(time.time() - _t, 2)
        _t = time.time()
        scores = util.cos_sim(query_emb, doc_emb)[0]
        similarity_seconds = round(time.time() - _t, 2)
        _LAST_EMBEDDING_TIMING.clear()
        _LAST_EMBEDDING_TIMING.update(
            {
                "model_was_cached": _model_was_cached,
                "model_load_seconds": model_load_seconds,
                "query_encode_seconds": query_encode_seconds,
                "docs_encode_seconds": docs_encode_seconds,
                "similarity_seconds": similarity_seconds,
                "docs_encoded": len(docs),
                "error": None,
            }
        )
        ranked = sorted(
            zip(candidates, scores.tolist()), key=lambda pair: pair[1], reverse=True
        )
        # Demo-only bias: human-added session cases are weighed
        # higher than pre-seed cases, by +0.10, capped at 1.0, and
        # ranked by the boosted score. This is ONLY because the
        # pre-seed cases are synthetic, not real historical data.
        # In prod with real historical decisions, the pre-seed would
        # be real data too, and this boost would not apply.
        # The boost is logged (raw_score, session_boost, score) and
        # the source is passed to Call 2, so it is not hidden.
        examples = []
        boosted_ranked = []
        for case, raw_score in ranked:
            is_session = str(case.get("case_id", "")).startswith("SESSION-")
            boost = 0.10 if is_session else 0.0
            final_score = min(1.0, float(raw_score) + boost)
            boosted_ranked.append((case, raw_score, boost, final_score))
        boosted_ranked.sort(key=lambda item: item[3], reverse=True)
        for case, raw_score, boost, final_score in boosted_ranked[:k]:
            is_session = str(case.get("case_id", "")).startswith("SESSION-")
            examples.append(
                {
                    "case_id": case.get("case_id"),
                    "action": case.get("action"),
                    "min_games": case.get("min_games"),
                    "max_games": case.get("max_games"),
                    "expected_games": case.get("expected_games"),
                    "expected_reason": case.get("expected_reason"),
                    "source": "human approved" if is_session else "synthetic preloaded",
                    "score": round(float(final_score), 3),
                    "raw_score": round(float(raw_score), 3),
                    "session_boost": boost,
                }
            )
        return examples, None
    except Exception as e:
        return [], f"Retrieval failed: {e}"


# ---------------------------------------------------------------------------
# Call 2: range recommendation (action only)
# ---------------------------------------------------------------------------

def get_ai_recommendations(client, model, range_facts, eval_cases):
    """Call 2, batched. Returns (recommendations_dict, warnings, retrieval_enabled, examples_by_case, retrieval_log_by_case, timings).

    recommendations_dict: case_id -> {recommended_games, recommended_reason, source}
    A per-case failure falls back to the handbook minimum, with a
    warning -- it never drops the case.
    retrieval_log_by_case: case_id -> {query_action, code, model_loaded,
    candidates_considered, top_examples, error, retrieval_seconds} --
    the embedding and similarity steps, shown in the app when retrieval
    detail is on. timings: {retrieval_seconds, call2_seconds} -- timing
    only, no speed fix yet, so we can see what actually takes time.
    """
    warnings = []
    retrieval_enabled = False
    examples_by_case = {}
    retrieval_log_by_case = {}
    cases_json = []
    retrieval_seconds_total = 0.0
    for fact in range_facts:
        _ret_start = time.time()
        examples, ret_error = retrieve_top_k(
            fact["action"], eval_cases, fact["misconduct_code"], k=3
        )
        _ret_seconds = round(time.time() - _ret_start, 2)
        retrieval_seconds_total += _ret_seconds
        examples_by_case[fact["case_id"]] = examples
        # Candidates are same-code cases only, exactly as
        # retrieve_top_k filters them. This log is computed here so
        # it stays true even if retrieve_top_k is mocked in tests.
        candidates_considered = [
            c for c in eval_cases if c.get("misconduct_code") == fact["misconduct_code"]
        ]
        retrieval_log_by_case[fact["case_id"]] = {
            "query_action": fact["action"],
            "code": fact["misconduct_code"],
            "model_loaded": _MODEL_CACHE["model"] is not None,
            "candidates_considered": len(candidates_considered),
            "candidate_case_ids": [c.get("case_id") for c in candidates_considered],
            "top_examples": examples,
            "error": ret_error,
            "retrieval_seconds": _ret_seconds,
            # Embedding breakdown, logged next to the embeddings:
            # model load, query encode, docs encode, similarity.
            # If retrieve_top_k was mocked in tests, this will be {}.
            "embedding": dict(_LAST_EMBEDDING_TIMING),
        }
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

    _call2_start = time.time()
    data, error = _call_llm_json(
        client, model, CALL_2_SYSTEM, CALL_2_USER.format(cases_json=json.dumps(cases_json))
    )
    call2_seconds = round(time.time() - _call2_start, 2)

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
    return (
        recommendations,
        warnings,
        retrieval_enabled,
        examples_by_case,
        retrieval_log_by_case,
        {
            "retrieval_seconds": round(retrieval_seconds_total, 2),
            "call2_seconds": call2_seconds,
            "call2_usage": dict(_LAST_LLM_USAGE),
        },
    )


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
    # Call 1 and Call 2 models are config, not hardcoded choices.
    # Simple names, set in Streamlit Secrets (or environment):
    #   light_model = Call 1, extraction only -- e.g. gpt-5-nano,
    #     normal default medium reasoning for now, no low/minimal
    #     yet, so the model change can be measured alone.
    #   full_model = Call 2, suspension judgment -- e.g. gpt-5.
    # "full" is used instead of "strong": strong is ambiguous in
    # this domain (a strong suspension), full means full capability.
    # If light_model is not set, Call 1 falls back to full_model.
    model = _get_config("full_model", "gpt-5")
    model_call1 = _get_config("light_model", model)

    # --- Call 1 ---
    _call1_start = time.time()
    raw_facts, call1_error = get_facts(client, model_call1, report_text)
    call1_seconds = round(time.time() - _call1_start, 2)
    call1_usage = dict(_LAST_LLM_USAGE)
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

    # Force a fail on missing required fields: a suspension cannot
    # be approved for "-" team / "-" player, and a report without a
    # minute cannot be placed in the match, so do not render a table
    # with blanks. Every report must have a player name, a team name,
    # and a minute. The minute is checked against the report text
    # itself, not just the LLM fact, so a minute the LLM drops still
    # counts as present.
    missing_required = []
    for fact in facts:
        missing = []
        if not (fact.get("player_name") or "").strip():
            missing.append("player name")
        if not (fact.get("team_name") or "").strip():
            missing.append("team name")
        if extract_minute(report_text, fact) is None:
            missing.append("minute")
        if missing:
            missing_required.append(
                f"case {fact.get('case_id', '?')}: missing {' and '.join(missing)}"
            )
    if missing_required:
        raise PipelineError(
            "Cannot run this report: it is missing required information. "
            + "; ".join(missing_required)
            + ". Add the player name, team name, and minute to the report and try again."
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
    retrieval_log_by_case = {}
    retrieval_seconds = 0.0
    call2_seconds = 0.0
    call2_usage = {}
    if range_facts:
        (
            recommendations,
            call2_warnings,
            retrieval_enabled,
            examples_by_case,
            retrieval_log_by_case,
            ai_timings,
        ) = get_ai_recommendations(client, model, range_facts, eval_cases)
        warnings.extend(call2_warnings)
        retrieval_seconds = ai_timings.get("retrieval_seconds", 0.0)
        call2_seconds = ai_timings.get("call2_seconds", 0.0)
        call2_usage = ai_timings.get("call2_usage", {})

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

    total_seconds = round(time.time() - start, 2)
    timings = {
        "call1_seconds": call1_seconds,
        "retrieval_seconds": retrieval_seconds,
        "call2_seconds": call2_seconds,
        "total_seconds": total_seconds,
        "call1_usage": call1_usage,
        "call2_usage": call2_usage,
    }
    # Console log, visible in Streamlit Cloud Manage app > Logs, so
    # timing can be checked without opening the Audit expander.
    logger.info(
        "[pipeline timing] call1=%ss retrieval=%ss call2=%ss total=%ss",
        call1_seconds,
        retrieval_seconds,
        call2_seconds,
        total_seconds,
    )
    logger.info(
        "[pipeline retrieval] enabled=%s eval_cases_used=%s",
        retrieval_enabled,
        len(eval_cases),
    )
    audit = {
        "model": model,
        "model_call1": model_call1,
        "model_call2": model,
        "reasoning_call1": "minimal",
        "reasoning_call2": "medium (default)",
        "rule_version": 1,
        "latency_seconds": total_seconds,
        "stage_seconds": timings,
        "retrieval_enabled": retrieval_enabled,
        "eval_cases_used": len(eval_cases),
        "stage_sources": sorted({row["source"] for row in stage}),
    }

    return {
        "facts": facts,
        "stage": stage,
        "final": final,
        "audit": audit,
        "timings": timings,
        "retrieval_log": retrieval_log_by_case,
        "warnings": warnings,
        "errors": errors,
    }
