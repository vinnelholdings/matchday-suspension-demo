"""Unit tests for pipeline.py -- pytest style, plain asserts.

No API key, no network, no model download. The OpenAI call, the
handbook load, the eval load, and retrieval are all mocked out, so
these tests check pipeline logic only: branch, clamp, fallback,
name-strip, warnings, and approval.

Run: pytest test_pipeline.py
Requires pytest locally only. Do NOT add pytest to requirements.txt --
Streamlit Cloud does not need it to run the app.
"""

import pytest

import pipeline
from pipeline import PipelineError


HANDBOOK = {
    "2CT": {
        "misconduct_code": "2CT",
        "min_games": 1,
        "max_games": 1,
        "handbook_section": "14.2",
        "handbook_text": "Second caution.",
    },
    "VC": {
        "misconduct_code": "VC",
        "min_games": 2,
        "max_games": 4,
        "handbook_section": "14.7",
        "handbook_text": "Violent conduct.",
    },
}


def _patch_basics(monkeypatch):
    monkeypatch.setattr(pipeline, "load_handbook", lambda *a, **k: HANDBOOK)
    monkeypatch.setattr(pipeline, "_get_client", lambda: object())
    monkeypatch.setattr(pipeline, "load_eval_cases", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "retrieve_top_k", lambda *a, **k: ([], None))


def _call1_fact(**overrides):
    fact = {
        "case_id": "CASE-1",
        "player_name": "Alex Test",
        "team_name": "Test FC",
        "misconduct_code": "2CT",
        "action": "Alex Test received a second caution.",
        "missing_facts": [],
    }
    fact.update(overrides)
    return fact


def test_player_name_stripped_from_action_but_kept_on_row(monkeypatch):
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_call_llm_json",
        lambda *a, **k: ({"facts": [_call1_fact()]}, None),
    )
    result = pipeline.run_pipeline("some report")
    assert len(result["final"]) == 1
    row = result["final"][0]
    # Identity is stitched back in code, not carried in the action.
    assert "Alex Test" not in row["action"]
    assert row["player_name"] == "Alex Test"
    assert row["team_name"] == "Test FC"


def test_automatic_branch_never_calls_call2(monkeypatch):
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_call_llm_json",
        lambda *a, **k: ({"facts": [_call1_fact()]}, None),
    )

    def _forbidden(*a, **k):
        raise AssertionError("Call 2 must not run for an automatic case")

    monkeypatch.setattr(pipeline, "get_ai_recommendations", _forbidden)
    result = pipeline.run_pipeline("some report")
    assert result["stage"][0]["source"] == "Automatic"
    assert result["stage"][0]["recommended_games"] == 1
    assert result["final"][0]["status"] == "pending"


def test_range_branch_clamps_to_handbook_max(monkeypatch):
    _patch_basics(monkeypatch)
    calls = iter(
        [
            ({"facts": [_call1_fact(misconduct_code="VC", action="Pushed opponent.")]}, None),
            ({"recommendations": [{"case_id": "CASE-1", "recommended_games": 99, "recommended_reason": "Very bad."}]}, None),
        ]
    )
    monkeypatch.setattr(pipeline, "_call_llm_json", lambda *a, **k: next(calls))
    result = pipeline.run_pipeline("some report")
    assert result["stage"][0]["recommended_games"] == 4
    assert result["stage"][0]["source"] == "AI"
    assert any("clamped" in w for w in result["warnings"])


def test_call2_failure_falls_back_to_minimum(monkeypatch):
    _patch_basics(monkeypatch)
    calls = iter(
        [
            ({"facts": [_call1_fact(misconduct_code="VC", action="Pushed opponent.")]}, None),
            (None, "API down"),
        ]
    )
    monkeypatch.setattr(pipeline, "_call_llm_json", lambda *a, **k: next(calls))
    result = pipeline.run_pipeline("some report")
    assert result["stage"][0]["recommended_games"] == 2
    assert result["stage"][0]["source"] == "fallback_minimum"
    assert result["warnings"], "fallback must be recorded as a warning"


def test_unknown_code_is_warned_not_silent(monkeypatch):
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_call_llm_json",
        lambda *a, **k: (
            {"facts": [_call1_fact(), _call1_fact(case_id="CASE-2", misconduct_code="XXX")]},
            None,
        ),
    )
    result = pipeline.run_pipeline("some report")
    assert len(result["final"]) == 1
    assert any("XXX" in w for w in result["warnings"])


def test_unknown_code_only_raises_pipeline_error(monkeypatch):
    _patch_basics(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_call_llm_json",
        lambda *a, **k: ({"facts": [_call1_fact(misconduct_code="XXX")]}, None),
    )
    with pytest.raises(PipelineError):
        pipeline.run_pipeline("some report")


def test_missing_api_key_raises_pipeline_error(monkeypatch):
    monkeypatch.setattr(pipeline, "load_handbook", lambda *a, **k: HANDBOOK)
    monkeypatch.setattr(
        pipeline, "_get_client", lambda: (_ for _ in ()).throw(PipelineError("OPENAI_API_KEY is missing."))
    )
    with pytest.raises(PipelineError, match="OPENAI_API_KEY"):
        pipeline.run_pipeline("some report")


def test_call1_empty_raises_pipeline_error(monkeypatch):
    _patch_basics(monkeypatch)
    monkeypatch.setattr(pipeline, "_call_llm_json", lambda *a, **k: ({"facts": []}, None))
    with pytest.raises(PipelineError):
        pipeline.run_pipeline("some report")


def test_approve_approved_vs_overridden_and_summary():
    base = {
        "team_name": "Test FC",
        "player_name": "Alex Test",
        "recommended_games": 2,
        "recommended_reason": "Minimum fits.",
    }
    approved = pipeline.approve(base)
    assert approved["status"] == "approved"
    assert approved["final_summary"] == "Test FC - Alex Test: Minimum fits."

    overridden = pipeline.approve(base, final_games=3, final_reason="Worse than minimum.")
    assert overridden["status"] == "overridden"
    assert overridden["final_games"] == 3
    assert overridden["final_summary"] == "Test FC - Alex Test: Worse than minimum."


def test_add_to_session_eval(monkeypatch):
    _patch_basics(monkeypatch)
    session_cases = []
    stage_row = {
        "case_id": "CASE-1",
        "misconduct_code": "VC",
        "min_games": 2,
        "max_games": 4,
        "final_games": 3,
        "final_reason": "Human decided 3.",
    }
    fact = {"misconduct_code": "VC", "min_games": 2, "max_games": 4, "player_name": "Alex Test", "team_name": "Test FC", "action": "Pushed opponent."}
    added = pipeline.add_to_session_eval(session_cases, stage_row, fact)
    assert len(session_cases) == 1
    assert added["case_id"] == "SESSION-001"
    assert added["expected_games"] == 3
    assert added["expected_reason"] == "Human decided 3."


def test_retrieval_disabled_without_model(monkeypatch):
    monkeypatch.setattr(pipeline, "_get_embedding_model", lambda: (None, None))
    examples, error = pipeline.retrieve_top_k("some action", [{"misconduct_code": "VC", "action": "other"}], "VC")
    assert examples == []
    assert error is not None
