from dataclasses import dataclass, field
from typing import Literal, Optional


# Call 1 output: one facts object per case.
# suspension_type is validated against discipline.json after Call 1,
# not trusted blindly from the LLM.
@dataclass
class Call1Facts:
    case_id: str
    player_name: str
    team_name: str
    misconduct_code: str
    suspension_type: Literal["automatic", "range"]
    action: str  # everything, including context. Player name removed.
    missing_facts: list[str] = field(default_factory=list)


# Call 2 input: action only, plus the handbook range.
# No team, no player, no full supplemental paragraph.
@dataclass
class Call2Case:
    case_id: str
    action: str
    misconduct_code: str
    handbook_section: str
    handbook_text: str
    min_games: int
    max_games: int


# Call 2 output, before clamping.
@dataclass
class Call2Output:
    case_id: str
    recommended_games: int
    recommended_reason: str


# Stage output: what goes into the stage JSON.
@dataclass
class StageRecommendation:
    case_id: str
    misconduct_code: str
    min_games: int
    max_games: int
    handbook_section: str
    handbook_text: str
    recommended_games: int
    recommended_reason: str
    recommendation_source: Literal[
        "AI", "Automatic", "fallback_minimum", "fallback_missing"
    ]


# Final output: human-approved version.
# final_reason replaces recommended_reason when a human overrides,
# per your design — the LLM reason is not the official record then.
@dataclass
class FinalRecommendation:
    stage: StageRecommendation
    final_games: Optional[int] = None
    final_reason: Optional[str] = None
    approved_by: Optional[str] = None
    status: Literal["pending", "approved", "overridden"] = "pending"

    @property
    def final_summary(self) -> str:
        # Team + player + statement, stitched in code.
        # The LLM never saw team/player, so this is added back here.
        raise NotImplementedError  # filled in pipeline.py, needs team/player
