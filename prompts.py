CALL_1_SYSTEM = """You extract facts from a soccer red card report.
Return JSON only, matching the schema in the user prompt exactly.
Do not add extra keys. Do not invent facts."""

CALL_1_USER = """Extract facts from the report below.

Report:
{report}

Return JSON only, matching this schema exactly:

{
  "facts": [
    {
      "player_name": "string, or null if not in the report",
      "team_name": "string, or null if not in the report",
      "misconduct_code": "string, e.g. VC, SFP, OL, 2CT, DH, DF, S",
      "action": "string, one blob encompassing everything, including context. Must NOT include the player name.",
      "missing_facts": ["string, ..."]
    }
  ]
}

Rules:
- Do NOT output suspension_type. The pipeline derives it in code
  from the handbook (min_games == max_games means automatic,
  otherwise range). You do not have the handbook table, so do not
  spend reasoning on it, and do not guess it.
- If a required fact is not in the report, set that field to null
  and list the fact name in missing_facts. Do not fill it in.
- action must NOT include the player name. That is the only
  name removed from the action text.

Return JSON only, in the schema above. No other text, no extra keys."""

CALL_2_SYSTEM = """You recommend suspension lengths for soccer misconduct cases. Return JSON only."""

CALL_2_USER = """You are helping recommend suspensions for multiple soccer misconduct incidents.

Treat each case independently. Do not mix facts between cases.

Rules:

* Return exactly one recommendation for each case_id provided.
* Copy each case_id exactly as given.
* Only recommend an integer number of games.
* You MUST stay within each case's handbook range.
* The misconduct type and handbook range are already given. Your job is only to recommend the appropriate number of games within that range.
* If the action is unclear, vague, or missing key details, recommend the minimum.
* Do not invent facts not present in the case.
* Keep each reason to 1–2 sentences in plain English.
* Be decisive. Do not hedge or use neutral/observational language.

---

Reason structure (REQUIRED):

Each reason MUST follow this logic:

1. Identify the key act (what happened).
2. Explain why it makes the incident more serious OR keeps it closer to the minimum.
3. Clearly justify why the chosen number fits.

The reason should sound like discipline logic, not a description of the play.

---

Writing guidance (VERY IMPORTANT):

* Always connect facts → severity → number.
* Do NOT stop at labels like "aggressive" or "deliberate."
* Do NOT compare to hypothetical worse incidents.
* Avoid vague phrasing like "less severe than…" unless explicitly stated in the report.
* Do NOT sound robotic (avoid repeating "high end of the range," etc.).

Instead, use natural conclusions tied to the number, such as:
- "so 2 games is appropriate"
- "so the minimum is appropriate"
- "which supports 3 games"
- "so a 3-game suspension fits"
- "which justifies 4 games"

---

Critical rule for mid-range (e.g., 3 in a 2–4 range):

When recommending above the minimum but below the maximum, you MUST explain BOTH:
* what makes the incident more serious, AND
* what prevents it from reaching the maximum

---

Severity guidance:

* For VC (violent conduct), increase severity when the act is clearly deliberate and aggressive. This includes punching, striking, kicking out, stomping, headbutting, retaliation, off-the-ball conduct, repeated actions, or forceful contact. Contact to vulnerable areas (face, head, neck, throat) or evidence of injury or treatment increases severity further.

* For SFP (serious foul play), increase severity when the challenge is dangerous or uncontrolled. This includes exposed studs, lunging, high or forceful contact (especially to shin, knee, ankle), excessive force, or actions that endanger safety. Injury or inability to continue increases severity further.

* For OL (offensive language/gestures), increase severity when the abuse is threatening, repeated, directed at officials/opponents, highly public, or includes gestures. Discriminatory or degrading language is more severe.

* These are signals, not strict rules. Base the recommendation on the totality of the facts.

* If the incident reflects a routine version of the offense with no strong aggravating details, stay at the minimum.

* Move to the maximum only when the action clearly shows strong aggravating facts or a serious outcome.

---

Examples of strong reasoning style:

* "The player threw a closed-fist punch to the opponent's face away from play, which is deliberate violent conduct to a vulnerable area, so 4 games is justified."

* "The player deliberately shoved an opponent off the ball, but there is no strike to the face or injury described, so the minimum of 2 games is appropriate."

* "The player swung an arm away from play and made forceful contact, which increases severity, but without contact to the face or injury described, a 3-game suspension fits."

* "The tackle involved exposed studs into the knee, creating clear danger to the opponent's safety, so 4 games is justified."

Cases:
{cases_json}

Return valid JSON only in this exact shape:
{{
  "recommendations": [
    {{
      "case_id": "<case_id>",
      "recommended_games": <integer>,
      "recommended_reason": "<short reason>"
    }}
  ]
}}
"""
