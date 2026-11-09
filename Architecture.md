# Architecture — Matchday Suspension Demo

This is the full design document. The README is first contact: pitch, try it, decisions in summary, headline evaluation, cost and timing, limitations. This document is where every decision lives in full, with trade-offs, rejected alternatives, failure modes, and detailed evaluation. It does not duplicate the README story, and the README does not duplicate this argument.

Where a decision was later superseded, this document states the current decision first and lists the superseded version in Appendix A, so a reader does not mistake an old test for current behaviour.

Live demo: https://matchday-suspension-demo.onrender.com/

## 1. Context and scope

### 1.1 Problem

The full Matchday app automated PDF game reports and suspensions for a huge amateur soccer league in the United States: 1,000 teams, 300 to 500 games a weekend, 100 suspensions a week minimum. Before the app, suspensions were judged manually, from referee supplemental reports sent by email as PDFs, with the handbook consulted by hand for each act.

In the full app, a referee report went from taking 45 minutes to about 8 minutes, league approval of a report took about 10 minutes, and a suspension went from about 3 hours to about 30 minutes. Those are full-app figures, not demo figures. Demo figures are in Section 9.

This demo takes just the suspension part from that application. It shows how a deterministic workflow, with LLM calls inside it only, can read a free-text referee report, apply a handbook, ground a recommendation in precedent cases, and require a human decision, without pretending to be the whole app or a production system.

### 1.2 In scope

- One free-text referee report at a time, in a Streamlit app.
- Classification and extraction from that report.
- A deterministic branch for automatic cases, and an LLM recommendation for range cases, clamped to the handbook.
- Retrieval of the top 3 precedent cases, by misconduct code first, then whole-action similarity.
- Human approval or override, with session learning from overrides.
- A Logs tab showing the last run only: timings, retrieved top 3, and audit for that run.
- Synthetic preloaded cases only, session-only persistence.

### 1.3 Out of scope, deliberately

- The rest of the Matchday app: game events, goals, cards, substitutions, check-ins, team documents, fines, bulk handling.
- A database, vector store, upload, download, cloud persistence, or any persistence beyond the session.
- Auth, tenancy, rate limiting, append-only audit, monitoring, or any production control. Those are prod gaps, listed in Section 11, not hidden omissions.
- Real data. No real reports, no real names, no real teams, no real users, no credentials, no real database in this demo. Handbook rules, codes, ranges, and text are the only prod-derived material, and they contain no people.

### 1.4 Readers

- A reviewer who wants to understand the design before judging the demo: start here, then try the demo.
- A future maintainer who wants to change a decision: read the relevant decision section first, not just the code. If a shape looks arbitrary, it is probably a decision recorded here.
- An interviewer: the README is the pitch, this document is the proof.

## 2. Invariants

These are the rules the code must not break, often stated as absences. If a change breaks one, it is not a refactor, it is a design change.

- **INV-CLAMP:** No sanction leaves the workflow except via the handbook clamp. A range recommendation is clamped to the handbook minimum to maximum for its code, in code, not by asking the LLM to stay in range.
- **INV-LLM:** The LLM never writes state, never sets a final sanction, never edits the handbook, and never sees team names in Call 2. Call 2 sees the action alone, plus retrieved precedents.
- **INV-HITL:** No outcome is final without a recorded human decision. Approval is the official record. A human reason replaces the LLM reason. Status is approved only if nothing changed, otherwise overridden.
- **INV-DATA:** Synthetic data only in this public demo. Preloaded cases are invented names and clubs only.
- **INV-SESSION:** Session learning is session-only. A new tab resets to the original preloaded cases. Session cases never rewrite the handbook, and never rewrite the preloaded corpus.
- **INV-HANDBOOK:** The handbook wins, always. If handbook minimum equals maximum, the case is automatic and deterministic, regardless of what any model output says.

## 3. Quality goals, ranked

1. **Correctness under the handbook.** A wrong code, or a number outside the range, is a failure, even if the reason reads well. Clamp violations must be 0 by construction, tested, not claimed.
2. **Human control that is real, not decorative.** The human must be able to see why, disagree, and change the outcome, and that change must be visible in the next similar case in the same session.
3. **Explainability to a non-technical reviewer.** A referee reviewer, a league authority, or an interviewer must be able to follow report, code, handbook, precedents, draft, clamped proposal, human decision, in that order.
4. **Demo speed.** A demo user will not wait 10 seconds for a first impression. Warm 5.71 seconds total was accepted. Cold start is parked, see Section 9.
5. **Honesty about scope.** Synthetic, session-only, and deferred prod controls must be stated, not implied away.

Non-goals, even though they would be reasonable goals in prod: prod accuracy claims, prod persistence, prod auth, prod bulk throughput, and a polished product UI. Those are listed as prod gaps, not as demo goals.

## 4. Constraints

- Cheap hosting. Render $7 tier, 512 MB, is the deployment target. That constraint forced the retrieval pivot in Section 7.3, it was not chosen after the design.
- Public repo. No real data, no secrets in code, no discriminatory slur quoted in a synthetic public file, even though prod quoting of offensive language is different, see Section 10.
- One file per concern in the demo, Streamlit frontend, pipeline, prompts, handbook JSON, preseed JSON. No framework beyond Streamlit and the OpenAI client, requirements are `openai` and `streamlit==1.66.1`.
- Model config is not hard-coded. `light_model` is Call 1, `full_model` is Call 2, set in Streamlit Secrets or environment, environment first in local code, then Secrets. If `light_model` is unset, Call 1 falls back to `full_model`. `OPENAI_API_KEY` is unchanged. On Render, both models are set to gpt-5-nano. "Full" was chosen over "strong" because strong is ambiguous in this domain, a strong suspension, while full means full capability. Model and reasoning are separate, light model is `gpt-5-nano`, not `gpt-5-nano-low`, there is no such model name, reasoning is a separate API parameter, in code only.

## 5. Architecture

### 5.1 Context diagram

```
Referee report (free text)
        |
        v
[ Streamlit app ] -- handbook (discipline.json)
        |           -- preloaded cases (eval_cases.json, 49 in repo)
        v
[ Pipeline ] -- OpenAI API (Call 1, Call 2, embeddings)
        |
        v
Human approver -- approves, or overrides with a reason
        |
        v
Final decision (approved / overridden), Logs for last run only
```

The user is a reviewer or a league authority in the demo. In prod, the user would be the designated authority, see Section 7.5. There is no other actor in the demo, no referee, no team, no external system writing to it.

### 5.2 Containers

- **Streamlit app (`app.py`):** UI only, adapted from the Matchday assignor suspensions page, with fines removed, no Matchday logo, no fines card, no bulk approve, no date column, because demo reports have no kickoff date and the demo must not fake one. Tabs, in order: Run a report, More examples, Preloaded cases, Logs. Heading is "Approve Suspensions" only, because no fines exist in this demo. Header right is just "Demo", the "Synthetic cases only" header label was removed on purpose, synthetic disclosure lives in the Preloaded tab subtitle, "Synthetic cases only, read-only", and in the README, not as header noise.
- **Pipeline (`pipeline.py`):** deterministic workflow, see Section 6. No UI code, no model prompting beyond calling the prompt functions.
- **Prompts (`prompts.py`):** Call 1 and Call 2 prompt text, in plain text, not schemas. See Section 6.2.
- **Handbook (`discipline.json`):** seven rules, field `misconduct_code` used everywhere. See Section 6.3.
- **Preloaded cases (`eval_cases.json`):** synthetic cases shipped in the repo, 49 cases in the repo, see Section 8. Session cases are added in memory only, never written to this file by the demo.
- **OpenAI API:** LLM calls and embeddings, via the same `OPENAI_API_KEY` already set on Render, no new secret for embeddings.

### 5.3 Components and code map

- `app.py`: Streamlit frontend, tabs, example, result table, Final Decision, Preloaded cases tab, Logs tab. No decision logic beyond calling the pipeline and writing back an approval.
- `pipeline.py`: `run_pipeline`, `approve`, handbook loading, required-fields check, branch, retrieval, clamp, fallback, timing and token logging. Case IDs are autoincremented in code, CASE-1, CASE-2, and so on per run, overwriting any ID the model supplies. Session eval cases keep SESSION-xxx IDs.
- `prompts.py`: Call 1 prompt, Call 2 prompt, severity guidance, general wording lines, worked example in neutral wording.
- `discipline.json`: handbook rules, see Section 6.3.
- `eval_cases.json`: synthetic preloaded cases, see Section 8.
- `test_pipeline.py`: local tests, pytest with plain asserts. Connector harness scripts exist locally for OpenAI calls, they are not part of the demo app.

### 5.4 Dynamic flow, one report

1. User pastes a report, or uses the example, and runs the pipeline.
2. Required-fields check: player name, team name, and minute must be present, or the pipeline fails with a plain message listing what is missing, instead of rendering a table with blanks. Minute is checked against the report text itself first, by regex for "Nth minute", "minute N", or "N'", and the model minute is used only if the text has none.
3. Call 1 classifies and extracts, see Section 6.2.
4. Branch in code, see Section 6.3. Automatic cases stop here, deterministic, no Call 2, no retrieval. Range cases continue.
5. Retrieval, top 3, see Section 7.
6. Call 2 recommends a number in range, with a 1 to 2 sentence reason, facts to severity to number, mid-range both sides, clamped in code, fallback to minimum on failure.
7. Result table shows the case, code, action, handbook, retrieved precedents, proposed games, and source label. Labels are Option C, table only, underlying JSON values unchanged: automatic is "Set by the rulebook", range is "Suggested by AI", fallback is "Rulebook minimum", not "Suggested by AI".
8. Human approves, or overrides games and reason. Approve writes the approved row back into `last_result`, so Final JSON is not left as pending after approval. That was a bug, fixed November 3, see Section 12.
9. The approved or overridden case becomes a session case, available for retrieval in that tab only, with a human-approved label and a session boost, see Section 7.4.
10. Logs shows the last run only: timings, Call 1 and Call 2 token usage, retrieved top 3 with scores, and audit. A new run overwrites Logs. Logs is kept on purpose, it is evidence for a reviewer, not a temporary debug tab to delete in cleanup. A session-only accumulating history view was proposed and not built, and is not promised here.

## 6. Core workflow decisions

### 6.1 Decision: deterministic workflow, not agents

**Context.** The sequence is fixed: read a report, classify it, apply a handbook, ground in precedent, propose a number, get a human decision. There is no multi-step planning, no tool choice, no negotiation between agents.

**Options considered.** A multi-agent system, with agents for extraction, handbook lookup, precedent search, and decision, talking to each other. A single LLM call that does everything, classification through final number, with no code branch.

**Decision.** Deterministic workflow, LLM calls inside, everything else code, templates, and human approval.

**Trade-off.** Less flexibility for novel workflows, more predictability, more testability, and a clear place to enforce the handbook. If the workflow ever needed planning or tool choice, agents would be reconsidered, but that is not this use case.

**Consequences.** The handbook branch, the clamp, and the human gate are all code, not prompt suggestions, so they cannot be talked out of by a model output. The LLM is used only where language understanding is needed: messy free text in, structured facts out, and severity judgment within a fixed range, grounded in precedent.

**Confirmation.** Every automatic case in testing must return the handbook number with no Call 2, and every range case must return a number inside its range, or the clamp test fails.

**Why this is not a missing feature.** A skimmer must not read the absence of agents as a gap. The absence is the design: a fixed sequence, deterministic tariff and history, LLM only where language is needed, human gate, and overrides feeding an eval loop.

### 6.2 Decision: Call 1, classify and extract, with definitions

**Context.** Referee reports are messy free text. The system must pull out who, what team, what minute, what misconduct code, and one action blob, without inventing facts.

**Decision.** Call 1 classifies and extracts. Action is one blob, and must not include the player name. Identity is stitched back in code at the end. The LLM never sees team names in Call 2, because Call 2 receives the action alone.

Fields removed from Call 1 over time, because they were not needed or were handbook work the model should not do: `deliberate`, `force_or_danger`, `injury_or_treatment`, `target_area`, and a separate `context`. `suspension_type` was also removed from Call 1, because the pipeline already re-derives it in code from the handbook, and the model was spending reasoning time on a handbook table it never sees, for a field that was discarded. The prompt now says do not output `suspension_type`, do not spend reasoning on it.

Prompt order matters, and is enforced in code: prompt starts "Extract facts from the report below", then "Report: {report}", then schema, then "Return JSON only" last. Code replaces exactly `{report}` or `{cases_json}` and never uses `.format()`, because schema braces broke `.format()` in an earlier bug.

Schemas were dropped: plain JSON, prompts define the exact JSON, types, null rule, and no extra keys, in the prompt itself.

**Call 1 definitions, operationalization, not handbook verbatim.** The handbook gives only short descriptions, so these definitions are an operationalization, and the README must say that, not present them as verbatim handbook text:

- Violent conduct: deliberate force against a person when not challenging for the ball.
- Serious foul play: a challenge for the ball that endangers safety, uses excessive force, lunges, or exposed studs.
- Tie-breaker: challenging for the ball goes to serious foul play, not violent conduct.
- Offensive language, second caution, DOGSO by handling, DOGSO by foul, and spitting are each one line in the prompt.

These definitions went live on Render with the general wording upload on November 4, bundled in the same prompts file, so the prompts file and the pipeline file must be uploaded together, never one without its matching partner.

**Trade-off.** Plain-text prompts are easier to read and change than schemas, but rely on prompt discipline and on code validation after the call. Missing required facts are not filled in by the model, they fail the run, see Section 6.4.

### 6.3 Decision: handbook as code, branch in code, clamp in code

**Handbook, `discipline.json`, seven rules:**

- 2CT, second caution in the same match: 1 to 1, automatic.
- DH, denying a goal by handling: 1 to 1, automatic.
- DF, denying a goal by a foul: 1 to 1, automatic.
- OL, offensive, insulting, or abusive language: 1 to 2.
- SFP, serious foul play: 1 to 3.
- VC, violent conduct: 2 to 4.
- S, spitting: 4 to 4, automatic.

Handbook sections in the file are 14.2, 14.6, and 14.7 as recorded there, with handbook text per rule. Automatic cases use the same deterministic reason as prod, and the handbook section 15.7 appeal text, email, evidence, 7 days, $250 fee, $50 admin if successful, is shown in the handbook cell, as in prod.

**Branch.** If handbook minimum equals maximum, the case is automatic and deterministic. Otherwise Call 2 runs on the action alone. Call 1 never sees the handbook, so it cannot know minimum equals maximum. Handbook wins, always.

**Clamp and fallback.** Call 2 output is clamped to the handbook range in code. On failure, the demo falls back to the rulebook minimum, labelled "Rulebook minimum". That fallback is safe and simple in a demo, but can be silently wrong, so prod should flag a failed call, retry once, and escalate to a human, not silently fall back, see Section 11.

**Bug history, kept because it proves the invariant.** Handbook loading once read `code` instead of `misconduct_code`, and because `discipline.json`, eval cases, Call 1 output, and pipeline all use `misconduct_code` everywhere, the one outlier emptied the handbook and caused "contains no usable rules". No `code` fallback remains. Stage and final rows once dropped `handbook_text`, so the Handbook Text column showed only the section number. Stage and final now carry `handbook_text`.

### 6.4 Decision: missing facts fail, do not guess

**Decision.** Every report must have a player name, a team name, and a minute, or it is rejected, with a plain message listing what is missing.

**Why minute, and why text-first.** Originally player and team only. Minute was added November 3, and is checked against the report text itself first, not just the LLM fact, so a minute the LLM drops still counts as present, and the model minute is used only if the text has none.

**Trade-off.** A reviewer cannot run a report with a missing minute just to see a code, which is friction in a demo, but it prevents a table with blanks being mistaken for a decision, and it mirrors prod, where a missing required fact returns the case to the referee noting what is missing.

### 6.5 Decision: human approval is the official record

**Decision.** Human approval is the official record. Human reason replaces LLM reason. Status is approved only if nothing changed, otherwise overridden. Gold standard is the human expert's final games plus final reason in `approve()`. Thumbs were dropped entirely, they were not a useful gold standard.

**Why not thumbs, and why not a score in the live UI.** A 1 to 5 score with a rubric belongs only to offline eval, not the live interface. The live interface carries thumbs plus human override in earlier designs, but the fifty preloaded cases carry approved expected games and reason as ground truth, with no invented thumbs, while a live run carries the human decision.

**Consequences.** Evaluation in Section 8 is against human final decisions, not against model self-scores. A human override is also the most valuable retrieval example, because it shows where the LLM was wrong and a human corrected it, see Section 7.4.

## 7. Retrieval and learning decisions

### 7.1 Decision: retrieval, same code first, then whole-action similarity, top 3

**Decision.** Retrieval is same misconduct code first, then whole-action semantic similarity, top k equals 3. The handbook still clamps the answer. Not phrase lookup alone, not every case in a code, not thumbs-filtered.

**Why whole-action, not phrase.** A phrase lookup retrieves on a shared word, for example a common word like "model" in another system, and misses the act. Whole-action similarity retrieves on the act, the target, the force, and the context together, which is what severity depends on.

**Why top 3.** Three precedents are enough for a reviewer to check the recommendation, and few enough to fit in a Call 2 prompt without drowning the current facts. More cases would add prompt tokens and cost, and would not necessarily add a new severity signal.

**Why same code first.** A precedent from another code cannot set the number, because the range is different, even if the action sounds similar. Same-code filtering keeps the retrieved set inside the same handbook range.

### 7.2 Decision: embeddings, and the pivot that forced them

**Original design, superseded.** Sentence-transformers, all-MiniLM-L6-v2, 384 dimensions, lazy and cached, loaded once per process. Automatic cases never loaded the model. Retrieval only ran inside Call 2 for range cases.

**Slimming, rejected.** On November 4, before any Render deploy, a slimmed version was built: word-overlap Jaccard retrieval, no torch, requirements only `openai` plus `streamlit==1.66.1`, 21 tests passed. Word-overlap top 3 on the five tricky cases mostly matched, except discriminatory offensive language, where the top was a 1-game profanity case and the 2-game discriminatory case was second at 0.136, the weakest retrieval, and Call 2 guidance would have had to carry it.

**Rejection.** Word-overlap was rejected by Emmy on November 4: semantic retrieval is a big part of the demo, word match cannot replace it. Plan agreed: try the original embeddings and torch version as-is on Render $7 first, pivot only if it does not work. Slimmed files are a fallback only, not to be uploaded for that test.

**Try-as-is failure.** Render as-is went live November 4 at https://matchday-suspension-demo.onrender.com/, page load was fine after about a 60-second cold start, showing 49 cases, but a run ran out of memory. Local torch embeddings ran out of memory on the 512 MB instance during a run. Page load itself was fine, the run was not.

**Current decision.** Retrieval uses OpenAI embeddings, `text-embedding-3-small`, 1,536 dimensions, via API, pure-Python cosine in `_cosine_similarity`, `_embed_texts` separated for test mocking, doc embeddings cached per process in a cache keyed on case ID and action. No torch, no sentence-transformers, requirements stay `openai` plus `streamlit==1.66.1`. Query and documents must use the same model. No new secret was needed, embeddings use the same `OPENAI_API_KEY` already set on Render. Minimal Render deploy for that batch was pipeline plus requirements only, the app change was cosmetic Logs wording only, tests are local-only.

**Trade-off.** API embeddings add per-run latency and a provider dependency, and cost input tokens, but they remove torch memory, which was the blocking constraint on a 512 MB host. At demo scale, embedding cost is negligible, see Section 9, while torch memory was fatal.

**Confirmation, connector harness, November 4, Call 2 nano, not Render-comparable for games.** 5 of 5 codes correct, 3 of 5 games matched expected. Single-insult offensive language now 1 game, via retrieval of two 1-game cases, versus 2 with retrieval off. Discriminatory offensive language retrieves the 2-game discriminatory case at the top at 0.555, fixing word-overlap's weakest case. Studs serious foul play 3 versus expected 2, face-push violent conduct 2 versus expected 3, both following their retrieved tops. That last pattern, retrieval following, is also the frontend weakest result in Section 8.

### 7.3 Decision: retrieval failure is recoverable, but not silent

**Decision.** If retrieval is disabled for a run, with a warning, the pipeline still runs, and Call 2 falls back to minimum if needed. A broken helper must not break the decision.

**Trade-off.** A run without retrieval is weaker, and a reviewer must be able to see that it ran without retrieval, from the warning and from the Logs, not from a silent change in quality. In prod, a retrieval failure should be flagged and escalated, not silently degraded, see Section 11.

### 7.4 Decision: session boost, demo-only, and immediate learning

**Decision.** Human session cases, SESSION-xxx, get +0.10, capped at 1.0, ranked by boosted score, logged as raw plus boost equals final, labelled "human approved" versus "synthetic preloaded", with score, raw score, and session boost all passed to Call 2 in the cases JSON.

**Caveat, load-bearing.** This boost applies only because the pre-seed is synthetic, not real historical data. In prod, with real historical decisions, pre-seed would be real data too, and this boost would not apply. Do not copy the boost into prod without that caveat.

**Identical-action fix.** An earlier version excluded a candidate whose action exactly matched the query, so an eval case could not retrieve itself. That broke live learning: a session case just approved, run again with identical wording, was excluded and never retrieved. Identical actions are now retrieved, with score 1.0. That fix is what makes immediate learning visible in a demo.

**Why immediate.** Learning must be immediate in a demo: one human approval must move a similar case into the top examples, not two incidents. A demo user will not add the same incident twice to prove learning. Session cases therefore help retrieval in that tab only, and a new tab resets to preloaded.

### 7.5 Decision: learning by repetition, not human-always-wins, not deterministic override

This is the most important learning decision, and it superseded an earlier design in the same day, so both are recorded here, with the superseded version marked as such.

**Trigger.** A push case retrieved two session cases at 3 games, with reasons reading "a push should always be 3 games", plus a synthetic case at 2 games, but Call 2 still recommended 2 games. The +0.10 ranking boost was preserved, session cases were top-ranked, and source labels, score, raw score, and session boost were all passed to Call 2, but the boost is ranking-only. The Call 2 prompt contained no instruction to weight human examples over synthetic examples in the decision, and Call 2 severity guidance pulled toward the minimum, and its worked example matched the synthetic minimum case's logic.

**Production framing, agreed.** For a production leadership direction change, the fix is neither runtime human-always-wins nor stacking new weighted cases on top of old ones. It is to curate the precedent set: remove or replace cases encoding the old rule, pre-seeded and session cases, any source, seed cases encoding the new rule, and keep old versions only in audit history. Remove all old-rule cases, not only one, or the next one will surface instead.

A single designated authority, centralised, reviews suspensions and sets direction. In real Matchday testing that authority was one person per conference, nationally it would have been one person for the nation. That person reviews suspensions and sets direction by adding human cases, with no code change needed. Access is not open to everyone. Demo containment is session-only, a new tab resets to the original pre-seed.

A direction change is narrow and targeted. Example: a trend in hair pulling, where leadership wants 4 games this season for that specific act, and the system must learn fast. A hair-pulling crackdown must not overgeneralize to all violent conduct at 4 games. Comparability, a factual tie, is what keeps a targeted change targeted.

**Why an LLM is still needed, even with human cases.** If human cases won deterministically in code, that is just a lookup table. The LLM does three things retrieval cannot do alone: Call 1 extraction and classification from messy free text, including the violent conduct versus serious foul play tie-break; generalization to novel reports and severity judgment within handbook ranges from specific facts; and comparability judgment itself, whether this is a factual tie, or whether facts differ, with an explanation of the difference. Retrieval grounds the LLM and shows learning, it does not replace the LLM.

**Superseded design choice 1, November 4, about 12:00, do not use.** A point-blank human-wins prompt line after severity guidance: "Human-approved examples always win: if a human-approved example is retrieved for a case, recommend its expected games, regardless of synthetic examples." Tested finding: nano does not follow it for blanket reasons, but does follow it for detailed reasons; full gpt-5 follows it for blanket reasons too. That finding is kept here for why a README must not over-claim prompt magic, and for why a prompt line alone was not the lever. Human reason quality was the lever in that test set, detailed reasons flipped nano, blanket reasons never did in that specific test set.

**Deterministic code override, rejected.** A code rule, human in top 3 within 0.15 of top synthetic follows human games deterministically, was proposed and rejected by Emmy: "I don't want it to be deterministic... I really don't love that." Determinism loses the LLM-role argument and turns a governance rule into a lookup.

**Current design choice 2, November 4, about 12:11 to 12:14, supersedes choice 1.** Learning by repetition, with no prompt override and no code override. One override is an exception, three consistent overrides is a direction.

Tested, Alexis Grant chest push, nano Call 2, with no human-win prompt line:

- 1 override, blanket reason: still 2 games.
- 3 overrides, blanket reason: flips to 3 games, with session case top at 0.929, synthetic minimum case second at 0.830.
- 3 overrides, detailed reasons: flips to 3 games, with session cases top at 0.993 and 0.915, synthetic minimum case third at 0.898.

The point-blank line was removed from local prompts after that test and was not restored for design choice 2. No upload, no model change, and no larger model is needed for this as the learning design.

**Demo flow for a reviewer.** Run a push case, override to 3, rerun, override again, rerun, override a third time, rerun, and watch it flip on the third override.

**Comparability limit, proved the same day.** A chest push report stayed at 2 games despite session neck cases at 3 games, because neck is a vulnerable area and a chest push is the synthetic minimum case. Neck cases are not comparable to a chest report, and a chest direction change needs chest session cases, not neck ones. A later chest case with one chest override plus two neck overrides also stayed at 2, because under design choice 2, chest at one override is an exception, not a direction. The settling test for that case is to override the chest case to 3, rerun, override again to 3, rerun, making three chest cases total, and only if it still stays at 2 after three chest cases is that a real failure of design choice 2.

### 7.6 Decision: general wording principle, no verb pair named

**Trigger.** A chest report gave different games for "push" versus "shove" on the same human top 3, proving Call 2 treated the verb alone as a severity signal, unstably. Local mixed neck plus chest test, no synonym line, with all top 3 human at 3 games, gave push chest 2, shove chest 3, on the same session set.

**Specific synonym line, tested then rejected.** A line naming push and shove as the same act made both give 3 in a mixed session, but Emmy rejected it: "no this is so specific. other words would have this same issue." Then a general line that still named the pair was also rejected: "no you cannot say push and shove. thats deterministic."

**Current decision, pair-free general principle, tested November 4 and verified live November 4 about 12:37.** Two lines only, no named verb pair anywhere in the prompts file, including the worked example, which was changed from "deliberately shoved an opponent off the ball" to "deliberately made contact with an opponent off the ball":

- "Wording alone is not severity. Do not treat a change in wording or verb alone as a change in severity. If the facts are otherwise the same, same act, same target area, same force, same repetition, same injury, the games should be the same. Judge severity by those facts and by retrieved cases, not by word choice."
- "If all retrieved examples agree on games, recommend that number unless the current facts clearly differ from those examples."

With those two lines, no named verb pair, mixed session nano gave push chest 3 and shove chest 3, consistent and in the human direction. Without any example and with only the abstract wording line, push was 2 and shove was 2, consistent but missing the human direction. Adding the retrieved-agreement line fixed both consistency and direction.

**Principle for prompt wording, from November 4 testing.** A prompt wording must be extremely strong and explicit for nano. Two layers were distinguished: approver reason quality, where detailed reasons flip nano and blanket reasons never did in that specific test set, and prompt wording, where a point-blank prompt flips a larger model even with a blanket reason but fails nano with a blanket reason. Those findings belong to that test set, they are now largely superseded as implementation rules by design choice 2 plus the general wording principle above, and are kept here for why a README must not over-claim prompt magic.

**Why not hardcode any pair.** There are multiple ways to say the same word, push, shove, and other verbs, same context. Hardcoding one pair overfits to one test case and misses the next pair with the same problem. The general principle is the only version that scales to unseen wording.

## 8. Evaluation

### 8.1 Design

- Gold standard is human final games plus final reason, not thumbs, not model self-score.
- Eval cases serve both retrieval and testing in the demo, cold-start seeding. A real project would hold out a separate eval set. That must be stated, not hidden, and it is stated in the README.
- Local eval file has 50 cases, SYN-001 to SYN-050. Repo canonical count is 49, one case was removed by Emmy on November 3, case not specified, do not chase 50. Preloaded tab summary counts are computed from the loaded list, never hard-coded, so 49 stays correct automatically. Until the repo file is checked against the local file, this document does not state a per-code distribution for the repo, because a frontend report on November 4 gave a distribution that does not match the local file, and that discrepancy is unresolved. See Section 12.
- Offline eval, if run, should use a 1 to 5 score with a rubric, plus per-category results, not one headline accuracy. Live interface does not carry that score.
- Gates that matter for this demo, in addition to headline accuracy: clamp violations equals 0, citation and precedent precision, abstention on no-evidence, that is, missing facts fail instead of guessing, and override rate and agreement, reported, not hidden.

### 8.2 Tricky cases, November 3, connector, retrieval off

All five tricky-case codes were correct, with retrieval disabled because sentence-transformers was not installed in the test environment at that time. Games: studs serious foul play 3 versus expected 2, with treatment pushing it to the top; single-insult offensive language 2 versus expected 1, with the reason explicitly saying directed abuse at an official escalates, which identified the later prompt issue; repeated offensive language 2, correct; discriminatory offensive language 2, correct, described not quoted; face-push violent conduct 3, correct, but the reason did not explicitly say apology. Connector-run times were much slower than live and are not comparable to live runs.

Tricky cases, in substance: studs exposed, shin, treatment on field, expected serious foul play; single general insult to a referee, once, walked away, expected offensive language 1, not maximum; repeated insult after warning, public, expected 2; discriminatory language, described not quoted, expected 2, do not quote a real slur in any test file; single face push, no injury, opponent stayed in game, immediate apology, expected violent conduct 3, reason must explain both mitigating and aggravating sides.

### 8.3 OpenAI embeddings harness, November 4

See Section 7.2. 5 of 5 codes correct, 3 of 5 games matched expected, Call 2 nano in the harness, not Render full model at that time, so games are not Render-comparable. Single-insult offensive language fixed by retrieval, discriminatory offensive language fixed by retrieval, studs and face-push both followed their retrieved tops, in opposite directions from expected.

### 8.4 Frontend test, November 4, Render, fresh tab

Measured runtime of the browser task: 12:54 to 13:01 CT, 6 minutes by harness stamp. Nothing was approved, and Reset session was not clicked during runs, so the count pill stayed at 49 preloaded, 0 added at start and after all 10 runs.

Upfront checks that passed: header "Matchday Suspension Demo", heading "Approve Suspensions", header right "Demo", pill "49 cases available for retrieval, 49 preloaded, 0 added this session", tabs in order Run a report, More examples, Preloaded cases, Logs, example label violent conduct and example text slapping-face case, Reset session button present, Preloaded tab full table with Case, Code, Action, Games, Reason.

Only upfront load issue: first load was blank and required one reload, after that the app loaded fully with no spinner or remaining error. That matches the parked Render cold-start issue, not a persistent deployment failure.

Per-case results, timings are total, Call 1, retrieval, Call 2 in seconds, captured from Logs immediately after each run because Logs shows only the last run:

1. Slap face, violent conduct, Suggested by AI, 2 games. Reason: slapped opponent in the face, direct strike to vulnerable area, serious but no stated injury or repetition, minimum 2 fits. Retrieved three 4-game violent conduct cases, scores not captured for this run because Logs was overwritten before scores were recorded. Timings: 9.69, 2.80, 1.59, 3.11.
2. Chest push, violent conduct, Suggested by AI, 2 games. Retrieved synthetic chest push 2 at 0.848, synthetic 2 at 0.685, synthetic 3 at 0.649. Timings: 5.15, 1.84, 0.36, 2.83.
3. Chest shove, violent conduct, Suggested by AI, 2 games. Retrieved synthetic both-hands shove 2 at 0.601, synthetic 2 at 0.580, synthetic 2 at 0.547. Push and shove are consistent at the no-session baseline on this run. Timings: 4.72, 1.64, 0.36, 2.67.
4. Studs, serious foul play, Suggested by AI, 2 games. Reason: lunged for ball with studs exposed, caught shin, opponent needed treatment, no indication of fracture, so 2 games. Retrieved studs 2 at 0.802, 3 at 0.741, 3 at 0.711. Timings: 5.43, 1.89, 1.05, 2.39.
5. Single insult, offensive language, Suggested by AI, 1 game. Retrieved 1-game cases at 0.677, 0.673, 0.616. Timings: 4.92, 1.48, 0.59, 2.76.
6. Repeated insult after warning, public, offensive language, Suggested by AI, 2 games. Retrieved 2-game cases at 0.725, 0.661, 0.649. Timings: 3.93, 1.49, 0.18, 2.16.
7. Discriminatory language, described not quoted, offensive language, Suggested by AI, 2 games. Retrieved 2-game case at 0.466, 1-game cases at 0.465 and 0.459. Scores markedly lower than other AI runs, typically 0.55 to 0.85. Timings: 5.35, 1.74, 0.22, 3.34.
8. Single face push, no injury, apology, violent conduct, Suggested by AI, 2 games. Reason: push to face is deliberate violent conduct, but single instance, no injury, opponent stayed in game, immediate apology, minimum 2 fits. Retrieved 2-game cases at 0.719 and 0.656, and a 4-game case at 0.633. This is a gap versus the tricky-case expectation of 3, noted as a README-worthy retrieval-following outcome, not as a UI bug. Described here generically, no name, per the November 4 decision. Timings: 6.07, 1.41, 0.52, 4.04.
9. Second caution, 2CT, Set by the rulebook, 1 game. Handbook shown: "14.3 Second caution in a match: 1 game." Retrieved: none, rulebook path. Timings: 2.84, 2.75, 0.0, 0.0, Call 2 not available for this run.
10. Spitting, S, Set by the rulebook, 4 games. Handbook shown: "14.7 Spitting: 4 games." Retrieved: none, rulebook path. Timings: 1.61, 1.52, 0.0, 0.0, Call 2 not available for this run.

Eight runs used "Suggested by AI", two used "Set by the rulebook", no run used "Rulebook minimum". AI runs were 3.93 to 9.69 seconds overall in this set, automatic runs were 1.61 to 2.84 seconds.

UI issues recorded: first load blank, needed one reload, judged likely Render cold start; supplemental text under results omits the player's name, for example "In the 34th minute, of Harbor FC lunged...", because the action blob intentionally strips the player name, the result table itself showed the correct player in its own column, so this is an action-blob artifact, not a table identity bug; Logs says "Shows the last run only", so per-run timings and scores are lost after the next run; retrieval-detail label CASE-1 on every run is expected per-run numbering, not a bug, case IDs autoincrement per run and overwrite any model-supplied ID; component sums do not always equal total, for example case 1 components 2.80 plus 1.59 plus 3.11 equals 7.50 versus total 9.69, suggesting unaccounted overhead.

Emmy accepted the results of this frontend test on November 4. Accepted does not mean all discrepancies are resolved, the distribution discrepancy in Section 12 remains open, and the face-push 2 versus expected 3 outcome remains a documented test observation.

### 8.5 Production evaluation position

Emmy stands by his position: in this case, real evaluation is actually new cases. Production evaluation is continuous on new incoming cases where the human final decision is ground truth, not an accuracy claim run on the same 49 cases. The textbook held-out objection still matters if anyone claims demo accuracy on the same cases used for retrieval, but his framing is that the demo proves the mechanism, retrieval plus learning, and prod evaluation happens on new cases. Do not turn that into a claim that a lecture on leakage should be re-delivered as if he had not considered it.

## 9. Speed, models, reasoning, cost, and latency

### 9.1 Speed history, measured, not estimated

Speed work was paused November 3 about 16:32, with quality testing next, then effectively resumed and re-closed November 4. Do not resume speed optimization without a new go from Emmy.

November 3, live, same kind of range run, warm retrieval:

- gpt-5, medium reasoning: total 22.98 seconds, Call 1 14.85 seconds, reasoning 576, total tokens 1,082.
- gpt-5-nano, medium reasoning: total 21.63 seconds, Call 1 13.01 seconds, reasoning 1,408, total tokens 1,936. Nano is faster per token, but generated 2.4 times as many reasoning tokens, wiping out the gain. Bottleneck is reasoning effort, not model size.
- Nano, low reasoning for Call 1: total 13.96 seconds, Call 1 4.89 seconds, reasoning 448, total tokens 950, Call 1 input 340 tokens, retrieval 0.59 seconds. Call 2 in that run was 8.47 seconds, reasoning 320, versus 576 in the previous nano-medium run, that Call 2 drop is observed, not explained, and is not attributed to the Call 1 change, since Call 2 reasoning was unchanged.
- Nano, minimal reasoning for Call 1: total 7.5 seconds, Call 1 2.07 seconds, reasoning 0, output 96, total 436, input 340, retrieval 0.56 seconds, Call 2 4.86 seconds with reasoning 320. Versus low, minimal drops Call 1 reasoning to 0 and Call 1 time to about 2 seconds. Extraction does not need chain-of-thought for this report, provided code, name, team, and action were still correct in that run.
- All nano, Call 1 minimal, Call 2 low: total 5.46 seconds, Call 1 1.78 seconds with reasoning 0, retrieval 0.63 seconds, Call 2 3.03 seconds with reasoning 192, total tokens 1,632, eval cases used 49. Quality check for nano-low Call 2 was still open at that point, especially mid-range and offensive language maximum cases, before nano-low became the default.
- Fastest run yet, Logs timing line only: total 3.7 seconds, Call 1 1.76 seconds, retrieval 0.57 seconds, Call 2 1.36 seconds. That is a 1.67 second cut on Call 2 versus the 3.03 second nano-low run, and total is inside the 4-second liveable range and close to the 3-second happy target set by Emmy on November 3: if total gets to 3 seconds, he is happy, and he can live with 4 seconds.

Current constants in local code: Call 1 reasoning effort minimal, Call 2 reasoning effort low. Audit log must never be hard-coded separately from the call: an earlier minimal run first showed reasoning Call 1 as low in the audit even though the call actually ran minimal, because the audit string was a separate literal from the reasoning argument. Fixed by a single constant, used both by the API call and by the audit log, so changing effort in one place changes both, and the log can no longer go stale. Repo version checked November 3 still used hard-coded strings, but with correct values for that test, so it was functionally correct.

Call 1 prompt input is about 340 to 362 tokens. Call 2 input is about 1,350 to 1,377 tokens in November 3 measurements. Call 1 slowness was not prompt size or API cold start.

### 9.2 Render speed, accepted

- First OpenAI-embeddings run, verified November 4 about 11:25 CT: total 9.71 seconds, Call 1 2.91 seconds, retrieval 1.56 seconds, Call 2 3.04 seconds. No out-of-memory, retrieval ran.
- Warm run, November 4 about 11:33 CT: about 5 seconds, down from 9.71 seconds, supporting the warming hypothesis. Warming fix is parked, flagged by Emmy as "we NEED to fix that" if it is a warming issue, and it is not solved.
- Speed bar, November 4 about 11:26 CT: a first-time user should not have to wait 10 seconds.
- Speed accepted and optimization stopped November 4 about 12:16 CT. Incognito warm run: total 5.71 seconds, Call 1 1.6 seconds, retrieval 0.91 seconds, Call 2 3.11 seconds. Call 2 and retrieval are near the LLM and API latency floor with nano, further gains would be under 1 second and risk breaking a working demo before Tampa or VAR. Incognito clears browser cache and session but not Render's warm process cache, so an incognito run is not a true post-deploy cold start.
- Speed levers discussed November 4, none approved or built: precomputed doc-embeddings file in the repo, about 300 KB for 50 times 1,536 vectors, surviving restarts, making retrieval query-only; Render Key Value Redis, free 25 MB without persistence, paid $10 per month for 256 MB, from web search returned November 4; a persistent disk at $0.25 per GB-month; a cron or background warm-up, which warms retrieval only and burns LLM calls; Call 2 nano, a quality trade-off; progressive UI, reducing perceived wait only. LLM calls dominate the 9.71 second run, about 6 seconds of it, so a cache alone cannot reach 3 to 4 seconds.

### 9.3 Cost per call, measured

Way to measure: pipeline already logs token usage per call in timings and in the Logs tab. Cost equals input tokens times input rate plus output tokens times output rate, with reasoning tokens billed at output rate. Embeddings are input-only.

Pricing used November 4, corroborated across secondary pricing sources returned by browser search in that conversation, not by opening OpenAI's official pricing page directly, so present them as corroborated and unverified before budgeting, not as official-page-verified:

- gpt-5-nano: $0.05 input, $0.40 output per 1 million tokens, cached input $0.005.
- gpt-5: $1.25 input, $10.00 output per 1 million tokens.
- text-embedding-3-small: $0.02 per 1 million input tokens, 1,536 dimensions, input-only pricing, max 8,191 tokens per input.
- A secondary pricing source in November 2026 said gpt-5-nano retires December 11, 2026. That is unverified and must not be presented as fact without checking OpenAI's official page.

Actual token usage, measured November 4 via connector run, both calls nano, Call 1 reasoning minimal, Call 2 reasoning low:

- Range case, slapping face: Call 1 input 566, output 83, reasoning 0. Call 2 input 1,418, output 613, reasoning 512.
- Range case cost, nano: Call 1 $0.000062 plus Call 2 $0.000316 equals $0.000378 per case. Embeddings warm query about 30 tokens equals $0.0000006, first cold run embedding all docs about 1,500 tokens equals $0.000030.
- Automatic second caution: Call 1 only, input 578, output 116, no Call 2, no retrieval, cost $0.000075 per case.
- Per 1,000 range cases, nano: $0.38. Per 400-case weekend, all range, nano: about $0.15.
- Same range case with full gpt-5 for Call 2: $0.007964 per case, $7.96 per 1,000 range cases, about 21 times nano. Comparison uses the corroborated $1.25 and $10.00 gpt-5 rates above. This comparison is for cost understanding only, the README does not talk about going back to gpt-5, per Emmy's November 4 decision, and live Call 2 on Render is nano, not gpt-5.

Range form, agreed as fair November 4, and used in the README:

- Range case, nano: $0.0003 to $0.0005 per case, centred on measured $0.000378.
- Automatic case: $0.00006 to $0.00010 per case, centred on measured $0.000075.
- 1,000 range cases, nano: $0.30 to $0.50.
- 400-case mixed weekend: about $0.10 to $0.16 LLM cost, plus hosting.
- gpt-5 Call 2 instead of nano, for comparison only: $0.006 to $0.010 per range case, $6 to $10 per 1,000.
- Hosting headline: Render $7 per month dominates LLM cost at demo scale by more than 95%, until volume becomes tens of thousands of cases per month or Call 2 is upgraded to a much more expensive model.

Small feature proposed November 4, answer pending, do not build without Emmy's answer: add a cost-per-run line to the Logs tab so a reviewer can see both tokens and dollars live, rather than only tokens.

## 10. Security, privacy, and guardrails

### 10.1 PII scope, corrected

Emmy's correction, accepted November 4, is load-bearing: real report PII scope is limited to victim and perpetrator player names, both team names, and player numbers, nothing else added. Referee names are never in reports, that is standard. Earlier concerns about referee names, age, and location were overblown and are struck.

Swapping those fields, deterministically from the roster, before the LLM sees the report, covers essentially all PII in this kind of report. Call 1 classification depends on action, not identity, so de-identification before Call 1 does not hurt code accuracy if the action text stays intact, including quoted offensive language.

### 10.2 Offensive language quoting, prod versus demo

Offensive language must be quoted in prod because exact words are the severity signal. Describing loses threat, repetition, discriminatory content, and intensity. Offensive language is never described in soccer, that is standard.

Public-demo exception stands: the discriminatory synthetic case deliberately does not quote an actual slur in the public file. Prod internal system quoting for rating and appeal evidence is different, access-restricted to the authority.

### 10.3 Retention distinction

Retaining PII in a third-party provider's logs, or allowing a provider to train on it, is a no-no. Retaining the report itself in the league's own database and continuing to train and learn from it is intended, not a problem. The risk is who else keeps a copy, for how long, and on what terms.

Player ID swap at ingestion, pipeline on ID plus action only, swap back only at display, is the best case for a third-party LLM. The only way to eliminate provider risk entirely is a private, on-prem model, which is a different architecture and cost decision, not a demo requirement.

### 10.4 Mapping-table control, the biggest real issue

Emmy's emphasis, November 4: of the remaining guardrail gaps, pseudonymity and mapping-table control is the biggest real issue, not prompt injection.

The ID swap is trivial to implement, but keeping ID and name apart after that is hard. If the LLM layer ever gets the mapping, or a log joins ID and name together and that log goes to a provider for eval or training, the swap fails silently. Stable IDs are needed for repeat-offender checks and learning by player, so rotating IDs per case would break prod value. Protection therefore must be architectural: ingestion does the swap, LLM sees ID plus action only, display reverse-swaps only for the authorized authority, audit stores ID not name, mapping table is access-controlled and audited.

This is the headline for gap 4 in the README, not just "we swap IDs", but "the swap is only as strong as the mapping separation".

### 10.5 Prompt injection, remaining gap

Prompt injection in free-text reports remains a risk the ID swap does nothing against. Mitigated, not eliminated, by Call 2 being action-only, Call 1 structured JSON forcing, handbook clamping, and human approval. No other guardrail is claimed in this demo. No auth, no rate limiting, no untrusted-user exposure is claimed either. The demo must not be exposed to untrusted users as if those controls existed.

### 10.6 Other security scope

- No secrets in code. `OPENAI_API_KEY` is in environment or Streamlit Secrets, never in a file, env value, or log in this document.
- No provider fallback claimed. If an LLM call fails, the demo falls back to minimum, see Section 6.3, and prod should flag and escalate instead.
- No compliance claim is made here. EU AI Act human oversight concepts, understand, monitor, resist automation bias, interpret, disregard, override, reverse, and stop, are useful design alignment, not a compliance claim for this demo.

## 11. Prod gaps and roadmap, in priority order

These are demo decisions, not bad decisions, provided the README says deferred, not done. A silent omission would be a bad decision, an explicit deferral is correct scoping.

P0, correctness and evidence:

- Persistent audit trail, append-only, who approved, denied, or overrode, and why, with ID not name.
- Real single-authority enforcement, access control, only the designated authority can set direction.
- Monitoring, alerts, and cost control, beyond Logs and token counts.
- Flag, retry once, and escalate to a human on LLM failure, instead of silently falling back to minimum.

P1, production controls:

- Prompt, model, and handbook versioning. In real life it will be in a DB, so versioning is easy, and rollback becomes a data problem. That does not by itself provide canary or A/B deploys, but it removes the demo's JSON and file limitation.
- Prod curation workflow. JSON versus DB is limited in the demo, DB makes prod curation easy. Curate the precedent set, remove or replace old-rule cases, seed new-rule cases, keep old versions only in audit history.
- Proper evaluation on new cases, continuous, with human final decisions as ground truth, per-category, with ablations and a raw-LLM baseline, not just golden cases.
- Retrieval hardening: re-ranking, deduplication, and date handling. Not league filtering, suspensions should be the same league-wide, so filtering by conference would shrink the precedent pool if the handbook is league-wide.

P2, product surface:

- Bulk and batch production workflow, one report at a time is a demo limit, prod ran at 3am in the full app and slow was fine there, interactive demo speed is a different constraint.
- Session history view, accumulating, not just last run, if a reviewer needs it, not as a replacement for Logs evidence.
- Cost-per-run line in Logs, tokens and dollars live, pending Emmy's answer, see Section 9.3.

## 12. Failure modes, tested and parked

| Failure | Symptom | Root cause | Fix or status |
|---|---|---|---|
| Preloaded eval file invalid | Pill said 0 cases, no clue | Repo eval file ended after SYN-049 with a trailing comma, load silently returned an empty list | App now warns with exact parser or file reason. Emmy fixed the comma himself, do not send a replacement file. Live count is 49, see discrepancy below. |
| Preloaded distribution discrepancy, unresolved | Frontend report said VC 26, SFP 11, OL 9, 2CT 2, S 1, total 49, and early rows did not match local file, for example SYN-001 as violent conduct | Not yet established whether repo file differs from local, whether the browser misread the table, or whether another deployed artifact was cached | Do not put a distribution claim in the README or in this document until confirmed with Emmy and against the actual repo file. Several retrieved IDs in the frontend run matched local expectations, including SYN-011 chest push, SYN-012 both-hands shove, SYN-022 punch face 4, SYN-032 studs 2, SYN-043 offensive language 1, SYN-049 offensive language 2, so the frontend report and local retrieval story are not uniformly inconsistent. |
| Widget chunk failure, three times | "Failed to fetch dynamically imported module" for NumberInput, then Selectbox, then Checkbox, after redeploys | Stale browser session requests a JS chunk hash that no longer exists, not a widget bug | Pin streamlit to 1.66.1, reboot cloud app and open a fresh tab, retrieval detail moved from checkbox to expander to reuse an already-loaded chunk. Games selectbox remains a candidate for the same failure if it appears next. |
| Final JSON not updating after Approve | Final JSON stayed final games null, status pending after approval | Approve added the case to session eval but never wrote the approved row back into last_result | Approve now writes the approved row back. Pipeline approve was already correct before that. |
| Handbook empty | "contains no usable rules" | Handbook loader read `code`, file uses `misconduct_code` everywhere | Fixed to read `misconduct_code` only, no fallback remains. |
| Handbook text missing | Handbook Text column showed only section number | Stage and final rows dropped handbook_text | Stage and final now carry handbook_text. |
| Call 1 format bug | Prompt failed or garbled | Schema braces broke `.format()` | Code uses string replacement, never `.format()`. |
| Audit log stale | Audit showed reasoning low when call ran minimal | Audit string was a separate literal from the reasoning argument | Single constant used by both call and audit. |
| Torch out of memory on Render | Run failed, page load fine | Torch embeddings on 512 MB instance during a run | Pivot to OpenAI embeddings, see Section 7.2. |
| Streamlit Cloud inconsistent load | Main app spun 7 plus minutes in Safari, over a minute in Chrome, hello test app spun, logs clean, reboot did not consistently fix | Platform frontend and WebSocket inconsistency, not code, not public or private state, not sleep, not pip | Parked, not abandoned. Emmy decided he will pay $7 a month for reliable hosting, reuse of Matchday frontend code was an option, no more time goes to debugging Streamlit Cloud. Do not keep telling Emmy to reboot, refresh, or wait. |
| Cold start, parked | First load blank, then reload works, page cold load about 60 seconds in an earlier check | Render cold start and warm process cache, incognito does not clear Render warm cache | Parked, not solved, despite warm 5.71 seconds being accepted. Speed optimization should not be resumed without a new go. |
| Supplemental text omits player name | "In the 34th minute, of Harbor FC..." | Action blob intentionally strips player name | Not a table identity bug, table shows correct player in its own column. Action-blob artifact, to be explained to a reviewer, not hidden. |
| System prompt duplication, agreed not done | System prompt for hierarchy protection should be kept but de-duplicated | Agreed November 3, not yet done | Open, do not claim done. |
| Missing facts schema, proposed not done | Removing `missing_facts` from Call 1 schema was proposed | Proposed November 3, not done | Open, do not claim done. |
| Test harness scope, unresolved | Whether harness should include all preloaded cases or stay with the tricky five | Unresolved November 4 | Open, do not start without Emmy's explicit go in his later time slot. |
| Next.js rebuild scope, unanswered | Switch now versus version 2 after Tampa, accumulating list versus one-report flow, two Render services versus single-service scope | Three questions still unanswered November 4, rebuild must not be started on assumption | Open, do not start without answers. |

## Appendix A. Superseded decisions, do not use as current

- **Point-blank human-wins prompt line, November 4 about 12:00, superseded by learning by repetition, Section 7.5.** Kept here only for test history and for why prompt magic must not be over-claimed.
- **Word-overlap retrieval, November 4 morning, rejected.** Slimmed files stay local as fallback only, not to be uploaded for the Render test. Semantic retrieval is a big part of the demo.
- **Torch and sentence-transformers retrieval, superseded by OpenAI embeddings, Section 7.2.** Original model was all-MiniLM-L6-v2, 384 dimensions.
- **Streamlit Cloud hosting, parked November 3 evening, superseded by Render $7, Section 9.2.** Do not resume Streamlit Cloud debugging.
- **Prompt 2, Call 2 offensive language single-insult wording, proposed November 3, closed November 4 as not needed.** Retrieval already gave single insult 1 game in the November 4 harness test, so the proposed wording, single general insult, even if directed at a referee or opponent, with no threat, repetition, public escalation, or discriminatory language, stays at the minimum, increase only when threatening, repeated, highly public, includes gestures, or is discriminatory, directed alone does not push to maximum, is not to be implemented. Prompt 2 no longer remains pending.
- **2CT main example, parked November 3, superseded November 4 by the slapping-face violent conduct example.** The 2CT example was the worst one because two cautions should really have two times. Main example is now Alex Carter of Riverside United, 20th minute, slapping an opponent in the face, with example label fixed to violent conduct, sent off.
- **Eval count 50, superseded by 49 canonical.** Local file still has 50 locally and is not the canonical count. Which specific case Emmy removed November 3 remains unspecified in the record.
- **"Approve Suspensions and Fines" title, superseded by "Approve Suspensions" only.** No fines exist in this demo.
- **Header "Synthetic cases only", removed on purpose.** Do not reintroduce it.
- **Preloaded cases as a collapsed expander, superseded by a full Preloaded cases tab.** Expander showed only case, code, and games, no action, no reason, no range, no context. Tab shows Case, Code, Action, Games, Reason, in loaded order, plus dynamic summary counts by code computed from the loaded list, never hard-coded. Session cases are not listed in that tab. No new Streamlit widget was added, specifically to avoid a new JS chunk risk.

## Appendix B. Glossary

- **Deterministic case:** handbook minimum equals maximum, number is set by code, no Call 2, no retrieval.
- **Range case:** handbook minimum is less than maximum, Call 2 recommends within range, clamped in code.
- **Preloaded case:** synthetic case shipped in the repo, labelled synthetic preloaded in retrieval.
- **Session case:** human-approved or overridden case added in the current tab only, labelled human approved, with a +0.10 session boost, capped at 1.0.
- **Clamp:** code enforcement of the handbook range on a Call 2 number.
- **Fallback minimum:** rulebook minimum used when an LLM call fails, labelled "Rulebook minimum", not "Suggested by AI".
- **Cold start:** first load or first run after a deploy or sleep, when caches and processes are not yet warm. Warm is a run after the process is already warm.
- **ADR:** architecture decision record, one decision per record, with context, options, decision, consequences, and confirmation. In this two-document shape, ADRs live as sections in this document, not as a separate folder, per Emmy's November 4 decision: one README plus one ARCHITECTURE.md.

## Docs map

- README: pitch, try it, architecture summary, decisions summary, headline evaluation, cost and timing, limitations.
- This ARCHITECTURE.md: full architecture, full decisions, full evaluation, failure modes, security and privacy, prod roadmap, superseded decisions.
