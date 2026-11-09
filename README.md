# Matchday Suspension Demo

[![License: Evaluation only](https://img.shields.io/badge/License-Evaluation%20only-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://matchday-suspension-demo.onrender.com/)

Live demo: https://matchday-suspension-demo.onrender.com/

> **Demo only. Synthetic data only. Session-only. A human decision is required.**
>
> This is a public demo of the suspension workflow from Matchday, a league operations application. It is not the full application, not production, and not real data.

## Table of contents

- [What this is](#what-this-is)
- [Try it](#try-it)
- [Installation](#installation)
- [How it works](#how-it-works)
- [Handbook ranges](#handbook-ranges)
- [Models and hosting](#models-and-hosting)
- [Decisions](#decisions)
- [Evaluation](#evaluation)
- [Cost and timing](#cost-and-timing)
- [Limitations and production gaps](#limitations-and-production-gaps)
- [What this demo is not](#what-this-demo-is-not)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Docs](#docs)

## What this is

Matchday automated game reports and suspensions for a 1,000-team amateur soccer league in the United States, running 300–500 games per weekend and 100+ suspensions per week. Before the application, suspensions were approved manually: referee supplemental reports arrived by email as PDFs, and each report had to be looked up in the handbook.

This demo isolates the suspension decision from that application:

1. Read a free-text referee report.
2. Extract the facts and classify the misconduct.
3. Apply the handbook deterministically where the sanction is fixed.
4. Where the handbook gives a range, ground a recommendation in precedent cases.
5. Require a human to approve or override the result.

No outcome is final without a human decision.

## Try it

Open the live demo and run the example report: Alex Carter, Riverside United, 20th minute, slapping an opponent in the face, a violent-conduct range case.

Inspect the result: the handbook range, the retrieved precedents, and the Logs tab for timings and the last run.

Override the same act, run it again, override it again, and override it a third time. Watch what changes on the third override.

One override is an exception. Three consistent overrides are a direction because k=3 and that segment's retrieval is now the human overrides.

## Installation

Prerequisites: Python 3.12, and an OpenAI API key.

```bash
git clone https://github.com/vinnelholdings/matchday-suspension-demo.git
cd matchday-suspension-demo
pip install -r requirements.txt
```

Set the environment variables below, then run:

```bash
streamlit run app.py
```

### Environment variables

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Set in the environment, or in Streamlit Secrets. |
| `light_model` | No | Call 1. On Render, set to `gpt-5-nano`. Falls back to `full_model` if unset. |
| `full_model` | No | Call 2. On Render, set to `gpt-5-nano`. |

### Tests

```bash
pytest test_pipeline.py
```

`pytest` is required locally only. Do not add it to `requirements.txt`.

### Files in this repo

```
app.py             — Streamlit frontend, adapted from the Matchday assignor suspensions page, with fines removed.
pipeline.py        — deterministic workflow: Call 1, handbook branch, retrieval, Call 2, clamp, human approve / override.
discipline.json    — the handbook.
eval_cases.json    — the synthetic preloaded set.
prompts.py         — Call 1 and Call 2 prompts.
test_pipeline.py   — local tests.
requirements.txt   — openai and streamlit==1.66.1.
README.md          — this file.
ARCHITECTURE.md    — full design document.
```

## How it works

The workflow is deterministic. Language models are called inside it, only where language understanding is required. There are no agents.

```
Referee report (free text)
        |
        v
Call 1 — classify and extract (player, team, minute, code, action)
        |
        v
Handbook branch (in code)
  ├── min == max → automatic, set by the rulebook. No Call 2, no retrieval.
  └── min < max  → retrieval, then Call 2
        |
        v
Retrieval — same misconduct code first, then whole-action
semantic similarity, top 3 (OpenAI text-embedding-3-small)
        |
        v
Call 2 — recommend a number within the handbook range,
with a 1–2 sentence reason. Clamped in code.
        |
        v
Human approver — approve, or override with a reason.
The human decision is the official record.
```

Call 1 classifies and extracts. The action is a single blob, without the player name; identity is stitched back in code. Call 1 never sees the handbook, so it cannot know whether a case is automatic.

Retrieval is same-code first, then whole-action semantic similarity, top 3. A precedent from another code cannot set the number, because its range differs. Whole-action similarity retrieves on the act, target, force, and context together rather than on a shared word.

Call 2 receives the action alone, plus the retrieved precedents. It recommends a number inside the handbook range. The recommendation is clamped in code; the model is never asked to stay in range. If an LLM call fails, the demo falls back to the rulebook minimum, labelled "Rulebook minimum", not "Suggested by AI".

Missing facts fail, they are not guessed. A report must contain a player name, a team name, and a minute, or the run is rejected with a plain message listing what is missing. The minute is checked against the report text first, not only against the extracted facts.

Human approval is the official record. A human reason replaces the model reason. Status is "approved" only if nothing changed, otherwise "overridden". Approved and overridden cases become session cases, available for retrieval in that tab only. A new tab resets to the preloaded set.

## Handbook ranges

From `discipline.json` — handbook rules only, no case data:

| Code | Misconduct | Games |
|---|---|---|
| 2CT | Second caution in the same match | 1, automatic |
| DH | Denying a goal by handling | 1, automatic |
| DF | Denying a goal by a foul | 1, automatic |
| OL | Offensive, insulting, or abusive language | 1–2 |
| SFP | Serious foul play | 1–3 |
| VC | Violent conduct | 2–4 |
| S | Spitting | 4, automatic |

Call 1 uses operational definitions for violent conduct and serious foul play (for example, challenging for the ball goes to serious foul play, not violent conduct). Those definitions are an operationalisation of the handbook's short descriptions, not verbatim handbook text.

## Models and hosting

Call 1 and Call 2: `gpt-5-nano` in the live deployment, configured via Streamlit Secrets / environment (`light_model` for Call 1, `full_model` for Call 2). Call 1 uses minimal reasoning effort; Call 2 uses low reasoning effort. The bottleneck in earlier testing was reasoning effort, not model size: a smaller model alone did not speed up Call 1 until reasoning was reduced.

Retrieval: OpenAI `text-embedding-3-small` (1,536 dimensions) via API, with pure-Python cosine similarity. Document embeddings are cached per process. The original local embedding model (sentence-transformers / torch) ran out of memory on the 512 MB Render instance during a run, which forced this pivot. Word-overlap retrieval was tested and rejected: semantic retrieval is a core part of the demo.

Hosting: Render, $7/month tier (512 MB). Streamlit Cloud was parked after inconsistent loading that was not attributable to the application code.

## Decisions

Full decisions, trade-offs, rejected alternatives, and failure modes are in [ARCHITECTURE.md](ARCHITECTURE.md). This table is the summary.

| Decision | Choice | Rationale |
|---|---|---|
| Workflow | Deterministic workflow; LLM calls inside only | The sequence is fixed — read, classify, apply handbook, ground in precedent, get a human decision. There is no planning or tool choice for agents to do. Agents would add complexity without adding capability. |
| Automatic vs range | Branch in code on handbook min == max | A fixed sanction is a handbook fact, not a judgement call. Automatic cases never reach Call 2. |
| Retrieval | Same code, whole-action similarity, top 3 | Keeps precedents inside the same handbook range and retrieves on the act as a whole, not a shared word. |
| Learning | Learning by repetition, with no prompt override and no code override | A single designated authority sets direction in production by curating the precedent set — removing or replacing cases encoding the old rule and seeding cases encoding the new rule — not by stacking weighted cases on top of old ones. A direction change is narrow and targeted (for example, a hair-pulling crackdown at 4 games for that act only), not a blanket change to all violent conduct. |
| Session boost | Human-approved session cases get +0.10, capped at 1.0, ranking only | This applies only because the preloaded cases are synthetic. In production, with real historical decisions as the pre-seed, this boost would not apply. Do not copy it into production without that caveat. |
| Wording | General principle; no verb pair named in the prompt | Wording alone is not severity. If the facts are otherwise the same, same act, same target area, same force, same repetition, same injury, the games should be the same. Judge severity by those facts and by retrieved cases, not by word choice. If all retrieved examples agree, recommend that number unless the current facts clearly differ. Hardcoding one verb pair (for example, push/shove) overfits to one test case. |
| Persistence | Session-only; no database, no upload, no download | Proves the learning mechanism in a demo. It deliberately does not prove production persistence. |
| Data | Synthetic only; 49 preloaded cases in the deployed demo | No real reports, names, or teams in a public demo. Production would use real, human-approved historical cases. Do not mistake synthetic cases for real precedent. |
| Privacy (prod) | Swap names for player IDs before the LLM sees a report | Real report PII is limited to player names (victim and perpetrator), both team names, and player numbers. Referee names are never in reports. The swap is only as strong as the separation of the ID↔name mapping afterwards, see Limitations. |

## Evaluation

Numbers here come from real runs. There is no held-out accuracy claim: the same 49 cases serve as both retrieval set and test set in this demo, so these results are mechanism proof, not a production accuracy claim. In production, evaluation is continuous, on new incoming cases, with the human final decision as ground truth.

Live frontend test, 4 November 2026, on Render, fresh tab, 10 cases:

- 8 suggested by AI, 2 set by the rulebook, 0 by rulebook-minimum fallback.
- AI runs: 3.93–5.69 seconds total. Automatic runs: 1.61–2.84 seconds.
- Warm run, accepted 4 November 2026: 5.71 seconds total (Call 1: 1.6s, retrieval: 0.91s, Call 2: 3.11s). Component timings do not always sum exactly to the total; there is unaccounted overhead.

Weakest result, stated plainly: a single face push with no injury and an immediate apology, expected 3 games, received 2, because it followed its retrieved 2-game cases. That is retrieval following its precedents, not a UI bug, and it is why it is documented here rather than hidden.

## Cost and timing

Cost is a measured range, not a single exact number, because token usage varies by case.

| Case type | Cost per case |
|---|---|
| Range case, nano, both calls | ~$0.0003–$0.0005, centred on a measured $0.000378 |
| Automatic case, Call 1 only | ~$0.00006–$0.00010, centred on a measured $0.000075 |
| 1,000 range cases, nano | ~$0.30–$0.50 |
| 400-case mixed weekend | ~$0.10–$0.16 in LLM cost, plus hosting |

For comparison only, the same range case with `gpt-5` for Call 2 would be ~$0.006–$0.010 per case, ~$6–$10 per 1,000 — about 21× nano. The live deployment uses nano for Call 2.

Hosting dominates at demo scale. Render at $7/month is more than 95% of the cost at this volume, until volume reaches tens of thousands of cases per month or Call 2 is upgraded to a much more expensive model.

Timing bar: a first-time user should not have to wait 10 seconds. Warm is accepted at 5.71 seconds.

Pricing note: the rates above were corroborated across secondary pricing summaries in November 2026, not verified against OpenAI's official pricing page in that check. Check the official page before budgeting.

## Limitations and production gaps

| Gap | Status in this demo |
|---|---|
| Persistent audit trail | Deferred. Logs shows the last run only. Production needs an append-only audit log: who approved, denied, or overrode, and why. |
| Single-authority enforcement | Deferred. The demo is session-only, one user. Production needs access control so only the designated authority can set direction. |
| Monitoring, alerts, cost control | Deferred. The demo has Logs and token counts only. |
| LLM failure handling | Demo falls back silently to the rulebook minimum. Production should flag a failed call, retry once, and escalate to a human. |
| Prompt, model, and handbook versioning | Deferred. The demo uses JSON files. Production would version these in a database, so rollback becomes a data problem. |
| Production curation workflow | Deferred. The demo can only add session cases. Production needs a reviewed workflow to curate the precedent set. |
| Proper evaluation | Demo proves the mechanism on 49 synthetic cases. Production evaluation is continuous, on new cases, with human final decisions as ground truth. |
| Retrieval hardening | Demo is same-code first, then whole-action similarity, top 3. Production needs re-ranking, deduplication, and date handling. Retrieval should not be filtered by conference, suspensions are league-wide, so filtering would shrink the precedent pool. |

### Privacy

The biggest real issue. Swapping names for IDs is trivial; keeping ID and name apart afterwards is hard. If the LLM layer ever gets the mapping, or a log joins ID and name and that log goes to a provider, the swap fails silently. Protection must be architectural: ingestion swaps, the LLM sees ID and action only, display reverse-swaps only for the authorised authority, audit stores ID not name, and the mapping table is access-controlled and audited. Stable IDs are needed for repeat-offender checks, so rotating IDs per case would break production value. The only way to eliminate provider risk entirely is a private, on-premises model, a different architecture and cost decision, not a demo requirement.

### Prompt injection

Remaining gap. Mitigated, not eliminated, by Call 2 being action-only, structured JSON output, handbook clamping, and human approval. The ID swap does nothing against it.

### Offensive language quoting

Production must quote exact words, because exact words are the severity signal. The public demo deliberately does not quote a discriminatory slur in its synthetic file; internal production quoting, access-restricted, is different.

## What this demo is not

- Not the full Matchday application.
- Not production: no auth, no tenancy, no rate limiting, no append-only audit log.
- Not real data: all preloaded cases are synthetic.
- Not a claim that a demo proves production accuracy.

## Troubleshooting

- **First load is blank:** reload the page. This is the parked Render cold start, not a persistent deployment failure.
- **A run is rejected:** the report must contain a player name, a team name, and a minute, or the run is rejected with a plain message listing what is missing.
- **No API key:** the app needs `OPENAI_API_KEY` in the environment or in Streamlit Secrets. Without it, runs fail.
- **Logs shows only the last run:** that is by design. Logs shows timings, retrieved top 3, and audit for the last run only.

## Contributing

This is a portfolio demo. No contributions are expected, please don't open PRs. Issues pointing out a bug are welcome.

## License

Evaluation only. See [LICENSE](LICENSE).

You may use and try this demo to evaluate it. You may not copy, replicate, redistribute, or reuse it.

## Author

Built by Emmy Nonso

## Docs

- This README: pitch, try it, architecture summary, decisions summary, headline evaluation, cost and timing, limitations.
- [ARCHITECTURE.md](ARCHITECTURE.md): full architecture, full decisions with trade-offs and rejected alternatives, full evaluation, failure modes, security and privacy, and the production roadmap.
