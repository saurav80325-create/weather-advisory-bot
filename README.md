# Weather-Advisory Support Bot (LangGraph + SOPs)

Answers "is it safe to ... outdoors?" using **live Open-Meteo data** and **written policies (SOPs)**.
The model never decides what good advice is: it extracts intent, picks a label for one fuzzy rule, and rephrases approved text.

## Setup & run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add ONE API key (.env is git-ignored)
streamlit run app.py                                   # frontend + backend (the graph runs in-process)
python -m evals.run_evals                              # eval suite (needs the API key); add case ids to run a subset, e.g. ... E1 S2
```
Model: `LLM_MODEL` env var, `provider:model`. The eval results below were produced with `google_genai:gemini-3.5-flash-lite` (set `GOOGLE_API_KEY`); other providers (default `anthropic:claude-haiku-4-5-20251001`, or `openai:gpt-4o-mini`) are supported by the code but I have not run the evals on them.

## Architecture
```mermaid
flowchart TD
  S([start]) --> U[understand - LLM fills schema]
  U -->|not outdoor| NG[no_guidance]
  U -->|no location| AL[ask_location]
  U --> RL[resolve_location]
  RL -->|error| FH[fail_honestly]
  RL --> FW[fetch_weather]
  FW -->|error| FH
  FW --> M[match_sops - deterministic]
  M -->|fuzzy SOP applies| J[judge_fuzzy - LLM picks 1 label]
  M --> SEL[select - rank + conflict rule]
  J --> SEL
  SEL -->|nothing triggered| NP[no_policy]
  SEL --> C[compose - LLM rephrases SOP text]
  C --> V[verify - number/ID guard]
  V -->|ok| R[respond]
  V -->|violation| FR[fallback_render - SOP text verbatim]
  NG & AL & FH & NP & FR --> R
  R --> E([end])
```

| Concern | Deterministic code | Model |
|---|---|---|
| What the user means (tags, place, time) | validates output against closed vocab | **extracts** into a schema |
| Weather numbers | `weather.py` computes every number | never |
| Which SOP matches | `policy.py` evaluates `when` conditions | only fuzzy SOP: picks one verdict label |
| Which SOP wins | `rank()` | never |
| Wording | fallback renders SOP text verbatim | rephrases approved text |
| Citation | footer built in `respond` | never |

## SOPs: form and why
`policies/sops.yaml` (+ `taxonomy.yaml`, `weather.yaml`): **declarative YAML data with a small condition language (`all`/`any`/field-op-value) validated at load time**, so non-engineers can edit policy, typos fail loudly, and no advice text or threshold exists in Python.
14 SOPs, 5 categories (general, travel, outdoor_exercise, vulnerable_groups, leisure), severities info to critical, SOP-13 is the fuzzy one (rubric + closed verdicts), SOP-01 is the situational heavy-rain override.

## Decisions you should be able to defend
- **Conflict rule** (`policy.rank`): lead policies first, then severity, then `priority`, then id. The primary SOP drives the answer; other triggered SOPs are surfaced as "also flagged" using their own headline text. Reason: hiding a high-severity flag is worse than a longer answer, but one primary avoids contradictory instructions. The footer states which was chosen and why.
- **SOP-01 (the case you care about)**: `applies_to: [any]`, `lead: true`, critical. It matches on situation (daily rain total >= 64.5 mm, the IMD "heavy rain" lower bound, or thunderstorm + >= 30 mm), not on the activity, so it fires for any question even when wind or UV look mild. **Honest limit:** Open-Meteo has no IMD warning feed, so this is a numeric proxy for "heavy-rain regime", not detection of a named low-pressure system. Production fix: ingest IMD/CAP warnings as a new weather field and reference it in SOP-01 (config only).
- **Prompt injection**: the raw user text reaches only `understand`, which can only output a schema whose tags are filtered against the taxonomy. `judge` and `compose` never see the raw message. Policy IDs and the citation footer come from code, and `verify` rejects any number not in the API facts or SOP text, and any SOP ID not selected. Defence in depth, not a single prompt.
- **Session memory**: LangGraph `MemorySaver` checkpointer, one `thread_id` per browser session = raw message history (`messages`) **plus** a structured `context` (location, tags, time frame, last SOP ids). Follow-up resolution (carry location/tags) is done in code, not left to the model. Consistency holds because every answer is re-derived from the same policies with a fresh forecast for the requested window. The bot does not currently say "this changed since earlier".
- **Missing data**: a condition on a missing field is False (never guessed). Geocoder or weather failure, empty geocode result, or a window that already passed all route to `fail_honestly`.

## The "add an 11th SOP live" test
- New rule on existing fields/tags: append to `sops.yaml`. Saved file is picked up on the next message (hot reload; bad edits are rejected and the last good set keeps serving). **No Python touched.**
- New activity: add a tag in `taxonomy.yaml`. New weather variable or window: add it in `weather.yaml`. Still no Python (eval C1 proves a new tag + rule end-to-end).
- **Needs code:** a new *kind* of rule (new operator, new SOP `type`), or data from a non-Open-Meteo source.

## Evals (`evals/run_evals.py`)
Two kinds of weather input, on purpose: **fixtures** (fixed hourly data fed through the real aggregation code) for deterministic cases, and **live API** for S1 only. S1's expectations are *derived from what the live API returned* (the engine independently recomputes the expected SOP from the same facts), and it reports INCONCLUSIVE rather than PASS when the weather is calm, so it keeps working after the Madhya Pradesh system passes. S2 is a pinned monsoon-style fixture so the severe path is always tested.

Cases: E1-E2 clear SOP; P1-P2 paraphrase; F1-F2 fuzzy; S1 live severe; S2 pinned severe; N1-N2 no SOP; U1-U3 API/geocode failure; A1-A2 adversarial; M1 multi-turn; M2 conflict; C1 add-SOP-by-config.

### Results 

Run on 2 Oct 2026 with `google_genai:gemini-3.5-flash-lite` (this model ignores `temperature=0`, so outputs can vary between runs). Full suite: **17 PASS, 0 FAIL, 1 INCONCLUSIVE**.

| Case | What it checks | Status |
|---|---|---|
| E1 | Strong wind + cycling gives SOP-03, quotes the real wind number | PASS |
| E2 | Child + heat gives SOP-09 | PASS |
| P1 | Paraphrase ("motorbike", "gusts") still matches SOP-03 | PASS |
| P2 | Paraphrase (grandmother, bitterly cold, tomorrow) gives SOP-11 | PASS |
| F1 | Fuzzy SOP-13, pleasant day, no word "picnic": verdict good | PASS |
| F2 | Fuzzy SOP-13, hot + 55% rain chance: verdict not good | PASS |
| S1 | LIVE Open-Meteo, Bhopal bike ride | **INCONCLUSIVE** |
| S2 | Pinned monsoon-style fixture: SOP-01 leads, quotes 144 mm | PASS |
| N1 | Scuba diving (no SOP covers it): "no guidance" | PASS (see notes) |
| N2 | Off-topic question: "no guidance", no weather fetched | PASS |
| U1-U3 | Weather API down / geocoder down / unknown place: honest failure, no numbers | PASS |
| A1 | Injection + fake "SOP-99": SOP-03 still wins, SOP-99 never appears | PASS |
| A2 | User dictates fake 15 degrees: number not repeated | PASS |
| M1 | Follow-up "what about this evening?" reuses city and activity | PASS |
| M2 | Wind + UV both high: SOP-03 primary, SOP-04 listed as also triggered | PASS |
| C1 | New tag + new SOP added via YAML only | PASS |

### Honest notes
- **S1 is INCONCLUSIVE, not passed.** Live weather in Bhopal was calm on the day I ran it (about 24-30 C, wind under 8 km/h, 0% rain chance), so nothing triggered and severity was never exercised against live data. The test reports INCONCLUSIVE instead of faking a pass. S2 covers the severe path, but it is a synthetic fixture, not a real IMD event. So **I have not verified a severe real-world case end to end.** For a suite that outlives the storm: keep S2 pinned, run S1 on a schedule and alert on INCONCLUSIVE streaks, and replay past real storms via Open-Meteo's historical API.
- **Failures found and fixed while building** (each is now covered by code):
  1. First run of E1/E2/P1 FAILED: the model picked the right SOP but its reply omitted the real numbers. Fix: prompt now requires the key values, and `guards.py` rejects replies that quote none of them (the verbatim SOP fallback is used instead).
  2. The model leaked an internal label ("PRIMARY:") and added "have a safe and pleasant time", a safety claim no SOP made. Fix: the guard rejects leaked labels and reassurance words not present in the SOP text.
  3. "Tomorrow morning" was evaluated over the whole day and flagged midday UV. Fix: finer windows in `weather.yaml` (config only).
  4. **N1 FAILED once** in a full run: the model tagged scuba diving as `outdoor_leisure`, so the picnic SOP told the user conditions were pleasant. Fix: narrowed that tag and added an `other_activity` catch-all tag no SOP uses. N1 then passed 3/3 repeated runs and the final full run. That is encouraging, not proof, since the model is non-deterministic.
- **Remaining weaknesses:** tag extraction is model-dependent (E2's tags differed between runs, and P2's window varied between `tomorrow` and `tomorrow_morning`; I loosened P2 to accept either). The evals check which SOP was selected and that real numbers appear. They do not grade the fluency or tone of replies. Fuzzy verdicts (F1/F2) are model judgment. The number guard is strict, so "about 65" for 64.8 falls back to verbatim SOP text. SOP-01 is a rain-total proxy, not an IMD warning feed. I did not measure how often the verbatim fallback triggers in normal use.