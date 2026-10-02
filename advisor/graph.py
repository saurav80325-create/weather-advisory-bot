"""The LangGraph agent. Control flow only: no advice text and no thresholds live in this file.

START -> understand --(not outdoor)--> no_guidance ----------------------------+
            |--(no location)--> ask_location -------------------------------+  |
            '--> resolve_location --(error)--> fail_honestly ---------------+  |
                      '--> fetch_weather --(error)--> fail_honestly          |  |
                              '--> match_sops --(fuzzy SOP?)--> judge_fuzzy  |  |
                                       '--------------'--> select --(none)--> no_policy --+
                                                              '--> compose --> verify --(ok)--+
                                                                                   '--(bad)--> fallback_render
All terminal branches -> respond (adds citation footer, updates session context) -> END
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import guards, llm_steps, weather
from .policy import PolicyStore, condition_fields, from_verdict, match, rank


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # session memory: raw chat history
    context: dict                             # session memory: structured facts from earlier turns
    user_input: str
    intent: dict
    location: dict
    facts: dict
    meta: dict
    error: str
    triggered: list
    fuzzy_candidates: list
    covered: bool
    selected: list
    shown_facts: dict
    key_facts: dict
    draft: str
    problems: list
    used_fallback: bool
    reply: str
    outcome: str
    trace: list


def _t(state, name):
    return (state.get("trace") or []) + [name]


def _why_primary(sel):
    p = sel[0]
    if p.get("lead"):
        return "it is a lead-with policy (a situational risk that outranks activity-specific advice)"
    if len(sel) == 1:
        return "it was the only policy triggered"
    ties = [s for s in sel if s["severity"] == p["severity"]]
    if len(ties) > 1:
        return f"highest severity ({p['severity']}); tied with {len(ties) - 1} other(s), broken by policy priority"
    return f"highest severity ({p['severity']}) among {len(sel)} triggered policies"


def build_graph(llm, store: PolicyStore, geocode_fn=weather.geocode, forecast_fn=weather.fetch_raw,
                checkpointer=None):
    """Weather fetchers and the LLM are injected so evals can simulate outages and fixed forecasts."""

    # ---------------- nodes ----------------
    def understand(state):
        pol, ctx = store.get(), state.get("context") or {}
        it = llm_steps.understand(llm, state["user_input"], ctx, (state.get("messages") or [])[-6:], pol)
        # deterministic carry-over from session memory (not left to the model)
        if not it["location"] and ctx.get("location_query"):
            it["location"], it["location_carried"] = ctx["location_query"], True
        if it["is_follow_up"] and not it["tags"]:
            it["tags"] = ctx.get("tags", [])
        if it["is_follow_up"] and ctx.get("tags"):
            it["is_outdoor_activity_question"] = True
        return {"messages": [HumanMessage(content=state["user_input"])], "intent": it, "error": None,
                "location": None, "facts": None, "meta": None, "triggered": [], "fuzzy_candidates": [],
                "covered": False, "selected": [], "shown_facts": {}, "key_facts": {}, "draft": None, "problems": [],
                "used_fallback": False, "reply": None, "outcome": None, "trace": ["understand"]}

    def route_understand(state):
        it = state["intent"]
        if not it["is_outdoor_activity_question"]:
            return "no_guidance"
        return "resolve_location" if it["location"] else "ask_location"

    def no_guidance(state):
        return {"outcome": "no_guidance", "trace": _t(state, "no_guidance"),
                "reply": "I'm sorry, I don't have guidance for that. I can only advise on outdoor-activity safety "
                         "(cycling, running, picnics, road trips, outings with children, older relatives or pets) "
                         "using our written safety policies."}

    def ask_location(state):
        return {"outcome": "needs_location", "trace": _t(state, "ask_location"),
                "reply": "Which city or town should I check the weather for?"}

    def resolve_location(state):
        try:
            return {"location": geocode_fn(state["intent"]["location"]), "trace": _t(state, "resolve_location")}
        except Exception as e:  # noqa: BLE001 - any failure is the same honest fallback
            return {"error": str(e), "trace": _t(state, "resolve_location")}

    def fetch_weather(state):
        pol, loc = store.get(), state["location"]
        try:
            raw = forecast_fn(loc["latitude"], loc["longitude"], sorted({f["var"] for f in pol.fields.values()}))
            facts, meta = weather.build_facts(raw, pol.fields, pol.windows, state["intent"]["time_frame"])
            return {"facts": facts, "meta": meta, "trace": _t(state, "fetch_weather")}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "trace": _t(state, "fetch_weather")}

    def route_on_error(next_node):
        return lambda state: "fail_honestly" if state.get("error") else next_node

    def fail_honestly(state):
        return {"outcome": "data_unavailable", "trace": _t(state, "fail_honestly"),
                "reply": f"I can't give you advice right now because I couldn't get verified weather data: "
                         f"{state['error']}. I won't guess. Please check the place name or try again shortly."}

    def match_sops(state):
        triggered, fuzzy, covered = match(store.get().sops, state["intent"]["tags"], state["facts"])
        return {"triggered": triggered, "fuzzy_candidates": fuzzy, "covered": covered,
                "trace": _t(state, "match_sops")}

    def route_match(state):
        return "judge_fuzzy" if state["fuzzy_candidates"] else "select"

    def judge_fuzzy(state):
        got = list(state["triggered"])
        for sop in state["fuzzy_candidates"]:
            label = llm_steps.judge(llm, sop, state["facts"], state["intent"]["tags"])
            if label:
                got.append(from_verdict(sop, label))
        return {"triggered": got, "trace": _t(state, "judge_fuzzy")}

    def select(state):
        pol, sel = store.get(), rank(state["triggered"])
        names = []
        for s in sel:
            names += sorted(condition_fields(s["when"])) if "when" in s else s.get("judge_fields", [])
        names += ["temp_c", "wind_kmh", "precip_prob_pct"]
        shown = {}
        for n in names:
            label = pol.fields[n].get("label", n) if n in pol.fields else n
            if state["facts"].get(n) is not None and label not in shown:
                shown[label] = state["facts"][n]
        key = {}
        if sel:
            p = sel[0]
            kn = sorted(condition_fields(p["when"])) if "when" in p else p.get("judge_fields", [])[:3]
            for n in kn:
                if state["facts"].get(n) is not None:
                    key[pol.fields[n].get("label", n)] = state["facts"][n]
        return {"selected": sel, "shown_facts": shown, "key_facts": key, "outcome": "answered" if sel else None,
                "trace": _t(state, "select")}

    def route_select(state):
        return "compose" if state["selected"] else "no_policy"

    def no_policy(state):
        it, loc, meta = state["intent"], state["location"], state["meta"]
        basics = "; ".join(f"{k}: {v}" for k, v in state["shown_facts"].items())
        if state["covered"]:
            msg = (f"None of our written safety policies are triggered for this activity in {loc['display']} "
                   f"({meta['window'].replace('_', ' ')}). That is not a guarantee that conditions are safe; it only means "
                   f"no policy flags a concern. Conditions I checked: {basics}.")
            outcome = "no_trigger"
        else:
            msg = ("I'm sorry, I don't have guidance for that activity: none of our written policies cover it, "
                   "and no general weather warning applies right now.")
            outcome = "no_guidance"
        return {"outcome": outcome, "reply": msg, "trace": _t(state, "no_policy")}

    def compose(state):
        sel, loc, meta = state["selected"], state["location"], state["meta"]
        payload = {"location": loc["display"], "window": f"{meta['window'].replace('_', ' ')} ({meta['hours']})",
                   "activity_tags": state["intent"]["tags"],
                   "primary_guidance": {"headline": sel[0]["headline"], "guidance": sel[0]["advice"]},
                   "also_flagged": [s["headline"] for s in sel[1:4]], "key_facts": state["key_facts"], "facts": state["shown_facts"]}
        try:
            draft = llm_steps.compose(llm, payload)
        except Exception:  # noqa: BLE001
            draft = None
        return {"draft": draft, "trace": _t(state, "compose")}

    def verify(state):
        draft, sel = state.get("draft"), state["selected"]
        if not draft:
            return {"problems": ["no draft produced"], "trace": _t(state, "verify")}
        texts = ([s["advice"] for s in sel] + [s["headline"] for s in sel] +
                 [state["meta"]["hours"], state["location"]["display"]])
        problems = guards.verify_reply(draft, state["shown_facts"].values(), texts, {s["id"] for s in sel},
                                       required_any=state["key_facts"].values())
        out = {"problems": problems, "trace": _t(state, "verify")}
        if not problems:
            out["reply"] = draft
        return out

    def route_verify(state):
        return "respond" if state.get("reply") else "fallback_render"

    def fallback_render(state):
        sel = state["selected"]
        parts = [sel[0]["advice"]]
        if len(sel) > 1:
            parts.append("Also flagged: " + " ".join(s["headline"] for s in sel[1:4]))
        parts.append("Conditions used: " + "; ".join(f"{k}: {v}" for k, v in state["shown_facts"].items()) + ".")
        return {"reply": "\n\n".join(parts), "used_fallback": True, "trace": _t(state, "fallback_render")}

    def respond(state):
        sel, outcome, loc, meta = state.get("selected") or [], state["outcome"], state.get("location"), state.get("meta")
        lines = []
        if outcome == "answered":
            p = sel[0]
            lines.append(f"Policy applied: {p['id']} - {p['title']} (severity: {p['severity']}"
                         + (f", verdict: {p['verdict']}" if p.get("verdict") else "") + ")")
            lines.append(f"Why this one: {_why_primary(sel)}")
            if len(sel) > 1:
                lines.append("Also triggered: " + ", ".join(f"{s['id']} ({s['severity']})" for s in sel[1:]))
        elif outcome == "no_trigger":
            lines.append("Policy applied: none triggered (activity is covered, no condition met)")
        else:
            lines.append("Policy applied: none")
        if meta and loc:
            lines.append(f"Data: Open-Meteo, {loc['display']}, {meta['window'].replace('_', ' ')} "
                         f"{meta['hours']} local, forecast time {meta['fetched_at']}")
        full = state["reply"] + "\n\n---\n" + "  \n".join(lines)
        ctx = dict(state.get("context") or {})
        if loc:
            it = state["intent"]
            ctx.update(location_query=it["location"], location_name=loc["display"], time_frame=it["time_frame"],
                       last_outcome=outcome, last_sop_ids=[s["id"] for s in sel])
            if it["tags"]:
                ctx["tags"] = it["tags"]
        return {"reply": full, "context": ctx, "messages": [AIMessage(content=full)],
                "trace": _t(state, "respond")}

    # ---------------- wiring ----------------
    g = StateGraph(State)
    for fn in (understand, no_guidance, ask_location, resolve_location, fetch_weather, fail_honestly,
               match_sops, judge_fuzzy, select, no_policy, compose, verify, fallback_render, respond):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route_understand,
                            {"no_guidance": "no_guidance", "ask_location": "ask_location",
                             "resolve_location": "resolve_location"})
    g.add_conditional_edges("resolve_location", route_on_error("fetch_weather"),
                            {"fetch_weather": "fetch_weather", "fail_honestly": "fail_honestly"})
    g.add_conditional_edges("fetch_weather", route_on_error("match_sops"),
                            {"match_sops": "match_sops", "fail_honestly": "fail_honestly"})
    g.add_conditional_edges("match_sops", route_match, {"judge_fuzzy": "judge_fuzzy", "select": "select"})
    g.add_edge("judge_fuzzy", "select")
    g.add_conditional_edges("select", route_select, {"compose": "compose", "no_policy": "no_policy"})
    g.add_edge("compose", "verify")
    g.add_conditional_edges("verify", route_verify, {"respond": "respond", "fallback_render": "fallback_render"})
    for terminal in ("no_guidance", "ask_location", "fail_honestly", "no_policy", "fallback_render"):
        g.add_edge(terminal, "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def build_default_graph():
    from .config import POLICY_DIR, get_llm
    return build_graph(get_llm(), PolicyStore(POLICY_DIR))
