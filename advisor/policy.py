"""Policy engine: loads, validates, evaluates and ranks SOPs. No LLM, no network.

This module is the ONLY place that knows how a policy 'matches'. Policies themselves live in
policies/*.yaml, so changing advice or thresholds never touches this file or the graph.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path

import yaml

SEVERITY = {"info": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
       "==": operator.eq, "in": lambda a, b: a in b}


class PolicyError(Exception):
    pass


@dataclass
class Policies:
    sops: list
    taxonomy: dict
    fields: dict
    windows: dict


# ---------- loading + validation (fail loudly at load time, not at 2am in production) ----------
def _read(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _check_cond(c, fields, where, out):
    if not isinstance(c, dict):
        out.append(f"{where}: condition must be a mapping")
        return
    for key in ("all", "any"):
        if key in c:
            for sub in c[key]:
                _check_cond(sub, fields, where, out)
            return
    for k in ("field", "op", "value"):
        if k not in c:
            out.append(f"{where}: condition missing '{k}'")
            return
    if c["field"] not in fields:
        out.append(f"{where}: unknown weather field '{c['field']}' (declare it in weather.yaml)")
    if c["op"] not in OPS:
        out.append(f"{where}: unknown operator '{c['op']}'")


def validate(sops, taxonomy, fields) -> list[str]:
    out, seen = [], set()
    for s in sops:
        sid = s.get("id", "<missing id>")
        if sid in seen:
            out.append(f"{sid}: duplicate id")
        seen.add(sid)
        for k in ("id", "title", "category", "applies_to"):
            if k not in s:
                out.append(f"{sid}: missing '{k}'")
        for t in s.get("applies_to", []):
            if t != "any" and t not in taxonomy:
                out.append(f"{sid}: unknown tag '{t}' (declare it in taxonomy.yaml)")
        kind = s.get("type", "rule")
        if kind == "rule":
            for k in ("severity", "headline", "advice", "when"):
                if k not in s:
                    out.append(f"{sid}: missing '{k}'")
            if s.get("severity") not in SEVERITY:
                out.append(f"{sid}: bad severity '{s.get('severity')}'")
            if "when" in s:
                _check_cond(s["when"], fields, sid, out)
        elif kind == "judgment":
            for k in ("rubric", "verdicts", "judge_fields"):
                if k not in s:
                    out.append(f"{sid}: missing '{k}'")
            for f in s.get("judge_fields", []):
                if f not in fields:
                    out.append(f"{sid}: unknown judge field '{f}'")
            for label, v in s.get("verdicts", {}).items():
                for k in ("severity", "headline", "advice"):
                    if k not in v:
                        out.append(f"{sid}/{label}: missing '{k}'")
                if v.get("severity") not in SEVERITY:
                    out.append(f"{sid}/{label}: bad severity")
        else:
            out.append(f"{sid}: unknown type '{kind}'")
    return out


def load_policies(directory) -> Policies:
    d = Path(directory)
    taxonomy = _read(d / "taxonomy.yaml")["tags"]
    wcfg = _read(d / "weather.yaml")
    sops = _read(d / "sops.yaml")["sops"]
    problems = validate(sops, taxonomy, wcfg["fields"])
    if problems:
        raise PolicyError("\n".join(problems))
    return Policies(sops, taxonomy, wcfg["fields"], wcfg["windows"])


class PolicyStore:
    """Hot-reloads policies when a yaml file changes, so an edit shows up on the next message
    with no restart. A bad edit is rejected and the last good policy set keeps serving."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self._sig, self._pol, self.last_error = None, None, None
        self.get()

    def _signature(self):
        return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(self.dir.glob("*.yaml")))

    def get(self) -> Policies:
        sig = self._signature()
        if sig != self._sig:
            try:
                self._pol, self.last_error = load_policies(self.dir), None
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)
                if self._pol is None:
                    raise
            self._sig = sig
        return self._pol


# ---------- evaluation ----------
def evaluate(cond: dict, facts: dict) -> bool:
    if "all" in cond:
        return all(evaluate(c, facts) for c in cond["all"])
    if "any" in cond:
        return any(evaluate(c, facts) for c in cond["any"])
    value = facts.get(cond["field"])
    if value is None:          # missing data never satisfies a condition
        return False
    return bool(OPS[cond["op"]](value, cond["value"]))


def condition_fields(cond: dict) -> set:
    if "all" in cond or "any" in cond:
        return set().union(*(condition_fields(c) for c in cond.get("all", cond.get("any"))))
    return {cond["field"]}


def applies(sop: dict, tags: list) -> bool:
    return "any" in sop["applies_to"] or bool(set(sop["applies_to"]) & set(tags))


def match(sops: list, tags: list, facts: dict):
    """Returns (triggered rule SOPs, fuzzy SOPs awaiting a verdict, covered_specifically).
    covered_specifically = some SOP names this activity/audience, not only catch-all 'any' SOPs."""
    triggered, fuzzy, specific = [], [], False
    for s in sops:
        if not applies(s, tags):
            continue
        if "any" not in s["applies_to"]:
            specific = True
        if s.get("type", "rule") == "judgment":
            fuzzy.append(s)
        elif evaluate(s["when"], facts):
            triggered.append(s)
    return triggered, fuzzy, specific


def rank(cands: list) -> list:
    """Conflict resolution (deliberate): lead policies first, then higher severity, then lower
    `priority` number, then id. Deterministic, so the same facts always give the same primary."""
    return sorted(cands, key=lambda s: (not s.get("lead", False), -SEVERITY[s["severity"]],
                                        s.get("priority", 50), s["id"]))


def from_verdict(sop: dict, label: str) -> dict:
    v = sop["verdicts"][label]
    return {"id": sop["id"], "title": sop["title"], "category": sop["category"], "type": "judgment",
            "severity": v["severity"], "priority": sop.get("priority", 50), "headline": v["headline"],
            "advice": v["advice"], "verdict": label, "judge_fields": sop["judge_fields"]}
