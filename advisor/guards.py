"""Deterministic output guard: the model may only compose language, never introduce facts.
Every number in the draft must appear in the API-derived facts or in the SOP text; every SOP id
mentioned must be one that was actually selected."""
import re

NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
SOP_ID = re.compile(r"SOP-\d+")
LABELS = ("PRIMARY", "ALSO_FLAGGED", "primary_guidance", "also_flagged")
# Words that would add a safety/comfort claim. Allowed only if the selected SOP text itself uses them.
REASSURANCE = ("safe", "enjoy", "great", "perfect", "fine", "pleasant", "wonderful", "fun", "relax")


def numbers_in(text: str) -> set:
    return {float(m) for m in NUM.findall(text)}


def verify_reply(reply: str, fact_values, source_texts, allowed_ids, required_any=None) -> list:
    allowed = {float(v) for v in fact_values}
    for t in source_texts:
        allowed |= numbers_in(t)
    problems = []
    bad = sorted(n for n in numbers_in(reply) if n not in allowed)
    if bad:
        problems.append(f"ungrounded numbers: {bad}")
    stray = set(SOP_ID.findall(reply)) - set(allowed_ids)
    if stray:
        problems.append(f"cites policies that were not selected: {sorted(stray)}")
    if required_any:
        nums = numbers_in(reply)
        if not any(float(v) in nums for v in required_any):
            problems.append("reply does not quote any of the numbers that triggered the policy")
    leaked = [l for l in LABELS if l in reply]
    if leaked:
        problems.append(f"leaked prompt labels: {leaked}")
    src = " ".join(source_texts).lower()
    low = reply.lower()
    extra = [w for w in REASSURANCE if re.search(rf"\b{w}\b", low) and not re.search(rf"\b{w}\b", src)]
    if extra:
        problems.append(f"unapproved reassurance wording: {extra}")
    return problems
