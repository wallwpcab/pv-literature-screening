import math
from dataclasses import dataclass

SEVERITY_WEIGHTS = {"fatal": 100, "severe": 60, "moderate": 30, "mild": 10, "unknown": 15}


def rank_review_queue(cases: list[dict]) -> list[dict]:
    ranked = []
    for case in cases:
        score = SEVERITY_WEIGHTS.get(case.get("severity", "unknown"), 15)
        reasons = [case.get("severity", "unknown")]
        if case.get("manufacturing_related"):
            score += 30; reasons.append("manufacturing-related")
        if case.get("causality") in {"definite", "probable"}:
            score += 20; reasons.append(case["causality"])
        if case.get("case_count", 1) > 1:
            score += min(int(case["case_count"]) * 2, 20); reasons.append(f"n={case['case_count']}")
        item = dict(case)
        item["priority_score"] = round(score, 1)
        item["priority_reasons"] = "; ".join(reasons)
        ranked.append(item)
    return sorted(ranked, key=lambda item: item["priority_score"], reverse=True)


@dataclass
class ContingencyTable:
    a: float
    b: float
    c: float
    d: float


def _correct(table: ContingencyTable) -> ContingencyTable:
    values = [table.a, table.b, table.c, table.d]
    if any(value == 0 for value in values):
        return ContingencyTable(*(value + 0.5 for value in values))
    return table


def compute_prr(table: ContingencyTable) -> tuple[float, float, float]:
    if table.a == 0:
        return 0.0, 0.0, 0.0
    t = _correct(table)
    prr = (t.a / (t.a + t.b)) / (t.c / (t.c + t.d))
    se = math.sqrt(1 / t.a - 1 / (t.a + t.b) + 1 / t.c - 1 / (t.c + t.d))
    lower = math.exp(math.log(prr) - 1.96 * se)
    total = sum([t.a, t.b, t.c, t.d])
    expected = (t.a + t.b) * (t.a + t.c) / total if total else 0
    chi2 = ((t.a - expected) ** 2) / expected if expected else 0
    return prr, lower, chi2


def compute_ror(table: ContingencyTable) -> tuple[float, float]:
    if table.a == 0:
        return 0.0, 0.0
    t = _correct(table)
    ror = (t.a * t.d) / (t.b * t.c)
    se = math.sqrt(1 / t.a + 1 / t.b + 1 / t.c + 1 / t.d)
    return ror, math.exp(math.log(ror) - 1.96 * se)


def rank_signals(cases: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str], int] = {}
    for case in cases:
        event = case.get("adverse_event_term") or (
            "manufacturing-related issue" if case.get("manufacturing_related")
            else f"{case.get('severity', 'unknown')} adverse event"
        )
        key = (case.get("drug_name", "unknown"), event)
        counts[key] = counts.get(key, 0) + int(case.get("case_count") or 1)
    total = sum(counts.values())
    drugs = {drug: sum(v for (d, _), v in counts.items() if d == drug) for drug, _ in counts}
    events = {event: sum(v for (_, e), v in counts.items() if e == event) for _, event in counts}
    out = []
    for (drug, event), a in counts.items():
        table = ContingencyTable(a, drugs[drug] - a, events[event] - a, total - drugs[drug] - events[event] + a)
        prr, prr_lower, chi2 = compute_prr(table)
        ror, ror_lower = compute_ror(table)
        out.append({
            "drug": drug, "event": event, "num_reports": a,
            "prr": round(prr, 2), "prr_lower_ci": round(prr_lower, 2),
            "chi_squared": round(chi2, 2), "ror": round(ror, 2),
            "ror_lower_ci": round(ror_lower, 2),
            "signal_prr": prr >= 2 and chi2 >= 4 and a >= 3,
            "signal_ror": ror_lower >= 1 and a >= 3,
        })
    return sorted(out, key=lambda item: item["prr"], reverse=True)
