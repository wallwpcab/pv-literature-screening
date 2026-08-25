import re

from .models import ExtractedCase

SEVERITY_KEYWORDS = {
    "fatal": ["fatal", "death", "died", "deceased", "mortality"],
    "severe": ["severe", "serious", "life-threatening", "hospitalization", "hospitalisation"],
    "moderate": ["moderate"],
    "mild": ["mild", "minor", "self-limiting"],
}
CAUSALITY_TERMS = ["definite", "probable", "possible", "unlikely", "unclassifiable", "temporally associated", "causally related", "not established"]
MANUFACTURING_TERMS = ["contamination", "recall", "batch defect", "lot defect", "impurity", "formulation defect", "packaging defect", "counterfeit", "substandard", "contaminated batch", "faulty batch", "manufacturing defect", "manufacturing error"]
EVENT_TERMS = ["hepatotoxicity", "liver injury", "rash", "anaphylaxis", "anaphylactic shock", "kidney injury", "nephrotoxicity", "cardiotoxicity", "seizure", "bleeding", "thrombosis", "hypertension", "hypotension", "nausea", "vomiting", "diarrhea", "diarrhoea", "headache", "allergic reaction"]
DEMOGRAPHIC_PATTERN = re.compile(r"(\d{1,3}[-\s]?year[-\s]?old\s+\w+|\baged?\s+\d{1,3}(?:[-\s]?(?:to|-)\s?\d{1,3})?\s*(?:years?)?)", re.I)
CASE_COUNT_PATTERN = re.compile(r"(?:n\s*=\s*(\d+))|(\d+)\s+(?:cases?|patients?|subjects?)", re.I)


def _positive_context(text: str, keyword: str) -> bool:
    for match in re.finditer(re.escape(keyword), text, re.I):
        before = text[max(0, match.start() - 25):match.start()].lower()
        if not any(neg in before for neg in ["no ", "not ", "without ", "absence of "]):
            return True
    return False


def extract_case(article: dict) -> dict:
    text = f"{article.get('title', '')} {article.get('abstract', '') or ''}"
    lower = text.lower()
    severity = "unknown"
    for level in ("fatal", "severe", "moderate", "mild"):
        if any(_positive_context(lower, term) for term in SEVERITY_KEYWORDS[level]):
            severity = level
            break
    causality = next((term for term in CAUSALITY_TERMS if term in lower), "")
    manufacturing = any(_positive_context(lower, term) for term in MANUFACTURING_TERMS)
    demographic_match = DEMOGRAPHIC_PATTERN.search(text)
    count_match = CASE_COUNT_PATTERN.search(text)
    case_count = 1
    if count_match:
        case_count = int(count_match.group(1) or count_match.group(2))
    event = next((term for term in EVENT_TERMS if term in lower), "")
    drug = (article.get("query_term") or (article.get("matched_terms") or [""])[0])
    notes = "; ".join(x for x in [
        f"severity={severity}",
        f"causality={causality}" if causality else "",
        "manufacturing-related" if manufacturing else "",
        f"case_count={case_count}",
    ] if x)
    return ExtractedCase(
        pmid=str(article.get("pmid", "")),
        drug_name=drug,
        severity=severity,
        demographics=demographic_match.group(1) if demographic_match else "",
        causality=causality,
        manufacturing_related=manufacturing,
        case_count=case_count,
        adverse_event_term=event,
        extraction_notes=notes,
    ).to_dict()


def extract_batch(articles: list[dict]) -> list[dict]:
    return [extract_case(article) for article in articles]
