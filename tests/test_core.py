from pv.analysis import ContingencyTable, compute_prr, compute_ror, rank_review_queue, rank_signals
from pv.extraction import extract_case
from pv.ingestion import merge_deduplicate


def test_extract_case_detects_severe_event():
    article = {
        "pmid": "x1",
        "title": "Severe liver injury after paracetamol",
        "abstract": "A 45-year-old woman developed hepatotoxicity; n=3 cases.",
        "query_term": "Paracetamol",
    }
    case = extract_case(article)
    assert case["severity"] == "severe"
    assert case["case_count"] == 3
    assert case["adverse_event_term"] == "hepatotoxicity"


def test_prioritization_puts_fatal_first():
    cases = [
        {"pmid": "1", "severity": "mild", "manufacturing_related": False, "case_count": 1},
        {"pmid": "2", "severity": "fatal", "manufacturing_related": False, "case_count": 1},
    ]
    assert rank_review_queue(cases)[0]["pmid"] == "2"


def test_signal_calculations_are_positive():
    prr, lower, chi2 = compute_prr(ContingencyTable(5, 2, 3, 20))
    ror, ror_lower = compute_ror(ContingencyTable(5, 2, 3, 20))
    assert prr > 0 and lower > 0 and chi2 >= 0
    assert ror > 0 and ror_lower > 0


def test_signal_ranking_returns_pair():
    cases = [
        {"drug_name": "A", "adverse_event_term": "rash", "case_count": 3},
        {"drug_name": "B", "adverse_event_term": "rash", "case_count": 1},
    ]
    result = rank_signals(cases)
    assert result[0]["drug"] == "A"
    assert result[0]["event"] == "rash"


def test_deduplication():
    one = {"title": "Same title", "pmid": "1"}
    two = {"title": "same title", "pmid": "1"}
    assert merge_deduplicate([one], [two]) == [one]


def test_disabled_audit_is_noop():
    from pv.audit import NullAuditRepository
    audit = NullAuditRepository()
    audit.log("x", "stage", "ok", "test")
    assert audit.rows() == []
    assert audit.enabled is False
