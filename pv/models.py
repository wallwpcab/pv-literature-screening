from dataclasses import dataclass, field
from typing import Any


@dataclass
class Article:
    pmid: str
    title: str = ""
    abstract: str = ""
    journal: str = ""
    journal_code: str = ""
    pub_date: str = ""
    authors: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    source: str = ""
    query_term: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "journal": self.journal,
            "journal_code": self.journal_code,
            "pub_date": self.pub_date,
            "authors": self.authors,
            "matched_terms": self.matched_terms,
            "source": self.source,
            "query_term": self.query_term,
        }


@dataclass
class ExtractedCase:
    pmid: str
    drug_name: str = ""
    severity: str = "unknown"
    demographics: str = ""
    causality: str = ""
    manufacturing_related: bool = False
    case_count: int = 1
    adverse_event_term: str = ""
    extraction_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
