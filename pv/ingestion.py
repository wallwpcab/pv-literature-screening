import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Callable

import requests

from .models import Article

JOURNAL_WATCHLIST = {
    "BJP": "https://www.banglajol.info/index.php/BJP",
    "BJMS": "https://www.banglajol.info/index.php/BJMS",
    "BMJK": "https://banglajol.info/index.php/BMJK",
    "BJMP": "https://www.banglajol.info/index.php/BJMP",
}
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "PV-Literature-Screening/1.0 (research use)"})
    return session


def _get(session: requests.Session, url: str, *, params: dict, timeout: int,
         retries: int = 2) -> requests.Response:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def _parse_record(record: ET.Element, journal_code: str) -> Article | None:
    header = record.find("oai:header", OAI_NS)
    if header is None or header.get("status") == "deleted":
        return None
    identifier = header.findtext("oai:identifier", "", OAI_NS)
    metadata = record.find(".//oai_dc:dc", OAI_NS)
    if metadata is None:
        return None

    def values(tag: str) -> list[str]:
        return [item.text.strip() for item in metadata.findall(f"dc:{tag}", OAI_NS)
                if item.text and item.text.strip()]

    titles, descriptions, creators, dates = (
        values("title"), values("description"), values("creator"), values("date")
    )
    return Article(
        pmid=identifier,
        title=titles[0] if titles else "",
        abstract=descriptions[0] if descriptions else "",
        journal=journal_code,
        journal_code=journal_code,
        pub_date=dates[0] if dates else "",
        authors=creators,
        source="banglajol",
    )


def harvest_journal(journal_code: str, base_url: str, days_back: int = 30,
                    progress: Callable[[str], None] | None = None) -> list[Article]:
    since = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {"verb": "ListRecords", "metadataPrefix": "oai_dc", "from": since}
    endpoint = f"{base_url.rstrip('/')}/oai"
    articles: list[Article] = []
    session = _session()

    while True:
        response = _get(session, endpoint, params=params, timeout=30)
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise ValueError(f"{journal_code} returned invalid XML: {exc}") from exc

        error = root.find(".//oai:error", OAI_NS)
        if error is not None:
            if error.get("code") == "noRecordsMatch":
                break
            raise ValueError(f"{journal_code} OAI error: {error.text or error.get('code')}")

        for record in root.findall(".//oai:record", OAI_NS):
            article = _parse_record(record, journal_code)
            if article:
                articles.append(article)
        if progress:
            progress(f"{journal_code}: collected {len(articles)} records")

        token = root.findtext(".//oai:resumptionToken", "", OAI_NS)
        if not token.strip():
            break
        params = {"verb": "ListRecords", "resumptionToken": token.strip()}
        time.sleep(0.5)
    return articles


def filter_by_products(articles: list[Article], catalog: dict[str, list[str]]) -> list[Article]:
    matched: list[Article] = []
    for article in articles:
        text = f"{article.title} {article.abstract}".lower()
        hits = [product for product, synonyms in catalog.items()
                if any(term.lower() in text for term in [product, *synonyms])]
        if hits:
            article.matched_terms = hits
            article.query_term = hits[0]
            matched.append(article)
    return matched


def run_banglajol(catalog: dict[str, list[str]], days_back: int,
                  journals: dict[str, str],
                  progress: Callable[[str], None] | None = None) -> tuple[list[dict], list[str]]:
    all_articles: list[Article] = []
    errors: list[str] = []
    for code, url in journals.items():
        try:
            all_articles.extend(harvest_journal(code, url, days_back, progress))
        except (requests.RequestException, ValueError, ET.ParseError) as exc:
            errors.append(f"BanglaJOL {code}: {type(exc).__name__}: {exc}")
    return [a.to_dict() for a in filter_by_products(all_articles, catalog)], errors


def run_pubmed(catalog: dict[str, list[str]], days_back: int = 30,
               progress: Callable[[str], None] | None = None) -> tuple[list[dict], list[str]]:
    session = _session()
    results: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    since = (date.today() - timedelta(days=days_back)).strftime("%Y/%m/%d")
    today = date.today().strftime("%Y/%m/%d")

    for product, synonyms in catalog.items():
        terms = " OR ".join(f'"{term}"[Title/Abstract]' for term in [product, *synonyms])
        query = f'({terms}) AND ("{since}"[Date - Publication] : "{today}"[Date - Publication])'
        try:
            search = _get(session, f"{EUTILS_BASE}/esearch.fcgi", params={
                "db": "pubmed", "term": query, "retmax": 200, "retmode": "json",
            }, timeout=30)
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                continue
            time.sleep(0.34)
            fetch = _get(session, f"{EUTILS_BASE}/efetch.fcgi", params={
                "db": "pubmed", "id": ",".join(ids), "retmode": "xml",
            }, timeout=60)
            root = ET.fromstring(fetch.content)
        except (requests.RequestException, ValueError, ET.ParseError, KeyError) as exc:
            errors.append(f"PubMed {product}: {type(exc).__name__}: {exc}")
            continue

        for item in root.findall(".//PubmedArticle"):
            pmid = item.findtext(".//PMID", "")
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            title = item.findtext(".//ArticleTitle", "")
            abstract = " ".join(x.text or "" for x in item.findall(".//AbstractText"))
            authors = []
            for author in item.findall(".//Author"):
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())
            results.append(Article(
                pmid=pmid, title=title, abstract=abstract,
                journal=item.findtext(".//Journal/Title", ""),
                journal_code="PubMed",
                pub_date=item.findtext(".//PubDate/Year", ""),
                authors=authors, matched_terms=[product], query_term=product,
                source="pubmed",
            ).to_dict())
        if progress:
            progress(f"PubMed {product}: collected {len(results)} total records")
        time.sleep(0.34)
    return results, errors


def merge_deduplicate(*groups: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for group in groups:
        for article in group:
            key = (str(article.get("title", "")).strip().lower(), str(article.get("pmid", "")))
            if key not in seen:
                seen.add(key)
                merged.append(article)
    return merged
