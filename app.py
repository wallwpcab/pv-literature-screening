import io
import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from pv.analysis import rank_review_queue, rank_signals
from pv.audit import AuditRepository, NullAuditRepository
from pv.extraction import extract_batch
from pv.ingestion import JOURNAL_WATCHLIST, merge_deduplicate, run_banglajol, run_pubmed
from pv.screening import load_classifier, screen_batch

st.set_page_config(page_title="PV Literature Screening", page_icon="⚕", layout="wide")
logging.basicConfig(filename="app_errors.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

AUDIT_ENABLED = os.getenv("PV_AUDIT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
DB_PATH = os.getenv("PV_AUDIT_DB_PATH", "pv_audit.db")
audit = AuditRepository(DB_PATH) if AUDIT_ENABLED else NullAuditRepository()


@st.cache_resource(show_spinner=False)
def get_classifier():
    return load_classifier()


def parse_catalog(text: str) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        name, synonyms = line.split(":", 1)
        name = name.strip()
        if name:
            catalog[name] = [item.strip() for item in synonyms.split(",") if item.strip()]
    return catalog


def initialize_state() -> None:
    defaults = {
        "articles": [], "screening": None, "cases": [],
        "ranked_cases": [], "signals": [], "ingestion_errors": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def progress_message(message: str) -> None:
    st.session_state["progress_message"] = message


def run_ingestion(catalog, days_back, journals, use_pubmed):
    errors: list[str] = []
    status = st.empty()
    bar = st.progress(0)
    try:
        status.info("Starting BanglaJOL ingestion...")
        banglajol, banglajol_errors = run_banglajol(catalog, days_back, journals, progress_message)
        errors.extend(banglajol_errors)
        bar.progress(0.5 if use_pubmed else 1.0)
        pubmed = []
        if use_pubmed:
            status.info("Starting PubMed ingestion...")
            pubmed, pubmed_errors = run_pubmed(catalog, min(days_back, 90), progress_message)
            errors.extend(pubmed_errors)
        merged = merge_deduplicate(banglajol, pubmed)
        st.session_state["articles"] = merged
        st.session_state["ingestion_errors"] = errors
        for article in merged:
            audit.log(article.get("pmid", ""), "ingested", "new", f"system:{article.get('source', 'unknown')}_ingest", query_term=article.get("query_term", ""))
        bar.progress(1.0)
        status.success(f"Ingestion complete: {len(merged)} unique articles")
    except Exception as exc:
        logging.exception("Ingestion failed")
        status.error(f"Ingestion failed: {type(exc).__name__}: {exc}")
        st.exception(exc)


def main() -> None:
    initialize_state()
    st.title("PV Literature Screening")
    st.caption("BanglaJOL + PubMed literature workflow for research and reviewer triage")

    with st.sidebar:
        st.header("Configuration")
        catalog_text = st.text_area(
            "Products and synonyms",
            value="Paracetamol: acetaminophen, napa\nOmeprazole: seclo, losec",
            height=120,
            help="One product per line: Product name: synonym1, synonym2",
        )
        catalog = parse_catalog(catalog_text)
        days_back = st.slider("Days back to harvest", 7, 730, 365, 7)
        selected = st.multiselect("BanglaJOL journals", list(JOURNAL_WATCHLIST), default=list(JOURNAL_WATCHLIST))
        use_pubmed = st.checkbox("Also harvest PubMed", value=True)
        threshold = st.slider("Screening confidence threshold", 0.0, 1.0, 0.6, 0.05)

    tabs = st.tabs(["1. Ingestion", "2. Screening", "3. Extraction", "4. Prioritization", "5. Signals", "6. Audit"])

    with tabs[0]:
        st.header("Step 1 — Harvest literature")
        if st.button("Run harvest", type="primary"):
            if not catalog:
                st.error("Enter at least one product in the sidebar.")
            elif not selected and not use_pubmed:
                st.error("Select a BanglaJOL journal or enable PubMed.")
            else:
                run_ingestion(catalog, days_back, {k: JOURNAL_WATCHLIST[k] for k in selected}, use_pubmed)
        if st.session_state["ingestion_errors"]:
            with st.expander("Source warnings", expanded=False):
                for error in st.session_state["ingestion_errors"]:
                    st.warning(error)
        if st.session_state["articles"]:
            df = pd.DataFrame(st.session_state["articles"])
            st.dataframe(df[["pmid", "title", "source", "journal_code", "pub_date"]], use_container_width=True)
            st.download_button("Download ingested CSV", df.to_csv(index=False), "pv_ingested_articles.csv", "text/csv")

    with tabs[1]:
        st.header("Step 2 — Relevance screening")
        articles = st.session_state["articles"]
        if not articles:
            st.info("Run ingestion first.")
        elif st.button("Run screening", type="primary"):
            with st.spinner("Loading the screening model and processing articles..."):
                progress = st.progress(0)
                try:
                    buckets = screen_batch(articles, threshold, get_classifier(), lambda i, n: progress.progress(i / n))
                    st.session_state["screening"] = buckets
                    for name, values in buckets.items():
                        for article in values:
                            audit.log(article.get("pmid", ""), "screened", name, "system:zero_shot_classifier", detail=str(article.get("screening")))
                    st.success(f"Relevant: {len(buckets['relevant'])}; needs review: {len(buckets['needs_review'])}; not relevant: {len(buckets['not_relevant'])}")
                except Exception as exc:
                    logging.exception("Screening failed")
                    st.error(f"Screening failed: {type(exc).__name__}: {exc}")
                    st.exception(exc)
        if st.session_state["screening"]:
            all_screened = sum(st.session_state["screening"].values(), [])
            st.dataframe(pd.DataFrame([
                {"pmid": x["pmid"], "title": x["title"], "relevant": x["screening"]["relevant"], "confidence": x["screening"]["confidence"], "reason": x["screening"]["reason"]}
                for x in all_screened
            ]), use_container_width=True)

    with tabs[2]:
        st.header("Step 3 — Structured extraction")
        relevant = st.session_state["screening"]["relevant"] if st.session_state["screening"] else []
        if not relevant:
            st.info("Run screening first.")
        elif st.button("Run extraction", type="primary"):
            try:
                st.session_state["cases"] = extract_batch(relevant)
                for case in st.session_state["cases"]:
                    audit.log(case["pmid"], "extracted", "success", "system:rule_based_extraction", detail=str(case))
                st.success(f"Extracted {len(st.session_state['cases'])} cases.")
            except Exception as exc:
                logging.exception("Extraction failed")
                st.error(f"Extraction failed: {type(exc).__name__}: {exc}")
                st.exception(exc)
        if st.session_state["cases"]:
            st.dataframe(pd.DataFrame(st.session_state["cases"]), use_container_width=True)

    with tabs[3]:
        st.header("Step 4 — Prioritization")
        if not st.session_state["cases"]:
            st.info("Run extraction first.")
        elif st.button("Rank review queue", type="primary"):
            st.session_state["ranked_cases"] = rank_review_queue(st.session_state["cases"])
            st.success(f"Ranked {len(st.session_state['ranked_cases'])} cases.")
        if st.session_state["ranked_cases"]:
            ranked = pd.DataFrame(st.session_state["ranked_cases"])
            st.dataframe(ranked, use_container_width=True)
            st.download_button("Download reviewer queue", ranked.to_csv(index=False), "pv_review_queue.csv", "text/csv")

    with tabs[4]:
        st.header("Step 5 — Signal detection")
        if not st.session_state["cases"]:
            st.info("Run extraction first.")
        elif st.button("Run signal detection", type="primary"):
            st.session_state["signals"] = rank_signals(st.session_state["cases"])
            st.success(f"Calculated {len(st.session_state['signals'])} drug-event pairs.")
        if st.session_state["signals"]:
            st.dataframe(pd.DataFrame(st.session_state["signals"]), use_container_width=True)

    with tabs[5]:
        st.header("Step 6 — Audit trail")
        if not audit.enabled:
            st.info("Audit logging is disabled. Set PV_AUDIT_ENABLED=true before starting the app to enable local SQLite audit records.")
        else:
            pmid = st.text_input("Filter by PMID or record ID")
            rows = audit.rows(pmid=pmid.strip() or None)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No audit records yet.")


if __name__ == "__main__":
    main()
