# PV Literature Screening

This repository is a clean Streamlit implementation of the pharmacovigilance literature workflow originally developed in Colab. It harvests BanglaJOL and PubMed metadata, screens articles for safety relevance, extracts structured case fields, ranks the review queue, calculates exploratory PRR/ROR statistics, exports CSV files, and records an audit trail.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The application does not use Colab, LocalTunnel, `npx`, or runtime notebook installation. The public URL is provided by the hosting platform.

## Deploy on Streamlit Community Cloud

Push this repository to GitHub. In Streamlit Community Cloud, create an app using the repository, select the desired branch, and set the main file to `app.py`. The platform installs the packages from `requirements.txt` and redeploys when repository files change.

The first screening run downloads the Hugging Face zero-shot model and may take several minutes. The model is loaded lazily and cached for the running process. For low-memory deployments, replace the zero-shot model with a smaller classifier or a separate worker service.

## Architecture

`app.py` is the Streamlit UI and request controller. The `pv` package contains the backend modules:

| Module | Responsibility |
|---|---|
| `pv/models.py` | Article and extracted-case data models |
| `pv/audit.py` | SQLite audit repository |
| `pv/ingestion.py` | BanglaJOL OAI-PMH and PubMed ingestion with retries |
| `pv/screening.py` | Lazy zero-shot model loading and screening buckets |
| `pv/extraction.py` | Transparent regex and keyword extraction |
| `pv/analysis.py` | Prioritization and PRR/ROR calculations |

## Important deployment limitation

Audit logging is **disabled by default**, so the app does not require `secrets.toml`, a database path, or a writable local database. The core ingestion, screening, extraction, prioritization, signals, and CSV export features work without the audit database. The Audit tab displays the disabled status.

To enable local SQLite audit records for a single-user or development run, set this environment variable before starting Streamlit:

```powershell
$env:PV_AUDIT_ENABLED = "true"
python -m streamlit run app.py
```

A production multi-user system should move audit records and job results to an external database, because hosted app filesystems may be temporary and concurrent sessions should not share an uncoordinated local SQLite file.

The PRR/ROR results are exploratory. The current extractor uses a limited rule-based event vocabulary and should not be treated as a replacement for validated pharmacovigilance coding or regulatory review.
