# NLP Project: Geopolitical News & Exchange Rate Alignment (Task 1)

This repository contains the data acquisition and preprocessing pipeline for Task 1 of our NLP project. The objective is to construct a continuous, 5-year aligned dataset connecting US geopolitical news with the USD/IDR exchange rate to serve as the foundational input for future predictive market modeling.

## Project Team (Team: Miguel!)
* **Muhammad Keenan Basyir**
* **[Team Member 2 Name]**
* **[Team Member 3 Name]**

## Data Sources
1. **The GDELT Project (Global Database of Events, Language, and Tone):** Provides continuous, 24/7 unstructured global news parsed into quantifiable geopolitical metrics (e.g., `EventCode`, `AvgTone`). Sourced via Google Cloud BigQuery to ensure a robust 5-year historical extraction without triggering web-scraping rate limits.
2. **Bank Indonesia:** Provides the historical business-day USD/IDR exchange rates. Sourced via direct CSV export to guarantee an uninterrupted 5-year sequence, bypassing ASP.NET session-timeout restrictions.

## Repository Structure
```text
nlp-project/
├── data/
│   ├── raw/                  # Unprocessed datasets (gdelt_news_raw.csv, usd_idr_raw.csv)
│   └── cleaned/              # Final output (aligned_dataset.csv)
├── src/
│   ├── news_scraper.py       # BigQuery extraction script for GDELT
│   └── preprocessing.py      # Cleansing and temporal alignment logic
├── .gitignore                # Security exclusions (credentials, venv)
└── README.md                 # Project documentation
