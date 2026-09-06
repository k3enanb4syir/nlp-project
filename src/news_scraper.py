# src/news_scraper.py
import os
import pandas as pd
from google.cloud import bigquery

# Authenticate using your JSON key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

def fetch_gdelt_news():
    client = bigquery.Client()
    
    # Query 5 years of US/Global geopolitical events
    query = """
        SELECT 
            SQLDATE as Date,
            SOURCEURL as URL,
            EventCode,
            GoldsteinScale,
            AvgTone
        FROM 
            `gdelt-bq.gdeltv2.events`
        WHERE 
            SQLDATE BETWEEN 20210901 AND 20260901
            AND ActionGeo_CountryCode = 'US'
        LIMIT 50000 
    """
    
    print("Querying GDELT database. This may take a few minutes...")
    query_job = client.query(query)
    df = query_job.to_dataframe()
    
    # Format the date to match your financial data
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.to_csv('data/raw/gdelt_news_raw.csv', index=False)
    print(f"Saved {len(df)} news records to data/raw/gdelt_news_raw.csv")

if __name__ == "__main__":
    fetch_gdelt_news()