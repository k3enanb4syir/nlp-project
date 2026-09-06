# src/bi_scraper.py
import pandas as pd
from selenium import webdriver
import time

def scrape_bank_indonesia_usd():
    # Initialize Chrome WebDriver 
    driver = webdriver.Chrome()
    driver.get("https://www.bi.go.id/en/statistik/informasi-kurs/jisdor/default.aspx")
    
    # Wait for the ASP.NET table to load
    time.sleep(5) 
    
    # Extract the HTML table via Pandas
    html = driver.page_source
    tables = pd.read_html(html)
    
    # The JISDOR table is usually the first table on the page
    df = tables[0]
    
    # Clean up column names and date formats
    df.columns = ['Date', 'Exchange_Rate']
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Note: To get the full 5-year span required for the project, 
    # you will need to add Selenium logic here to input '01-Sep-2021' 
    # and '01-Sep-2026' into the page's date-picker fields.
    
    # Save to your raw data folder
    df.to_csv('data/raw/usd_idr_raw.csv', index=False)
    driver.quit()
    print("Financial data saved to data/raw/usd_idr_raw.csv")

if __name__ == "__main__":
    scrape_bank_indonesia_usd()