# src/preprocessing.py
import pandas as pd

def clean_and_align_data():
    print("Loading raw datasets...")
    # Load the raw datasets
    news_df = pd.read_csv('data/raw/gdelt_news_raw.csv')
    
    # Skip the first 4 rows of metadata in the Bank Indonesia file
    fin_df = pd.read_csv('data/raw/usd_idr_raw.csv', skiprows=4)
    
    # Rename the 'Exchange Rates' column to match our alignment logic
    fin_df = fin_df.rename(columns={'Exchange Rates': 'Exchange_Rate'})

    # 1. DATA CLEANSING 
    print("Cleaning data...")
    # Ensure Dates are properly formatted datetime objects
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    fin_df['Date'] = pd.to_datetime(fin_df['Date'], dayfirst=True, format="mixed") 
    
    # Drop news rows with missing crucial text elements or metric scores
    news_df = news_df.dropna(subset=['EventCode', 'AvgTone'])
    
    # Sort both dataframes chronologically
    news_df = news_df.sort_values('Date')
    fin_df = fin_df.sort_values('Date')

    # 2. TEMPORAL DATA ALIGNMENT
    print("Aligning 24/7 news with business trading days...")
    # STRATEGY: Use a forward-looking merge (merge_asof). 
    # If news is published on Saturday (non-trading), it aligns to Monday's exchange rate.
    # If news is published on a Holiday, it aligns to the day after the holiday.
    
    aligned_df = pd.merge_asof(
        news_df,               # Left dataframe (24/7 news)
        fin_df,                # Right dataframe (Business days only)
        on='Date',             # Align based on the Date column
        direction='forward'    # Map to the exact day OR the NEXT available day
    )

    # Drop any leftover news at the very end of 2026 that has no future trading day to map to
    aligned_df = aligned_df.dropna(subset=['Exchange_Rate'])

    # 3. SAVE FINAL DATASET
    aligned_df.to_csv('data/cleaned/aligned_dataset.csv', index=False)
    print("Success! Cleaned and aligned dataset saved to data/cleaned/aligned_dataset.csv")

if __name__ == "__main__":
    clean_and_align_data()