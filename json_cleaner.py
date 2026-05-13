import pandas as pd
import json
import re
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
ENRICHED_FILE = "./phillips_data/contemporary_lots_enriched.jsonl"
CLEANED_FILE = "./phillips_data/contemporary_lots_cleaned.jsonl"
OUTPUT_CSV = "./phillips_data/final_cv_dataset.csv"

# Converted to US currency, website: https://www.xe.com/en-us/currencycharts/?from=HKD&to=USD&view=10Y
# GBP to USD has been pretty volatile over past 10ish years, 1.3 is a loose average to avoid overcomplicating
# HKD pretty stable
# Probably wont encounter Geneva auctions since we are focusing on art but its volatile like GBP so its averaged
CURRENCY_FIX = {'London': 1.30, 'Hong Kong': 0.13, 'Geneva': 1.10, 'New York': 1.0}

# Inflation numbers from https://www.usinflationcalculator.com/
CPI_MAP = {
    2014: 1.33, 2015: 1.32, 2016: 1.30, 2017: 1.27, 2018: 1.24,
    2019: 1.22, 2020: 1.21, 2021: 1.15, 2022: 1.06, 2023: 1.03, 2024: 1.0
}

def parse_estimate(est_str):
    """Extracts low and high numbers from estimate string."""
    nums = re.findall(r'\d[\d,]*', str(est_str))
    nums = [float(n.replace(',', '')) for n in nums]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None

def get_surface_code(medium):
    """Categorizes the surface into numeric codes."""
    m = str(medium).lower()
    if 'canvas' in m: return 0  # On Canvas: 0
    if 'paper' in m: return 1  # On Paper: 1
    if any(x in m for x in ['wood', 'board', 'panel']): return 2  # On Wood, board, panel: 2
    if 'linen' in m: return 3  # On Linen: 3
    return 4  # Otherwise: 4

def process_data():
    # 1. Load Data
    with open(ENRICHED_FILE, 'r') as f:
        df_e = pd.DataFrame([json.loads(line) for line in f])
    
    with open(CLEANED_FILE, 'r') as f:
        df_c = pd.DataFrame([json.loads(line) for line in f])
        
    # 2. Merge for Surface Area
    # We include local_image_path here to link to the physical files
    df = pd.merge(df_e, df_c[['lot_url', 'surface_area']], on='lot_url', how='left')
    
    # 3. Standardization Logic
    df['auction_year'] = df['auction_date'].str.extract(r'(\d{4})').astype(float).fillna(2024).astype(int)
    
    def get_std_factor(row):
        c_factor = 1.0
        for city, rate in CURRENCY_FIX.items():
            if city in str(row['auction_date']):
                c_factor = rate
                break
        return c_factor * CPI_MAP.get(row['auction_year'], 1.0)

    # Convert to US dollars then multiply by inflation rate
    df['std_factor'] = df.apply(get_std_factor, axis=1)

    # 4. Price & Year Processing
    df['sold_for'] = df['sold_price_int'] * df['std_factor']
    
    estimates = df['estimate'].apply(parse_estimate)
    df['estimate_low'] = estimates.apply(lambda x: x[0]) * df['std_factor']
    df['estimate_high'] = estimates.apply(lambda x: x[1]) * df['std_factor']

    # Clean creation year: extract only digits (handles "c. 1990" or "1990-95")
    df['creation_year'] = df['year_created'].str.extract(r'(\d{4})').astype(float)

    # 5. Material & Surface Coding
    df['material'] = df.apply(lambda r: 0 if r['is_oil'] else (1 if r['is_acrylic'] else 2), axis=1)  # Oil: 0, Acrylic: 1, Otherwise: 2
    df['surface'] = df['medium'].apply(get_surface_code)

    # 6. Final Column Selection & Cleanup
    # mapping 'local_image_path' to 'image_path' for the CV loader
    final_cols = {
        'artist': 'artist_name',
        'estimate_high': 'estimate_high',
        'estimate_low': 'estimate_low',
        'sold_for': 'sold_price',
        'auction_year': 'auction_year',
        'creation_year': 'creation_year',
        'material': 'material',
        'surface': 'surface',
        'surface_area': 'surface_area',
        'local_image_path': 'image_path'
    }
    
    result = df[list(final_cols.keys())].rename(columns=final_cols)
    
    # 7. Quality Control for Machine Learning
    # Drop rows without labels (sold_price) or critical features (surface_area)
    initial_len = len(result)
    result = result.dropna(subset=['sold_price', 'surface_area', 'image_path'])
    
    # Remove rows where image file doesn't actually exist to prevent DataLoader crashes
    # Comment this out if you haven't downloaded images yet
    # result = result[result['image_path'].apply(lambda x: Path(x).exists())]

    result.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Dataset Refined:")
    print(f"- Rows retained: {len(result)} (Dropped {initial_len - len(result)} invalid rows)")
    print(f"- Columns: {list(result.columns)}")

if __name__ == "__main__":
    process_data()