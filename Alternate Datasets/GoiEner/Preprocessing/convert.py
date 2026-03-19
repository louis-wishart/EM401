import pandas as pd
import os

# --- Configuration ---
INPUT_CSV = 'dataset.csv'
# OUTPUT_PARQUET is no longer hardcoded for saving, 
# the script will generate a name based on the date range.

def interactive_convert():
    print(f"--- Interactive Data Converter ---")
    
    # 1. Load the CSV (Scanning for dates)
    print(f"Reading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Read CSV
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    print(f"\nData Loaded: {len(df):,} rows.")
    print(f"Date Range Available: {min_date.date()} to {max_date.date()}")
    
    # 2. Ask for Date Range
    print("\n--- Filter Data (Optional) ---")
    print("Press Enter to keep all data, or type a date (YYYY-MM-DD).")
    
    start_input = input(f"Start Date [{min_date.date()}]: ").strip()
    end_input = input(f"End Date   [{max_date.date()}]: ").strip()
    
    # Set defaults if empty
    start_date = pd.to_datetime(start_input) if start_input else min_date
    end_date = pd.to_datetime(end_input) if end_input else max_date
    
    # 3. Filter
    if start_date != min_date or end_date != max_date:
        print(f"\nFiltering from {start_date.date()} to {end_date.date()}...")
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df = df.loc[mask].copy()
        print(f"Rows remaining: {len(df):,}")
    else:
        print("\nKeeping full date range.")

    # 4. Save as Parquet (Best Format) with Dynamic Name
    # Format: dataset_YYYY-MM-DD_to_YYYY-MM-DD.parquet
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    output_filename = f"dataset_{start_str}_to_{end_str}.parquet"

    print(f"\nSaving to {output_filename} (Parquet format)...")
    # Parquet requires 'pyarrow' or 'fastparquet'
    # pip install pyarrow
    df.to_parquet(output_filename, index=False)
    
    # 5. Show Stats
    csv_size = os.path.getsize(INPUT_CSV) / (1024 * 1024)
    par_size = os.path.getsize(output_filename) / (1024 * 1024)
    
    print("-" * 30)
    print(f"Success! Data ready for modeling.")
    print(f"File saved as: {output_filename}")
    print(f"Original CSV: {csv_size:.2f} MB")
    print(f"New Parquet:  {par_size:.2f} MB")
    print(f"Compression:  {csv_size/par_size:.1f}x smaller")

if __name__ == "__main__":
    try:
        interactive_convert()
    except ImportError:
        print("\nError: 'pyarrow' library missing.")
        print("Please run: pip install pyarrow")