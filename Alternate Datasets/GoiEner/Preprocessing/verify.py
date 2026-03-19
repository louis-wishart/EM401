import pandas as pd
import os
import glob

def inspect_parquet_files():
    print("--- Parquet File Inspector ---")
    
    # 1. Find all parquet files in the current directory
    parquet_files = glob.glob("*.parquet")
    
    if not parquet_files:
        print("No .parquet files found in the current folder.")
        return

    # 2. Let user choose a file
    print("\nFound the following files:")
    for i, f in enumerate(parquet_files):
        print(f"{i+1}: {f}")
    
    try:
        selection = int(input("\nEnter the number of the file to inspect: ")) - 1
        if selection < 0 or selection >= len(parquet_files):
            print("Invalid selection.")
            return
        target_file = parquet_files[selection]
    except ValueError:
        print("Invalid input.")
        return

    # 3. Load and Display Info
    print(f"\nLoading {target_file}...")
    try:
        df = pd.read_parquet(target_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print("-" * 40)
    print(f"Basic Info:")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print("-" * 40)
    
    print("\nFirst 5 Rows:")
    print(df.head().to_string())
    
    print("\n\nLast 5 Rows:")
    print(df.tail().to_string())
    
    print("\n\nColumn Types:")
    print(df.dtypes)
    
    # Check for 'date' or 'timestamp' columns
    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    elif 'timestamp' in df.columns:
        date_col = 'timestamp'
        
    if date_col:
        print("\nDate Range Verified:")
        print(f"  Start: {df[date_col].min()}")
        print(f"  End:   {df[date_col].max()}")

if __name__ == "__main__":
    inspect_parquet_files()