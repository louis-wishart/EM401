import os
import glob
import random
import pandas as pd
from collections import Counter

def create_sample_files(
    source_folder, 
    output_data_csv, 
    output_summary_csv, 
    output_global_report_txt, 
    sample_size
):
    print(f"--- Starting CSV Sample Creation ---")
    
    # --- Step 1: Find all files ---
    print(f"Step 1: Searching for .csv files in '{source_folder}'...")
    all_files = glob.glob(os.path.join(source_folder, '*.csv'))
    
    if not all_files:
        print(f"ERROR: No .csv files found in '{source_folder}'.")
        return

    print(f"Found {len(all_files)} total customer files.")

    # --- Step 2: Select sample ---
    if len(all_files) < sample_size:
        sample_size = len(all_files)
        
    print(f"Step 2: Randomly selecting {sample_size} files...")
    sampled_files = random.sample(all_files, sample_size)
    
    # --- Step 3: Process files ---
    print(f"Step 3: Processing {sample_size} files in memory...")

    all_data_dfs = []
    customer_summary_list = []
    all_imputed_timestamps = []
    total_readings_processed = 0

    for i, file_path in enumerate(sampled_files):
        try:
            filename = os.path.basename(file_path)
            customer_id = os.path.splitext(filename)[0]
            
            df = pd.read_csv(file_path)
            customer_total_readings = len(df)
            total_readings_processed += customer_total_readings
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # KEEP THE IMPUTED COLUMN
            data_df = df[['timestamp', 'kWh', 'imputed']].copy()
            data_df['meter'] = customer_id
            # Reorder: meter, date, data, imputed
            data_df = data_df[['meter', 'timestamp', 'kWh', 'imputed']]
            all_data_dfs.append(data_df)
            
            # Summary data
            imputed_rows = df[df['imputed'] == 1]
            total_imputed = len(imputed_rows)
            
            first_imputed = None
            last_imputed = None
            if total_imputed > 0:
                first_imputed = imputed_rows['timestamp'].min()
                last_imputed = imputed_rows['timestamp'].max()
                all_imputed_timestamps.extend(list(imputed_rows['timestamp']))
            
            customer_summary_list.append({
                'meter': customer_id,
                'total_imputed_rows': total_imputed,
                'total_readings': customer_total_readings,
                'first_imputed_timestamp': first_imputed,
                'last_imputed_timestamp': last_imputed
            })
            
            if (i + 1) % 100 == 0:
                print(f"  ...processed {i + 1} / {sample_size} files...")
                
        except Exception as e:
            print(f"ERROR processing file {file_path}: {e}")

    # --- Step 4: Write main data CSV ---
    print(f"Step 4: Writing main data to '{output_data_csv}'...")
    if all_data_dfs:
        main_df = pd.concat(all_data_dfs, ignore_index=True)
        main_df = main_df.rename(columns={'timestamp': 'date', 'kWh': 'data'})
        main_df.to_csv(output_data_csv, index=False)
        print(f"  Successfully wrote {len(main_df)} rows.")
        
    # --- Step 5: Write summary CSV ---
    print(f"Step 5: Writing summary to '{output_summary_csv}'...")
    if customer_summary_list:
        customer_summary_list.sort(key=lambda x: x['total_imputed_rows'], reverse=True)
        cols = ['meter', 'total_imputed_rows', 'total_readings', 'first_imputed_timestamp', 'last_imputed_timestamp']
        pd.DataFrame(customer_summary_list, columns=cols).to_csv(output_summary_csv, index=False)

    # --- Step 6: Write report ---
    print(f"Step 6: Writing report to '{output_global_report_txt}'...")
    with open(output_global_report_txt, 'w') as f:
        f.write(f"Total Readings Processed: {total_readings_processed}\n")
        f.write(f"Total Imputed Timestamps: {len(all_imputed_timestamps)}\n")

    print(f"--- Done! ---")

if __name__ == "__main__":
    # UPDATE THIS PATH BEFORE RUNNING
    SOURCE_FOLDER = r'/Users/louis.wishart/Library/CloudStorage/OneDrive-UniversityofStrathclyde/EM401/Code/EM401/All Data'
    OUTPUT_DATA_CSV = 'dataset1.csv'
    OUTPUT_SUMMARY_CSV = 'imputed_summary.csv'
    OUTPUT_GLOBAL_REPORT_TXT = 'imputed_summary.txt'
    SAMPLE_SIZE = 1500

    if not os.path.isdir(SOURCE_FOLDER):
        print(f"Error: Folder not found: {SOURCE_FOLDER}")
    else:
        create_sample_files(SOURCE_FOLDER, OUTPUT_DATA_CSV, OUTPUT_SUMMARY_CSV, OUTPUT_GLOBAL_REPORT_TXT, SAMPLE_SIZE)