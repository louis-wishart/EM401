import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
SOURCE_DATA_FILE = 'dataset1.csv'
CLEAN_OUTPUT_FILE = 'dataset.csv'
REPORT_FILE = 'data_summary.txt'
PLOT_FILE = 'imputed_plot.png' 
# This is now the default, which can be overridden by user input
IMPUTATION_THRESHOLD_PERCENT = 3.0 

# Time Window (365 days = 8760 hours)
START_DATE = '2019-03-01 00:00:00'
END_DATE = '2020-02-28 23:00:00' 
EXPECTED_HOURS = 8760
# --- --- --- --- --- ---

def prepare_data_for_modeling(source_file, output_file, report_file, plot_file, start, end, expected_hours, threshold):
    print(f"--- Starting Modeling Data Preparation ---")
    print(f"Using imputation threshold: {threshold}%")
    
    # --- Step 1: Load the source data ---
    print(f"Loading '{source_file}'...")
    try:
        df = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"ERROR: File not found: '{source_file}'")
        print("Please run the (updated) 'create_sample_files.py' script first.")
        return
    except Exception as e:
        print(f"Error loading {source_file}: {e}")
        return
        
    df['date'] = pd.to_datetime(df['date'])

    # --- Step 2: Filter all data to the time window ---
    print(f"Filtering all data between {start} and {end}...")
    
    df_windowed = df.set_index('date').sort_index().loc[start:end].reset_index()

    if df_windowed.empty:
        print("ERROR: No data found within this time window.")
        return

    # --- Step 3: Check for "Complete" Meters ---
    print(f"Time window is {expected_hours} hours ({expected_hours / 24} days).")
    print(f"Checking for 'complete' meters (must have exactly {expected_hours} readings)...")
    
    readings_per_meter = df_windowed.groupby('meter').size()
    complete_meters = readings_per_meter[readings_per_meter == expected_hours].index
    
    total_meters_in_window = len(readings_per_meter)
    total_complete_meters = len(complete_meters)
    
    if total_complete_meters == 0:
        print("ERROR: No meters with complete data for this period were found.")
        return
        
    print(f"Found {total_complete_meters} / {total_meters_in_window} meters with complete data.")

    # --- Step 4: Filter to *only* the complete meters ---
    df_complete = df_windowed[df_windowed['meter'].isin(complete_meters)].copy()
    
    # --- Step 5: Calculate Imputation % for this clean set ---
    print("Calculating in-window imputation percentage for these meters...")
    
    imputation_counts = df_complete.groupby('meter')['imputed'].sum()
    imputation_percent = (imputation_counts / expected_hours) * 100
    
    # --- Step 6: Write the final report ---
    print(f"Writing final analysis report to '{report_file}'...")
    # 'w' mode = OVERWRITE the file
    with open(report_file, 'w') as f:
        f.write("--- Modeling Data Preparation Report ---\n\n")
        f.write(f"Time Window: {start} to {end}\n")
        f.write(f"Expected Readings: {expected_hours}\n\n")
        
        f.write(f"Meters found in window: {total_meters_in_window}\n")
        f.write(f"Meters with complete data: {total_complete_meters}\n")
        f.write(f"Meters discarded (incomplete): {total_meters_in_window - total_complete_meters}\n\n")
        
        f.write("--- Imputation Statistics (for COMPLETE meters only) ---\n")
        f.write(f"These are the meters you can use for modeling.\n")
        f.write(imputation_percent.describe().to_string())
        
        f.write(f"\n\n--- Meters with > {threshold}% Imputation (to consider removing) ---\n")
        bad_meters_report = imputation_percent[imputation_percent > threshold]
        if bad_meters_report.empty:
            f.write(f"No meters found with > {threshold}% imputation. Your data is very clean.")
        else:
            # Sort by percentage, descending, for the report
            f.write(bad_meters_report.sort_values(ascending=False).to_string())

    # --- Step 7: Create a 2-PART PLOT ---
    try:
        print(f"Saving sorted imputation plot to '{plot_file}'...")
        
        sorted_imputation = imputation_percent.sort_values().reset_index(drop=True)
        
        fig, (ax_main, ax_zoom) = plt.subplots(nrows=2, ncols=1, figsize=(12, 14), 
                                               gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Plot 1: The "Overall" View ---
        ax_main.plot(sorted_imputation, marker='.', linestyle='-', markersize=2, label='Imputation %')
        ax_main.set_title(f'Plot 1: Overall View - All {total_complete_meters} Complete Meters')
        ax_main.set_xlabel('Meters (Sorted from Cleanest to Dirtiest)')
        ax_main.set_ylabel('Imputation Percentage')
        ax_main.grid(True, linestyle='--')
        ax_main.axhline(y=threshold, color='red', linestyle=':', label=f'{threshold:.1f}% Threshold')
        ax_main.legend()

        # --- Plot 2: The "Bad Meters > threshold" View ---
        bad_meters = imputation_percent[imputation_percent > threshold]
        bad_meters = bad_meters.sort_values(ascending=True) 

        if bad_meters.empty:
            ax_zoom.text(0.5, 0.5, f'No meters found with > {threshold:.1f}% imputation.', 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=ax_zoom.transAxes)
            ax_zoom.set_title(f'Plot 2: Meters with > {threshold:.1f}% Imputation')
        else:
            bar_positions = np.arange(len(bad_meters))
            ax_zoom.barh(bar_positions, bad_meters.values, color='tomato') 
            
            truncated_labels = ['...' + mid[-10:] for mid in bad_meters.index]
            ax_zoom.set_yticks(bar_positions)
            ax_zoom.set_yticklabels(truncated_labels) 
            
            ax_zoom.set_title(f'Plot 2: Zoomed View - Only the {len(bad_meters)} Meters with > {threshold:.1f}% Imputation')
            ax_zoom.set_ylabel(f'Meter ID (Truncated)')
            ax_zoom.set_xlabel('Imputation Percentage')
            ax_zoom.grid(True, linestyle='--', axis='x')
            ax_zoom.text(0.98, 0.01, 'Full meter IDs are in the report file',
                         ha='right', transform=ax_zoom.transAxes, style='italic', fontsize=9)

        plt.tight_layout()
        plt.savefig(plot_file)
        print("Plot saved successfully. Please review it now.")
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("Please ensure you have 'matplotlib' installed: python3 -m pip install matplotlib")

    # --- --- --- --- --- --- --- --- --- --- --- ---
    # --- Step 8: User Decision ---
    # --- --- --- --- --- --- --- --- --- --- --- ---
    print("\n" + "-"*34)
    print("--- User Decision ---")
    
    # Re-calculate bad meters just to be safe
    bad_meters_to_remove = imputation_percent[imputation_percent > threshold]
    num_bad_meters = len(bad_meters_to_remove)
    
    if num_bad_meters == 0:
        print(f"No meters found above your {threshold:.1f}% threshold.")
        # We will save all meters by default
        final_df_to_save = df_complete
    else:
        print(f"Analysis found {num_bad_meters} meters with > {threshold:.1f}% imputation.")
        print("Please review the plot and report files before answering.")
        
        while True:
            choice = input(f"Do you want to REMOVE these {num_bad_meters} meters from the final dataset? (y/n): ").strip().lower()
            if choice == 'y':
                print(f"Removing {num_bad_meters} meters.")
                # Get the *good* meter IDs by finding those *not* in the bad list
                good_meter_ids = imputation_percent[imputation_percent <= threshold].index
                final_df_to_save = df_complete[df_complete['meter'].isin(good_meter_ids)]
                break
            elif choice == 'n':
                print("Keeping all meters (including those above the threshold).")
                final_df_to_save = df_complete
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
                
    # --- --- --- --- --- --- --- --- --- --- --- ---
    # --- Step 9: Save the final, clean dataset ---
    # --- --- --- --- --- --- --- --- --- --- --- ---
    
    # We drop the 'imputed' column *after* the decision is made.
    final_df_to_save = final_df_to_save.drop(columns=['imputed'])
    
    # --- --- --- FINAL SANITY CHECK --- --- ---
    num_meters_saved = len(final_df_to_save['meter'].unique())
    actual_rows_saved = len(final_df_to_save)
    expected_rows_saved = num_meters_saved * expected_hours

    print("\n" + "-"*34)
    print("--- Final Sanity Check ---")
    print(f"Meters to save: {num_meters_saved}")
    print(f"Expected hours per meter: {expected_hours}")
    print(f"Expected total rows: {num_meters_saved} * {expected_hours} = {expected_rows_saved}")
    print(f"Actual total rows to save: {actual_rows_saved}")
    
    if actual_rows_saved == expected_rows_saved:
        print("SANITY CHECK: PASSED. Row count is perfect.")
        sanity_check_result = "PASSED"
    else:
        print("SANITY CHECK: FAILED. Row count does not match expected count.")
        print("This may indicate a bug in the script.")
        sanity_check_result = f"FAILED (Expected {expected_rows_saved})"
    print("-" * 34 + "\n")
    # --- --- --- END OF SANITY CHECK --- --- ---

    print(f"Saving {num_meters_saved} meters to '{output_file}'...")
    final_df_to_save.to_csv(output_file, index=False)
    
    # --- --- --- NEW: Append final summary to report --- --- ---
    try:
        # 'a' mode = APPEND to the file
        with open(report_file, 'a') as f: 
            f.write("\n\n" + "="*34 + "\n")
            f.write("--- Final Dataset Summary ---\n")
            f.write(f"This is the final state of '{output_file}'.\n\n")
            f.write(f"Total Meters Saved: {num_meters_saved}\n")
            f.write(f"Total Readings Saved: {actual_rows_saved}\n")
            f.write(f"Sanity Check: {sanity_check_result}\n")
    except Exception as e:
        print(f"Warning: Could not append final summary to report: {e}")
    # --- --- --- END OF NEW BLOCK --- --- ---
    
    print("\n--- Done ---")
    print(f"Your final data for modeling is ready in '{output_file}'.")
    print(f"A full analysis is in '{report_file}'.")


if __name__ == "__main__":
    
    # --- --- --- USER INPUT SECTION --- --- ---
    default_threshold = IMPUTATION_THRESHOLD_PERCENT
    print(f"--- Set Imputation Threshold ---")
    
    user_input = input(f"Enter a percentage threshold (e.g., 3.0) [default: {default_threshold}]: ")
    
    final_threshold = default_threshold
    
    if user_input.strip() == "":
        print(f"Using default threshold: {default_threshold}%")
    else:
        try:
            final_threshold = float(user_input)
            print(f"Using user-defined threshold: {final_threshold}%")
        except ValueError:
            print(f"Invalid input '{user_input}'. Using default threshold: {default_threshold}%")
    
    print("-" * 34) 

    try:
        prepare_data_for_modeling(
            SOURCE_DATA_FILE, 
            CLEAN_OUTPUT_FILE, 
            REPORT_FILE, 
            PLOT_FILE,
            START_DATE, 
            END_DATE,
            EXPECTED_HOURS,
            final_threshold 
        )
    except ImportError:
        print("\n--- ERROR ---")
        print("Module 'matplotlib' or 'pandas' not found.")
        print("Please install them by running (inside your venv): python3 -m pip install matplotlib pandas")
        print("Then, re-run this script.")