import pandas as pd
import sys
import numpy as np


INTERVAL_FILE = "/Users/louis.wishart/Library/CloudStorage/OneDrive-UniversityofStrathclyde/EM401/Code/EM401/Consumption Only/consumption_readings.csv"


REPORT_DAYS = "missing_days_report_consum.csv"
REPORT_NULLS = "missing_values_report_consum.csv"
REPORT_GAPS = "missing_timestamp_gaps_consum.csv"
REPORT_SUMMARY = "missing_data_summary_consum.csv" 


def missing():

    try:
        interval_df = pd.read_csv(INTERVAL_FILE, parse_dates=['timestamp'])
    except FileNotFoundError:
        print(f"'{INTERVAL_FILE}' not found")
        sys.exit()
    except Exception as e:
        print(f"error {INTERVAL_FILE}: {e}")
        sys.exit()

# Missing Days 
    missing_days_report_list = []
    
    daily_meter_data = interval_df[['meterID', 'timestamp']].copy()
    daily_meter_data['date'] = daily_meter_data['timestamp'].dt.date
    unique_days_per_meter = daily_meter_data.drop_duplicates(subset=['meterID', 'date'])
    
    min_date = unique_days_per_meter['date'].min()
    max_date = unique_days_per_meter['date'].max()
    expected_days_index = pd.date_range(min_date, max_date, freq='D')
    
    all_meter_ids = interval_df['meterID'].unique()
    
    for meter_id in all_meter_ids:

        actual_days = pd.DatetimeIndex(
            unique_days_per_meter[unique_days_per_meter['meterID'] == meter_id]['date']
        )
        missing_days = expected_days_index.difference(actual_days)
        
        if not missing_days.empty:
            print(f"Meter {meter_id} is missing {len(missing_days)} days")
            for day in missing_days:
                missing_days_report_list.append({
                    'meterID': meter_id,
                    'missing_date': day.date()
                })

    if not missing_days_report_list:
        print("No missing days")
        daily_report_df = pd.DataFrame(columns=['meterID', 'missing_date']) 
    else:
        daily_report_df = pd.DataFrame(missing_days_report_list)
        daily_report_df.to_csv(REPORT_DAYS, index=False)
        print(f" Missing days saved to {REPORT_DAYS}")
        
    
    # Missing Values
    null_data = interval_df[interval_df['consumption'].isnull()]
    
    if null_data.empty:
        print("No null values found")
        null_report = pd.DataFrame(columns=['meterID', 'timestamp', 'missing_column']) 
    else:
        print(f"Found {len(null_data)} rows with null values")
        null_report = null_data[['meterID', 'timestamp']].copy()
        null_report['missing_column'] = 'consumption'
        null_report.to_csv(REPORT_NULLS, index=False)
        print(f"Null values saved to {REPORT_NULLS}")
        
    # Missing Timestamps

    all_gaps_list = []
    total_meters = len(all_meter_ids)

   
    for i, meter_id in enumerate(all_meter_ids):
        print(f"Processing meter {i+1} of {total_meters} (ID: {meter_id})...")
        
        meter_df = interval_df[interval_df['meterID'] == meter_id]
        actual_timestamps = pd.DatetimeIndex(meter_df['timestamp']).sort_values()
        
        if actual_timestamps.empty:
            print(f"Empty, Meter {meter_id}")
            continue
            
        start_date = actual_timestamps.min()
        end_date = actual_timestamps.max()
        expected_range = pd.date_range(start=start_date, end=end_date, freq='15T')
        
        missing_timestamps = expected_range.difference(actual_timestamps)
        
        if missing_timestamps.empty:
            continue

        ts_series = missing_timestamps.to_series()
        ts_diff = ts_series.diff()
        gap_groups = (ts_diff != pd.Timedelta('15 minutes')).cumsum()

        for group_id, timestamps in ts_series.groupby(gap_groups):
            all_gaps_list.append({
                'meterID': meter_id,
                'gap_start': timestamps.min(),
                'gap_end': timestamps.max(),
                'missing_intervals': len(timestamps),
                'duration_hours': round((len(timestamps) * 15) / 60, 2)
            })
    
    if not all_gaps_list:
        print("No missing timestamps for any meter")
        gaps_report_df = pd.DataFrame(columns=['meterID', 'missing_intervals']) 
    else:
        print(f"\n Found {len(all_gaps_list)} total timestamp gaps")
        gaps_report_df = pd.DataFrame(all_gaps_list)
        gaps_report_df = gaps_report_df.sort_values(by=['meterID', 'gap_start'])
        gaps_report_df.to_csv(REPORT_GAPS, index=False)
        print(f"Missing timestamps saved to {REPORT_GAPS}")

    # Summary Report
    
    summary_df = pd.DataFrame(all_meter_ids, columns=['meterID'])
    
    days_summary = daily_report_df.groupby('meterID').size().reset_index(name='total_missing_days')
    nulls_summary = null_report.groupby('meterID').size().reset_index(name='total_null_values')
    gaps_summary = gaps_report_df.groupby('meterID')['missing_intervals'].sum().reset_index(name='total_missing_intervals')
    
    summary_df = pd.merge(summary_df, days_summary, on='meterID', how='left')
    summary_df = pd.merge(summary_df, nulls_summary, on='meterID', how='left')
    summary_df = pd.merge(summary_df, gaps_summary, on='meterID', how='left')
    
    summary_df = summary_df.fillna(0).astype({
        'total_missing_days': int,
        'total_null_values': int,
        'total_missing_intervals': int
    })
    
    
    summary_df.to_csv(REPORT_SUMMARY, index=False)
    print(f"Summary saved to {REPORT_SUMMARY}")
    
    # Printed Summary
  
    print(f"Total Meters:           {len(summary_df)}")
    print(f"Meters with Gaps:     {len(gaps_summary)}")
    print(f"Meters with Nulls:    {len(nulls_summary)}")
    print(f"Meters with Missing Days: {len(days_summary)}")
    print(f"Total Missing Intervals:  {summary_df['total_missing_intervals'].sum()}")


if __name__ == "__main__":
    missing()