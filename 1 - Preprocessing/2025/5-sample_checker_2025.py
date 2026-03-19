import pandas as pd
import os

INPUT_FILE = "200_2025.parquet"

df = pd.read_parquet(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])
df['date'] = df['timestamp'].dt.date

# Date Range 
min_date = df['timestamp'].min()
max_date = df['timestamp'].max()
days_span = (max_date - min_date).days + 1  #inclusive 

# Theoretical Max
expected_vectors = 200 * days_span

# Feeder Count
daily_counts = df.groupby(['lv_feeder_unique_id', 'date']).size()
actual_vectors = len(daily_counts)

# Quality
perfect_vectors = len(daily_counts[daily_counts == 48])
precision_pct = (perfect_vectors / expected_vectors) * 100

print(f"""
   Start: {min_date.date()}
   End:   {max_date.date()}
   Span:  {days_span} Days

   Expected Readings: {expected_vectors:,}
   Actual:       {actual_vectors:,}
   Missing:       {expected_vectors - actual_vectors:,}

   Complete Days(48): {perfect_vectors:,} [{precision_pct:.1f}%]
  
""")