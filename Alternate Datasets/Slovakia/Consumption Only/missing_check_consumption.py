import pandas as pd

INTERVAL_FILE = "consumption_readings.csv"
REPORT_NULLS  = "missing_values_report_consum.csv"
REPORT_GAPS   = "missing_timestamp_gaps_consum.csv"

interval_df = pd.read_csv(INTERVAL_FILE, parse_dates=['timestamp'])

# Nulls
nulls = interval_df[interval_df['consumption'].isnull()]
if not nulls.empty:
    out = nulls[['meterID', 'timestamp']].copy()
    out['missing_column'] = 'consumption'
    out.to_csv(REPORT_NULLS, index=False)

# Timestamp gaps
all_gaps = []
for mid, grp in interval_df.groupby('meterID'):
    ts = pd.DatetimeIndex(grp['timestamp']).sort_values()
    expected = pd.date_range(ts.min(), ts.max(), freq='15T')
    missing  = expected.difference(ts)

    if missing.empty:
        continue

    ts_s   = missing.to_series()
    groups = (ts_s.diff() != pd.Timedelta('15 minutes')).cumsum()
    for _, block in ts_s.groupby(groups):
        all_gaps.append({
            'meterID':           mid,
            'gap_start':         block.min(),
            'gap_end':           block.max(),
            'missing_intervals': len(block),
            'duration_hours':    round((len(block) * 15) / 60, 2)
        })

if all_gaps:
    gaps_df = pd.DataFrame(all_gaps).sort_values(['meterID', 'gap_start'])
    gaps_df.to_csv(REPORT_GAPS, index=False)

# Summary
n_meters      = interval_df['meterID'].nunique()
meters_nulls  = nulls['meterID'].nunique() if not nulls.empty else 0
meters_gaps   = gaps_df['meterID'].nunique() if all_gaps else 0
total_missing = gaps_df['missing_intervals'].sum() if all_gaps else 0

print(f"Total Meters:            {n_meters}")
print(f"Meters with Nulls:       {meters_nulls}")
print(f"Meters with Gaps:        {meters_gaps}")
print(f"Total Missing Intervals: {total_missing}")