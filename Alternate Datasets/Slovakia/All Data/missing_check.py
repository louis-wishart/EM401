import pandas as pd

DAILY_FILE    = "daily_summary.csv"
INTERVAL_FILE = "interval_readings.csv"
REPORT_DAYS   = "missing_days_report.csv"
REPORT_NULLS  = "missing_values_report.csv"
REPORT_GAPS   = "missing_timestamp_gaps.csv"

# Load
interval_df = pd.read_csv(INTERVAL_FILE, parse_dates=['timestamp'])

# Check 1 - Missing days
daily_df  = pd.read_csv(DAILY_FILE, parse_dates=['date'])
min_date  = daily_df['date'].min()
max_date  = daily_df['date'].max()
full_range = pd.date_range(min_date, max_date, freq='D')

missing_days = []
for mid, grp in daily_df.groupby('meterID'):
    gaps = full_range.difference(pd.DatetimeIndex(grp['date']))
    for d in gaps:
        missing_days.append({'meterID': mid, 'missing_date': d.date()})

if missing_days:
    pd.DataFrame(missing_days).to_csv(REPORT_DAYS, index=False)
    print(f"{len(missing_days)} missing days -> {REPORT_DAYS}")

# Check 2 - Null values
nulls = interval_df[interval_df['consumption'].isnull()]
if not nulls.empty:
    out = nulls[['meterID', 'timestamp']].copy()
    out['missing_column'] = 'consumption'
    out.to_csv(REPORT_NULLS, index=False)
    print(f"{len(nulls)} null rows -> {REPORT_NULLS}")

# Check 3 - Timestamp gaps
all_gaps = []
for mid, grp in interval_df.groupby('meterID'):
    ts = pd.DatetimeIndex(grp['timestamp']).sort_values()
    expected = pd.date_range(ts.min(), ts.max(), freq='15T')
    missing  = expected.difference(ts)

    if missing.empty:
        continue

    ts_s    = missing.to_series()
    groups  = (ts_s.diff() != pd.Timedelta('15 minutes')).cumsum()
    for _, block in ts_s.groupby(groups):
        all_gaps.append({
            'meterID':          mid,
            'gap_start':        block.min(),
            'gap_end':          block.max(),
            'missing_intervals': len(block),
            'duration_hours':   round((len(block) * 15) / 60, 2)
        })

if all_gaps:
    gaps_df = pd.DataFrame(all_gaps).sort_values(['meterID', 'gap_start'])
    gaps_df.to_csv(REPORT_GAPS, index=False)
    print(f"{len(all_gaps)} gaps -> {REPORT_GAPS}")