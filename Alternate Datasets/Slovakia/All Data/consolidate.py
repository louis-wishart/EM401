import glob
import pandas as pd
import json
from datetime import datetime, timedelta

DATA_PATH   = "/Users/louis.wishart/Library/CloudStorage/OneDrive-UniversityofStrathclyde/EM401/Code/slovakia"
METER_INFO  = "meter_info.csv"
OUT_DAILY   = "daily_summary_all_meters.csv"
OUT_INTERVAL = "interval_readings_all_meters.csv"

# Load meter info
meter_df = pd.read_csv(f"{DATA_PATH}/{METER_INFO}")
meter_df['meterID'] = meter_df['meterID'].astype(str)

json_files = glob.glob(f"{DATA_PATH}/*.json")
print(f"{len(json_files)} JSON files found")

daily_rows    = []
interval_rows = []

for path in json_files:
    with open(path, 'r') as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping: {path}")
            continue

    for r in records:
        try:
            date = datetime(r['year'], r['month'], r['day']).date()
        except (ValueError, TypeError):
            continue

        daily_rows.append({
            'meterID':                 r['meterID'],
            'date':                    date,
            'lowConsumptionSum':       r.get('lowConsumptionSum'),
            'highConsumptionSum':      r.get('highConsumptionSum'),
            'maxConsumption':          r.get('maxConsumption'),
            'laggingReactivePowerSum': r.get('laggingReactivePowerSum'),
            'leadingReactivePowerSum': r.get('leadingReactivePowerSum')
        })

        base_ts  = datetime.combine(date, datetime.min.time())
        cons     = r.get('consumption', [])
        lagging  = r.get('laggingReactivePower', [])
        leading  = r.get('leadingReactivePower', [])

        for i in range(len(cons)):
            interval_rows.append({
                'meterID':              r['meterID'],
                'timestamp':            base_ts + timedelta(minutes=i*15),
                'consumption':          cons[i],
                'laggingReactivePower': lagging[i] if i < len(lagging) else None,
                'leadingReactivePower': leading[i] if i < len(leading) else None
            })

# Daily
daily_df = pd.DataFrame(daily_rows)
daily_df['meterID'] = daily_df['meterID'].astype(str)
daily_df = pd.merge(daily_df, meter_df, on='meterID', how='left')
daily_df.to_csv(OUT_DAILY, index=False)
print(f"Saved {OUT_DAILY}")

# Interval
pd.DataFrame(interval_rows).to_csv(OUT_INTERVAL, index=False)
print(f"Saved {OUT_INTERVAL}")