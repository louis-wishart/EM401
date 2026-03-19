import glob
import pandas as pd
import json
from datetime import datetime, timedelta

DATA_PATH = "/Users/louis.wishart/Library/CloudStorage/OneDrive-UniversityofStrathclyde/EM401/Code/data set"
OUT_FILE  = "consumption_readings.csv"

json_files = glob.glob(f"{DATA_PATH}/*.json")
print(f"{len(json_files)} JSON files found")

rows = []

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

        base_ts = datetime.combine(date, datetime.min.time())
        cons    = r.get('consumption', [])

        for i, val in enumerate(cons):
            rows.append({
                'meterID':     r['meterID'],
                'timestamp':   base_ts + timedelta(minutes=i*15),
                'consumption': val
            })

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print(f"Saved {OUT_FILE}")