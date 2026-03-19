import pandas as pd
import numpy as np


INPUT_FILE = "pysindy_ratio_2025.csv"  
OUTPUT_FILE = "lv_dataset.csv" 



# Load data, format date
df = pd.read_csv(INPUT_FILE)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
else:
    col_0 = df.columns[0]
    df[col_0] = pd.to_datetime(df[col_0])
    df = df.sort_values(col_0)
    df = df.rename(columns={col_0: 'date'})

# Extract
df['Predator_y'] = df['high_user_ratio_smooth']
df['Prey_x'] = 1.0 - df['Predator_y']

# Context
def context(date):
    m = date.month
    is_weekend = (date.dayofweek >= 5)
    
    # Winter (Nov, Dec, Jan, Feb)
    if m in [11, 12, 1, 2]:
        return "Winter WE" if is_weekend else "Winter WD"
    # Summer (May, Jun, Jul, Aug)
    elif m in [5, 6, 7, 8]:
        return "Summer WE" if is_weekend else "Summer WD"
    else:
        return "Transition"

df['Context'] = df['date'].apply(context)

df.to_csv(OUTPUT_FILE, index=False)
