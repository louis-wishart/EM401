import pandas as pd
import numpy as np
import pysindy as ps
import warnings
import json
warnings.filterwarnings("ignore")

INPUT_FILE = "lv_dataset.csv"
OUTPUT_JSON = "coefficients.json"
CONTEXTS = ["Winter WD", "Winter WE", "Summer WD", "Summer WE"]

# Load and Format Data
df = pd.read_csv(INPUT_FILE)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

# PySindy Configuration
# include_bias=False removes C, interaction_only=True removes x^2 and y^2
poly_library = ps.PolynomialLibrary(
    degree=2, 
    include_bias=False, 
    interaction_only=True
)
optimizer = ps.STLSQ(threshold=0.005)

# Initialise 
model = ps.SINDy(
    feature_library=poly_library,
    optimizer=optimizer
)

# Derivatives 
data_buckets = {ctx: {"x": [], "dx": []} for ctx in CONTEXTS}

for i in range(len(df) - 1):
    row_today = df.iloc[i]
    row_tomorrow = df.iloc[i+1]
    
    # Continuity Check 
    days_diff = (row_tomorrow['date'] - row_today['date']).days
    if days_diff != 1:
        continue
        
    # Context Match 
    current_context = row_today['Context']
    if current_context != row_tomorrow['Context']: 
        continue
    
    # Skip bad transitions 
    if current_context not in CONTEXTS:
        continue
        
    # Calc States 
    x_today = [row_today['Prey_x'], row_today['Predator_y']]
    x_tomorrow = [row_tomorrow['Prey_x'], row_tomorrow['Predator_y']]
    
    # dx/dt = State(Tomorrow) - State(Today)
    dx = [x_tomorrow[0] - x_today[0], x_tomorrow[1] - x_today[1]]
    
    # Store
    data_buckets[current_context]["x"].append(x_today)
    data_buckets[current_context]["dx"].append(dx)

# Extraction
equations_dict = {}

for context in CONTEXTS:
    X_train = np.array(data_buckets[context]["x"])
    X_dot_train = np.array(data_buckets[context]["dx"])
    
    # Calculate total possible days for this context in the raw dataframe
    possible_days = (df['Context'] == context).sum()
    
    if len(X_train) < 5:
        print(f"[{context.upper()}] Insufficient Data ({len(X_train)} / {possible_days} possible days)")
        continue
        
    # Fit Sindy Model
    model.fit(X_train, t=1.0, x_dot=X_dot_train)
    
    
    print(f"[{context.upper()}] LV Equations (Based on {len(X_train)} / {possible_days} days):")

    eqs = model.equations()
    eq_x = eqs[0].replace('x0', 'x').replace('x1', 'y')
    eq_y = eqs[1].replace('x0', 'x').replace('x1', 'y')
    
    print(f"x' =  {eq_x}")
    print(f"y' =  {eq_y}")
    print()
    print()
    
    # Store coefficients
    coeffs = model.coefficients()
    equations_dict[context.upper()] = {
        "A": float(coeffs[0, 0]), 
        "B": float(coeffs[0, 1]), 
        "C": float(coeffs[0, 2])
    }


with open(OUTPUT_JSON, 'w') as f:
    json.dump(equations_dict, f, indent=4)