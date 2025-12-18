import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from modules.Plot2 import load_and_prep_data,analyze_single_run , get_option_contributions, plot_contributions
delta_df = pd.read_csv("data/Delta.csv", parse_dates=["Periode"])
filepath = 'results/simulation_log.csv'
filepath_returns = 'data/RealeReturns.csv' 
filepath_rates = 'data/DSG1MO_fred.csv' 
filepath_spx="data/SpxDaten.csv"

df, df_returns, w_cols, l_cols, s_cols , sp500_df= load_and_prep_data(filepath, filepath_returns, filepath_rates,filepath_spx)


unique_runs = sorted(df['RunID'].unique())

#print(f"Gefundene Runs in der Datei: {len(unique_runs)}")
for i, r in enumerate(unique_runs):
    run_exposure = df[df['RunID'] == r]
    is_long_only = run_exposure['Short_Exposure'].sum() == 0
    type_str = "LONG ONLY" if is_long_only else "LONG & SHORT"
    marker = "  <-- AKTUELLSTER" if i == len(unique_runs) - 1 else ""
    #print(f"[{i}] ID: {r}  ({type_str}){marker}")
if len(unique_runs) > 0:
    selected_run_id = unique_runs[-1] 
    
    print(f"\nStarte Analyse für den aktuellsten Run: {selected_run_id}")
    analyze_single_run(df, df_returns, selected_run_id, w_cols, sp500_df)




