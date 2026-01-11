import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import gmean, kurtosis, skew
from modules.Plot2 import load_and_prep_data, auswertung


SINGLE_MODE = False
SINGLE_INDEX = -1
BATCH_SIZE = 7


pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 500)

filepath = "results/simulation_log.csv"
filepath_returns = "data/RealeReturns_final.csv"
filepath_rates = "data/DSG1MO_fred.csv"
filepath_spx = "data/SpxDaten.csv"
filepath_vix = "data/VIX.csv"
df_vix = pd.read_csv(filepath_vix, parse_dates=["Date"], index_col="Date")

df, df_returns, w_cols, l_cols, s_cols, sp500_df = load_and_prep_data(
    filepath, filepath_returns, filepath_rates, filepath_spx
)


unique_runs = sorted(df["RunID"].unique())


if SINGLE_MODE:
    try:
        target_run = unique_runs[SINGLE_INDEX]
        runs_to_analyze = [target_run]
        print(f"MODUS: Einzel-Analyse für RunID: {target_run}")
    except IndexError:
        print(
            f"FEHLER: Der Index {SINGLE_INDEX} existiert nicht. Es gibt {len(unique_runs)} Runs."
        )
        exit()
else:

    runs_to_analyze = unique_runs[::-1][:BATCH_SIZE]
    print(f"MODUS: Batch-Vergleich der letzten {len(runs_to_analyze)} Runs.")

metrics_data = {}
for run_id in runs_to_analyze:
    run_label = str(run_id)

    result, sharpe, cagr, options_corr, max_draw, kurt, skew_val, vix_corr = auswertung(
        run_id=run_id, df=df, df_returns=df_returns, df_vix=df_vix
    )

    if result is not None:
        metrics_data[run_label] = {
            "CAGR": f"{cagr:.2%}",
            "Sharpe": f"{sharpe:.2f}",
            "MaxDD": f"{max_draw:.2%}",
            "Kurtosis": f"{kurt:.2f}",
            "Skew": f"{skew_val:.2f}",
            "VIX-Corr": f"{vix_corr:.2f}",
        }

df_metrics = pd.DataFrame(metrics_data)

print("\n" + "=" * 80)
if SINGLE_MODE:
    print(f"PERFORMANCE-KENNZAHLEN: {runs_to_analyze[0]}")
else:
    print("VERGLEICH DER PERFORMANCE-KENNZAHLEN (Mehrere Runs)")
print("=" * 80)
print(df_metrics)
