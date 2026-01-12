import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

def get_precise_data():
    # 1. Tägliche Daten laden (WICHTIG: Das Paper nutzt tägliche Daten für Volatilität)
    # Wir laden S&P 500 (^GSPC)
    print("Lade tägliche Daten für S&P 500...")
    df_daily = yf.download("^GSPC", start="1950-01-01", end="2020-01-31", auto_adjust=False)
    
    # 2. Tägliche Log-Returns berechnen
    # r_{t,d}
    df_daily['Log_Ret'] = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
    df_daily = df_daily.dropna()
    
    # 3. Aggregation auf Monatsebene
    # Wir brauchen zwei Dinge pro Monat:
    # A) Die Summe der Returns (für den Monats-Return r_t)
    # B) Die Realized Volatility (Wurzel aus Summe der quadrierten täglichen Returns)
    
    monthly_stats = df_daily['Log_Ret'].resample('ME').agg(
        Monthly_Return='sum', 
        Squared_Sum=lambda x: np.sum(x**2)
    )
    
    # Realized Volatility rv_t = sqrt(sum(r_{t,d}^2))
    monthly_stats['Realized_Vol'] = np.sqrt(monthly_stats['Squared_Sum'])
    
    # 4. Berechnung der "Standardized Returns" nach Formel (9) im Paper
    # x_t = r_t / rv_{t-1}
    # Wir shiften die Volatilität um 1, um durch den Vormonat zu teilen
    monthly_stats['Prev_Realized_Vol'] = monthly_stats['Realized_Vol'].shift(1)
    monthly_stats['Std_Return'] = monthly_stats['Monthly_Return'] / monthly_stats['Prev_Realized_Vol']
    
    # Erste Zeile entfernen (da Vormonats-Volatilität fehlt)
    monthly_stats = monthly_stats.dropna()
    print(monthly_stats.info())
    return monthly_stats

def calculate_table1_stats(series):
    """
    Berechnet die Statistiken exakt wie in Table 1 definiert.
    """
    # N
    n_obs = len(series)
    
    # Skewness & Excess Kurtosis (Fisher)
    sk = skew(series)
    ku = kurtosis(series) # Fisher per default (Normal = 0)
    
    # Autokorrelationen (Lag 1) für Returns z
    acf_z = acf(series, nlags=1, fft=False)
    rho1_z = acf_z[1] if len(acf_z) > 1 else np.nan
    
    # Autokorrelationen (Lag 1) für quadrierte Returns z^2
    acf_z2 = acf(series**2, nlags=1, fft=False)
    rho1_z2 = acf_z2[1] if len(acf_z2) > 1 else np.nan
    
    # Ljung-Box Test Q(1)
    # Test auf Autokorrelation im Return
    lb_test = acorr_ljungbox(series, lags=[1], return_df=True)
    q1_stat = lb_test.iloc[0]['lb_stat']
    q1_pval = lb_test.iloc[0]['lb_pvalue']
    
    # ARCH(1) Test (Engle's LM Test)
    # Testet auf ARCH-Effekte (Volatilitätscluster)
    lm_stat, lm_pval, _, _ = het_arch(series, ddof=1, nlags=1)
    
    return {
        "Obs": n_obs,
        "Skew": round(sk, 2),
        "Exc Kurt": round(ku, 2),
        "rho1(z)": round(rho1_z, 2),
        "rho1(z^2)": round(rho1_z2, 2),
        "Q1(z) Stat": round(q1_stat, 2),
        "Q1(z) p-val": round(q1_pval, 2),
        #"ARCH(1) Stat": round(lm_stat, 2),
        #"ARCH(1) p-val": round(lm_pval, 2)
    }

# --- HAUPTPROGRAMM ---

df = get_precise_data()

# Zeiträume definieren (Paper nutzt 1950-1995 und 1996-2008/2013)
periods = {
    "1950-1995": df.loc["1950":"1995"],
    "1996-2020": df.loc["1996":"2020"], # Dein Bild ging bis 2013
    "1950-2020": df.loc["1950":"2020"]
}

results_raw = {}
results_std = {}

for period_name, data_slice in periods.items():
    # Raw Returns Statistiken
    results_raw[period_name] = calculate_table1_stats(data_slice['Monthly_Return'])
    
    # Standardized Returns Statistiken (nach Paper Methodik)
    results_std[period_name] = calculate_table1_stats(data_slice['Std_Return'])

# --- AUSGABE ---
print("\n=== RAW RETURNS (Table 1 Panel Left) ===")
print(pd.DataFrame(results_raw))

print("\n=== STANDARDIZED RETURNS (Table 1 Panel Right) ===")
print(pd.DataFrame(results_std))