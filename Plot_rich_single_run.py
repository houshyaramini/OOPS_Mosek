import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()
from scipy.stats import gmean, kurtosis, skew
from modules.Plot2 import load_and_prep_data,analyze_single_run 
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option('display.max_rows', 500)
delta_df = pd.read_csv("data/Delta.csv", parse_dates=["Periode"])
filepath = 'results/simulation_log.csv'
filepath_returns = 'data/RealeReturns_final.csv' 
filepath_rates = 'data/DSG1MO_fred.csv' 
filepath_spx="data/SpxDaten.csv"
filepath_vix = "data/VIX.csv"
df_vix = pd.read_csv(filepath_vix, parse_dates=["Date"], index_col="Date")

df, df_returns, w_cols, l_cols, s_cols , sp500_df= load_and_prep_data(filepath, filepath_returns, filepath_rates,filepath_spx)


unique_runs = sorted(df['RunID'].unique())
last_10_runs = unique_runs[-1] #[-1:-7:-1]
last_run = True
id = unique_runs[-1]

def auswertung(last_run: int = -1):
        df_vix = pd.read_csv(filepath_vix, parse_dates=["Date"], index_col="Date")
        test = df[df['RunID'] == id].copy()
        test= test.dropna(axis=0)
        test.loc[:,'RiskFreeRate'] = test.loc[:,'RiskFreeRate'] / 12
        test['zinsen'] = test['Cash_Weight'] * test['RiskFreeRate']
        test = test.set_index('Periode')
        test_series = test.index.to_series()
        N = len(test_series) 
        Start_vec = test_series[:N]
        Mat_vix = df_vix["Close"].pct_change().reindex(Start_vec, method="ffill")
        w_df = test.filter(like="W_")
        ret_df = (df_returns.set_index("Periode").filter(like="W_"))
        port_df = ret_df * w_df
        port_df["zinsen"] = test['zinsen']
        port_df["ATM_Put"] = w_df["W_Long_P1"] - w_df["W_Short_P1"]
        port_df["ATM_Call"] = w_df["W_Long_C1"] - w_df["W_Short_C1"]  
        port_df["OTM_Put"] = w_df["W_Long_P2"] - w_df["W_Short_P2"]
        port_df["OTM_Call"] = w_df["W_Long_C2"] - w_df["W_Short_C2"]
        port_df["Excess_ret"] = port_df[w_cols].sum(axis=1)
        port_df['TotalRet'] = (port_df["zinsen"] + port_df["Excess_ret"])
        port_df["Wealth"] = 100 * (port_df['TotalRet'] +1).cumprod(axis=0)
        port_df["ExcessWealth"] = 100 * (port_df["Excess_ret"] +1).cumprod(axis=0)
        op = port_df.iloc[:,-8:]
        op['VIX'] = Mat_vix
        op['Peak'] = op['Wealth'].cummax()
        op['Drawdown'] = (op['Wealth'] - op['Peak']) / op['Peak']
        op = op.dropna(axis=0)
        sharpe = (op['Excess_ret'].mean() / op['Excess_ret'].std()) * np.sqrt(12)
        total_growth = (op['Excess_ret'] + 1).prod() 
        n_months = len(op)
        n_years = n_months / 12
        cagr_variante_2 = (total_growth) ** (1 / n_years) - 1
        asset_corr = op.iloc[:,:4].corr()
        max_drawdown = op['Drawdown'].min()
        ret_kurt = kurtosis(op['Excess_ret'], fisher=True)
        ret_skew = skew(op['Excess_ret'])
        ret_vix_corr = op['Excess_ret'].corr(op['VIX'])
        return op , sharpe, cagr_variante_2, asset_corr, max_drawdown,ret_kurt, ret_skew,ret_vix_corr
   

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

console = Console()

# --- FARBKONFIGURATION (HIER ÄNDERN) ---

# THEMA 1: "Modern Purple" (Lila / Türkis / Weiß) - Sehr elegant
c_border = "purple"          # Rahmenfarbe
c_header = "bold plum1"      # Spaltenüberschriften
c_title  = "bold cyan"       # Haupttitel
c_label  = "orchid1"         # Beschriftungen (links)
c_val_Hi = "green"           # Gute Werte (z.B. CAGR)
c_val_Lo = "red"             # Schlechte Werte (z.B. Drawdown)
c_text   = "white"           # Standard Text




# --- HILFSFUNKTION: Pandas DataFrame -> Rich Table ---
def df_to_rich_table(df, title, header_style="bold white", border_style="blue"):
    """Wandelt einen Pandas DataFrame in eine Rich Table um."""
    df_temp = df.reset_index()
    
    table = Table(
        title=title, 
        box=box.ROUNDED, 
        header_style=header_style, 
        border_style=border_style,
        title_style=c_title,
        expand=True
    )
    
    for col in df_temp.columns:
        # Index-Spalte bekommt die Label-Farbe
        style = c_label if col == df_temp.columns[0] else c_text
        justify = "left" if col == df_temp.columns[0] else "right"
        table.add_column(str(col), style=style, justify=justify)

    for row in df_temp.values:
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.3f}")
            else:
                formatted_row.append(str(val))
        table.add_row(*formatted_row)
        
    return table


if last_run == True:
    result, sharpe, cagr2, options_corr, max_draw, excess_ret_kurtosis, excess_ret_skew, ret_vix_cor = auswertung(last_run=id)
    
    # 1. Header (Trenner)
    console.print("\n")
    # Benutzt die Rahmenfarbe für die Linie und Titel-Farbe für den Text
    console.rule(f"[{c_title}]ANALYSE RUN ID: {id}[/]", style=c_border)
    console.print("\n")

    # 2. Performance Kennzahlen Grid
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    
    # Linke Seite: Hauptmetriken
    t_main = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_main.add_column("Label", style=c_label)
    t_main.add_column("Value", justify="right", style="bold white")
    
    t_main.add_row("CAGR", f"[{c_val_Hi}]{cagr2:.2%}[/]") 
    t_main.add_row("Sharpe Ratio", f"[{c_text}]{sharpe:.2f}[/]")
    t_main.add_row("Max Drawdown", f"[{c_val_Lo}]{max_draw:.2%}[/]")
    t_main.add_row("Wealth Ende", f"[{c_text}]{result['Wealth'].iloc[-1]:.2f}[/]")

    # Rechte Seite: Statistik
    t_stats = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t_stats.add_column("Label", style=c_label)
    t_stats.add_column("Value", justify="right", style=c_text)
    
    t_stats.add_row("Kurtosis", f"{excess_ret_kurtosis:.2f}")
    t_stats.add_row("Skew", f"{excess_ret_skew:.2f}")
    t_stats.add_row("Corr (Ret/VIX)", f"{ret_vix_cor:.2f}")
    
    # Layout zusammenfügen
    metrics_layout = Table.grid(padding=2)
    metrics_layout.add_column()
    metrics_layout.add_column()
    metrics_layout.add_row(t_main, t_stats)
    
    console.print(Panel(
        metrics_layout, 
        title=f"[{c_title}]Performance Übersicht[/]", 
        border_style=c_border,
        expand=False
    ))
    console.print("\n")

    # 3. Korrelationsmatrix
    # border_style="dim "+c_border macht den Rahmen etwas dunkler/dezenter
    table_corr = df_to_rich_table(
        options_corr, 
        "Asset Korrelationen", 
        header_style=c_header, 
        border_style="dim " + c_border
    )
    console.print(table_corr)
    console.print("\n")

    # 4. Statistische Zusammenfassung
    table_desc = df_to_rich_table(
        result.describe().T, 
        "Statistische Details", 
        header_style=c_header, 
        border_style="dim " + c_border
    )
    console.print(table_desc)