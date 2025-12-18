

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def load_and_prep_data(filepath, filepath_ret, filepath_rates, filepath_spx):
    """
    Lädt die Simulations-Daten, Returns und Zinsraten.
    Merged die Zinsraten basierend auf dem Datum in den DataFrame.
    """
    df = pd.read_csv(filepath)
    df_ret = pd.read_csv(filepath_ret)
    df_rates = pd.read_csv(filepath_rates)
    sp500_df = pd.read_csv(filepath_spx)
    df['Periode'] = pd.to_datetime(df['Periode'])
    df_ret['Periode'] = pd.to_datetime(df_ret['Periode'])
    df_rates['Date'] = pd.to_datetime(df_rates['Date'])
    df = df.sort_values('Periode')
    df_rates = df_rates.sort_values('Date')
    try:
        sp500_df = pd.read_csv(filepath_spx)
        sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
    except FileNotFoundError:
        print(f"Warnung: {filepath_spx} nicht gefunden. S&P Vergleich wird übersprungen.")
        sp500_df = pd.DataFrame(columns=['Date', 'Close'])
    df = pd.merge_asof(df, df_rates, left_on='Periode', right_on='Date', direction='backward')
    df.rename(columns={'DGS1MO': 'RiskFreeRate'}, inplace=True)
    
    weight_cols = [c for c in df.columns if c.startswith('W_')]
    
    df['Total_Exposure'] = df[weight_cols].sum(axis=1)
    
    long_cols = [c for c in weight_cols if 'Long' in c]
    short_cols = [c for c in weight_cols if 'Short' in c]
    
    df['Long_Exposure'] = df[long_cols].sum(axis=1)
    df['Short_Exposure'] = df[short_cols].sum(axis=1)
    df['Cash_Weight'] = (1.0 - df['Long_Exposure'] + df['Short_Exposure'] ).clip(lower=0.0)
    
    return df, df_ret, weight_cols, long_cols, short_cols, sp500_df

def analyze_single_run(df, df_ret, run_id, weight_cols, sp500_df):
    """
    Erstellt Statistiken, Performance-Berechnung (inkl. Zinsen) und Plots.
    Berechnet Annualisierte Returns, Volatilität und Sharpe Ratio.
    Vergleicht mit S&P 500 Benchmark.
    Zusätzlich: Portfolio-Delta aus Delta.csv (anstelle des Zins-Plots).
    """
    
    # 1. Daten filtern für diesen Run
    run_data = df[df['RunID'] == run_id].sort_values('Periode').copy()
    
    if run_data.empty:
        print(f"Warnung: Keine Daten für RunID {run_id} gefunden.")
        return

    # sicherstellen, dass Periode datetime ist
    run_data['Periode'] = pd.to_datetime(run_data['Periode'])

    # 2. Merge mit Returns-Daten
    merged_data = pd.merge(
        run_data,
        df_ret,
        on='Periode',
        how='left',
        suffixes=('', '_ret')
    )

    # ---------------------------------------------------------
    # S&P 500 MERGE & VORBEREITUNG
    # ---------------------------------------------------------
    if sp500_df is not None and not sp500_df.empty:
        # falls Date im Index von sp500_df steckt -> in Spalte umwandeln
        if 'Date' not in sp500_df.columns:
            sp500_tmp = sp500_df.copy()
            sp500_tmp = sp500_tmp.reset_index().rename(columns={'index': 'Date'})
        else:
            sp500_tmp = sp500_df.copy()

        merged_data = pd.merge(
            merged_data,
            sp500_tmp[['Date', 'Close']],
            left_on='Periode',
            right_on='Date',
            how='left'
        )

        # Lücken füllen (falls Simulationstag ein Feiertag war, nimm letzten Preis)
        merged_data['Close'] = merged_data['Close'].ffill()

        # Normierung auf Startwert 100
        start_price_sp = merged_data['Close'].iloc[0]
        merged_data['SP500_Indexed'] = (merged_data['Close'] / start_price_sp) * 100

        # Date-Spalte nach Merge nur löschen, wenn sie existiert
        if 'Date' in merged_data.columns:
            merged_data = merged_data.drop(columns=['Date'])
    else:
        merged_data['SP500_Indexed'] = np.nan

    # ---------------------------------------------------------
    # DELTA-DATEI LADEN & PORTFOLIO-DELTA BERECHNEN
    # ---------------------------------------------------------
    # Delta.csv mit Spalten: 'Periode' + W_-Spalten (wie weight_cols)
    delta_df = pd.read_csv("data/Delta.csv")
    delta_df['Periode'] = pd.to_datetime(delta_df['Periode'])

    # Delta-Spalten aus Datei (alle W_-Spalten)
    delta_cols = [c for c in delta_df.columns if c.startswith('W_')]
    delta_df = delta_df[['Periode'] + delta_cols]

    # Delta-Spalten umbenennen, damit sie nicht mit Gewichten kollidieren
    delta_rename = {c: c + "_delta" for c in delta_cols}
    delta_df = delta_df.rename(columns=delta_rename)

    # Delta an merged_data mergen
    merged_data = pd.merge(
        merged_data,
        delta_df,
        on='Periode',
        how='left'
    )

    # nur Spalten verwenden, die es sowohl als Gewicht als auch als Delta gibt
    common_weight_cols = [w for w in weight_cols if w + "_delta" in merged_data.columns]

    delta_arr = np.nan_to_num(
        merged_data[[w + "_delta" for w in common_weight_cols]].to_numpy(dtype=float),
        nan=0.0
    )
    weight_arr = np.nan_to_num(
        merged_data[common_weight_cols].to_numpy(dtype=float),
        nan=0.0
    )

    # dein Schema: (delta_arr * rr_arr).sum(axis=1)
    merged_data['Port_Delta'] = (delta_arr * weight_arr).sum(axis=1)

    # ---------------------------------------------------------
    # 3. Performance Berechnung Portfolio
    # ---------------------------------------------------------
    merged_data['Portfolio_Ret'] = 0.0
    
    # A) Beitrag aus Assets (Gewicht * Asset-Return)
    for col in weight_cols:
        # Bestimme den Namen der Return-Spalte
        if f"{col}_ret" in merged_data.columns:
            ret_col_name = f"{col}_ret"
        elif col in df_ret.columns:
            ret_col_name = col
        else:
            continue
            
        contribution = merged_data[col] * merged_data[ret_col_name].fillna(0)
        merged_data['Portfolio_Ret'] += contribution
    
    # B) Beitrag aus Zinsen auf Cash (Cash-Gewicht * Monatszins)
    merged_data['Monthly_RiskFree'] = merged_data['RiskFreeRate'].fillna(0) / 12
    merged_data['Interest_Ret'] = merged_data['Cash_Weight'] * merged_data['Monthly_RiskFree']
    merged_data['Portfolio_Ret'] += merged_data['Interest_Ret']

    # Wealth Index berechnen
    merged_data['Wealth_Index'] = 100 * (1 + merged_data['Portfolio_Ret']).cumprod()
    
    # ---------------------------------------------------------
    # KENNZAHLEN BERECHNUNG
    # ---------------------------------------------------------
    n_months = len(merged_data)
    
    # 1. Annualisierter Return (CAGR)
    total_growth = merged_data['Wealth_Index'].iloc[-1] / 100
    ann_ret = (total_growth ** (12 / n_months)) - 1
    
    # 2. Annualisierte Volatilität
    monthly_vol = merged_data['Portfolio_Ret'].std()
    ann_vol = monthly_vol * np.sqrt(12)
    
    # 3. Sharpe Ratio
    merged_data['Excess_Ret'] = merged_data['Portfolio_Ret'] - merged_data['Monthly_RiskFree']
    
    mean_excess = merged_data['Excess_Ret'].mean()
    std_excess = merged_data['Excess_Ret'].std()
    
    if std_excess > 0:
        sharpe_ratio = (mean_excess / std_excess) * np.sqrt(12)
    else:
        sharpe_ratio = 0.0

    # Statistiken berechnen
    total_return_pct = (merged_data['Wealth_Index'].iloc[-1] - 100)
    avg_rf_ann = merged_data['RiskFreeRate'].mean()
    
    print(f"\n{'='*40}")
    print(f"ANALYSE FÜR RUN: {run_id}")
    print(f"{'='*40}")
    
    print("\n--- Exposure & Cash Statistik (Ø) ---")
    print(f"Total Exposure:    {merged_data['Total_Exposure'].mean():.4f}")
    print(f"Cash Quote:        {merged_data['Cash_Weight'].mean():.4f}")
    print(f"Risk Free Rate (Ø):{avg_rf_ann*100:.2f}% p.a.")
    
    print("\n--- Performance Statistik ---")
    print(f"Endwert (Start 100): {merged_data['Wealth_Index'].iloc[-1]:.2f}")
    print(f"Total Return:        {total_return_pct:.2f}%")
    print(f"Ann. Return (CAGR):  {ann_ret*100:.2f}%")
    print(f"Ann. Volatilität:    {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio:        {sharpe_ratio:.2f}")

    # 4. Visualisierung
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    
    # Plot A: Allokation
    axes[0].stackplot(
        merged_data['Periode'], 
        [merged_data[c] for c in weight_cols], 
        labels=weight_cols, alpha=0.8
    )
    axes[0].set_title(f'Portfolio Allokation ({run_id} )', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Gewichtung')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    # Plot B: Long/Short Exposure
    axes[1].plot(
        merged_data['Periode'],
        merged_data['Long_Exposure'],
        label='Long Exposure',
        color='green'
    )
    axes[1].plot(
        merged_data['Periode'],
        -merged_data['Short_Exposure'],
        label='Short Exposure',
        color='red'
    )
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_title('Long vs. Short Exposure', fontsize=12)
    axes[1].legend()

    # Plot C: Portfolio-Delta (statt Zinsumfeld)
    axes[2].plot(
        merged_data['Periode'],
        merged_data['Port_Delta'],
        label='Portfolio-Delta'
    )
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].set_title('Portfolio-Delta', fontsize=12)
    axes[2].set_ylabel('Delta')
    axes[2].legend()
    
    # Plot D: Performance Vergleich
    title_perf = (
        f'Performance Kurve | CAGR: {ann_ret*100:.1f}% | '
        f'Vol: {ann_vol*100:.1f}% | Sharpe: {sharpe_ratio:.2f}'
    )
    
    axes[3].plot(
        merged_data['Periode'],
        merged_data['Wealth_Index'], 
        color='blue',
        linewidth=2,
        label='Portfolio Wert'
    )
    
    if 'SP500_Indexed' in merged_data.columns and not merged_data['SP500_Indexed'].isnull().all():
        axes[3].plot(
            merged_data['Periode'],
            merged_data['SP500_Indexed'], 
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            label='S&P 500 Benchmark'
        )

    axes[3].set_title(title_perf, fontsize=12)
    axes[3].set_ylabel('Index (Start=100)')
    axes[3].axhline(100, color='black', linestyle=':', linewidth=1)
    axes[3].legend(loc='upper left')
    
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.show()


def get_option_contributions(df, df_ret, run_id, weight_cols):
    """
    Erstellt ein DataFrame, das die Beiträge (Contributions) der einzelnen 
    Optionen zum Portfolio-Return in jeder Periode enthält.
    
    Contribution = Gewicht der Option * Return der Option
    """
    
    # 1. Daten filtern für diesen Run
    run_data = df[df['RunID'] == run_id].sort_values('Periode').copy()
    
    if run_data.empty:
        print(f"Warnung: Keine Daten für RunID {run_id} gefunden.")
        return None

    # Sicherstellen, dass Periode datetime ist
    run_data['Periode'] = pd.to_datetime(run_data['Periode'])

    # 2. Merge mit Returns-Daten (identisch zur Hauptfunktion)
    merged_data = pd.merge(
        run_data,
        df_ret,
        on='Periode',
        how='left',
        suffixes=('', '_ret')
    )

    # 3. Contributions berechnen
    contribution_dict = {'Periode': merged_data['Periode']}
    
    for col in weight_cols:
        # Bestimme den Namen der Return-Spalte (Logik aus analyze_single_run übernommen)
        if f"{col}_ret" in merged_data.columns:
            ret_col_name = f"{col}_ret"
        elif col in df_ret.columns:
            ret_col_name = col
        else:
            # Falls keine Return-Daten gefunden werden, Contribution = 0
            contribution_dict[col] = 0.0
            continue
            
        # Berechnung: Gewicht * Return
        contribution = merged_data[col] * merged_data[ret_col_name].fillna(0)
        contribution_dict[col] = contribution

    # 4. Als DataFrame formatieren
    df_contrib = pd.DataFrame(contribution_dict)
    df_contrib.set_index('Periode', inplace=True)
    
    return df_contrib


import matplotlib.pyplot as plt
import pandas as pd

def plot_contributions(df_contrib):
    """
    Plottet die Contributions als Stackplot.
    Gewinne werden nach oben, Verluste nach unten gestapelt.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 1. Daten in Positiv und Negativ trennen
    # Alles was > 0 ist, kommt in den positiven Stack, Rest ist 0
    df_pos = df_contrib.clip(lower=0)
    # Alles was < 0 ist, kommt in den negativen Stack, Rest ist 0
    df_neg = df_contrib.clip(upper=0)
    
    # 2. Farben definieren (damit Asset A oben und unten die gleiche Farbe hat)
    # Wir nehmen die Standard-Palette von Matplotlib
    colors = plt.cm.tab10.colors 
    if len(df_contrib.columns) > len(colors):
        # Falls mehr Assets als Farben, Palette wiederholen oder erweitern
        colors = plt.cm.get_cmap('tab20').colors
        
    cols = df_contrib.columns
    color_map = {col: colors[i % len(colors)] for i, col in enumerate(cols)}
    asset_colors = [color_map[c] for c in cols]

    # 3. Positive Beiträge stapeln (nach oben)
    ax.stackplot(
        df_pos.index, 
        df_pos.T, 
        labels=cols, 
        colors=asset_colors, 
        alpha=0.8
    )
    
    # 4. Negative Beiträge stapeln (nach unten)
    # Hinweis: Labels hier weglassen, sonst erscheinen sie doppelt in der Legende
    ax.stackplot(
        df_neg.index, 
        df_neg.T, 
        colors=asset_colors, 
        alpha=0.8
    )

    # 5. Netto-Kurve (Schwarze Linie) hinzufügen (Optional)
    # Zeigt den tatsächlichen Portfolio-Return (Summe aller Contributions)
    total_ret = df_contrib.sum(axis=1)
    ax.plot(
        total_ret.index, 
        total_ret, 
        color='black', 
        linestyle='--', 
        linewidth=1.5, 
        label='Netto Return'
    )

    # 6. Formatierung
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title('Portfolio Return Attribution (Contributions)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Beitrag zum Return')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    # Datumsformatierung x-Achse
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

# --- Anwendung ---
# df_contributions wurde im vorherigen Schritt erstellt
# plot_contributions(df_contributions)