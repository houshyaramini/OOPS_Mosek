import pandas as pd
import numpy as np
from tqdm import tqdm # Optional: Für Fortschrittsbalken (pip install tqdm)

class BatchRunAnalyzer:
    def __init__(self, sim_path, ret_path, rates_path, spx_path=None):
        """
        Initialisiert den Analyzer und lädt die Daten in den Speicher.
        """
        self.paths = {
            'sim': sim_path,
            'ret': ret_path,
            'rates': rates_path,
            'spx': spx_path
        }
        self.data = self._load_data()
        self.prepared_data = None

    def _load_data(self):
        """Interne Ladefunktion."""
        data = {}
        # Returns laden & Index setzen
        data['ret'] = pd.read_csv(self.paths['ret'], parse_dates=['Periode']).set_index('Periode')
        
        # Simulation laden
        data['sim'] = pd.read_csv(self.paths['sim'], parse_dates=['Periode'])
        
        # Zinsen laden & sortieren
        data['rates'] = pd.read_csv(self.paths['rates'], parse_dates=['Date']).sort_values('Date')
        
        # Optional: Benchmark
        if self.paths['spx']:
            try:
                data['spx'] = pd.read_csv(self.paths['spx'], parse_dates=['Date']).set_index('Date')
            except FileNotFoundError:
                data['spx'] = pd.DataFrame()
        return data

    def prepare(self):
        """
        Bereitet die Simulationsdaten vor: Merged Zinsen und berechnet 
        Cash/Exposure für ALLE Runs auf einmal (vektorisiert).
        """
        print("Bereite Daten vor...")
        df_sim = self.data['sim'].sort_values('Periode')
        df_rates = self.data['rates']

        # 1. Merge mit Risk Free Rate (Global für alle Runs)
        df_merged = pd.merge_asof(df_sim, df_rates, left_on='Periode', right_on='Date', direction='backward')
        
        # Zinsberechnung (p.a. -> monatlich dezimal)
        # Annahme: Input ist % (z.B. 2.5), daher /100. Falls Input 0.025 ist, das /100 entfernen.
        #df_merged['RiskFreeRate_Mo'] = (df_merged['DGS1MO'] / 100) / 12
        df_merged['RiskFreeRate_Mo'] = df_merged['DGS1MO'] / 12

        # 2. Exposure Berechnung (Vektorisiert über gesamten DF)
        w_cols = [c for c in df_merged.columns if c.startswith('W_')]
        long_cols = [c for c in w_cols if 'Long' in c]
        short_cols = [c for c in w_cols if 'Short' in c]

        df_merged['Long_Exp'] = df_merged[long_cols].sum(axis=1)
        df_merged['Short_Exp'] = df_merged[short_cols].sum(axis=1)
        # Cash Weight darf nicht negativ sein
        df_merged['Cash_Weight'] = (1.0 - df_merged['Long_Exp'] + df_merged['Short_Exp']).clip(lower=0.0)
        
        self.prepared_data = df_merged
        self.w_cols = w_cols
        print("Datenvorbereitung abgeschlossen.")

    def _calc_metrics(self, series, exposure_series=None):
        """Berechnet die KPIs für eine Zeitreihe."""
        if len(series) < 2: return {}
        
        # Basisdaten
        freq = 12
        total_ret = (1 + series).prod() - 1
        n_years = len(series) / freq
        
        # 1. CAGR
        cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # 2. Volatilität & Downside
        vol = series.std() * np.sqrt(freq)
        neg_rets = series[series < 0]
        downside_std = neg_rets.std() * np.sqrt(freq)
        
        # 3. Drawdown
        wealth = (1 + series).cumprod()
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        max_dd = dd.min()
        
        # 4. Profit Factor
        gross_win = series[series > 0].sum()
        gross_loss = abs(series[series < 0].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else np.inf
        
        # 5. Impliziter Leverage (Avg Exposure)
        avg_exp = exposure_series.mean() if exposure_series is not None else 1.0
        
        return {
            'CAGR': cagr,
            'Vol_p.a.': vol,
            'Sharpe': (cagr - 0.02) / vol if vol > 0 else 0, # Annahme 2% RFR fix für Sharpe
            'Sortino': (cagr - 0.02) / downside_std if downside_std > 0 else 0,
            'MaxDD': max_dd,
            'Calmar': cagr / abs(max_dd) if max_dd != 0 else 0,
            'ProfitFactor': pf,
            'WinRate': (series > 0).mean(),
            'Avg_Exposure': avg_exp
        }

    def run_analysis(self):
        """
        Führt die Analyse über alle RunIDs durch.
        """
        if self.prepared_data is None:
            self.prepare()

        results = []
        grouped = self.prepared_data.groupby('RunID')
        
        # Loop über jeden Run
        # tqdm zeigt einen Fortschrittsbalken an (optional)
        iterator = tqdm(grouped) if 'tqdm' in globals() else grouped
        
        for run_id, group in iterator:
            # Gemeinsame Daten abgleichen
            common_idx = group['Periode']
            # Returns für diesen Zeitraum filtern
            # Achtung: Reindex ist teuer, wir nutzen intersection logic via loc
            current_rets = self.data['ret'].loc[common_idx]
            
            # Asset Returns (Matrix Multiplikation: (T x Assets) * (T x Weights))
            # Wir nutzen numpy values für Speed
            w_mat = group[self.w_cols].values
            r_mat = current_rets[self.w_cols].values
            
            # Summe der gewichteten Returns pro Zeitschritt
            asset_ret = np.sum(w_mat * r_mat, axis=1)
            
            # Cash Return
            cash_ret = group['Cash_Weight'].values * group['RiskFreeRate_Mo'].values
            
            # Total Return Series
            total_ret_series = pd.Series(asset_ret + cash_ret, index=common_idx)
            
            # Metriken berechnen
            metrics = self._calc_metrics(total_ret_series, group['Long_Exp'] + group['Short_Exp'])
            metrics['RunID'] = run_id
            results.append(metrics)
            
        return pd.DataFrame(results).set_index('RunID')

# --- ANWENDUNG ---
if __name__ == "__main__":
    # Pfade anpassen
    analyzer = BatchRunAnalyzer(
        sim_path='results/simulation_log.csv',
        ret_path='data/RealeReturns.csv',
        rates_path='data/DSG1MO_fred.csv'
    )
    
    # Analyse starten
    df_results = analyzer.run_analysis()
    
    # Ergebnis sortiert nach Calmar Ratio anzeigen
    print("\n--- TOP 10 RUNS (nach Calmar Ratio) ---")
    print(df_results.sort_values('Calmar', ascending=False).head(10))
    
    # Speichern für Excel/CSV Übersicht
    #df_results.to_csv("results/batch_overview.csv")
    #print("\nÜbersicht gespeichert unter results/batch_overview.csv")