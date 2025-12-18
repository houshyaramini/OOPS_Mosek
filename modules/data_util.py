import pandas as pd
import numpy as np
from arch import arch_model
from dataclasses import dataclass
from typing import Union

def daten_laden(Index, Derivate, RiskFreeRate):
    df = pd.read_csv(Index, parse_dates=["Date"], index_col="Date")
    df_options = pd.read_csv(
        Derivate, parse_dates=["BewertungsDatum"], index_col="BewertungsDatum"
    )
    rf = pd.read_csv(RiskFreeRate, parse_dates=["Date"], index_col="Date")
    return df, df_options, rf

def real_vola2(ret, start, d, Skalierung):
    #start = start - pd.Timedelta(days=1)
    werte = ret[: start][-d:]
    return np.sqrt(werte.sum()) * np.sqrt(Skalierung / d)

def option_payoff(Spot, Strike, type):
    if type == "c":
        return np.maximum(Spot - Strike, 0)
    if type == "p":
        return np.maximum(Strike - Spot, 0)

def payoff_vektor(s_t, k, options_type):
    if s_t.ndim == 3:
        k = np.array(k)
        k_array = k[:, np.newaxis, np.newaxis]
        if options_type == "c":
            return np.maximum(s_t - k_array, 0)
        if options_type == "p":
            return np.maximum(k_array - s_t, 0)
    else:
        return option_payoff(Spot=s_t, Strike=k, type=options_type)
    
def vek_ret_long(payoff, ask):
    if payoff.ndim == 4:
        ask_array = ask[:, np.newaxis, np.newaxis, :]
    else:
        ask_array = ask
    return (payoff / ask_array) - 1

def vek_ret_short(payoff, bid):
    if payoff.ndim == 4:
        bid_array = bid[:, np.newaxis, np.newaxis, :]
    else:
        bid_array = bid
    return 1 - (payoff / bid_array)

@dataclass
class OptionsData:
    ask: np.ndarray
    bid: np.ndarray
    strikes_all: np.ndarray
    strikes_atm: np.ndarray
    strikes_otm_call: np.ndarray
    strikes_otm_put: np.ndarray
    maturity: pd.Series
    StartDatumPeriode: pd.Series
    EndDatumPeriode: pd.Series
    N: int

def get_options_data(df_options: pd.DataFrame) -> OptionsData:
    df_clean = df_options.dropna()
    df_strikes = df_clean.filter(like="Strike")
    maturity= df_options.index.to_series().dropna()
    N = len(maturity)-1
    return OptionsData(
        ask = df_clean.filter(like="Ask").to_numpy(),
        bid = df_clean.filter(like="Bid").to_numpy(),
        strikes_all = df_strikes.to_numpy(),
        strikes_atm = df_strikes.filter(like="ATM").squeeze().to_numpy(),
        strikes_otm_call = df_strikes.filter(like="Call").squeeze().to_numpy(),
        strikes_otm_put = df_strikes.filter(like="Put").squeeze().to_numpy(),
        maturity= maturity,
        StartDatumPeriode = maturity[:N],
        EndDatumPeriode = maturity[-N:],
        N= N
    )

@dataclass
class ReturnsData:
    """
    Container für berechnete Returns und Volatilitäten.
    """
    log_d: pd.Series          
    log_d_sqr: pd.Series      
    log_m: pd.Series          
    rv_m: pd.Series           
    stand_log_m: pd.Series    
    z_score: pd.Series        
    start_prices: Union[pd.Series, float] 
    end_prices: np.ndarray


def calculate_returns(
    df: pd.DataFrame,
    start_date_vec: pd.Series, 
    end_date_vec: pd.Series,
    min_periods_z: int = 60
) -> ReturnsData:
    """
    Berechnet Return-Metriken basierend auf einem DataFrame mit 'Close'-Spalte.
    """
    log_d = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    log_d_sqr = log_d**2
    monthly_close = df["Close"].resample("ME").last()
    log_m = np.log(monthly_close / monthly_close.shift(1))
    rv_m = np.sqrt(log_d_sqr.resample("ME").sum()).dropna()
    stand_log_m = (log_m / rv_m.shift(1)).dropna()
    expanding_mean = log_m.expanding(min_periods=min_periods_z).mean().shift(1)
    z = (log_m - expanding_mean) / rv_m.shift(1)
    start_prices = df["Close"].reindex(start_date_vec, method="ffill").squeeze()
    end_prices = (
        df["Close"]
        .reindex(end_date_vec - pd.Timedelta(days=1), method="ffill")
        .to_numpy()
    )

    return ReturnsData(
        log_d=log_d,
        log_d_sqr=log_d_sqr,
        log_m=log_m,
        rv_m=rv_m,
        stand_log_m=stand_log_m,
        z_score=z,
        start_prices=start_prices,
        end_prices=end_prices
    )
