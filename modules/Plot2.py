import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import gmean, kurtosis, skew


def load_and_prep_data(filepath, filepath_ret, filepath_rates, filepath_spx):
    df = pd.read_csv(filepath)
    df_ret = pd.read_csv(filepath_ret)
    df_rates = pd.read_csv(filepath_rates)
    sp500_df = pd.read_csv(filepath_spx)
    df["Periode"] = pd.to_datetime(df["Periode"])
    df_ret["Periode"] = pd.to_datetime(df_ret["Periode"], dayfirst=False)
    df_rates["Date"] = pd.to_datetime(df_rates["Date"])
    df = df.sort_values("Periode")
    df_rates = df_rates.sort_values("Date")
    sp500_df = pd.read_csv(filepath_spx)
    sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])
    df = pd.merge_asof(
        df, df_rates, left_on="Periode", right_on="Date", direction="backward"
    )
    df.rename(columns={"DGS1MO": "RiskFreeRate"}, inplace=True)

    weight_cols = [c for c in df.columns if c.startswith("W_")]

    df["Total_Exposure"] = df[weight_cols].sum(axis=1)

    long_cols = [c for c in weight_cols if "Long" in c]
    short_cols = [c for c in weight_cols if "Short" in c]

    df["Long_Exposure"] = df[long_cols].sum(axis=1)
    df["Short_Exposure"] = df[short_cols].sum(axis=1)
    df["Cash_Weight"] = (1.0 - df["Long_Exposure"] + df["Short_Exposure"]).clip(
        lower=0.0
    )

    return df, df_ret, weight_cols, long_cols, short_cols, sp500_df


def auswertung(run_id, df, df_returns, df_vix):
    test = df[df["RunID"] == run_id].copy()
    test = test.dropna(axis=0)
    test.loc[:, "RiskFreeRate"] = test.loc[:, "RiskFreeRate"] / 12
    test["zinsen"] = test["Cash_Weight"] * test["RiskFreeRate"]
    test = test.set_index("Periode")
    vix_pct_change = df_vix["Close"].pct_change()
    Mat_vix = vix_pct_change.reindex(test.index, method="ffill")
    w_df = test.filter(like="W_")
    ret_df = df_returns.set_index("Periode").filter(like="W_")
    ret_df = ret_df.reindex(test.index)
    port_df = ret_df * w_df
    port_df["zinsen"] = test["zinsen"]
    port_df["ATM_Put"] = w_df["W_Long_P1"] - w_df["W_Short_P1"]
    port_df["ATM_Call"] = w_df["W_Long_C1"] - w_df["W_Short_C1"]
    port_df["OTM_Put"] = w_df["W_Long_P2"] - w_df["W_Short_P2"]
    port_df["OTM_Call"] = w_df["W_Long_C2"] - w_df["W_Short_C2"]
    w_cols = [c for c in df.columns if c.startswith("W_")]
    port_df["Excess_ret"] = port_df[[c for c in w_cols if c in port_df.columns]].sum(
        axis=1
    )
    port_df["TotalRet"] = port_df["zinsen"] + port_df["Excess_ret"]
    port_df["Wealth"] = 100 * (port_df["TotalRet"] + 1).cumprod(axis=0)
    target_cols = [
        "ATM_Put",
        "ATM_Call",
        "OTM_Put",
        "OTM_Call",
        "Excess_ret",
        "TotalRet",
        "Wealth",
    ]
    existing_cols = [c for c in target_cols if c in port_df.columns]
    op = port_df[existing_cols].copy()

    op["VIX"] = Mat_vix
    op["Peak"] = op["Wealth"].cummax()
    op["Drawdown"] = (op["Wealth"] - op["Peak"]) / op["Peak"]
    op = op.dropna(axis=0)
    sharpe = (op["Excess_ret"].mean() / op["Excess_ret"].std()) * np.sqrt(12)
    total_growth = (op["Excess_ret"] + 1).prod()
    n_months = len(op)
    n_years = n_months / 12 if n_months > 0 else 0
    cagr_variante_2 = (total_growth) ** (1 / n_years) - 1 if n_years > 0 else 0

    asset_corr = op.iloc[:, :4].corr()

    max_drawdown = op["Drawdown"].min()
    ret_kurt = kurtosis(op["Excess_ret"], fisher=True)
    ret_skew = skew(op["Excess_ret"])
    ret_vix_corr = op["Excess_ret"].corr(op["VIX"])

    return (
        op,
        sharpe,
        cagr_variante_2,
        asset_corr,
        max_drawdown,
        ret_kurt,
        ret_skew,
        ret_vix_corr,
    )
