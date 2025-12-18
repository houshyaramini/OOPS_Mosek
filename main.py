import pandas as pd
import numpy as np
import cvxpy as cp
import os
import warnings
import mosek
from joblib import Parallel, delayed
from datetime import datetime


from modules.config import SimulationConfig
from modules.data_util import (
    daten_laden,
    get_options_data,
    calculate_returns,
    real_vola2,
    payoff_vektor,
    vek_ret_long,
    vek_ret_short,
)
from modules.optimization import optimize_period

warnings.filterwarnings(
    "ignore", message=".*Incorrect array format.*", category=UserWarning, module="mosek"
)

pd.set_option("display.float_format", "{:.3f}".format)


def run_simulation(config: SimulationConfig = None):

    if config is None:
        config = SimulationConfig()
    run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if config.use_seed:
        np.random.seed(config.seed)

    df, df_options, rf = daten_laden(
        Index=config.file_spx,
        Derivate=config.file_options,
        RiskFreeRate=config.file_risk_free,
    )
    options_data = get_options_data(df_options)
    returns_data = calculate_returns(
        df=df,
        start_date_vec=options_data.StartDatumPeriode,
        end_date_vec=options_data.EndDatumPeriode,
        min_periods_z=60,
    )

    K = len(config.d_window)

    rf = rf.dropna()
    rf_t = rf.reindex(options_data.StartDatumPeriode, method="ffill").to_numpy()
    rf_t = rf_t / 12

    S_t_arr = np.empty((options_data.N, K, config.n_wiederholungen), dtype=float)
    for i in range(options_data.N):
        start = options_data.StartDatumPeriode.iloc[i]
        S_0 = float(returns_data.start_prices.loc[start])
        pool = returns_data.stand_log_m.loc[:start].dropna().to_numpy()
        stand_draws = np.random.choice(pool, size=config.n_wiederholungen, replace=True)
        for j, d in enumerate(config.d_window):
            vol_real = float(
                real_vola2(returns_data.log_d_sqr, start, d=d, Skalierung=21)
            )
            S_t_arr[i, j, :] = S_0 * np.exp(stand_draws * vol_real)

    Sim_payoff_array = np.stack(
        [
            payoff_vektor(S_t_arr, np.array(options_data.strikes_atm), "p"),
            payoff_vektor(S_t_arr, np.array(options_data.strikes_atm), "c"),
            payoff_vektor(S_t_arr, np.array(options_data.strikes_otm_put), "p"),
            payoff_vektor(S_t_arr, np.array(options_data.strikes_otm_call), "c"),
        ],
        axis=-1,
    )
    sim_vek_ret_long = vek_ret_long(Sim_payoff_array, options_data.ask)
    sim_vek_ret_short = vek_ret_short(Sim_payoff_array, options_data.bid)
    sim_ret_array = np.stack([sim_vek_ret_long, sim_vek_ret_short], axis=-1)

    N_perioden = sim_ret_array.shape[0]
    N_d_werte = sim_ret_array.shape[1]
    N_assets = sim_ret_array.shape[3] * sim_ret_array.shape[4]
    best_d = np.zeros(N_perioden, dtype=float)
    best_util = np.full(N_perioden, -np.inf)
    best_w_matrix = np.zeros((N_perioden, N_assets))

    print("Starte parallele Optimierung...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(optimize_period)(i, rf_t[i], sim_ret_array[i], config)
        for i in range(N_perioden)
    )

    print("Optimierung abgeschlossen. Starte Post-Processing...")

    all_best_weights = []
    vol_scaler = []

    cols = [
        "W_Long_P1",
        "W_Short_P1",
        "W_Long_C1",
        "W_Short_C1",
        "W_Long_P2",
        "W_Short_P2",
        "W_Long_C2",
        "W_Short_C2",
    ]

    for i in range(N_perioden):
        start = options_data.StartDatumPeriode.iloc[i]
        end = options_data.EndDatumPeriode.iloc[i]
        rf_T_scalar = rf_t[i].item()
        S_0_scalar = returns_data.start_prices.iloc[i]

        best_expected_utility, best_d_index, best_weights, best_R = results[i]

        if best_d_index < 0:
            best_d_value = np.nan
        else:
            best_d_value = config.d_window[int(best_d_index)]

        w = np.asarray(best_weights, dtype=float)
        cash = 1.0 - np.sum(w[config.long_idx]) + np.sum(w[config.short_idx])

        vol_scaler.append(pd.Series(best_d_value))
        all_best_weights.append(pd.Series(best_weights, index=cols, name=start))
    df_best_w_pro_t = pd.concat(all_best_weights, axis=1).T

    log_df = df_best_w_pro_t.copy()
    log_df["RunID"] = run_id
    log_df["Gamma"] = config.gamma
    log_df["Seed"] = config.seed if config.use_seed else np.nan
    log_df = log_df.reset_index().rename(columns={"index": "Periode"})

    file_exists = os.path.isfile(config.log_file)
    log_df.to_csv(config.log_file, mode="a", index=False, header=not file_exists)

    print(f"Fertig! Log geschrieben in {config.log_file}")

cfg = SimulationConfig(
    use_crra=True,
    n_wiederholungen=3000,
    use_seed=True,
    seed=252,
    gamma=10,
    bounds= [(0.0 , 1.0)] * 8,
    pair_idx=False

)

if __name__ == "__main__":
    run_simulation(config=cfg)
