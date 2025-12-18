import numpy as np
import cvxpy as cp
import mosek
import warnings

warnings.filterwarnings(
    "ignore", message=".*Incorrect array format.*", category=UserWarning, module="mosek"
)
def crra_utility(returns, gamma):
    returns = np.asarray(returns)
    if gamma == 1:
        return np.mean(np.log(1 + returns))
    else:
        return np.mean(((1 + returns) ** (1 - gamma)) / (1 - gamma))


def optimize_with_mosek(
    returns,
    rf,
    gamma,
    long_idx,
    short_idx,
    bounds,
    pair_idx,
    eps,
    w0=None,
    use_crra: bool = False,
    n_threads=1,
):
    n_samples = returns.shape[0]
    n_assets = returns.shape[1]
    w = cp.Variable(n_assets)
    rf_t = float(rf.item())
    cash = 1.0 - cp.sum(w[long_idx]) + cp.sum(w[short_idx])
    r_port = cash * rf_t + returns @ w
    x = cp.Variable(n_assets, boolean=True)
    constraints = []

    if use_crra:
        utility_vals = cp.mean(cp.power(1 + r_port, 1 - gamma) / (1 - gamma))
        objective = cp.Maximize(utility_vals)
        constraints.append(1 + r_port >= eps)

    else:
        objective = cp.Maximize(cp.mean(r_port - gamma * 0.5 * cp.power(r_port, 2)))

    constraints.append(cp.sum(w[long_idx]) + cp.sum(w[short_idx]) <= 1.0)
    # constraints.append(cp.sum(w[short_idx]) <= 0.0)
    # constraints.append(cp.sum(w[long_idx]) <= 0.0)
    for i, (lb, ub) in enumerate(bounds):
        constraints.append(w[i] >= lb * x[i])
        constraints.append(w[i] <= ub * x[i])

    if pair_idx:
        for i, j in pair_idx:
            constraints.append(x[i] + x[j] <= 1)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=cp.MOSEK,
            mosek_params={
                "MSK_IPAR_NUM_THREADS": n_threads,
            },
            verbose=False,
        )

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            w_opt = w.value
            w_opt[np.abs(w_opt) < 1e-6] = 0.0
            cash_val = 1.0 - np.sum(w_opt[long_idx]) + np.sum(w_opt[short_idx])
            r_port_val = cash_val * rf_t + returns @ w_opt
            exp_utility = crra_utility(r_port_val, gamma)
            exp_return = np.mean(r_port_val) - gamma / 2 * np.mean(r_port_val**2)
            return {
                "success": True,
                "x": w_opt,
                "fun": exp_utility,
                "port": exp_return,
                "status": problem.status,
            }
        else:
            return {"success": False, "status": problem.status}
    except Exception as e:
        print(f"MOSEK optimization failed: {e}")
        return {"success": False, "status": "error", "message": str(e)}


def optimize_period(i, rf_T, period_returns, config):
    gamma = config.gamma
    bounds = config.bounds
    long_idx = config.long_idx
    short_idx = config.short_idx
    pair_idx = config.pair_idx
    w0 = config.w0
    worst_case = config.worst_case
    big_array = config.big_array

    eps_val = config.eps
    N_assets = config.n_assets
    use_crra = config.use_crra
##################################
    d1,d2,d3,d4 = period_returns.shape
    if big_array == True:
        period_returns = period_returns.reshape(1, d1*d2,d3,d4)


##############################
    N_d_werte = period_returns.shape[0]
    best_util_in_t = -np.inf
    best_d_in_t = -1
    best_w_in_t = np.zeros(N_assets)
    best_ret_array_in_t = np.array([])

    for j in range(N_d_werte):
        sim_ret_3 = period_returns[j, :, :, :]

        ret_mosek = sim_ret_3.reshape(-1, N_assets)
        ret_mosek = ret_mosek.astype(float)
        nan_mask = np.isnan(ret_mosek).any(axis=1)
        filtered_ret = ret_mosek[~nan_mask]

        if filtered_ret.shape[0] == 0:
            current_u = -np.inf
            current_w = np.zeros(N_assets)
        else:
            result = optimize_with_mosek(
                returns=filtered_ret,
                rf=rf_T,
                gamma=gamma,
                long_idx=long_idx,
                short_idx=short_idx,
                bounds=bounds,
                pair_idx=pair_idx,
                w0=w0,
                eps=eps_val,
                use_crra=use_crra,
            )
            if result["success"]:
                if use_crra == True:
                    current_u = result["fun"]
                else:
                    current_u = result["port"]
                current_w = result["x"]
                if worst_case == True:
                    current_u = -current_u
            else:
                current_u = -np.inf
                current_w = np.zeros(N_assets)

        if current_u > best_util_in_t:
            best_util_in_t = current_u
            best_d_in_t = j
            best_w_in_t = current_w
            best_ret_array_in_t = filtered_ret

    return (best_util_in_t, best_d_in_t, best_w_in_t, best_ret_array_in_t)
