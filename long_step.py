import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
from helper_functions import step_size_common, initial_point_standard_heuristic


# --- KKT Solver for Primal-Dual Long-Step ---
def newton_direction_pd(Rb, Rc, Rxs_target_for_Sdx_Xds_eq_neg_this, A, m, n, x, s, error_check=False, reg_param=1e-9):
    """ KKT solver for Primal-Dual Long-Step. S dx + X ds = -Rxs_target_... """
    x_safe = np.maximum(x, 1e-14)
    rhs1 = -Rb
    rhs2 = -Rc + Rxs_target_for_Sdx_Xds_eq_neg_this / x_safe  # Consistent with Sdx+Xds = -Rxs_target
    rhs_augmented = np.vstack([rhs1, rhs2])

    s_over_x = s.flatten() / x_safe.flatten()
    D_S_over_X_diag = -np.minimum(1e16, s_over_x) - reg_param

    B_00 = sp.diags(-reg_param * np.ones(m), format='csc') if m > 0 else sp.csc_matrix((0, 0))
    B_01 = sp.csc_matrix(A)
    B_10 = sp.csc_matrix(A.T)
    B_11 = sp.diags(D_S_over_X_diag, format='csc') if n > 0 else sp.csc_matrix((0, 0))

    nan_vec_n = np.full((n, 1), np.nan);
    nan_vec_m = np.full((m, 1), np.nan)
    if m == 0 and n == 0: return nan_vec_n, nan_vec_m, nan_vec_n

    B = sp.bmat([[B_00, B_01], [B_10, B_11]], format='csc')

    try:
        sol_kkt = spla.spsolve(B, rhs_augmented.flatten()).reshape(-1, 1)
        if np.isnan(sol_kkt).any(): return nan_vec_n, nan_vec_m, nan_vec_n
    except Exception as e:
        print(f"Error (Newton PD): Solve failed: {e}")
        return nan_vec_n, nan_vec_m, nan_vec_n

    dy = sol_kkt[:m];
    dx = sol_kkt[m:m + n]
    ds = (-Rxs_target_for_Sdx_Xds_eq_neg_this - s * dx) / x_safe

    if error_check:  # Simplified check
        err_c = np.linalg.norm(s * dx + x * ds + Rxs_target_for_Sdx_Xds_eq_neg_this)
        print(f"[Newton PD System Check (Complementarity Target)] Rc_err: {err_c:.1e}")
    return dx, dy, ds


# --- Primal-Dual Long-Step Path-Following Solver (Main Function) ---
def pd_long_step_solver(A_std, b_std, c_std, param_in=None, name='LP_LongStep'):  # Renamed params
    t_start = time.time()
    m, n = A_std.shape
    c_std_flat = c_std.flatten()  # Prepare 1D c
    b_std_flat = b_std.flatten()  # Ensure b is 1D

    param_pd = {
        'maxN': 100, 'eps_mu': 1e-7, 'eps_feas': 1e-7, 'theta': 0.9995,
        'sigma_centering': 0.2, 'verbose': 1, 'min_step_size': 1e-8,
        'reg_param': 1e-9
    }
    if param_in:
        for k_in, v_in in param_in.items():  # More careful update
            if k_in in param_pd: param_pd[k_in] = v_in
    options = {'errorCheck': False}

    x, y, s = initial_point_standard_heuristic(A_std, b_std, c_std)  # Call unified
    x_traj =[]

    norm_b = np.linalg.norm(b_std) if np.linalg.norm(b_std) > 1e-10 else 1.0
    norm_c = np.linalg.norm(c_std) if np.linalg.norm(c_std) > 1e-10 else 1.0
    status = "MaxIter";
    iterations = 0;
    mu = np.inf;
    prim_obj = np.nan;
    relative_gap = np.inf;
    res_p_norm = np.inf;
    res_d_norm = np.inf;

    '''if param_pd['verbose'] > 0:
        print(f"\n===== {name} (Uses common initial_point, step_size) =====")
        print(
            f"Problem: m={m}, n={n}. Params: eps_mu={param_pd['eps_mu']:.1e}, eps_feas={param_pd['eps_feas']:.1e}, sigma={param_pd['sigma_centering']:.2f}")'''
    '''if param_pd['verbose'] > 1:
            header = f"{'IT':>3} {'P.Obj':>10} {'D.Obj':>10} {'mu':>9} {'RelGap':>9} {'ResP':>9} {'ResD':>9} {'alphP':>6} {'alphD':>6}"
            print(header);
            print("-" * len(header))'''

    for iteration in range(param_pd['maxN'] + 1):
        iterations = iteration
        x = np.maximum(x, 1e-14);
        x_traj.append(x.flatten())
        s = np.maximum(s, 1e-14)

        Rb = A_std @ x - b_std.reshape(-1, 1)
        Rc = A_std.T @ y + s - c_std.reshape(-1, 1)
        Rxs_current = x * s

        prim_obj = float(np.dot(c_std_flat, x.flatten()) if n > 0 else 0.0)
        dual_obj = float(np.dot(b_std_flat, y.flatten()) if m > 0 else 0.0)
        mu = np.sum(Rxs_current) / n if n > 0 else 0.0

        duality_gap_abs = prim_obj - dual_obj
        rel_gap_denom = max(1.0, abs(prim_obj), abs(dual_obj))
        relative_gap = abs(duality_gap_abs) / rel_gap_denom if rel_gap_denom > 1e-12 else abs(duality_gap_abs)
        res_p_norm = np.linalg.norm(Rb) / (1 + norm_b) if m > 0 else 0.0
        res_d_norm = np.linalg.norm(Rc) / (1 + norm_c) if n > 0 else 0.0

        '''if param_pd['verbose'] > 1:
            print(
                f"{iteration:3d} {prim_obj:10.3e} {dual_obj:10.3e} {mu:9.2e} {relative_gap:9.2e} {res_p_norm:9.2e} {res_d_norm:9.2e}",
                end="")'''

        if mu < param_pd['eps_mu'] and res_p_norm < param_pd['eps_feas'] and res_d_norm < param_pd['eps_feas']:
            status = "Optimal";
            #print(f"\n{status}.") if param_pd['verbose'] >= 1 else None;
            break
        if iteration >= param_pd['maxN']:
            status = "MaxIter";
            break  # Verbose print after loop for this
        if np.isnan(x).any() or np.isnan(s).any() or np.isnan(y).any():
            status = "NaN Detected";
            prim_obj = np.nan;
            break

        sigma = param_pd['sigma_centering']
        # Target for S_k dx + X_k ds = sigma * mu_k * e - X_k S_k e
        # newton_direction_pd expects Rxs_target such that Sdx+Xds = -Rxs_target
        # So, pass -(sigma*mu*e - XSe) = XSe - sigma*mu*e
        Rxs_target_for_newton_pd = Rxs_current - sigma * mu * np.ones_like(Rxs_current)

        dx, dy, ds = newton_direction_pd(
            Rb, Rc, Rxs_target_for_newton_pd,  # Pass this target
            A_std, m, n, x, s,
            error_check=options['errorCheck'],
            reg_param=param_pd['reg_param']
        )

        if np.isnan(dx).any(): status = "NewtonFail"; prim_obj = np.nan; break
        alphax, alphas = step_size_common(x, s, dx, ds, eta=param_pd['theta'])  # Call unified
        '''if param_pd['verbose'] > 1: print(f" {alphax:6.3f} {alphas:6.3f}")'''
        if alphax < param_pd['min_step_size'] and alphas < param_pd['min_step_size'] and iteration > 5:
            status = "Stalled";
            break

        x += alphax * dx;
        y += alphas * dy;
        s += alphas * ds

    total_time = time.time() - t_start
    final_objective_val = np.dot(c_std_flat, x.flatten()) if not np.isnan(x).any() and x.size > 0 and n > 0 else np.nan
    final_objective = float(final_objective_val) if not np.isnan(final_objective_val) else np.nan
    '''if param_pd['verbose'] > 0 and status != "Optimal":  # Print if not optimal or if already printed
        print(f"\n{name} finished: {status}. Obj={final_objective:.4e}, mu={mu:.2e}, Iter={iterations}")'''
    return x.flatten(), final_objective, iterations, total_time, status, x_traj