import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
from helper_functions import step_size_common, initial_point_standard_heuristic

#--- KKT Solver for Mehrotra's Predictor-Corrector ---
def newton_direction_mehrotra(Rb, Rc, Rxs_arg_for_neg_Sdx_Xds, A, m, n, x, s, factor_obj_in=None, error_check=False, reg_param=1e-9):
    """ KKT solver for Mehrotra. S dx + X ds = -Rxs_arg_for_neg_Sdx_Xds """
    x_safe = np.maximum(x, 1e-14)
    # This uses the RHS formulation from your original Mehrotra newton_direction,
    # which was ds = -(Rxs_arg + s*dx)/x. This implies Sdx+Xds = -Rxs_arg.
    # So, A.T dy - (S/X)dx = -Rc + (Rxs_arg / X)
    rhs1 = -Rb
    rhs2 = -Rc + Rxs_arg_for_neg_Sdx_Xds / x_safe
    rhs_augmented = np.vstack([rhs1, rhs2])

    s_over_x = s.flatten() / x_safe.flatten()
    D_S_over_X_diag = -np.minimum(1e16, s_over_x) - reg_param

    B_00 = sp.diags(-reg_param * np.ones(m), format='csc') if m > 0 else sp.csc_matrix((0, 0))
    B_01 = sp.csc_matrix(A)
    B_10 = sp.csc_matrix(A.T)
    B_11 = sp.diags(D_S_over_X_diag, format='csc') if n > 0 else sp.csc_matrix((0, 0))

    nan_vec_n = np.full((n, 1), np.nan);
    nan_vec_m = np.full((m, 1), np.nan)
    if m == 0 and n == 0: return nan_vec_n, nan_vec_m, nan_vec_n, None

    B = sp.bmat([[B_00, B_01], [B_10, B_11]], format='csc')

    factor_obj_out = factor_obj_in
    if factor_obj_out is None:
        try:
            factor_obj_out = spla.splu(B)
        except Exception as e:
            print(f"Error (Mehrotra Newton): LU factorization failed: {e}")
            return nan_vec_n, nan_vec_m, nan_vec_n, None

    if factor_obj_out is None: return nan_vec_n, nan_vec_m, nan_vec_n, None  # Factorization failed

    try:
        sol_kkt = factor_obj_out.solve(rhs_augmented.flatten()).reshape(-1, 1)
        if np.isnan(sol_kkt).any(): return nan_vec_n, nan_vec_m, nan_vec_n, factor_obj_out
    except Exception as e:  # Solve failed, even with potentially existing factor_obj
        print(f"Error (Mehrotra Newton): LU solve failed: {e}.")
        # If it was a reuse, try re-factorizing once
        if factor_obj_in is not None:
            print("Attempting re-factorization...")
            try:
                factor_obj_out = spla.splu(B)  # New factorization object
                sol_kkt = factor_obj_out.solve(rhs_augmented.flatten()).reshape(-1, 1)
                if np.isnan(sol_kkt).any(): return nan_vec_n, nan_vec_m, nan_vec_n, factor_obj_out
            except Exception as e2:
                print(f"Error (Mehrotra Newton): Re-factorization and solve also failed: {e2}")
                return nan_vec_n, nan_vec_m, nan_vec_n, factor_obj_out  # Return potentially new factor_obj
        else:  # Factorization was done in this call and solve failed
            return nan_vec_n, nan_vec_m, nan_vec_n, factor_obj_out

    dy = sol_kkt[:m];
    dx = sol_kkt[m:m + n]
    ds = -(Rxs_arg_for_neg_Sdx_Xds + s * dx) / x_safe  # Consistent with Sdx+Xds = -Rxs_arg_...

    if error_check:  # Simplified check
        err_c = np.linalg.norm(s * dx + x * ds + Rxs_arg_for_neg_Sdx_Xds)
        print(f"[Mehrotra Newton System Check (Complementarity Target)] Rc_err: {err_c:.1e}")
    return dx, dy, ds, factor_obj_out

#--- Mehrotra's Predictor-Corrector Method (Main Function) ---
def mpc_solver(A_std, b_std, c_std, param_in=None, name='LP_Mehrotra'):  # Renamed params
    t0 = time.time()
    m, n = A_std.shape
    c_std_flat = c_std.flatten()  # Prepare 1D c
    b_std_flat = b_std.flatten()  # Ensure b is 1D
    param_mpc = {
        'maxN': 100, 'eps_mu': 1e-7, 'eps_feas': 1e-7, 'theta': 0.9995,
        'verbose': 1, 'min_step_size': 1e-8,
        'reg_param_pred': 1e-9, 'reg_param_corr': 1e-9
    }
    if param_in:
        for k_in, v_in in param_in.items():  # More careful update
            if k_in in param_mpc: param_mpc[k_in] = v_in
    options = {'errorCheck': False}

    x, y, s = initial_point_standard_heuristic(A_std, b_std, c_std)  # Call unified
    x_traj = []

    norm_b = np.linalg.norm(b_std) if np.linalg.norm(b_std) > 1e-10 else 1.0
    norm_c = np.linalg.norm(c_std) if np.linalg.norm(c_std) > 1e-10 else 1.0
    status = "MaxIter";
    iterations = 0;
    KKT_factor_predictor = None;
    mu = np.inf;
    prim_obj = np.nan;
    relative_gap = np.inf;
    res_p_norm = np.inf;
    res_d_norm = np.inf;

    '''if param_mpc['verbose'] > 0:
        print(f"\n===== {name} (Uses common initial_point, step_size) =====")
        print(
            f"Problem: m={m}, n={n}. Params: eps_mu={param_mpc['eps_mu']:.1e}, eps_feas={param_mpc['eps_feas']:.1e}, theta={param_mpc['theta']:.4f}")
        if param_mpc['verbose'] > 1:
            header = f"{'IT':>3} {'P.Obj':>10} {'D.Obj':>10} {'mu':>9} {'RelGap':>9} {'ResP':>9} {'ResD':>9} {'alphP':>6} {'alphD':>6}"
            print(header);
            print("-" * len(header))'''

    for iteration in range(param_mpc['maxN'] + 1):
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

        '''if param_mpc['verbose'] > 1:
            print(
                f"{iteration:3d} {prim_obj:10.3e} {dual_obj:10.3e} {mu:9.2e} {relative_gap:9.2e} {res_p_norm:9.2e} {res_d_norm:9.2e}",
                end="")'''

        if mu < param_mpc['eps_mu'] and res_p_norm < param_mpc['eps_feas'] and res_d_norm < param_mpc['eps_feas']:
            status = "Optimal";
            #print(f"\n{status}.") if param_mpc['verbose'] >= 1 else None;
            break
        if iteration >= param_mpc['maxN']:
            status = "MaxIter";
            break
        if np.isnan(x).any() or np.isnan(s).any() or np.isnan(y).any():
            status = "NaN Detected";
            prim_obj = np.nan;
            break

        # ----- Predictor Step -----
        # Target for S dx + X ds is -XSe. So, Rxs_arg for newton_direction_mehrotra is XSe.
        dx_aff, dy_aff, ds_aff, KKT_factor_predictor = newton_direction_mehrotra(
            Rb, Rc, Rxs_current, A_std, m, n, x, s,
            factor_obj_in=None,  # Predictor always (re)factors
            error_check=options['errorCheck'],
            reg_param=param_mpc['reg_param_pred']
        )
        if np.isnan(dx_aff).any(): status = "NewtonFailPred"; prim_obj = np.nan; break


        alphax_aff, alphas_aff = step_size_common(x, s, dx_aff, ds_aff, eta=1.0)  # Call unified

        mu_aff_numerator = (x + alphax_aff * dx_aff).T @ (s + alphas_aff * ds_aff)
        mu_aff = mu_aff_numerator[0, 0] / n if n > 0 else 0.0

        sigma = (mu_aff / max(mu, 1e-12)) ** 3
        sigma = np.clip(sigma, 0.001, 0.999)


        # ----- Corrector Step -----
        # Target for S dx + X ds = -(XSe + dXadSa - sigma*mu*e)
        # So, Rxs_arg for newton_direction_mehrotra is (XSe + dXadSa - sigma*mu*e)
        Rxs_target_corrector = Rxs_current + (dx_aff * ds_aff) - sigma * mu * np.ones((n, 1))


        dx_total, dy_total, ds_total, _ = newton_direction_mehrotra(
            Rb, Rc, Rxs_target_corrector, A_std, m, n, x, s,
            factor_obj_in=KKT_factor_predictor,  # Reuse factorization
            error_check=options['errorCheck'],
            reg_param=param_mpc['reg_param_corr']
        )
        if np.isnan(dx_total).any(): status = "NewtonFailCorr"; prim_obj = np.nan; break



        alphax, alphas = step_size_common(x, s, dx_total, ds_total, eta=param_mpc['theta'])  # Call unified
        #if param_mpc['verbose'] > 1: print(f" {alphax:6.3f} {alphas:6.3f}")
        if alphax < param_mpc['min_step_size'] and alphas < param_mpc['min_step_size'] and iteration > 5:
            status = "Stalled";
            break

        x += alphax * dx_total;
        y += alphas * dy_total;
        s += alphas * ds_total
        #print(f"{iteration:3d} {prim_obj:10.3e} {dual_obj:10.3e} {mu:9.2e} {relative_gap:9.2e} {res_p_norm:9.2e} {res_d_norm:9.2e}",end = "")
        '''if iteration == 9:
            print(f"\nIteration 9: x={x.flatten()}, s={s.flatten()}, y={y.flatten()}")
            print(f"dx_cc={dx_total.flatten()}, dy_cc={dy_total.flatten()}, ds_cc={ds_total.flatten()}")
            print(f"Corrector RHS components: -XSe={-Rxs_current.flatten()}, -DX_aff_DS_aff_e={-(dx_aff * ds_aff).flatten()}, sigma_mu_e={(sigma * mu * np.ones((n, 1))).flatten()}")
            print(f"dx_aff={dx_aff.flatten()}, dy_aff={dy_aff.flatten()}, ds_aff={ds_aff.flatten()}")
            print(f"sigma={sigma}")
            print(f"alphP_aff={alphax_aff}, alphD_aff={alphas_aff}, mu_aff={mu_aff}")'''

    total_time = time.time() - t0
    final_objective_val = np.dot(c_std_flat, x.flatten()) if not np.isnan(x).any() and x.size > 0 and n > 0 else np.nan
    final_objective = float(final_objective_val) if not np.isnan(final_objective_val) else np.nan
    '''if param_mpc['verbose'] > 0 and status != "Optimal":  # Print if not optimal or if already printed
        print(f"\n{name} finished: {status}. Obj={final_objective:.4e}, mu={mu:.2e}, Iter={iterations}")'''
    return final_objective, x.flatten(), y.flatten(), s.flatten(), iterations, total_time, status, x_traj