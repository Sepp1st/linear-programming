import numpy as np
import scipy.sparse as sp


# --- COMMON PRE-PROCESSING AND UTILITIES ---
def convert_to_standard_form(c_orig, A_orig, b_orig, constraint_types):
    # (Using the version from your last successful run)
    m, n_orig = A_orig.shape
    A_mod = A_orig.copy()  # Make copies to avoid in-place modification of inputs
    b_mod = b_orig.copy()
    constraint_types_mod = list(constraint_types)  # Make a mutable copy

    num_slacks = 0
    for i in range(m):  # Iterate with index to modify constraint_types_mod
        if constraint_types_mod[i] == 'G':
            A_mod[i, :] *= -1
            b_mod[i] *= -1
            constraint_types_mod[i] = 'L'  # Now treat as 'L'
            # num_slacks will be incremented when 'L' is processed
        # No explicit increment for num_slacks here for 'G' because it's now 'L'
        # 'E' constraints also don't add to num_slacks here for this counting logic

    # Count slacks needed for L (original L and G-converted-to-L)
    for ct in constraint_types_mod:
        if ct == 'L':
            num_slacks += 1
        elif ct == 'E':
            pass
        else:  # Should not happen if G was converted
            raise ValueError(f"Unknown constraint type after G-conversion: {ct}")

    n_std = n_orig + num_slacks
    A_std = np.zeros((m, n_std))
    # Ensure c_orig is 1D before assignment, and handle if it's already the correct length (e.g. for testing)
    c_flat_orig = c_orig.flatten()
    if len(c_flat_orig) == n_std:  # If c_orig was already for standard form (e.g. test cases)
        c_std = c_flat_orig
    elif len(c_flat_orig) == n_orig:
        c_std = np.zeros(n_std)
        c_std[:n_orig] = c_flat_orig
    else:
        raise ValueError(f"c_orig length {len(c_flat_orig)} incompatible with n_orig {n_orig} or n_std {n_std}")

    A_std[:, :n_orig] = A_mod

    current_slack_idx = n_orig
    for i in range(m):
        if constraint_types_mod[i] == 'L':
            A_std[i, current_slack_idx] = 1.0
            current_slack_idx += 1

    b_std = b_mod
    return c_std, A_std, b_std

# --- Standard Heuristic Initial Point ---
def initial_point_standard_heuristic(A_std, b_std, c_std, verbose_debug=False):
    """
    A more standard heuristic for initial point in IPMs.
    Based on Nocedal & Wright, Chapter 14, Algorithm 14.2.
    Ensures x0, s0 > 0 and aims for a reasonable initial mu.
    """
    if verbose_debug: print("<<<< USING initial_point_standard_heuristic >>>>")
    m, n = A_std.shape
    e_n = np.ones((n, 1))
    b_flat = b_std.flatten()
    c_flat = c_std.flatten()

    x_tilde, y_tilde, s_tilde = np.ones((n, 1)) * 1e-1, np.zeros((m, 1)), np.ones((n, 1)) * 1e-1  # Fallback defaults

    try:
        if n > 0:  # Solve for x_tilde: Ax ~ b
            if m > 0:  # Normal equations if A has rows
                # x_tilde = A.T @ solve(A @ A.T, b) - More robust for rank deficiency but A@A.T can be dense
                # For now, using lstsq directly on A
                x_tilde_flat, res_x, rank_A, sing_vals_A = np.linalg.lstsq(
                    A_std.toarray() if sp.issparse(A_std) else A_std, b_flat, rcond=None)  # lstsq prefers dense
                x_tilde = x_tilde_flat.reshape(-1, 1)
                if verbose_debug: print(
                    f"  IP: x_tilde norm={np.linalg.norm(x_tilde):.1e}, min={np.min(x_tilde):.1e}, max={np.max(x_tilde):.1e}")
            elif m == 0 and n > 0:  # Unconstrained in primal variables (least norm x)
                x_tilde_flat, _, _, _ = np.linalg.lstsq(np.eye(n), np.zeros(n), rcond=None)  # x_tilde = 0
                x_tilde = x_tilde_flat.reshape(-1, 1)

        if m > 0 and n > 0:  # Solve for y_tilde, s_tilde: A.T y + s ~ c
            AT = A_std.T.toarray() if sp.issparse(A_std.T) else A_std.T
            # One way: solve A.T y_temp = c (if n >= m)
            # Then s_tilde = c - A.T y_temp
            # Or solve min ||s_tilde|| s.t. A.T y + s_tilde = c
            # Let's try solving for y that minimizes ||A.T y - c||
            # This is lstsq(A.T, c)
            y_tilde_flat, res_s, rank_AT, sing_vals_AT = np.linalg.lstsq(AT, c_flat, rcond=None)
            y_tilde = y_tilde_flat.reshape(-1, 1)
            s_tilde = c_flat.reshape(-1, 1) - AT @ y_tilde
            if verbose_debug:
                print(
                    f"  IP: y_tilde norm={np.linalg.norm(y_tilde):.1e}, min={np.min(y_tilde):.1e}, max={np.max(y_tilde):.1e}")
                print(
                    f"  IP: s_tilde (from y_tilde) norm={np.linalg.norm(s_tilde):.1e}, min={np.min(s_tilde):.1e}, max={np.max(s_tilde):.1e}")
        elif n > 0:  # m == 0
            s_tilde = c_flat.reshape(-1, 1)  # y_tilde is empty or zero
            if verbose_debug: print(f"  IP: s_tilde (m=0) norm={np.linalg.norm(s_tilde):.1e}")


    except np.linalg.LinAlgError as e:
        if verbose_debug: print(f"  IP: LinAlgError in initial point lstsq: {e}. Using robust fallbacks.")
        x_tilde = np.ones((n, 1)) * 10.0  # Fallback to ensure some scale
        y_tilde = np.zeros((m, 1))
        s_tilde = np.ones((n, 1)) * 10.0
    except ValueError as ve:  # Catches issues like empty arrays for lstsq
        if verbose_debug: print(f"  IP: ValueError in initial point lstsq: {ve}. Using robust fallbacks.")
        if n > 0:
            x_tilde = np.ones((n, 1)) * 10.0
        else:
            x_tilde = np.zeros((0, 1))
        if m > 0:
            y_tilde = np.zeros((m, 1))
        else:
            y_tilde = np.zeros((0, 1))
        if n > 0:
            s_tilde = np.ones((n, 1)) * 10.0
        else:
            s_tilde = np.zeros((0, 1))

    # Ensure x0 and s0 are strictly positive
    # Calculate delta_x_hat and delta_s_hat (N&W p. 409)
    delta_x_hat_val = 0.0
    if n > 0:
        min_x_val = np.min(x_tilde) if x_tilde.size > 0 else 0
        delta_x_hat_val = max(0, -1.5 * min_x_val)  # Ensure shift is non-negative
        if x_tilde.size > 0 and (min_x_val + delta_x_hat_val) < 1e-4:  # If still too small after shift
            delta_x_hat_val += 1e-2  # Add a bit more

    x0 = x_tilde + delta_x_hat_val * e_n

    delta_s_hat_val = 0.0
    if n > 0:
        min_s_val = np.min(s_tilde) if s_tilde.size > 0 else 0
        delta_s_hat_val = max(0, -1.5 * min_s_val)
        if s_tilde.size > 0 and (min_s_val + delta_s_hat_val) < 1e-4:
            delta_s_hat_val += 1e-2

    s0 = s_tilde + delta_s_hat_val * e_n

    # Further adjustment to balance x_i*s_i (N&W p. 410)
    if n > 0:
        x0_s0_dot = (x0.T @ s0)[0, 0]
        s0_sq_norm = (s0.T @ s0)[0, 0]
        x0_sq_norm = (x0.T @ x0)[0, 0]

        delta_x_bar_val = delta_x_hat_val
        if s0_sq_norm > 1e-8:  # Avoid division by zero
            delta_x_bar_val += 0.5 * x0_s0_dot / s0_sq_norm

        delta_s_bar_val = delta_s_hat_val
        if x0_sq_norm > 1e-8:
            delta_s_bar_val += 0.5 * x0_s0_dot / x0_sq_norm

        x0 = x_tilde + delta_x_bar_val * e_n  # Add the full delta_x_bar to original x_tilde
        s0 = s_tilde + delta_s_bar_val * e_n  # Add the full delta_s_bar to original s_tilde

        # Final floor to ensure they are not pathologically small
        x0 = np.maximum(x0, 1e-2)
        s0 = np.maximum(s0, 1e-2)

    y0 = y_tilde  # y_tilde should be (m,1)

    # Final safety/reshape
    if x0.shape != (n, 1): x0 = np.ones((n, 1))
    if y0.shape != (m, 1): y0 = np.zeros((m, 1))
    if s0.shape != (n, 1): s0 = np.ones((n, 1))

    if verbose_debug and n > 0:
        mu0_calc = (x0.T @ s0)[0, 0] / n if n > 0 else 0
        print(f"  IP: Final x0 min={np.min(x0):.1e}, max={np.max(x0):.1e}, norm={np.linalg.norm(x0):.1e}")
        print(f"  IP: Final s0 min={np.min(s0):.1e}, max={np.max(s0):.1e}, norm={np.linalg.norm(s0):.1e}")
        print(f"  IP: Resulting mu_0: {mu0_calc:.2e}")
    if verbose_debug: print("<<<< EXITING initial_point_standard_heuristic >>>>")
    return x0, y0, s0

def step_size_common(x, s, dx, ds, eta=0.9995):
    x_flat = x.flatten()
    s_flat = s.flatten()
    dx_flat = dx.flatten()
    ds_flat = ds.flatten()

    alphax = 1.0
    if dx_flat.size > 0:  # Only if dx exists
        blocking_indices_x = dx_flat < -1e-12
        if np.any(blocking_indices_x):
            ratios_x = -x_flat[blocking_indices_x] / dx_flat[blocking_indices_x]
            if ratios_x.size > 0:
                valid_ratios_x = ratios_x[~np.isnan(ratios_x) & (ratios_x >= -1e-12)]
                if valid_ratios_x.size > 0:
                    min_ratio_x = np.min(valid_ratios_x)
                    alphax = min(1.0, eta * min_ratio_x)
                # If no valid positive ratios, alphax remains 1.0

    alphas = 1.0
    if ds_flat.size > 0:  # Only if ds exists
        blocking_indices_s = ds_flat < -1e-12
        if np.any(blocking_indices_s):
            ratios_s = -s_flat[blocking_indices_s] / ds_flat[blocking_indices_s]
            if ratios_s.size > 0:
                valid_ratios_s = ratios_s[~np.isnan(ratios_s) & (ratios_s >= -1e-12)]
                if valid_ratios_s.size > 0:
                    min_ratio_s = np.min(valid_ratios_s)
                    alphas = min(1.0, eta * min_ratio_s)
                # If no valid positive ratios, alphas remains 1.0
    return alphax, alphas

